module LocalSearchFunctions

include("MVCEs.jl")
include("DMDFunctions.jl")
include("Outputs.jl")
import .MVCEs, .DMDFunctions
using Clustering
using ProgressMeter

export local_search_ell, find_closest_to_one_indices

const TOLERANCE = 1e-5



function local_search_ell(X::Matrix, clusters, e::Float64; max_iteration::Int64 = 100)

    d, n = size(X)
    K = size(clusters.labels, 2)
    bestClusters = clusters
    
    Δ = max_iteration / 10

    iteration = 1

    n_candidatesMax = compute_dist_clusters(X, clusters, e)

    while n_candidatesMax / 2 ^ (iteration - 1) > K * max_iteration / Δ
        n_candidatesInit = n_candidatesMax / 2 ^ (iteration - 1)

        no_change_counter = 0

        
        for iter in 1:max_iteration

            clusters = manage_overlap_points(X, clusters)
        
            if sum(clusters.vols) < sum(bestClusters.vols)
                bestClusters = clusters
            end
            
            n_candidates = Int(ceil((1 - (iter-1) % Δ / Δ) * n_candidatesInit))
        
            clusters = manage_boundary_points(X, clusters, n_candidates, e)
        
            if sum(clusters.vols) < sum(bestClusters.vols)
                bestClusters = clusters
                no_change_counter = 0
            else
                no_change_counter += 1
                if no_change_counter >= Δ
                    break
                end
            end
                
        end

        Outputs.printStatus(clusters, bestClusters, n_candidatesInit, iteration)

        iteration += 1

    end
    
    return bestClusters
    
end


function popin_candidates(distk::Vector, L, k)
    popin = []
    i = 0
    sorted_indices = sortperm(distk)
    while length(unique(popin)) < k
        i += 1
        if sum(L, dims=2)[sorted_indices[i]] == 0
            push!(popin, sorted_indices[i])
        end
    end

    return popin
end


# This function updates the cluster assignments based on the distances 
# from each data point to each cluster center and the indices of the outliers.
function update_L(dist, L_in, outliers)
    # n is the number of data points, and K is the number of clusters.
    n, K = size(dist)
    
    # L_out is a matrix where each row corresponds to a data point and each column corresponds to a cluster.
    # The element at the i-th row and k-th column will be true if the i-th data point is assigned to the k-th cluster, and false otherwise.
    L_out = falses(n, K)
    
    # outliers_set is a set that contains the indices of the outliers.
    # Using a set for this allows for efficient membership checks.
    outliers_set = Set(outliers)
    
    # Loop over each data point.
    for i in 1:n
        # If the i-th data point is an outlier, it is not assigned to any cluster.
        if i in outliers_set
            L_out[i, :] .= false
        # Otherwise, the i-th data point is assigned to the cluster to which it has the smallest distance.
        else
            L_out[i, argmin(dist[i, :])] = true
        end
    end
    
    # Return the updated cluster assignments.
    return L_out
end

function manage_overlap_points(X, clusters)
    L = clusters.labels
    n, n_clusters = size(L)
    d = size(X, 1)

    L_shrinked1 = copy(L)
    L_shrinked2 = copy(L)

    for j = 1:n_clusters - 1
        for k = j+1:n_clusters
            overlap_indices = findall(x -> x[j] && x[k], eachrow(L))
            L_shrinked1[overlap_indices, k] .= false
            L_shrinked2[overlap_indices, j] .= false

            for shrinkedL in (L_shrinked1, L_shrinked2)
                cluster_sum = sum(shrinkedL, dims=1)
                if any(cluster_sum .< d + 1)
                    i = argmax(cluster_sum)[2]
                    R = kmeans(X[:, shrinkedL[:, i]], 2)
                    iter = 0
                    for point = 1:n
                        if shrinkedL[point, i]
                            iter += 1
                            if R.assignments[iter] == 2
                                shrinkedL[point, i] = false
                                shrinkedL[point, [j, k][shrinkedL === L_shrinked1 ? 2 : 1]] = true
                            end
                        end
                    end
                end
            end

            shrinkedClusters1 = MVCEs.minimum_volume_ell_labeled(X, L_shrinked1)
            Ovals1 = sum(shrinkedClusters1.vols)

            shrinkedClusters2 = MVCEs.minimum_volume_ell_labeled(X, L_shrinked2)
            Ovals2 = sum(shrinkedClusters2.vols)

            clusters = Ovals1 <= Ovals2 ? shrinkedClusters1 : shrinkedClusters2
        end
    end

    return clusters
end


function compute_dist_clusters(X, clusters, e)
    # Get the number of points and clusters
    d, n = size(X)
    K = length(clusters.vols)
    L_shrinked = copy(clusters.labels)
    cluster_sum = sum(L_shrinked, dims=1)
    for k in 1:K
        if cluster_sum[k] < d + 1
            i = argmax(cluster_sum)[2]
            R = kmeans(X[:, L_shrinked[:, i]], 2)
            iter = 0
            for point = 1:n
                if L_shrinked[point, i]
                    iter += 1
                    if R.assignments[iter] == 2
                        L_shrinked[point, i] = false
                        L_shrinked[point, k] = true
                    end
                end
            end
        end
    end

    clusters = MVCEs.minimum_volume_ell_labeled(X, L_shrinked)

    # Initialize a matrix to store the distances
    dists = DMDFunctions.dist_clusters(X, clusters)

    # Initialize an array to store the specific distance values for each cluster
    specific_dists = zeros(K)

    # For each cluster, find the specific distance value
    for k in 1:K
        # Get the distances of the points from this cluster
        cluster_dists = dists[clusters.labels[:, k], k]

        # Sort the distances
        sorted_dists = sort(cluster_dists)

        # Find the specific distance value such that only d+1 points have a smaller distance
        specific_dists[k] = sorted_dists[d+1]
    end

    specific_dist = maximum(specific_dists)

    # Update labels in one pass
    L = dists .<= specific_dist

    max_num_candidates = sum(sum(L, dims=2) .== 0) - e * n

    return max_num_candidates
end


function find_closest_to_one_indices(matrix, num_elements)

# Initialize an array to store distances and corresponding row and column indices
distances = Float64[]
row_indices = Int[]

# Get the dimensions of the matrix
n, K = size(matrix)

# Iterate through the matrix to find elements less than d and calculate distances
for row in 1:n
    for col in 1:K
        val = matrix[row, col]
        if val < 1 + TOLERANCE * 10
            distance = abs(val - 1)
            push!(distances, distance)
            push!(row_indices, row)
        end
    end
end

# Sort the distances and corresponding row and column indices based on distance
sorted_indices = sortperm(distances)

num_elements = minimum([num_elements, length(sorted_indices)])

# Return sorted row and column indices
return row_indices[sorted_indices[1: num_elements]]
end


function manage_boundary_points(X, clusters, n_candidates, e)

    L = clusters.labels
    n, n_clusters = size(L)
    d = size(X, 1)
    dist = DMDFunctions.dist_clusters(X, clusters)

    candidates = find_closest_to_one_indices(dist, n_candidates)
    L_shrinked = copy(L)
    L_shrinked[candidates, :] .= false

    cluster_sum = sum(L_shrinked, dims=1)
    for k in 1:n_clusters
        if cluster_sum[k] < d + 1
            i = argmax(cluster_sum)[2]
            R = kmeans(X[:, L_shrinked[:, i]], 2)
            iter = 0
            for point = 1:n
                if L_shrinked[point, i]
                    iter += 1
                    if R.assignments[iter] == 2
                        L_shrinked[point, i] = false
                        L_shrinked[point, k] = true
                    end
                end
            end
        end
    end

    bestClusters = clusters
    Totalt = Inf

    neededPoint = 1
    while neededPoint > 0

        shrinkedClusters = MVCEs.minimum_volume_ell_labeled(X, L_shrinked)

        dist = DMDFunctions.dist_clusters(X, shrinkedClusters)

        # How many points needed to add entirely
        neededPoint = sum(sum(L_shrinked, dims=2) .== 0) - e * n
        # How many points will be added this iteration based on pigeonhole principle
        addpoint = ceil(neededPoint / n_clusters)

        # Objective function of the whole clusters volumes
        Totalt = Inf

        for j in 1:n_clusters
            L_copy = copy(L_shrinked)
            candidates = popin_candidates(dist[:, j], L_copy, addpoint)
            L_copy[candidates, j] .= true
            
            clusters = MVCEs.minimum_volume_ell_labeled(X, L_copy)
            cluster_vols_sum = sum(clusters.vols)

            if cluster_vols_sum < Totalt
                Totalt = cluster_vols_sum
                bestClusters = clusters
            end
        end

        L_shrinked = bestClusters.labels

    end

    return bestClusters

end


end