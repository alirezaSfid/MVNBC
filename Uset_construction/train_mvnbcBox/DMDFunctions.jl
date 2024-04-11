module DMDFunctions

include("MVBBs.jl")
import .MVBBs

using LinearAlgebra
using Clustering
using Random
using GaussianMixtures
using Distributions

export DMD_box
export DMDgmm

# export initialize_clusters, expectation_step, maximization_step, distL_update, DMD_box, MVBCResult, dist_clusters

const MAX_ITERATION_MULTIPLIER = 5
const TERMINATION_INDEX_THRESHOLD = 10
const TOLERANCE = 1e-5

function DMD_box(X::Matrix, n_clusters::Int64, e::Float64; max_iteration::Int64=10)
    clusters = initialize_clusters(X, n_clusters)

    K = n_clusters
    n = size(X, 2)
    bestClusters = clusters
    bestVolTotal = Inf

    for iteration in 1:max_iteration
        iter = 0
        termination_index = 0
        last_log_p = 0

        gamma_nk = falses(n, K)


        while termination_index <= TERMINATION_INDEX_THRESHOLD && iter <= max_iteration * MAX_ITERATION_MULTIPLIER
            iter += 1

            gamma_nk, p_nk = expectation_step(X, clusters)

            r = 1

            clusters = maximization_step(X, clusters, gamma_nk, e, r)

            
    
            log_p = sum(log.(p_nk))
            
            if abs(log_p - last_log_p) <= MIN_GAMMA
                termination_index +=1
            else
                # termination_index = 0
                last_log_p = log_p
            end
    
        end

        labels = falses(n, K)
        [labels[i, argmax(gamma_nk[i, :])] = true for i in 1:n]
        
        dist = dist_clusters(X, clusters)
        outliers = outlier_detect(dist, e)
        labels[outliers, :] .= false
    
        clusters = MVBBs.minimum_volume_box_labeled(X, labels)
    
        if sum(clusters.vols)/sum(sum(clusters.labels, dims=2) .> 0) <= bestVolTotal/sum(sum(bestClusters.labels, dims=2) .> 0)
            bestVolTotal = sum(clusters.vols)
            bestClusters = clusters
        end
    end

    return bestClusters

end


# This function takes points, and the number of clusters and return a initial solution for the MVBC
# clustering result will be stored as a ::MVBCResult
function initialize_clusters(X, n_clusters)
    d, n = size(X)
    R = kmeans(X, n_clusters)
    # R = dbscan(X', .25, min_neighbors=30, min_cluster_size=10)

    Hs = Matrix{Float64}[]
    x_hats = Vector{Float64}[]
    vols = []
    L = falses(n, n_clusters)

    for k in 1:n_clusters
        x = X[:, R.assignments .== k]

        H, x_hat, vol = MVBBs.minimum_volume_box(x[:, randperm(size(x, 2))[1:Int(d + 1)]])
        # H, x_hat, vol = MVBBs.minimum_volume_box(X[:, randperm(n)[1:Int(d+1)]])

        for i in 1:n
            L[i, k] = norm(H * (X[:, i] - x_hat), Inf) <= NORM_THRESHOLD
        end

        push!(Hs, H)
        push!(x_hats, x_hat)
        push!(vols, vol)
    end
    
    return MVBBs.MVBCResult(Hs, x_hats, L, vols)
end


function cluster_density(clusters)
    L =  clusters.labels
    vols = clusters.vols
    n, K = size(L)
    dens = zeros(K)
    for k = 1:K
        H = clusters.transformation_matrices[k]
        d = size(vols, 1)
        dens[k] = sum(L[:, k]) / vols[k]
    end

    return dens
end

# This function calculates the Norm_2 distance from each data point to each cluster center.
function dist_clusters(X, clusters)
    # L is a matrix where each row corresponds to a data point and each column corresponds to a cluster.
    # The element at the i-th row and k-th column is true if the i-th data point is assigned to the k-th cluster, and false otherwise.
    L = clusters.labels

    # n is the number of data points, and K is the number of clusters.
    n, K = size(L)
    d = size(X, 1)

    # dist is a matrix where each row corresponds to a data point and each column corresponds to a cluster.
    # The element at the i-th row and k-th column will be the Norm_2 distance from the i-th data point to the k-th cluster center.
    dist = zeros(n, K)

    # Loop over each cluster.
    for k in 1:K
        # H is the transformation matrix for the k-th cluster.
        H = clusters.transformation_matrices[k]

        # x_hat is the translation vector for the k-th cluster.
        x_hat = clusters.translation_vectors[k]

        # Loop over each data point.
        for i in 1:n
            # Calculate the Norm_Inf distance from the i-th data point to the k-th cluster center.
            dist[i, k] = norm(H * (X[:, i] - x_hat), Inf)
        end
    end

    # Return the matrix of distances.
    return dist
end

# it takes X and clusters and return gammas which show the responsiveness of each point to each cluster, based on its distance from the border of the cluster
function expectation_step(X, clusters)
    d, n =size(X)
    L =  clusters.labels
    K = size(L, 2)
    gamma_nk = zeros(n, K)
    
    density_k = cluster_density(clusters)
    
    dist = dist_clusters(X, clusters)
    
    totals = zeros(n)
    for k in 1:K
        for i in 1:n
            gamma_nk[i, k] = density_k[k] * exp(-0.5 * (dist[i, k]))
            # gamma_nk[i, k] = exp(-(dist[i, k])^2)
        end
    end
    
    totals = sum(gamma_nk, dims=2)
    gamma_nk ./= totals
    gamma_nk = [gamma = max(gamma, MIN_GAMMA * 10) for gamma in gamma_nk]    
    
    return gamma_nk, totals
end

# This function identifies the indices of the `e` proportion of points 
# that have the greatest minimum distance to any cluster.
function outlier_detect(dist, e::Float64)
    # n is the number of data points.
    n = size(dist, 1)
    
    # num_outliers is the number of outliers to detect, 
    # calculated as a proportion `e` of the total number of data points.
    num_outliers = Int(round(e * n))
    
    # dist_min is a vector that contains the minimum distance 
    # from each data point to any cluster.
    dist_min = [minimum(dist[row_idx, :]) for row_idx in 1:n]
    
    # outlier_indices is a vector that contains the indices of the `num_outliers` data points 
    # that have the greatest minimum distance to any cluster.
    outlier_indices = partialsortperm(dist_min, 1:num_outliers, rev=true)
    
    # Return the indices of the outliers.
    return outlier_indices
end

function outlier_gamma_update(gamma_nk, outliers)
    gamma_nk[outliers, :] .= MIN_GAMMA
    return gamma_nk
end


function maximization_step(X, clusters, gamma_nk, e, r=1)
    d, n =size(X)
    K = size(gamma_nk, 2)
    L =  falses(n, K)
    dist = dist_clusters(X, clusters)
        
    outliers = outlier_detect(dist, e)
    gamma_nk = outlier_gamma_update(gamma_nk, outliers)
    
    Hs = Matrix{Float64}[]
    x_hats = Vector{Float64}[]
    vols = []
    for k in 1:K
        H, x_hat, vol = MVBBs.minimum_volume_box_weighted(X, gamma_nk[:, k])
        for i in 1:n
            L[i, k] = norm(H * (X[:, i] - x_hat), Inf) <= NORM_THRESHOLD
        end
    
        push!(Hs, H)
        push!(x_hats, x_hat)
        push!(vols, vol)
    end
    
    return MVBBs.MVBCResult(Hs, x_hats, L, vols)
end

function DMDgmm(X::Matrix, n_clusters::Int64, e)

    X_transposed = Matrix(X')

    n, d = size(X_transposed)
    gmm = GMM(n_clusters, X_transposed; method=:kmeans, kind=:full)
    posteriors = gmmposterior(gmm, X_transposed)
    labels = falses(n, n_clusters)
    for i in 1:n
        labels[i, argmax(posteriors[1][i, :])] = true
    end

    clusters = MVBBs.minimum_volume_box_labeled(X, labels)

    dist = dist_clusters(X, clusters)

    outliers = outlier_detect(dist, e)

    clusters.labels[outliers, :] .= false

    clusters = MVBBs.minimum_volume_box_labeled(X, clusters.labels)

    L_shrinked = copy(clusters.labels)

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

    clusters = MVBBs.minimum_volume_box_labeled(X, L_shrinked)


    return clusters
    
end


end