include("MVCEs.jl")
include("DMDFunctions.jl")
include("LocalSearchFunctions.jl") 
include("MVPlots.jl") 
include("Outputs.jl")

const DMD_MAX_ITERATION = 10

# This function calculates the distances from each data point to each cluster center 
# and identifies the outliers.
function calculateDistOutliers(X, clusters, e)
    # dist is a matrix where each row corresponds to a data point and each column corresponds to a cluster.
    # The element at the i-th row and k-th column is the distance from the i-th data point to the k-th cluster center.
    dist = DMDFunctions.dist_clusters(X, clusters)
    
    # outliers is a vector that contains the indices of the data points that are considered outliers.
    # A data point is considered an outlier if it is among the `e` proportion of data points 
    # that have the greatest minimum distance to any cluster.
    outliers = DMDFunctions.outlier_detect(dist, e)
    
    # Return the matrix of distances and the indices of the outliers.
    return dist, outliers
end

function norm2BasedUncertaintySet(X::AbstractArray, n_clusters::Int64, e::Float64, max_iteration::Int64 = 100)
    # clusters = DMDFunctions.DMD_ellipsoid(X, n_clusters, e; max_iteration = DMD_MAX_ITERATION)

    clusters = DMDFunctions.DMDgmm(X, n_clusters, e)

    dist, outliers = calculateDistOutliers(X, clusters, e)
    
    L = LocalSearchFunctions.update_L(dist, clusters.labels, outliers)
    
    clusters = MVCEs.minimum_volume_ell_labeled(X, L)
    
    clusters = LocalSearchFunctions.local_search_ell(X, clusters, e; max_iteration)

    dist, outliers = calculateDistOutliers(X, clusters, e)

    return clusters, outliers
end

# n_clusters = 3
# e = 0.05
# max_iteration = 100

# clusters, outliers = norm2BasedUncertaintySet(X, n_clusters, e, max_iteration)

# MVPlots.plot_MVCE(X, clusters)
# MVPlots.plot_MVCE(X, clustersgmm)