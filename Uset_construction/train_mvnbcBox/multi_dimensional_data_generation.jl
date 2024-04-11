import Plots
using Random
using LinearAlgebra

function generate_point_cloud(
    n;                 # number of points
    dimensions = 2,    # number of dimensions
    scaling_factors = ones(dimensions),  # scaling factors in each dimension
    rotation_matrix = Matrix(I, dimensions, dimensions),     # rotation matrix
    translation_vector = zeros(dimensions),
    random_seed = 1,
)
    rng = Random.MersenneTwister(random_seed)
    P = randn(rng, Float64, dimensions, n)
    S = rotation_matrix * P .* scaling_factors .+ translation_vector
    return S
end

function generate_clusterable_clouds(
    no_of_points_per_cluster,
    no_of_clusters,
    dimension,
    batch_number = 1)
    

    i = 0
    clusters = no_of_clusters

    S = zeros(Float64, dimension, 0)
    while i < clusters
        i += 1

        n = no_of_points_per_cluster
        rndnmbr = i + batch_number
        rng1 = Random.MersenneTwister(rndnmbr)
        rng2 = Random.MersenneTwister(rndnmbr + 1)
        rng3 = Random.MersenneTwister(rndnmbr + 2)

        s = generate_point_cloud(
            n,
            dimensions=dimension,
            scaling_factors = 0.5 .+ randn(rng1, Float64, dimension),
            rotation_matrix = randn(rng2, Float64, dimension, dimension),
            translation_vector = 6 * randn(rng3, Float64, dimension),
            random_seed = i
        )
        S = hcat(S, s)
    end
    return S[shuffle(1:size(S, 1)), :]
end


function generate_noise(X, number_of_noise_points)
    d, n = size(X)
    S = zeros(Float64, d, n + number_of_noise_points)
    S[:, 1:n] = X
    for j = 1:d
        S[j, n + 1:end] = rand(minimum(X[j, :]):maximum(X[j, :]), number_of_noise_points)
    end

    return S[shuffle(1:size(S, 1)), :]
end
