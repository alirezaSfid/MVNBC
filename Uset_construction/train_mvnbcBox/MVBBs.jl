module MVBBs

using JuMP
using MosekTools
using LinearAlgebra
import MathOptInterface as MOI
using Plots



export minimum_volume_box, minimum_volume_box_weighted, minimum_volume_box_labeled, MVBCResult

# Define a constant for labeling the points for each cluster
const TOLERANCE = 1.0000001

# Define a struct to hold the result of MVBC
struct MVBCResult
    transformation_matrices::Vector{Matrix{Float64}}
    translation_vectors::Vector{Vector{Float64}}
    labels::Matrix{Bool}
    vols::Vector{Float64}
end


"""
    minimum_volume_box(X)

Find the minimum volume box that contains all points in X.

# Arguments
- `X`: a d x n matrix, where each column is a point in d-dimensional space.

# Returns
- `H`: a d x d matrix representing the box.
- `x_hat`: a d-dimensional vector representing the center of the box.
- `(2 ^ d) / det(H)`: the volume of the box.
"""
function minimum_volume_box(X)
    d, n = size(X)
    model = Model(MosekTools.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-7)
    set_silent(model)

    @variable(model, t)
    @variable(model, H[1:d, 1:d], Symmetric)
    @variable(model, x_hat[1:d])

    @objective(model, Max, 1 * t + 0)

    @constraint(model, [t; triangle_vec(H)] in MOI.RootDetConeTriangle(d))
    for i in 1:n
        @constraint(model, [1; H * X[:, i] - x_hat] in MOI.NormInfinityCone(d+1))
    end

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        # p = scatter(X[1, :], X[2, :], label = "Datapoints", color = :gray)
        # display(p)
        println("Optimization did not reach optimal solution. Status: $(termination_status(model)), Message: $(raw_status(model))")
    end

    H = value.(H)
    x_hat = H \ value.(x_hat)

    return H, x_hat, (2 ^ d) / det(H)
end

"""
    minimum_volume_box_weighted(X, γ, r=1)

Find the minimum volume box that contains all points in X, with different coefficients for each point constraint.

# Arguments
- `X`: a d x n matrix, where each column is a point in d-dimensional space.
- `γ`: a vector of coefficients for each point constraint.
- `r`: the rhs value for the constraint 𝐸(𝐷,𝑐)={𝑥:(𝑥−𝑐)⊤𝐷(𝑥−𝑐)≤1}.

# Returns
- `H`: a d x d matrix representing the box.
- `x_hat`: a d-dimensional vector representing the center of the box.
- `(2 ^ d) / det(H)`: the volume of the box.
"""
function minimum_volume_box_weighted(X, γ; r=1)
    d, n = size(X)
    model = Model(MosekTools.Optimizer)
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-10)
    set_silent(model)

    @variable(model, t)
    @variable(model, H[1:d, 1:d], Symmetric)
    @variable(model, x_hat[1:d])

    @objective(model, Max, 1 * t + 0)

    @constraint(model, [t; triangle_vec(H)] in MOI.RootDetConeTriangle(d))
    for i in 1:n
        @constraint(model, [r / γ[i]; H * X[:, i] - x_hat] in MOI.NormInfinityCone(d+1))
    end

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        println("Optimization did not reach optimal solution. Status: $(termination_status(model)), Message: $(raw_status(model))")
    end

    H = value.(H)
    x_hat = H \ value.(x_hat)

    return H, x_hat, (2 ^ d) / det(H)
end


"""
    minimum_volume_box_labeled(X, L)

Find the minimum volume box for each label in L that contains all corresponding points in X.

# Arguments
- `X`: a d x n matrix, where each column is a point in d-dimensional space.
- `L`: a n x K matrix, where each column is a label for the points in X.

# Returns
- `MVBCResult(Hs, x_hats, L)`: a MVBCResult object containing the matrices Hs, the vectors x_hats, and the labels L.
- `vols`: a vector of volumes for each box.
"""
function minimum_volume_box_labeled(X, L)
    n, K = size(L)
    d = size(X, 1)
    
    Hs = Matrix{Float64}[]
    x_hats = Vector{Float64}[]
    vols = []
    labels = falses(n, K)

    for k in 1:K
        H, x_hat, vol = minimum_volume_box(X[:, L[:, k]])

        for i in 1:n
            if !labels[i, k]
                labels[i, k] = norm(H * (X[:, i] - x_hat), Inf) <= TOLERANCE
            end
        end

        push!(Hs, H)
        push!(x_hats, x_hat)
        push!(vols, vol)
    end

    return MVBCResult(Hs, x_hats, L, vols)
end

end