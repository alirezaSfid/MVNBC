using JuMP
using MosekTools
using Statistics
using DelimitedFiles
using CSV
function boxUncertaintySet(X, outlier_rate)

    n, d = size(X)

    M = maximum(X) - minimum(X)

    model = Model(Mosek.Optimizer)
    @variable(model, z[1:n], Bin)
    @variable(model, l[1:d])
    @variable(model, u[1:d])

    @objective(model, Min, sum(u - l))
    
    for i in 1:d
        for j in 1:n
            @constraint(model, l[i] - M * z[j] <= X[j, i])
            @constraint(model, X[j, i] <= u[i] + M * z[j])
        end
    end

    @constraint(model, sum(z) <= n * outlier_rate)
    optimize!(model)

    return value.(l), value.(u)
end

