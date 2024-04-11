module MVPlots

using Plots
using LinearAlgebra

export plot_MVBB, plot_outliers

function plot_MVBB(X, boxes)
    p = scatter(X[1, :], X[2, :], label = "Datapoints", markershape = :x)
    no_boxes = size(boxes.labels, 2)
    for k in 1:no_boxes
        H = boxes.transformation_matrices[k]
        x_hat = boxes.translation_vectors[k]
        x1 = (H[1, 2] - H[2, 2]) / det(H) + x_hat[1]
        y1 = (-H[1, 1] + H[2, 1]) / det(H) + x_hat[2]
        
        x2 = (H[1, 2] + H[2, 2]) / det(H) + x_hat[1]
        y2 = (-H[1, 1] - H[2, 1]) / det(H) + x_hat[2]
        
        x3 = (-H[1, 2] + H[2, 2]) / det(H) + x_hat[1]
        y3 = (H[1, 1] - H[2, 1]) / det(H) + x_hat[2]
        
        x4 = (-H[1, 2] - H[2, 2]) / det(H) + x_hat[1]
        y4 = (H[1, 1] + H[2, 1]) / det(H) + x_hat[2]
        
        shape = Shape([(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x1,y1)])
        p = plot!(p, shape; fillalpha=0, linewidth=2, linecolor=k, label = "cluster $k-MVB", legend=:bottomright, aspect_ratio=:equal)
    end
    return p
end

function plot_outliers(X, outliers; markershape=:circle, markersize=4, label="Outliers")
    scatter!(X[1, outliers], X[2, outliers], color=:yellow, markershape=:o, markersize=markersize, label=label, legend=:bottomright, aspect_ratio=:equal)
end


end