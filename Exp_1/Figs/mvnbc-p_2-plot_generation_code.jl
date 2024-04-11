using Clustering
using PlotThemes
using DelimitedFiles
using DataFrames
using Plots
using CSV

include("../../Uset_construction/train_mvnbc/main.jl")
# include("../../Uset_construction/train_mvnbc/DMDFunctins2.jl")
theme(:bright)

data_dir = "Exp_1/Data/2d/data_1500-clstrs_3-seed_15.txt"
rndidx = 15
figs_path = "Exp_1/Figs/2d/"


# if !isdir("your_folder_name")
#     mkdir("your_folder_name")
# end

X = copy(readdlm(data_dir, ',', Float64)')

p = scatter(X[1, :], X[2, :], label = "Datapoints", markershape = :x, color=:gray, markersize=3, markerstrokewidth=0.5)
plot!(xlims = [minimum(X[1, :]) - 5, maximum(X[1, :]) + 5], ylims = [minimum(X[2, :]) - 5, maximum(X[2, :]) + 5])
plot!(legend=:best, aspect_ratio=:equal)
savefig(figs_path * "scatter-1500_3_15.pdf")

n_clusters = 3

e = 0.1
max_iteration = 50

# Create an empty DataFrame
df = DataFrame(name = String[], time = Float64[], vol = Float64[])

#random
start_time = time()
clusters, volSum = DMDFunctions.initialize_clusters_randomly(X, n_clusters, e, rndidx)

dist, outliers = calculateDistOutliers(X, clusters, e)
    
L = LocalSearchFunctions.update_L(dist, clusters.labels, outliers)

clusters = MVCEs.minimum_volume_ell_labeled(X, L)

clusters = LocalSearchFunctions.local_search_ell(X, clusters, e; max_iteration)

dist, outliers = calculateDistOutliers(X, clusters, e)
end_time = time()

push!(df, (name = "Random Initialization", time = end_time - start_time, vol = sum(clusters.vols)))

MVPlots.plot_MVCE(X, clusters)
plot!(xlims = [minimum(X[1, :]) - 5, maximum(X[1, :]) + 5], ylims = [minimum(X[2, :]) - 5, maximum(X[2, :]) + 5])
p1 = plot!(legend=:best)

savefig(figs_path * "/Random_$(max_iteration)-data_15003$(rndidx)-nClusters_$(n_clusters)-e_$(e).pdf")


#random2
start_time = time()
dist, outliers = calculateDistOutliers(X, clusters, e)
    
L = LocalSearchFunctions.update_L(dist, clusters.labels, outliers)

clusters = MVCEs.minimum_volume_ell_labeled(X, L)

clusters = LocalSearchFunctions.local_search_ell(X, clusters, e; max_iteration)

dist, outliers = calculateDistOutliers(X, clusters, e)
end_time = time()

push!(df, (name = "Random Initialization * 2", time = end_time - start_time, vol = sum(clusters.vols)))

MVPlots.plot_MVCE(X, clusters)
plot!(xlims = [minimum(X[1, :]) - 5, maximum(X[1, :]) + 5], ylims = [minimum(X[2, :]) - 5, maximum(X[2, :]) + 5])
p1 = plot!(legend=:best)

savefig(figs_path * "/Random_$(max_iteration*2)-data_15003$(rndidx)-nClusters_$(n_clusters)-e_$(e).pdf")



#K-means
start_time = time()
clusters, volSum = DMDFunctions.initialize_clusters(X, n_clusters, e)

dist = DMDFunctions.dist_clusters(X, clusters)

outliers = DMDFunctions.outlier_detect(dist, e)

clusters.labels[outliers, :] .= false

clusters = MVCEs.minimum_volume_ell_labeled(X, clusters.labels)
end_time = time()

push!(df, (name = "K-means", time = end_time - start_time, vol = sum(clusters.vols)))

MVPlots.plot_MVCE(X, clusters)
plot!(xlims = [minimum(X[1, :]) - 5, maximum(X[1, :]) + 5], ylims = [minimum(X[2, :]) - 5, maximum(X[2, :]) + 5])
p1 = plot!(legend=:best)

savefig(figs_path * "/Kmeans-data_15003$(rndidx)-nClusters_$(n_clusters)-e_$(e).pdf")

#K-means+LocalSearch
start_time = time()

clusters, volSum = DMDFunctions.initialize_clusters(X, n_clusters, e)

dist, outliers = calculateDistOutliers(X, clusters, e)
    
L = LocalSearchFunctions.update_L(dist, clusters.labels, outliers)

clusters = MVCEs.minimum_volume_ell_labeled(X, L)

clusters = LocalSearchFunctions.local_search_ell(X, clusters, e; max_iteration)

dist, outliers = calculateDistOutliers(X, clusters, e)
end_time = time()

push!(df, (name = "K-means Initialization", time = end_time - start_time, vol = sum(clusters.vols)))

MVPlots.plot_MVCE(X, clusters)
plot!(xlims = [minimum(X[1, :]) - 5, maximum(X[1, :]) + 5], ylims = [minimum(X[2, :]) - 5, maximum(X[2, :]) + 5])

p2 = plot!(legend=:best)

savefig(figs_path * "/Kmeans+MVNBC_$(max_iteration)-data_15003$(rndidx)-nClusters_$(n_clusters)-e_$(e).pdf")


#GMM
start_time = time()

clusters = DMDFunctions.DMDgmm(X, n_clusters, e)
end_time = time()

push!(df, (name = "GMM", time = end_time - start_time, vol = sum(clusters.vols)))

MVPlots.plot_MVCE(X, clusters)
plot!(xlims = [minimum(X[1, :]) - 5, maximum(X[1, :]) + 5], ylims = [minimum(X[2, :]) - 5, maximum(X[2, :]) + 5])

p4 = plot!(legend=:best)
savefig(figs_path * "/GMM-data15003$(rndidx)-nClusters_$(n_clusters)-e_$(e).pdf")


#GMM+LocalSearch
start_time = time()

clusters, outliers = norm2BasedUncertaintySet(X, n_clusters, e)
end_time = time()

push!(df, (name = "GMM Initialization", time = end_time - start_time, vol = sum(clusters.vols)))

MVPlots.plot_MVCE(X, clusters)
plot!(xlims = [minimum(X[1, :]) - 5, maximum(X[1, :]) + 5], ylims = [minimum(X[2, :]) - 5, maximum(X[2, :]) + 5])

p5 = plot!(legend=:best)
savefig(figs_path * "/GMM+MVNBC_$(max_iteration)-data15003$(rndidx)-n_Clusters$(n_clusters)-e_$(e).pdf")

# Save the DataFrame to a csv file
CSV.write(figs_path * "results_$(max_iteration)-data15003$(rndidx)-n_Clusters$(n_clusters)-e_$(e).csv", df)

#sums1:random - sums2:random200 - sums3:K-means+LocalSearch - sums5:GMM+LocalSearch


# For trend of total volume you should store these volumes as a vector from each iteration then you can use following for the trend
# Convert the vectors to strings
# str1 = join(sums1, ",")
# str2 = join(sums2, ",")
# str3 = join(sums3, ",")
# str5 = join(sums5, ",")

# # Write the strings to a CSV file
# open(figs_path * "results-sums.csv", "w") do f
#     write(f, str1 * "\n")
#     write(f, str2 * "\n")
#     write(f, str3 * "\n")
#     write(f, str5 * "\n")
# end

# # df = DataFrame(column_name = sums2)

# # CSV.write(figs_path * "results-sums.csv", df)


# # Concatenate sums1 and sums2
# sums1_2 = vcat(sums1, sums2)

# # Create a new plot
# p = plot()

# # Add a line for the concatenated vector
# plot!(p, sums1_2, label="Random initialization", linewidth=2)

# # Add lines for the other methods
# plot!(p, sums3, label="K-means Initialization", linewidth=2)
# plot!(p, sums5, label="GMM Initialization", linewidth=2)

# # Add labels
# xlabel!(p, "Iterations")
# ylabel!(p, "Total volume")

# # Set the y-axis to a logarithmic scale
# yaxis!(p, :log)

# savefig(figs_path * "trend_data15003205$(i)_nClusters$(n_clusters)_e$(e).pdf")
