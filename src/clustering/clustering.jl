function cluster(data, alpha, mu0, sigma0, sigma_y, c_init, maxiter = 1000)
    # Dimensionality and number of data points
    D, N = size(data)

    # Result matrix
    res_assignments = Matrix{Int64}(undef, N, maxiter)

    # Precisions of priors on data/clusters
    # Precision of the full data
    precision_0 = inv(sigma0)

    # Prior precision of each cluster
    precision_C = inv(sigma_y)

    z = deepcopy(c_init)

    # Count occupancy of each cluster
    nk = [sum(z .== c) for c in 1:maximum(z)]
    n_clusters = length(nk)

    # Iterate maxiter times
    for iteration in 1:maxiter
        # Iterate over data points i
        for i in 1:N
            # Remove data point from its cluster
            c = z[i]
            nk[c] -= 1

            # If this cluster is now empty, delete it and update n_clusters
            if iszero(nk[c])
                # Move data in the last cluster to the now empty one
                last_inds::Vector{Int64} = findall(x -> x == n_clusters, z)
                nk[c] = nk[n_clusters]
                pop!(nk)
                @views z[last_inds] .= c
                n_clusters -= 1
            end

            # Vector of cluster probabilities including unoccupied cluster
            logp_clusters::Vector{Float64} = Vector{Float64}(undef, n_clusters + 1)

            for c_i::Int64 in 1:n_clusters

                # Find all data points in cluster c_i
                data_inds = findall(x -> x == c_i, z)
                # Sum all datapoints in cluster
                datasum = sum(@view(data[:, data_inds]), dims = 2)

                # Cluster precision posterior
                prec_posterior = @. precision_0 + nk[c_i] * precision_C
                σ_posterior = inv(prec_posterior)

                # compute posterior mean for cluster based on its current occupants
                mean_posterior = σ_posterior *
                                 (precision_C * datasum .+ precision_0 * mu0) |> vec

                dist = MvNormal(mean_posterior, σ_posterior .+ sigma_y)

                logp_clusters[c_i] = log(nk[c_i]) + logpdf(dist, data[:, i])
            end # Cluster loop

            # Compute probability of a new cluster
            logp_clusters[n_clusters + 1] = log(alpha) + logpdf(
                MvNormal(mu0, sigma0 .+ sigma_y), data[:, i])

            # Compute probabilities from log probs

            max_logp_clusters = maximum(logp_clusters)
            @. logp_clusters -= max_logp_clusters

            loc_probs = exp.(logp_clusters)
            loc_probs ./= sum(loc_probs)

            # Draw cluster for data point i
            z_new::Int64 = rand(Categorical(loc_probs))

            # If new cluster, create it. Otherwise update existing cluster
            if z_new == n_clusters + 1
                push!(nk, 0)
                n_clusters += 1
            end
            z[i] = z_new
            nk[z_new] += 1
        end # Data point loop
        res_assignments[:, iteration] = z
    end # Iteration loop
    return res_assignments
end

function cluster2(data, alpha, mu0, sigma0, sigma_y, c_init, maxiter = 1000)
    # Dimensionality and number of data points
    D, N = size(data)

    # Result matrix
    res_assignments = Matrix{Int64}(undef, N, maxiter)

    # Hyperparameter priors for component means

    lambda_prior = rand(
        MvNormal(mu0, sigma0)
    )

    r_prior = rand(
        Wishart(D, inv(sigma0))
    )

    # Prior on component means
    c_mean_prior = rand(MvNormal(lambda_prior, inv(r_prior)))

    # Hyperparameters for precision prior
    w = rand(Wishart(D, sigma0))

    # Precision prior
    c_precision_prior = rand(Wishart(D, inv(w)))

    # Precisions of priors on data/clusters
    # Precision of the full data
    precision_0 = inv(sigma0)

    # Prior precision of each cluster
    precision_C = inv(sigma_y)

    z = deepcopy(c_init)

    # Count occupancy of each cluster
    nk = [sum(z .== c) for c in 1:maximum(z)]
    n_clusters = length(nk)

    base_distr = MvNormal(mu0, sigma0 .+ sigma_y)

    # Initialize a D x D x N array for precisions for each cluster
    posterior_precisions = Array{Float64}(undef, D, D, N)

    # Initialize a D x N Array for posterior means of clusters
    posterior_means = Matrix{Float64}(undef, D, N)
    posterior_vars = Matrix{Float64}(undef, D, N)

    # Initialize a Vector for log probabilities of clusters
    logp_clusters = Vector{Float64}(undef, N)

    # Assign precisions and means from prior to initial clusters
    @views for c in z
        posterior_precisions[:, :, c] = rand(Wishart(beta, inv(w)))
        posterior_means[:, :, c] = rand(MvNormal(lambda, inv(r)))
        posterior_vars[:, :, c] = inv(posterior_precisions[:, :, c])
    end

    # Iterate maxiter times
    for iteration in 1:maxiter
        # Iterate over data points i
        for i in 1:N
            # Remove data point from its cluster
            c = z[i]
            nk[c] -= 1

            # If this cluster is now empty, delete it and update n_clusters
            if iszero(nk[c])
                # Move data in the last cluster to the now empty one
                last_inds::Vector{Int64} = findall(x -> x == n_clusters, z)
                nk[c] = nk[n_clusters]
                pop!(nk)
                z[last_inds] .= c

                # Update posteriors etc
                posterior_vars[:, :, c] = posterior_vars[:, :, n_clusters]
                posterior_precisions[:, :, c] = posterior_precisions[:, :, n_clusters]
                posterior_means[:, c] = posterior_means[:, n_clusters]

                # Bump n_clusters down
                n_clusters -= 1
            end

            for c_i::Int64 in 1:n_clusters

                # Find all data points in cluster c_i
                data_inds = findall(x -> x == c_i, z)
                # Sum all datapoints in cluster
                datasum = sum(@view(data[:, data_inds]), dims = 2)

                # Cluster precision posterior
                @views posterior_precisions[:, :, c_i] = @. precision_0 +
                                                            nk[c_i] * precision_C
                @views posterior_vars[:, :, c_i] = inv(posterior_precisions[:, :, c_i])

                # compute posterior mean for cluster based on its current occupants
                posterior_means[:, c_i] = posterior_vars[:, :, c_i] *
                                          (precision_C * datasum .+ precision_0 * mu0) |>
                                          vec

                # Distribution of current cluster
                @views dist = MvNormal(
                    posterior_means[:, c_i], posterior_vars[:, :, c_i] .+ sigma_y)

                logp_clusters[c_i] = log(nk[c_i]) + logpdf(dist, data[:, i])
            end

            # Compute probability of a new cluster
            logp_clusters[n_clusters + 1] = log(alpha) + logpdf(base_distr, data[:, i])

            # Compute probabilities from log probs

            max_logp_clusters = maximum(@view(logp_clusters[begin:(n_clusters + 1)]))
            @. logp_clusters -= max_logp_clusters

            loc_probs = exp.(@view(logp_clusters[begin:(n_clusters + 1)]))
            loc_probs ./= sum(loc_probs)

            # Draw cluster for data point i
            z_new::Int64 = rand(Categorical(loc_probs))

            # If new cluster, create it. Otherwise update existing cluster
            if z_new == n_clusters + 1
                push!(nk, 0)
                n_clusters += 1
            end
            z[i] = z_new
            nk[z_new] += 1
        end # Data point loop
        res_assignments[:, iteration] = z
    end # Iteration loop
    return res_assignments
end

function lambda_posterior(mean_data, prec_data, cluster_means, r)
    # r = prior precision(?)
    # prec_data = sum of precisions of clusters
    k = length(cluster_means)
    posterior_mean = inv(prec_data + k .* r) *
                     (prec_data * sum(cluster_means, dims = 2) + k .* r * mean_data)
    posterior_var = inv(prec_data + k .* r)
    MvNormal(posterior_mean, posterior_var)
end

function r_posterior(cluster_means, prec_data, lambda)
    deg = size(cluster_means) |> sum
    scale_m = inv(prec_data + pairwisedeviationpsum(cluster_means, lambda))
    Wishart(deg, scale_m)
end

function w_posterior(cluster_precisions)
    deg = size(cluster_precisions, 1) + size(cluster_precisions, 3)
    scale_m = inv()
end

function beta_posterior(precisions_c, w)
    k = size(precisions_c, 3)
    return function (β)
        gamma(β / 2)^(-k) * exp(-1 / (2 * β)) * (β / 2)^((k * β - 3) / 2) *
        prod([(precisions_c[:, :, j] * w)^(β / 2) *
              exp(-(β .* precisions_c[:, :, j] * w) ./ 2) for j in 1:k])
    end
end

function calc_contour(dist, xrange, yrange)
    [logpdf(dist, [x, y]) for x in xrange, y in yrange]
end

function MAP_assignments(immresult::AbstractMatrix{<:Integer})
    result = zeros(Int64, maximum(immresult), size(immresult, 1))
    for col in axes(immresult, 2), row in axes(immresult, 1)
        result[immresult[row, col], row] += 1
    end
    out = [findall(result[:, i] .== maximum(result[:, i]))[1] for i in axes(result, 2)]
    return out
end

function pairwisedeviationpsum(
        m::AbstractMatrix{<:AbstractFloat}, μ::Vector{<:AbstractFloat})
    sum = zeros(length(μ))
    for i in axes(m, 2)
        @views @. sum += (m[:, i] - μ)^2
    end
    return sum
end

function adjgraph(immresult::AbstractMatrix{<:Integer})
    D = size(immresult, 1)
    out = zeros(D, D)
    for x in 1:D
        for j in axes(immresult, 2)
            xgroup = immresult[x, j]
            for i in axes(immresult, 1)
                if immresult[i, j] == xgroup
                    out[x, i] += 1.0
                end
            end
        end
    end
    out ./= maximum(out)
    return out
end

function plotadjgraph(vars, adjm, cutoff, truelabs)
    fixed_layout(_) = [(x[1], x[2]) for x in eachcol(vars)]
    g = SimpleGraph(adjm)
    colors = [adjm[e.src, e.dst] for e in edges(g)]
    colors[colors .< cutoff] .= 0.0
    graphplot(g, layout = fixed_layout, edge_width = colors,
        edge_color = colors, colormap = :viridis, node_color = truelabs)
end
