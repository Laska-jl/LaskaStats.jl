# TODO: Possibly use a struct with a vector of component structs, each holding 
# SA:s of their parameters
struct ComponentBuffer{D, N, T}
    means::Vector{MVector{D, T}}
    prec::Vector{MMatrix{D, D, T, N}}
    nk::Vector{Int64}
    n_active::Base.RefValue{Int64}
    data::Matrix{T}
    z::Vector{Int64}
end

function ComponentBuffer(data::AbstractArray{T}, D::Integer, N::Integer) where {T <: Number}
    msize = D * D
    z = zeros(Int64, N)
    z[1] = size(data, 2)
    ComponentBuffer(
        [MVector{D, T}(undef) for _ in 1:N],
        [MMatrix{D, D, T, msize}(undef) for _ in 1:N],
        z,
        Base.RefValue{Int64}(0),
        data,
        ones(Int64, size(data, 2))
    )
end

function __data(b::ComponentBuffer)
    b.data
end

function __data(b::ComponentBuffer, i::Union{Integer, AbstractUnitRange})
    @view b.data[:, i]
end

function __logprobs_t_dist!(
        logv::AbstractVector{T}, b::ComponentBuffer{D, N, T}, data) where {D, N, T}
    n_clusters = __nactive(b)
    nk = __nk(b)
    cmeans = __means(b)
    d = length(cmeans[1])
    cprec = __precisions(b)

    m = @. inv((nk / (d + 1)) * cprec)

    n = 0
    for i in 1:__nactive(b)
        n += 1
        logv[i] = log(nk[i]) +
                  logpdf(
            MvTDist(
                1,
                Vector{T}(cmeans[i]),
                PDMats.PDMat(m[i], cholesky(Hermitian(m[i])))
            ), data)
    end
end

function __update_component_params!(b::ComponentBuffer, component)
    m = __nk(b, component)
    datainds = findall(x -> x == component, __assignments(b))
    data = __data(b)[:, datainds]
    data_mean = vec(mean(data, dims = 2))
    mu0 = __means(b, component)[]
    prec0 = __precisions(b, component)[]
    __setparams!(b, component,
        ((m + 1) * mu0 + data_mean * m) / (m * 2 + 1),
        inv(
            prec0 + pairwisedeviationpsum(data, data_mean) +
            ((m^2 + 1) / (m * 2 + 1)) * (data_mean - mu0) *
            transpose(data_mean - mu0)
        )
    )
end

# NOTE: Do we need to check n_active etc here?
# Unnecessary since the state literally is from the last iteration

# function __assignfromprior!(
#         b::ComponentBuffer, means::AbstractVector, precisions::AbstractVector)
#     n_components = length(means)
#     b.means[begin:n_components] = means
#     b.prec[begin:n_components] = precisions
# end

# Remove component 'n' and, if the component becomes empty, remove it, lower number of active
# components and return 0. Otherwise, return the index of the component that has been changed
function __removedata!(b::ComponentBuffer, n::Integer)
    c = __assignment(b, n)
    b.z[n] = 0
    __lowernk!(b, c)
    # If this was the only member of the component, remove the component
    if iszero(__nk(b, c))
        # If c is the last component then none need to be moved
        if c == __nactive(b)
            __lowernactive!(b)
            return 0
        else # If this was not the last component, move the last one to take its place
            n_components = __nactive(b)
            last_inds = findall(x -> x == n_components, __assignments(b))
            # Move the count for the last component to the emtied one
            __setnk!(b, c, __nk(b, n_components))
            # Move the parameters of the last component to the emptied one
            __setparams!(b, c, __getparams(b, n_components))
            __lowernactive!(b)
            __assign!(b, last_inds, c)
            return 0
        end
    end
    return c
end

__assignments(b::ComponentBuffer) = b.z
# get assignment of data point n
__assignment(b::ComponentBuffer, n::Integer) = b.z[n]
# Assign datapoint 'n' to luster 'c'
function __assign!(b::ComponentBuffer, n::Integer, c::Integer)
    if c > __nactive(b)
        throw(ArgumentError("Attempted to assign data to component $c of ComponentBuffer with $(__nactive(b)) active components"))
    end
    b.z[n] = c
    __upnk!(b, c)
end

# Assign datapoints 'n' to luster 'c'
function __assign!(b::ComponentBuffer, n::UnitRange, c::Integer)
    if c > __nactive(b)
        throw(ArgumentError("Attempted to assign data to component $c of ComponentBuffer with $(__nactive(b)) active components"))
    end
    b.z[n] .= c
    __upnk!(b, c, length(n))
end

function __assign!(b::ComponentBuffer, n::Vector{Int64}, c::Integer)
    if c > __nactive(b)
        throw(ArgumentError("Attempted to assign data to component $c of ComponentBuffer with $(__nactive(b)) active components"))
    end
    b.z[n] .= c
    __upnk!(b, c, length(n))
end

# Unassign datapoint
# Sets the indicator for the data point to 0 and lowers the count in the nk
function __unassign!(b::ComponentBuffer, n::Integer, prev_assignment::Integer)
    if prev_assignment > __nactive(b)
        throw(ArgumentError("Attempted to unassign data to component $prev_assignment of ComponentBuffer with $(__nactive(b)) active components"))
    end
    b.z[n] = 0
    __lowernk!(b, prev_assignment)
end

__nactive(b::ComponentBuffer) = b.n_active[]
__bumpnactive!(b::ComponentBuffer) = b.n_active[] += 1
__lowernactive!(b::ComponentBuffer) = b.n_active[] -= 1

function __means(b::ComponentBuffer)
    @view b.means[begin:__nactive(b)]
end

function __means(b::ComponentBuffer, n::Integer)
    if n > __nactive(b)
        throw(ArgumentError("Attempted to access uninitialized/inactive components at index $n of ComponentBuffer with $(__nactive(b)) active components"))
    end
    @view b.means[n]
end

__precisions(b::ComponentBuffer) = @view b.prec[begin:__nactive(b)]

function __precisions(b::ComponentBuffer, n::Integer)
    if n > __nactive(b)
        throw(ArgumentError("Attempted to access uninitialized/inactive components at index $n of ComponentBuffer with $(__nactive(b)) active components"))
    end
    @view b.prec[n]
end

# Add parameters for component
function __setparams!(b::ComponentBuffer{D, N, T}, component::Integer,
        mean::AbstractVector{T}, prec::AbstractArray{T}) where {D, N, T}
    if component > __nactive(b) + 1
        throw(ArgumentError("Attempted to assign new component at non-consecutive index from the last active component at $(__nactive(b))"))
    end
    b.means[component] = mean
    b.prec[component] = prec
    component > __nactive(b) && __bumpnactive!(b)
end

function __setparams!(b::ComponentBuffer{D, N, T}, component::Integer,
        params::Tuple{<:AbstractVector{T}, <:AbstractMatrix{T}}) where {D, N, T}
    if component > __nactive(b) + 1
        throw(ArgumentError("Attempted to assign new component at non-consecutive index from the last active component at $(__nactive(b))"))
    end
    b.means[component] = params[1]
    b.prec[component] = params[2]
    component > __nactive(b) && __bumpnactive!(b)
end

function __getparams(b::ComponentBuffer, c::Integer)
    (__means(b, c)[], __precisions(b, c)[])
end

# Get nk
__nk(b::ComponentBuffer) = @view b.nk[begin:__nactive(b)]

function __nk(b::ComponentBuffer, n::Integer)
    if n > __nactive(b)
        throw(ArgumentError("Attempted to access uninitialized/inactive components at index $n of ComponentBuffer with $(__nactive(b)) active components"))
    end
    b.nk[n]
end

# Change nk
function __setnk!(b::ComponentBuffer, n::Integer, nk::Integer)
    if n > __nactive(b)
        throw(ArgumentError("Attempted to access uninitialized/inactive components at index $n of ComponentBuffer with $(__nactive(b)) active components"))
    end
    b.nk[n] = nk
end

function __upnk!(b::ComponentBuffer, c::Integer, n::Integer = 1)
    if c > __nactive(b)
        throw(ArgumentError("Attempted to access uninitialized/inactive components at index $c of ComponentBuffer with $(__nactive(b)) active components"))
    end
    b.nk[c] += n
end

function __lowernk!(b, c)
    if c > __nactive(b)
        throw(ArgumentError("Attempted to access uninitialized/inactive components at index $c of ComponentBuffer with $(__nactive(b)) active components"))
    end
    b.nk[c] -= 1
end

function cluster3(data, z_init, alpha, maxiter = 1000)
    D, N = size(data)
    res_assignments = Matrix{Int64}(undef, N, maxiter)
    z = deepcopy(z_init)
    res_assignments[:, 1] = z

    cbuffer = ComponentBuffer(data, D, N)

    cvar_data = cov(data, dims = 2)
    prec_data = inv(cvar_data)
    mean_data = mean(data, dims = 2)

    # Initialize parameter vectors/arrays
    r = Array{Float64}(undef, D, D, maxiter) # Precision hyperparameter of means
    lambda = Matrix{Float64}(undef, D, maxiter) # Mean hyperparameter for means

    w = Array{Float64}(undef, D, D, maxiter) # Scale matrix for component precision prior

    # Sample initial values for component means
    r[:, :, 1] = rand(Wishart(D, cvar_data^(-2)))
    lambda[:, 1] = rand(MvNormal(vec(mean(data, dims = 2)), cvar_data))

    component_means = Vector{Vector{MVector{D, Float64}}}(undef, 0)
    component_precisions = Vector{Vector{MMatrix{D, D, Float64, D * D}}}(undef, 0)

    push!(component_means, [MVector{D, Float64}(undef)])
    component_means[1][1] = rand(MvNormal(lambda[:, 1], Symmetric(inv(r[:, :, 1]))))

    # Sample initial values for component precisions
    w[:, :, 1] = rand(Wishart(D, cvar_data))
    push!(component_precisions,
        [MMatrix{D, D, Float64, D * D}(undef)])
    component_precisions[1][1] = rand(Wishart(
        D, Matrix{Float64}(Symmetric(inv(w[:, :, 1])))))

    # Set initial params from prior above

    __setparams!(cbuffer, 1, component_means[1][1], component_precisions[1][1])

    # Preallocate vector for log probs of each cluster. Will be the length of data since
    # having more clusters than datapoints would be useless
    logp_clusters = Vector{Float64}(undef, N)

    # Main iteration loop
    for iteration in 2:maxiter
        iszero(iteration % 100) && println(iteration)

        # Use posteriors from last iteration as priors

        for i in Base.OneTo(N)
            # Remove datapoint from its cluster

            changed_component = __removedata!(cbuffer, i)

            cluster_means = clustermeans(
                data[:, Not(i)], z[Not(i)], __nactive(cbuffer), __nk(cbuffer))
            n_clusters = __nactive(cbuffer)

            # update non-cluster specific parameters
            ## r
            r_deg, r_scale = conditional_mvn_prec(
                n_clusters, D, cvar_data,
                cluster_means, vec(lambda[:, iteration - 1]))
            r[:, :, iteration] = rand(Wishart(
                r_deg, Matrix{Float64}(Symmetric(inv(r_scale)))))
            ## lambda
            lambda_mean, lambda_sig = conditional_mvn_mu(
                prec_data, r[:, :, iteration], mean_data, n_clusters, mean_data)
            lambda[:, iteration] = rand(MvNormal(vec(lambda_mean), lambda_sig))

            # Update parameters of the changed cluster
# NOTE: Check data of changed component here!
            if !iszero(changed_component)
                __update_component_params!(cbuffer, changed_component)
            end
            # Calculate probabilities for each cluster
            __logprobs_t_dist!(logp_clusters, cbuffer, __data(cbuffer, i))

            # Probability of new cluster
            mu_new = rand(MvNormal(lambda[:, iteration], Symmetric(r[:, :, iteration])))
            # TODO: Figure out updating precision priors!
            prec_new = rand(Wishart(D, Matrix{Float64}(Symmetric(inv(w[:, :, 1])))))
            logp_clusters[n_clusters + 1] = log(alpha) + logpdf(
                MvNormal(mu_new, Symmetric(inv(prec_new))), __data(cbuffer, i))

            max_logp_clusters = maximum(@view(logp_clusters[begin:(__nactive(cbuffer) + 1)]))
            logp_clusters[begin:(__nactive(cbuffer) + 1)] .-= max_logp_clusters

            loc_probs = exp.(@view(logp_clusters[begin:(__nactive(cbuffer) + 1)]))
            loc_probs ./= sum(loc_probs[begin:(__nactive(cbuffer) + 1)])

            z_new = rand(Categorical(loc_probs))

            # Add new params if new cluster
            if z_new == n_clusters + 1
                __setparams!(cbuffer, n_clusters + 1, mu_new, prec_new)
            end
            # Assign datapoint to selected cluster
            __assign!(cbuffer, i, z_new)
        end # Closes data point iteration loop
        res_assignments[:, iteration] = __assignments(cbuffer)
        push!(component_means, __means(cbuffer))
        push!(component_precisions, __precisions(cbuffer))
    end # Closes main iteration loop
    return (
        assignments = res_assignments,
        means = component_means,
        prec = component_precisions
    )
end

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
                z[last_inds] .= c
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

function cluster2(data, z_init, alpha, maxiter = 1000)
    D, N = size(data)
    res_assignments = Matrix{Int64}(undef, N, maxiter)
    z = deepcopy(z_init)
    res_assignments[:, 1] = z

    cvar_data = cov(data, dims = 2)
    prec_data = inv(cvar_data)
    mean_data = mean(data, dims = 2)

    # Initialize parameter vectors/arrays
    r = Array{Float64}(undef, D, D, maxiter) # Precision hyperparameter of means
    lambda = Matrix{Float64}(undef, D, maxiter) # Mean hyperparameter for means
    component_means = Vector{Matrix{Float64}}(undef, maxiter)

    w = Array{Float64}(undef, D, D, maxiter) # Scale matrix for component precision prior
    component_precisions = Vector{Array{Float64}}(undef, maxiter)

    # Sample initial values for component means
    r[:, :, 1] = rand(Wishart(D, cvar_data^(-2)))
    lambda[:, 1] = rand(MvNormal(vec(mean(data, dims = 2)), cvar_data))
    component_means[1] = Matrix{Float64}(undef, D, maximum(z_init))
    @views for c in 1:maximum(z_init)
        component_means[1][:, c] = rand(MvNormal(lambda[:, 1], Symmetric(inv(r[:, :, 1]))))
    end

    # Sample initial values for component precisions
    w[:, :, 1] = rand(Wishart(D, cvar_data))
    component_precisions[1] = Array{Float64}(undef, D, D, maximum(z_init))
    @views for c in 1:maximum(z_init)
        component_precisions[1][:, :, c] = rand(Wishart(
            D, Matrix{Float64}(Symmetric(inv(w[:, :, 1])))))
    end

    nk = [count(z .== c) for c in 1:maximum(z)]
    n_clusters = length(nk)

    # Preallocate vector for log probs of each cluster. Will be the length of data since
    # having more clusters than datapoints would be useless
    logp_clusters = Vector{Float64}(undef, N)

    # Main iteration loop
    for iteration in 2:maxiter
        iszero(iteration % 100) && println(iteration)

        # Use posteriors from last iteration as priors
        component_means[iteration] = hcat(component_means[iteration - 1], zeros(D, 1))
        component_precisions[iteration] = cat(
            component_precisions[iteration - 1], zeros(D, D), dims = 3)

        for i in Base.OneTo(N)
            # Remove datapoint from its cluster
            c = z[i]
            nk[c] -= 1

            # If cluster is empty, remove it and move the data in the last clusters to it
            if iszero(nk[c])
                last_inds = findall(x -> x == n_clusters, z)
                nk[c] = nk[n_clusters]
                pop!(nk)
                z[last_inds] .= c
                n_clusters -= 1
            end

            cluster_means = clustermeans(data[:, Not(i)], z[Not(i)], n_clusters, nk)

            # update non-cluster specific parameters
            ## r
            r_deg, r_scale = conditional_mvn_prec(
                n_clusters, D, cvar_data,
                cluster_means, vec(lambda[:, iteration - 1]))
            r[:, :, iteration] = rand(Wishart(
                r_deg, Matrix{Float64}(Symmetric(inv(r_scale)))))
            ## lambda
            lambda_mean, lambda_sig = conditional_mvn_mu(
                prec_data, r[:, :, iteration], mean_data, n_clusters, mean_data)
            lambda[:, iteration] = rand(MvNormal(vec(lambda_mean), lambda_sig))

            for c in Base.OneTo(n_clusters)
                data_inds = findall(x -> x == c, z)

                component_means[iteration][:, c], component_precisions[iteration][:, :, c] = conditional_component_params(
                    nk[c], component_means[iteration][:, c], data[:, data_inds],
                    cluster_means[:, c], component_precisions[iteration][:, :, c])

                logp_clusters[c] = log(nk[c]) + logpdf(
                    conditional_t_dist(D, D, component_means[iteration][:, c], nk[c],
                        component_precisions[iteration][:, :, c]), data[:, i])
            end # Closes cluster iteration loop

            # Probability of new cluster
            mu_new = rand(MvNormal(lambda[:, iteration], Symmetric(r[:, :, iteration])))
            component_means[iteration][:, n_clusters + 1] = mu_new
            # TODO: Figure out updating precision priors!
            prec_new = rand(Wishart(D, Matrix{Float64}(Symmetric(inv(w[:, :, 1])))))
            component_precisions[iteration][:, :, n_clusters + 1] = prec_new
            logp_clusters[n_clusters + 1] = log(alpha) + logpdf(
                MvNormal(mu_new, Symmetric(inv(prec_new))), data[:, i])
            logp_clusters[begin:(n_clusters + 1)]

            max_logp_clusters = maximum(@view(logp_clusters[begin:(n_clusters + 1)]))
            @. logp_clusters[begin:(n_clusters + 1)] -= max_logp_clusters

            loc_probs = exp.(@view(logp_clusters[begin:(n_clusters + 1)]))
            loc_probs ./= sum(loc_probs[begin:(n_clusters + 1)])

            z_new = rand(Categorical(loc_probs))

            # TODO: Add handling of new cluster parameters
            if z_new == n_clusters + 1
                push!(nk, 0)
                n_clusters += 1
            end
            z[i] = z_new
            nk[z_new] += 1
        end # Closes data point iteration loop
        res_assignments[:, iteration] = z
    end # Closes main iteration loop
    return (
        assignments = res_assignments,
        means = component_means,
        prec = component_precisions
    )
end

function conditional_t_dist(n_m, d, mu_m, k_m, prec_m)
    MvTDist(n_m - d + 1, mu_m,
        Matrix{Float64}(Symmetric(inv((k_m * (n_m - d + 1) / (k_m + 1)) * prec_m))))
end

function conditional_component_params(k0, mu0, x, x_mean, prec0)
    mu_m = (k0 * mu0 + x_mean) / k0
    prec_m = inv(prec0 + pairwisedeviationpsum(x, x_mean) +
                 (x_mean - mu0) * transpose(x_mean - mu0))
    return mu_m, prec_m
end

"""
    conditional_mvn_mu(Λ₀, Λ, μ₀, n, x̄)

Assuming MvN prior on mean vector
Λ₀ = Precision prior
Λ = Measeured precision
μ₀ = Mean vector prior
n = Number of data points 
x̄ = sample mean
Returns the mean vector and covariance matrix for a MvN distribution representing
the prior on the normal vector of a Mvn
"""
function conditional_mvn_mu(Λ₀, Λ, μ₀, n, x̄)
    mu = inv(Λ₀ + n * Λ) * (Λ₀ * μ₀ + n * Λ * x̄)
    sig = Λ₀ + n * Λ
    return mu, sig
end

"""
    conditional_mvn_prec(n, v, V, x, μ)

Assuming Wishart prior on precision matrix
n = number of new data points(?)
v = Original number of data points used to estimate precision matrix
V = Sum of pairwise deviation products
x = Data points as a D x N Matrix
μ = Mean of MvN distribution, assumed known
Returns the conditional degree and scale matrix for a Wishart distribution
"""
function conditional_mvn_prec(n, v, V, x, μ)
    deg = n + v
    # scale = inv(V) + pairwisedeviationpsum(x, μ)
    scale = V + pairwisedeviationpsum(x, μ)
    return deg, scale
end

function clustermeans(data, z, n_j, nk)
    means = zeros(size(data, 1), n_j)

    for i in eachindex(z)
        means[:, z[i]] += data[:, i]
    end

    for i in eachindex(nk)
        means[:, i] ./= nk[i]
    end
    return means
end

function conditional_mean(r::AbstractMatrix, sj::AbstractMatrix, mu0::AbstractVector,
        nj::Integer, samplemean::AbstractVector)
    mu = inv(r + nj * sj) * (r * mu0 + sj * samplemean)
    s = r + nj * sj
    return mu, s
end

function conditional_precision(data_j, nj, β, mu_j, cluster_means)
    deg = β + nj
    scale = inv(inv(pairwisedeviationpsum(data_j, cluster_means)) +
                pairwisedeviationpsum(data_j, mu_j))
    return deg, scale
end

# Multivariate normal with known precision (r)
function conditional_lambda(r::AbstractMatrix, s_y::AbstractMatrix, mu_y::AbstractVector,
        n_j::Integer, samplemean::AbstractVector)
    mu = inv(s_y + n_j * r) * (s_y * mu_y + n_j * r * samplemean)
    s = s_y + n_j * r
    return mu, s
end

function conditional_r(s_y, k, λ, cluster_means)
    deg = k + 1
    scale = inv(inv(s_y) + pairwisedeviationpsum(cluster_means, λ))
    return deg, scale
end

# function r_posterior(cluster_means, prec_data, lambda)
#     deg = length(cluster_means) + length(lambda)
#     scale_m = inv(prec_data + pairwisedeviationpsum(cluster_means, lambda))
#     Wishart(deg, scale_m)
# end

# function w_posterior(cluster_precisions)
#     deg = size(cluster_precisions, 1) + size(cluster_precisions, 3)
#     scale_m = inv()
# end

function beta_posterior(precisions_c, w, k)
    return function (β)
        gamma(β / 2)^(-k) * exp(-1 / (2 * β)) * (β / 2)^((k * β - 3) / 2) *
        prod([(precisions_c[:, :, j] * w)^(β / 2) *
              exp(-(β .* precisions_c[:, :, j] * w) ./ 2) for j in 1:k])
    end
end

function calc_contour(dist, xrange, yrange)
    [logpdf(dist, [x, y]) for x in xrange, y in yrange]
end

function MAP_assignments(immresult::AbstractMatrix{<:Integer}, burnin, skip)
    result = zeros(Int64, maximum(immresult), size(immresult, 1))
    for col in burnin:skip:size(immresult, 2), row in axes(immresult, 1)
        result[immresult[row, col], row] += 1
    end
    out = [findall(result[:, i] .== maximum(result[:, i]))[1] for i in axes(result, 2)]
    return out
end

function pairwisedeviationpsum(
        m::AbstractMatrix{<:AbstractFloat}, μ::Vector{<:AbstractFloat})
    sum = m .- μ
    sum * transpose(sum)
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

# function plotadjgraph(vars, adjm, cutoff, truelabs)
#     fixed_layout(_) = [(x[1], x[2]) for x in eachcol(vars)]
#     g = SimpleGraph(adjm)
#     colors = [adjm[e.src, e.dst] for e in edges(g)]
#     colors[colors .< cutoff] .= 0.0
#     graphplot(g, layout = fixed_layout, edge_width = colors,
#         edge_color = colors, colormap = :viridis, node_color = truelabs)
# end
