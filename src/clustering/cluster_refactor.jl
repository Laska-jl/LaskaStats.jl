
function immcluster(data, alpha, maxiter = 1000)
    D, N = size(data)
    z_init = ones(Int64, N)
    res_assignments = Matrix{Int64}(undef, N, maxiter)
    res_assignments[:, 1] = z_init

    # Keep precision prior constant for now
    cvar_data = cov(data, dims = 2)

    # Mean of data
    mean_all_data = mean(data, dims = 2)

    # Matrix for lambda of each iteration
    # lambdas = Matrix{Float64}(undef, D, maxiter)

    # Assign first lambda from prior
    # lambdas[:, 1] = rand(MvNormal(mean_all_data, cvar_data))

    # Array for r of each iteration
    # r = Array{Float64}(undef, D, D, maxiter)

    v0 = D

    # Assign first r from prior
    # r[:, :, 1] = rand(Wishart(v0, cvar_data^(-2)))

    # Initialize vectors for storing component means and precisions of each iteration
    component_means = Vector{Vector{MVector{D, Float64}}}(undef, 0)
    component_precisions = Vector{Vector{MMatrix{D, D, Float64, D * D}}}(undef, 0)

    # Push vectors for initial means and precisions
    push!(component_means, [MVector{D, Float64}(undef)])
    push!(component_precisions, [MMatrix{D, D, Float64, D * D}(undef)])

    component_means[1][1] = rand(MvNormal(mean_all_data, cvar_data))
    # Just keep the v0 and cvar_data for precision priors for now
    component_precisions[1][1] = rand(Wishart(v0, cvar_data))

    # Initialize buffer for iterations
    cbuffer = __initcbuffer(
        data, z_init, [component_means[1][1]], [component_precisions[1][1]])

    # Buffer for cluster logprobs
    logp_clusters = Vector{Float64}(undef, N)

    for iteration in 2:maxiter
        iszero(iteration % 100) && println(iteration)
        for i in 1:N
            changed_component = __removedata!(cbuffer, i)
            __logprobs_mvn_dist!(logp_clusters, cbuffer, __data(cbuffer, i))

            mu_new = rand(MvNormal(mean_all_data, cvar_data))
            prec_new = rand(Wishart(v0, cvar_data))
            logp_clusters[__nactive(cbuffer) + 1] = __logprob_mvn_dist_new(
                mu_new, prec_new, alpha, __data(cbuffer, i)
            )
        end
    end
end

# tau0 = precision hos data = Λ₀
# tau = r = Λ
# mu0 = mean av data = μ₀
# xi = mean av alla aktiva kluster = x̄
# n = Antal kluster
function __conditional_lambda_params(
        Λ₀::AbstractMatrix, Λ::AbstractMatrix, n::AbstractVector,
        μ₀::AbstractVector, x̄::AbstractVector)
    mean = inv(Λ₀ + n * Λ) * (Λ₀ * μ₀ + n * Λ * x̄)
    prec = Λ₀ + n * Λ
    return mean, prec
end

function __conditional_r_params(v::AbstractFloat, n::AbstractFloat, V::AbstractMatrix,
        ujs::AbstractVector{<:AbstractVector}, λ::AbstractVector)
    degree = v + n
    scale = inv(V + pairwisedeviationpsum(ujs, λ))
    return degree, scale
end

# Λ₀ = r
# n  = nk
# Λ  = sⱼ
# μ₀ = λ
# x̄  = ȳⱼ
function __conditional_mean_params(r, nk, sj, λ, ȳj)
    mu = inv(r + nk * sj) * (r * λ + nk * sj * ȳj)
    prec = r + nk * sj
    return mu, prec
end

function __conditional_w_params()
end

function __sample_β()
end

# n = nk
# v = D eller β
# V = w eller precision hos hela datan
# μj = medel hos component j
# x = data i component j
function __conditional_precision_params(nk, v, V, μj, x)
    df = nk + v
    scale = inv(V + pairwisedeviationpsum(x, μj))
    return df, scale
end
