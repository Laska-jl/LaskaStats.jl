#TODO: Replace inv with transpose
#TODO: fix the constructor
#TODO: Figure out the dimensions of the hess_r

struct OmegaEstimatorGaussian{
    T <: AbstractArray{<:Number}, U <: Integer, V <: Integer, X <: Integer,
    Y <: AbstractArray{<:Number}, Z <: AbstractArray{<:Number},
    S <: Union{<:Number, <:AbstractArray{<:Number}},
    R <: Union{<:Number, <:AbstractArray{<:Number}}}
    data::T
    p::U
    d::V
    n::X
    mu0::Y
    Sigma0Inv::Z
    Sigma0::Z
    A::S
    v::R
end

function NewOmegaEstimatorGaussian(data::T,
        mu0::Y,
        Sigma0::Z,
        p,
        m,
        grad_m) where {
        T <: AbstractArray{<:Number}, U <: Integer, V <: Integer, X <: Integer,
        Y <: AbstractArray{<:Number}, Z <: AbstractArray{<:Number},
        S <: Union{<:Number, <:AbstractArray{<:Number}},
        R <: Union{<:Number, <:AbstractArray{<:Number}}}
    n = size(data)[1]
    d = 1
    Sigma0Inv = collect(inv(Sigma0))
    A = (sum([Ax(OmegaEstimatorGaussian, m, x) for x in data], dims = 1))[1] / n
    v = 2 .*
        sum([vx(OmegaEstimatorGaussian, m, grad_m, d, p, x) for x in data]) / n

    OmegaEstimatorGaussian(
        data,
        p,
        d,
        n,
        mu0,
        Sigma0Inv,
        Sigma0,
        A,
        v
    )
end

function grad_r(::Union{OmegaEstimatorGaussian, Type{OmegaEstimatorGaussian}}, x)
    [1.0 -x[1]]
end

function hess_r(::Union{OmegaEstimatorGaussian, Type{OmegaEstimatorGaussian}}, x)
    reshape([0.0, -1.0], (1, 1, 2))
end

function grad_b(::Union{OmegaEstimatorGaussian, Type{OmegaEstimatorGaussian}}, x)
    reshape([0.0], (1, 1))
end

function Ax(o::Union{OmegaEstimatorGaussian, Type{OmegaEstimatorGaussian}}, m, x)
    transpose(grad_r(o, x)) * m(x) * transpose(m(x)) * grad_r(o, x)
end

function vx(
        o::Union{OmegaEstimatorGaussian, Type{OmegaEstimatorGaussian}}, m, grad_m, d, p, x)
    v1 = transpose(grad_r(o, x)) * m(x) * m(x) * grad_b(o, x)

    div_mm = sum(grad_m(x) * transpose(m(x)), dims = (1, 3)) .+
             sum(grad_m(x) * transpose(m(x)), dims = (1, 2))
    div_mm = reshape(div_mm, (d, 1))

    v2 = transpose(grad_r(o, x)) * div_mm
    hess = hess_r(o, x)
    v3 = [(m(x) * transpose(m(x)) * hess[:, :, i])[1] for i in axes(hess, 3)]
    v3 = reshape(v3, (p, 1))
    @show v3

    return v1 .+ v2 .+ v3
end

function var_dsm_full(o::OmegaEstimatorGaussian, b)
    inv(o.Sigma0Inv .+ (2 .* b .* o.n .* o.A))
end

# var_dsm_full with explicit parameters
function var_dsm_full(Sigma0Inv, n, A, b)
    inv(Sigma0Inv .+ (2 .* b .* n .* A))
end

function mu_dsm_full(o::OmegaEstimatorGaussian, b)
    var_dsm_full(o, b) * (o.Sigma0Inv * o.mu0 .- b .* o.n .* o.v)
end

# mu_dsm_full with explicit parameters
function mu_dsm_full(Sigma0Inv, mu0, n, v, A, b)
    var_dsm_full(Sigma0Inv, n, A, b) * (Sigma0Inv * mu0 .- b .* n .* v)
end

function log_norminvgamma_posterior(o::OmegaEstimatorGaussian, mu, sigma2, prior_parameters)
    alpha, beta, u, k = prior_parameters
    sample_mean = mean(o.data)

    alpha = alpha + o.n / 2

    beta = beta +
           0.5 * sum(
        (o.data .- sample_mean) .^ 2 .+
        o.n * k * (sample_mean - u)^2 / (k + o.n)
    )
    k = k + o.n

    t1 = 0.5 * log(k) .- log.(2 * π * sigma2)
    t2 = alpha * log(beta) - loggamma(alpha)
    t3 = -(alpha + 1) .* log.(sigma2)
    t4 = -(2 .* beta .+ k .* (mu .- u) .^ 2) ./ (2 .* sigma2)

    return @. t1 + t2 + t3 + t4
end

function kl(o, omega, prior_parameters = [1.0, 1.0, 0.0, 1.0], n_samples = 1000)
    var = var_dsm_full(o, omega)
    mu = mu_dsm_full(o, omega)

    q1 = Normal(mu[1], sqrt(var[1, 1]))
    q2 = truncated(Normal(mu[2], sqrt(var[2, 2])), 0.0, 1.0e10)

    sample_set_1 = rand(q1, n_samples)
    sample_set_2 = rand(q2, n_samples)

    q1_log = logpdf.(q1, sample_set_1)
    q2_log = logpdf.(q2, sample_set_2)
    posterior = log_norminvgamma_posterior(
        o,
        sample_set_1 ./ sample_set_2,
        1 ./ sample_set_2,
        prior_parameters
    )

    return mean(@. q1_log + q2_log - posterior)
end

function estimateomega(o::OmegaEstimatorGaussian, omega0; lr = 0.01,
        niter = 1000, prior_parameters = [1.0, 1.0, 0.0, 1.0], n_samples = 1000)
    param = [omega0]
    optimizer = Optimisers.setup(Descent(lr), param)
    costs = Vector{Float64}(undef, niter)
    params = Vector{Float64}(undef, niter)

    klfunc = x -> kl(o, x, prior_parameters, n_samples)

    for i in 1:niter
        costs[i] = klfunc(param)
        grads = Zygote.gradient(klfunc, param)[1]
        optimizer, param, = Optimisers.update!(optimizer, param, grads)
        params[i] = param[1]
    end
    return params, costs
end

# @model function omega(o::OmegaEstimatorGaussian, omega, prior_parameters,
#         lr = 0.01, niter = 1000, n_samples = 1000)
#     mu = mu_dsm_full(o, omega)
#     var = var_dsm_full(o, omega)
#
#     q1 = Normal(mu[1], var[1, 1])
#     q2 = truncated(Normal(mu[2], sqrt(var[2, 2])), 0.0, 1.0e10)
#
#     sample_set_1 ~ filldist(q1, n_samples)
#     sample_set_2 ~ filldist(q2, n_samples)
#
#     q1_log = logpdf.(q1, sample_set_1)
#     q2_log = logpdf.(q2, sample_set_2)
#
#     posterior = log_norminvgamma_posterior(
#         o,
#         sample_set_1 ./ sample_set_2,
#         1 ./ sample_set_2,
#         prior_parameters
#     )
#     return posterior
# end

# https://fluxml.ai/Optimisers.jl/stable/api/#Model-Interface
