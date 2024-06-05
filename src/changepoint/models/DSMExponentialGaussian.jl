
mutable struct DSMExponentialGaussian{T, O, M, S, U, I} <:
               AbstractDSMModel where {
    T <: AbstractArray{<:AbstractFloat}, O <: AbstractFloat, M <:
                                                             AbstractArray{<:AbstractFloat},
    S <: AbstractArray{<:AbstractFloat}, U <: Number, I <: Integer}
    data::T
    omega::O
    mu0::M
    mu::M
    Sigma0Inv::S
    SigmaInv::S
    Sigma0::S
    Sigma::S
    p::I
    d::U
    b::I

    function DSMExponentialGaussian(
            data::AbstractArray{T},
            omega::AbstractFloat,
            mu0::AbstractArray{T},
            Sigma0::AbstractArray{T};
            p::I = 2,
            d::U = 1,
            b::I = 20
    ) where {T, U, I}
        mu0 = reshape(mu0, (size(mu0)..., 1))
        mu = deepcopy(mu0)

        Sigma0Inv = reshape(LinearAlgebra.inv(Sigma0), (size(Sigma0)..., 1))
        SigmaInv = deepcopy(Sigma0Inv)
        Sigma0 = reshape(Sigma0, (size(Sigma0)..., 1))
        Sigma = deepcopy(Sigma0)
        new{typeof(data), typeof(omega), typeof(mu0), typeof(Sigma0), U, I}(
            data, omega, mu0, mu, Sigma0Inv, SigmaInv, Sigma0, Sigma, p, d, b)
    end
end

function grad_r(::DSMExponentialGaussian, x::AbstractVector{T}) where {T}
    grad_t1 = zeros(T, 1, 6)
    grad_t1[:, 1] .= -one(T)
    grad_t1[:, 5] .= one(T)
    grad_t1[:, 6] .= -x[2]
    # grad_t2 = zeros(T, 1, 3)
    # grad_t2[:, 2] .= one(T)
    # grad_t2[:, 3] .= -x[2]
    return grad_t1
end

function hess_r(::DSMExponentialGaussian, x::T) where {T}
    hess_t = zeros(eltype(x), 2, 2, 3)
    hess_t[2, 2, 3] = -one(eltype(x))
    return hess_t
end

function grad_b(::DSMExponentialGaussian, x::T) where {T}
    reshape([zero(eltype(x)), zero(eltype(x))], 2, 1)
end

function log_pred_prob(m::DSMExponentialGaussian, t::Integer, data, indices)
    _log_pred_prob(m, t, data, indices, m.mu, m.Sigma, m.b, m.p)
end

function _log_pred_prob(
        ::DSMExponentialGaussian, t::Integer, data, indices, mu, Sigma, b, p = 3)
    # TODO: Figure out the dimensions of data
    x = data[t - 1]
    log_pred_prob = zeros(length(indices))
    lb = [0.0, -Inf, 0.0]
    ub = fill(Inf, p)

    for (k, i) in enumerate(indices)
        eta = sample(
            TruncatedMVNormal(
                reshape(mu[:, :, i], p),
                Sigma[:, :, i],
                lb,
                ub
            ),
            b
        )
        eta1 = eta[1, :]
        eta2 = eta[2, :]
        eta3 = eta[3, :]

        expd = [Exponential(1.0 / s) for s in eta1]
        sample_exp = pdf.(expd, x[1])

        normd = [Normal(μ, σ) for (μ, σ) in zip(eta2 ./ eta3, @.(sqrt(1 / eta3)))]
        sample_normal = pdf.(normd, x[2])

        log_pred_prob[k] = log(mean(sample_normal .* sample_exp))
    end
    return log_pred_prob
end
