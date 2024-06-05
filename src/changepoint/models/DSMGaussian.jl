
mutable struct DSMGaussian{T, O, M, S, U, I} <:
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

    function DSMGaussian(
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

function grad_r(::DSMGaussian, x::AbstractVector{T}) where {T}
    [one(T) -x[1]]
end

function grad_r(::DSMGaussian, x::T) where {T <: Number}
    [one(T) -x]
end

function hess_r(::DSMGaussian, ::Union{T, AbstractVector{T}}) where {T}
    reshape([zero(T) -one(T)], (1, 1, 2))
end

function grad_b(::DSMGaussian, ::Union{T, AbstractVector{T}}) where {T}
    reshape([zero(T)], (1, 1))
end

function log_pred_prob(model::DSMGaussian, t, data, indices)
    x = data[t - 1]
    logpredprobs = zeros(Float64, length(indices))
    lb = [-Inf, 0.0]
    ub = [Inf, Inf]
    for (k, i) in enumerate(indices)
        dist = TruncatedMVNormal(
            reshape(model.mu[:, :, i], model.p),
            model.Sigma[:, :, i],
            lb,
            ub
        )
        eta = TruncatedMVN.sample(dist, model.b)
        eta1 = eta[1, :]
        eta2 = eta[2, :]
        # eta1 = rand(truncated(Normal(model.mu[2, 1, i], model.Sigma[1, 1, i]),
        #         lb[1],
        #         Inf
        #     ),
        #     model.b
        # )
        # eta2 = rand(
        #     truncated(
        #         Normal(model.mu[2, 1, i], model.Sigma[2, 2, i]),
        #         lb[2],
        #         Inf
        #     ),
        #     model.b
        # )
        #
        # eta = rand(
        #     truncated(MvNormal(reshape(model.mu[i], model.p), model.Sigma[i]), lb, ub),
        #     model.b)
        # eta1 = eta[:, 1]
        # eta2 = eta[:, 2]

        loc = eta1 ./ eta2
        scale = @. sqrt(1 / eta2)
        distr = @. Normal(loc)
        preds = @. pdf(distr, (x - loc) / scale) / scale
        logpredprobs[k] = log(mean(preds))
    end
    return logpredprobs
end
