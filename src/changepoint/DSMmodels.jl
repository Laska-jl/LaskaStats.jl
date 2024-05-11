
abstract type AbstractDSMModel end

mutable struct DSMGaussian{T, M, S, U, I} <:
               AbstractDSMModel where {
    T <: AbstractArray{<:AbstractFloat}, M <: AbstractArray{<:AbstractFloat},
    S <: AbstractArray{<:AbstractFloat}, U <: AbstractFloat, I <: Integer}
    data::T
    omega::U
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
            mu::AbstractArray{T},
            Sigma0::AbstractArray{T};
            p::I = 2,
            d::U = 1.0,
            b::I = 20
    ) where {T, U, I}
        Sigma0Inv = LinearAlgebra.inv(Sigma0)
        SigmaInv = deepcopy(Sigma0Inv)
        Sigma = deepcopy(Sigma0)
        new{typeof(data), typeof(mu0), typeof(Sigma0), U, I}(
            data, omega, mu0, mu, Sigma0Inv, SigmaInv, Sigma0, Sigma, p, d, b)
    end
end

function grad_r(::DSMGaussian, x::AbstractVector{T}) where {T}
    [one(T) -x[1]]
end

function hess_r(::DSMGaussian, ::AbstractVector{T}) where {T}
    reshape([zero(T) -one(T)], (1, 1, 2))
end

function grad_b(::DSMGaussian, ::AbstractVector{T}) where {T}
    reshape([zero(T)], (1, 1))
end

function log_pred_prob(model::DSMGaussian, t, data, indices)
    x = data[t]
    logpredobs = zeros(Float64, length(indices))
    lb = [-Inf, 0.0]
    ub = fill(Inf, model.p)
    for (k, i) in enumerate(indices)
        eta = rand(truncated(MvNormal(model.mu[i], model.Sigma[i]), lb, ub), model.b)
        eta1 = eta[:, 1]
        eta2 = eta[:, 2]

        loc = eta1 ./ eta2
        scale = @. sqrt.(1 / eta2)
        distr = @. Normal(loc)
        preds = @. pdf(distr, (x - loc) / scale) / scale
        logpredobs[k] = log(mean(preds))
    end
    return logpredobs
end

function calclambda(model::AbstractDSMModel, m, x)
    transpose(grad_r(model, x)) * m(x) * transpose(m(x)) * grad_r(model, x)
end

function calcnu(model::AbstractDSMModel, m, grad_m, x)
    v1 = transpose(grad_r(model, x)) * m(x) * grad_b(model, x)
    div_mm = (
        sum(grad_m(x) * transpose(m(x)), dims = (1, 3)) .+
        sum(grad_m(x) * transpose(m(x)), dims = (1, 2))
    )
    div_mm = reshape(div_mm, (model.p + 1, 2))

    v2 = transpose(grad_r(model, x)) * div_mm

    v3 = tr(m(x) * transpose(m(x)) * hess_r(model, x))
    v3 = reshape(v3, (model.p + 1, 2))

    return v1 .+ v2 .+ v3
end

function update_params!(model::AbstractDSMModel, m, grad_m, t, data)
    x = data[t - 1]
    new_SigmaInv = model.SigmaInv + 2 * model.omega .* calcnu(model, m, grad_m, x)
    new_Sigma = [inv(new_SigmaInv[i]) for i in 1:t]
    new_mu = new_Sigma *
             (model.SigmaInv * model.mu .- 2 * model.omega .* calcnu(model, m, grad_m, x))

    model.SigmaInv = cat(model.Sigma0Inv, new_SigmaInv, dims = 1)
    model.Sigma = cat(model.Sigma0, new_Sigma, dims = 1)
    model.mu = cat(model.mu0, new_mu, dims = 1)
end
