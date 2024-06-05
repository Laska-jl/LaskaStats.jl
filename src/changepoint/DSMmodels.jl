
abstract type AbstractDSMModel end

function A(model::AbstractDSMModel, m, x)
    transpose(grad_r(model, x)) * m(x) * transpose(m(x)) * grad_r(model, x)
end

function v(model::AbstractDSMModel, m, grad_m, x)
    v1 = transpose(grad_r(model, x)) * m(x) * m(x) * grad_b(model, x)
    div_mm = (
        sum(grad_m(x) * transpose(m(x)), dims = (1, 3)) .+
        sum(grad_m(x) * transpose(m(x)), dims = (1, 2))
    )
    div_mm = reshape(div_mm, (model.d, 1))

    v2 = transpose(grad_r(model, x)) * div_mm

    # v3 = tr(m(x) * transpose(m(x)) .* hess_r(model, x))
    mtm = m(x) * transpose(m(x))
    hess = hess_r(model, x)
    v3 = [tr(mtm * hess[:, :, i]) for i in axes(hess, 3)]
    v3 = reshape(v3, (model.p, 1))

    return @. v1 + v2 + v3
end

function update_params!(model::AbstractDSMModel, m, grad_m, t, data)
    x = data[t - 1]
    a = A(model, m, x)
    new_SigmaInv = model.SigmaInv .+ 2.0 .* model.omega .* a
    new_Sigma = similar(new_SigmaInv)
    for i in axes(new_Sigma, 3)
        @views new_Sigma[:, :, i] = inv(new_SigmaInv[:, :, i])
    end
    # new_mu = new_Sigma *
    #          (model.SigmaInv * model.mu .- 2 * model.omega .* v(model, m, grad_m, x))
    new_mu = similar(model.mu)
    for i in axes(new_mu, 3)
        @views new_mu[:, :, i] = new_Sigma[:, :, i] *
                                 (model.SigmaInv[:, :, i] * model.mu[:, :, i] .-
                                  2 * model.omega .* v(model, m, grad_m, x))
    end

    model.SigmaInv = cat(model.Sigma0Inv, new_SigmaInv, dims = 3)
    model.Sigma = cat(model.Sigma0, new_Sigma, dims = 3)
    model.mu = cat(model.mu0, new_mu, dims = 3)
end
