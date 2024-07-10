
@doc raw"""
    struct InvChiSq{T}
        θ::T
    end

Inverse Chi^2 distribution. PDF defined as:

``p_{inverse}\chi^2(\alpha|\theta) = \frac{\alpha^{-\frac{\theta}{2}-1}exp(-\frac{1}{2\alpha})}{\Gamma(\frac{\theta}{2})2^{\frac{\theta}{2}}}``

Adapted from: https://ojs.aaai.org/index.php/AAAI/article/view/17079

"""
struct InvChiSq{T}
    θ::T
end

function pdf(d::InvChiSq, α::Number)
    (α^(-d.θ * 0.5 - 1.0) * exp(-1.0 / (2.0 * α))) /
    (gamma(d.θ / 2.0) * 2.0^(d.θ * 0.5))
end


function pdf(d::InvChiSq, α::AbstractVector{<:Number})
    @. (α^(-d.θ * 0.5 - 1.0) * exp(-1.0 / (2.0 * α))) /
    (gamma(d.θ / 2.0) * 2.0^(d.θ * 0.5))
end
