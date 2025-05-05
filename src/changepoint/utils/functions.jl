
@inline function generic_m(x::AbstractVector{T}) where {T <: AbstractFloat}
    d = Vector{T}(undef, length(x))
    @. d = __M(x)
    return diagm(d)
end

@inline function generic_m(x::T) where {T <: AbstractFloat}
    d = Vector{T}(undef, length(x))
    @. d = __M(x)
    return diagm(d)
end

@inline function __M(x::T) where {T}
    (one(T) + x^2)^(-0.5)
end

@inline function generic_m_grad(x::AbstractVector{T}) where {T <: AbstractFloat}
    d = Vector{T}(undef, length(x))
    @. d = __M_grad(x)
    return diagm(d)
end

@inline function generic_m_grad(x::T) where {T <: AbstractFloat}
    d = Vector{T}(undef, 1)
    @. d = __M_grad(x)
    return diagm(d)
end
# -x[0]/((1+x[0]**2)**(3/2))
@inline function __M_grad(x::T) where {T}
    -x / ((one(T) + x^2) * (3.0 / 2.0))
end
