
function zstandardize!(v::AbstractVector{T}, mean::T, sd::T) where {T <: AbstractFloat}
    sdfac = 1 / sd
    @inbounds @simd for i in eachindex(v)
        v[i] = (v[i] - mean) * sdfac
    end
end

"""
    zstandardize!(v::AbstractVector{T}) where {T <: AbstractFloat}

Standardize the `Vector` `v` in-place, setting its mean to 0 and standard deviation to 1.
"""
function zstandardize!(v::AbstractVector{T}) where {T <: AbstractFloat}
    zstandardize!(v, mean(v), std(v))
end

"""
    zstandardize(v::AbstractVector{T}) where {T <: AbstractFloat}

Standardize the `Vector` `v`, setting its mean to 0 and standard deviation to 1.
"""
function zstandardize(v::AbstractVector{T}) where {T <: AbstractFloat}
    out = deepcopy(v)
    zstandardize!(out)
    return out
end
