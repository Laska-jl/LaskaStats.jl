#------------------------------------------------#
# Different normalizations of Vectors and Arrays #
#------------------------------------------------#

# Normalize with manual min/max and range

@doc raw"""
    rangenormalize!(v::AbstractVector{T}, min_v::T, max_v::T, range::NTuple{2, T}) where {T<:AbstractFloat}

Normalize an `AbstractVector` in-place to values between `range[1]` (``a``) and `range[2]` (``b``). `min_v` and `max_v` may be provided as "manual" extrema of the vector.

```math
X' = a + \frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
```
"""
function rangenormalize!(v::AbstractVector{T},
        min_v::T,
        max_v::T,
        range::NTuple{2, T}) where {T <: AbstractFloat}
    diffx = max_v - min_v
    rangediff = range[2] - range[1]
    @. v = (((v - min_v) * rangediff) / diffx) + range[1]
end

@doc raw"""
    rangenormalize(v::AbstractVector{T}, min_v::T, max_v::T, range::NTuple{2, T}) where T


Normalize an `AbstractVector` to values between `range[1]` (``a``) and `range[2]` (``b``). `min_v` and `max_v` may be provided as "manual" extrema of the vector.

```math
X' = a + \frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
```
"""
function rangenormalize(v::AbstractVector{T},
        min_v::T,
        max_v::T,
        range::NTuple{2, T}) where {T}
    out = Vector{Float64}(deepcopy(v))
    rangenormalize!(out, min_v, max_v, range)
    return out
end

# Normalize with manual min/max of input in range 0-1

@doc raw"""
    rangenormalize!(v::AbstractVector{T}, min_v::T, max_v::T) where {T<:AbstractFloat}

Normalize an `AbstractVector` in-place to values between 0 and 1. `min_v` and `max_v` may be provided as "manual" extrema of the vector.

```math
X' = \frac{(X-X_{min})}{X_{max}-X_{min}}
```
"""
function rangenormalize!(v::AbstractVector{T},
        min_v::T,
        max_v::T) where {T <: AbstractFloat}
    diffx = max_v - min_v
    @. v = (v - min_v) / diffx
end

function derangenormalize!(v::AbstractVector{T},
        premin::T,
        premax::T) where {T <: AbstractFloat}
    diffx = premax - premin
    @. v = v * diffx + premin
end

@doc raw"""
    rangenormalize(v::AbstractVector{T}, min_v::T, max_v::T) where T

Normalize an `AbstractVector` to values between 0 and 1. `min_v` and `max_v` may be provided as "manual" extrema of the vector.

```math
X' = \frac{(X-X_{min})}{X_{max}-X_{min}}
```
"""
function rangenormalize(v::AbstractVector{T}, min_v::T, max_v::T) where {T}
    out = Vector{Float64}(deepcopy(v))
    rangenormalize!(out, min_v, max_v)
    return out
end

# Normalize with auto min/max to a manual range

@doc raw"""
    rangenormalize!(v::AbstractVector{T}, range::NTuple{2,T}) where {T<:AbstractFloat}

Normalize an `AbstractVector` in-place to values between `range[1]` (``a``) and `range[2]` (``b``).
```math
X' = a + \frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
```
"""
function rangenormalize!(v::AbstractVector{T},
        range::NTuple{2, T}) where {T <: AbstractFloat}
    min_v, max_v = extrema(v)
    diffx = max_v - min_v
    rangediff = range[2] - range[1]
    @. v = (((v - min_v) * rangediff) / diffx) + range[1]
end

@doc raw"""
    rangenormalize(v::AbstractVector{T}, range::NTuple{2,T}) where T

Normalize a `Vector`  to values between `range[1]` (``a``) and `range[2]` (``b``).

```math
X' = a + \frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
```
"""
function rangenormalize(v::AbstractVector{T}, range::NTuple{2, T}) where {T}
    out = Vector{Float64}(deepcopy(v))
    rangenormalize!(out, range)
    return out
end

# Normalize with automatic minimum to range 0-1

@doc raw"""
    rangenormalize!(v::AbstractVector{T}) where {T<:AbstractFloat}

Normalize a `Vector` to values between 0--1 in-place.

```math
X' = \frac{(X-X_{min})}{X_{max}-X_{min}}
```
"""
function rangenormalize!(v::AbstractVector{T}) where {T <: AbstractFloat}
    rangenormalize!(v, minimum(v), maximum(v))
end

@doc raw"""
    rangenormalize(v::AbstractVector{T}) where T

Normalize a `Vector` to values between 0--1.

```math
X' = \frac{(X-X_{min})}{X_{max}-X_{min}}
```
"""
function rangenormalize(v::AbstractVector)
    out = Vector{Float64}(deepcopy(v))
    rangenormalize!(out)
    return out
end

# Normalize columns of a Matrix

@doc raw"""
    rangenormalizecols!(m::Matrix{T}, min_v::Vector{T}, max_v::Vector{T}, range::Vector{NTuple{2, T}}) where {T<:AbstractFloat}

Normalize each column of the `Matrix` `m` in-place to values between `range[j][1]` (``a``) and `range[j][2]` (``b``). `min_v[j]` and `max_v[j]` may be provided as "manual" extrema of the vector.

The `min_v`, `max_v` and `range` vectors should be the same length as `size(m, 2)`. For each of these vectors index `j` will be used to normalize column `j`.

```math
X' = a + \frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
```
"""
function rangenormalizecols!(m::Matrix{T},
        min_v::Vector{T},
        max_v::Vector{T},
        range::Vector{NTuple{2, T}}) where {T <: AbstractFloat}
    for j in 1:size(m, 2)
        @views rangenormalize!(m[:, j], min_v[j], max_v[j], range[j])
    end
end

@doc raw"""
    rangenormalizecols!(m::Matrix{T}, min_v::Vector{T}, max_v::Vector{T}) where {T<:AbstractFloat}

Normalize each column of the `Matrix` `m` in-place to values between 0--1. min_v[j]` and `max_v[j]` may be provided as "manual" extrema of the vector.

The `min_v` and `max_v` vectors should be the same length as `size(m, 2)`. For each of these vectors index `j` will be used to normalize column `j`.

```math
X' = \frac{(X-X_{min})}{X_{max}-X_{min}}
```
"""
function rangenormalizecols!(m::Matrix{T},
        min_v::Vector{T},
        max_v::Vector{T}) where {T <: AbstractFloat}
    for j in 1:size(m, 2)
        @views rangenormalize!(m[:, j], min_v[j], max_v[j])
    end
end

@doc raw"""
    rangenormalizecols!(m::Matrix{T}, min_v::Vector{T}, max_v::Vector{T}, range::NTuple{2, T}) where {T<:AbstractFloat}

Normalize each column of the `Matrix` `m` in-place to values between `range[1]` (``a``) and `range[2]` (``b``). `min_v[j]` and `max_v[j]` may be provided as "manual" extrema of the vector.

The `min_v` and `min_v` vectors should be the same length as `size(m, 2)`. For each of these vectors index `j` will be used to normalize column `j`.

```math
X' = a + \frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
```
"""
function rangenormalizecols!(m::Matrix{T},
        min_v::Vector{T},
        max_v::Vector{T},
        range::NTuple{2, T}) where {T <: AbstractFloat}
    for j in 1:size(m, 2)
        @views rangenormalize!(m[:, j], min_v[j], max_v[j], range)
    end
end

@doc raw"""
    rangenormalizecols!(m::Matrix{T}, min_v::T, max_v::T, range::NTuple{2, T}) where {T<:AbstractFloat}

Normalize each column of the `Matrix` `m` in-place to values between `range[1]` (``a``) and `range[2]` (``b``). `min_v` and `max_v` may be provided as "manual" extrema of the vector.

The same `min_v`, `max_v` and `range` values are used for every column.

```math
X' = a + \frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
```
"""
function rangenormalizecols!(m::Matrix{T},
        min_v::T,
        max_v::T,
        range::NTuple{2, T}) where {T <: AbstractFloat}
    for j in 1:size(m, 2)
        @views rangenormalize!(m[:, j], min_v, max_v, range)
    end
end

@doc raw"""
    rangenormalizecols!(m::Matrix{T}) where {T<:AbstractFloat}

Normalize each column of the `Matrix` `m` in-place to values between 0--1. Each column is normalized separately.

```math
X' = \frac{(X-X_{min})}{X_{max}-X_{min}}
```
"""
function rangenormalizecols!(m::Matrix{T}) where {T <: AbstractFloat}
    for j in 1:size(m, 2)
        @views rangenormalize!(m[:, j], minimum(m[:, j]), maximum(m[:, j]))
    end
end

@doc raw"""
    rangenormalizecols!(m::Matrix{T}, range::NTuple{2, T}) where {T<:AbstractFloat}

Normalize each column of the `Matrix` `m` in-place to values between `range[1]` (``a``) and `range[2]` (``b``). 

Each column is normalized separately.

```math
X' = a + \frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
```
"""
function rangenormalizecols!(m::Matrix{T}, range::NTuple{2, T}) where {T <: AbstractFloat}
    for j in 1:size(m, 2)
        @views rangenormalize!(m[:, j], minimum(m[:, j]), maximum(m[:, j]), range)
    end
end

@doc raw"""
    rangenormalizecols!(m::Matrix{T}, range::Vector{NTuple{2, T}}) where {T<:AbstractFloat}

Normalize each column of the `Matrix` `m` in-place to values between `range[j][1]` (``a``) and `range[j][2]` (``b``).

The `range` vector should be the same length as `size(m, 2)`. Index `j` will be used as the range to which column `j` will be normalized.

```math
X' = a + \frac{(X-X_{min})(b-a)}{X_{max}-X_{min}}
```
"""
function rangenormalizecols!(m::Matrix{T},
        range::Vector{NTuple{2, T}}) where {T <: AbstractFloat}
    for j in 1:size(m, 2)
        @views rangenormalize!(m[:, j], minimum(m[:, j]), maximum(m[:, j]), range[j])
    end
end
