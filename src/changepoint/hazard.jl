
"""
    abstract type AbstractHazard end


Supertype for run-length priors used in [`LaskaStats.bocpd`](@ref).
"""
abstract type AbstractHazard end

"""
    struct ConstantHazard{T} <: AbstractHazard
        lambda::T
    end

Constant run-length prior for use with [`LaskaStats.bocpd`](@ref).
Assigning a higher prior will cause the result to tend towards higher run lengths.
"""
struct ConstantHazard{T} <: AbstractHazard
    lambda::T
end

function (hazard::ConstantHazard{T})(::U) where {T, U <: Number}
    return one(T) / hazard.lambda
end

function (hazard::ConstantHazard{T})(r::AbstractArray{U}) where {T, U}
    return ones(T, size(r)) ./ hazard.lambda
end
