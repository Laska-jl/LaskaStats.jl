
abstract type AbstractHazard end

struct ConstantHazard{T} <: AbstractHazard
    lambda::T
end

function (hazard::ConstantHazard{T})(::U) where {T, U <: Number}
    return one(T) / hazard.lambda
end

function (hazard::ConstantHazard{T})(r::AbstractArray{U}) where {T, U}
    return ones(T, size(r)) ./ hazard.lambda
end
