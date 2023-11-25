#---------------------------------------------------------------
# Median absolute difference from the median interspike interval
#---------------------------------------------------------------

"""

    mad(cluster::T) where {T<:Laska.AbstractCluster}

Calculate the median absolute difference from the median interspike interval of a cluster.

"""
function mad(cluster::T) where {T<:AbstractCluster}
    return mad(spiketimes(cluster))
end

function mad(vec::Vector{T}) where {T<:Real}
    isis::Vector{Float64} = isi(vec)
    medianisi::Float64 = median(isis)
    @inbounds @views for i in eachindex(isis)
        isis[i] = abs(isis[i] - medianisi)
    end
    return median(isis)
end
