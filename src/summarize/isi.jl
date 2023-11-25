#------------------------------
# Calculate ISI:s for a cluster
#------------------------------


"""

    isi(cluster::Cluster{T}) where {T<:Real}

Calculate the ISI of each spike in `cluster`.

"""
function isi(cluster::Cluster{T}) where {T<:Real}
    return isi(spiketimes(cluster))
end

function isi(spikes::Vector{T}) where {T<:Real}
    out::Vector{T} = Vector(undef, length(spikes) - 1)
    isi!(out, spikes)
    return out
end

function isi!(out::Vector{T}, spikes::Vector{T}) where {T<:Real}
    @assert length(out) == length(spikes) - 1
    @inbounds @views for i in 1:length(out)
        out[i] = spikes[i+1] - spikes[i]
    end
end
