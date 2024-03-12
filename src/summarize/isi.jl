#------------------------------
# Calculate ISI:s for a cluster
#------------------------------

"""

    isi(cluster::Cluster{T}) where {T}

Calculate the ISI of each spike in `cluster`.

"""
function isi(cluster::AbstractCluster{T, U}) where {T, U}
    return isi(spiketimes(cluster))
end

function isi(spikes::AbstractVector{T}) where {T}
    len = length(spikes) - 1
    out = Vector{T}(undef, len)
    isi!(out, spikes)
    return out
end

"""
    isi!(out::Vector{T}, spikes::Vector{T}) where {T<:Real}

In-place version of [`isi`](@ref). Length of `out` must be equal to `length(spikes) - 1`.
"""
function isi!(out::Vector{T}, spikes::AbstractVector{T}) where {T}
    @assert length(out) == length(spikes) - 1
    @inbounds @views for i in 1:length(out)
        out[i] = spikes[i + 1] - spikes[i]
    end
end

# Isi of RelativeCluster

function isi(spikes::Vector{Vector{T}}) where {T}
    out = Vector{Vector{T}}(undef, 0)
    for i in eachindex(spikes)
        out[i] = length(spikes[i]) > 1 ? isi(spikes[i]) : T[]
    end
    return out
end

function isi(spikes::RelativeSpikeVector{T, U}) where {T, U}
    out = Vector{Vector{T}}(undef, length(spikes))
    for i in eachindex(spikes)
        out[i] = length(spikes[i]) > 1 ? isi(spikes[i]) : T[]
    end
    return out
end
# Mean isi
"""
    meanisi!(exp::AbstractExperiment)

Add a column (named `mean_isi`) to the `info` DataFrame of `exp` containing the mean ISI of each `Cluster`.
"""
function meanisi!(exp::AbstractExperiment)
    clusters = @views clustervector(exp)
    exp.info.mean_isi = meanisi.(spiketimes.(clusters))
end

function meanisi(spikes::Vector{T}) where {T}
    out = zero(T)
    len = length(spikes) - 1
    @inbounds @views for i in 1:len
        out += spikes[i + 1] - spikes[i]
    end
    out / len
end
