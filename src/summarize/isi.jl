#------------------------------
# Calculate ISI:s for a cluster
#------------------------------

"""

    isi(cluster::Cluster{T}) where {T}

Calculate the ISI of each spike in `cluster`.

"""
function isi(cluster::AbstractCluster)
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
    @views for i in 1:length(out)
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

function meanisi(spikes::AbstractVector)
    out = 0.0
    len = length(spikes) - 1
    @inbounds @views for i in 1:len
        out += spikes[i + 1] - spikes[i]
    end
    out / len
end

function meanisi(cluster::Cluster)
    meanisi(spiketimes(cluster))
end

"""
    percentileisi(spikes, perc)

Returns the percentile of the ISI distribution specifed by `perc`.
Accepts a `Vector` of spiketimes, a [`LaskaCore.Cluster`](@ref) or a [`LaskaCore.PhyOutput`](@ref)
in which case a `Vector` with results for each `Cluster` is returned.

# See also 

[`LaskaStats.percentileisi!`](@ref) which adds the result to the `info` dataframe of a `PhyOutput`.

"""
function percentileisi end

function percentileisi(spikes::AbstractVector, perc::AbstractFloat)
    isis = isi(spikes)
    StatsBase.percentile(isis, perc)
end

function percentileisi(cluster::Cluster, perc::AbstractFloat)
    percentileisi(spiketimes(cluster), perc)
end

function percentileisi(exp::PhyOutput, perc::AbstractFloat)
    clusters = clustervector(exp)
    @. percentileisi(spiketimes(clusters), perc)
end

"""
    percentileisi!(exp::PhyOutput, perc::AbstractFloat)

Calculates the specified percentile of the ISI-distribution for each cluter in `exp`
and stores the result in the `info` dataframe of `exp`.

"""
function percentileisi!(exp::PhyOutput, perc::AbstractFloat)
    exp.info[:, "ISI_perc" * split(repr(round(perc, sigdigits = 2)), ".")[2]] = percentileisi(
        exp, perc)
end

function medianisi(spikes::AbstractVector)
    median(isi(spikes))
end

function medianisi(cluster::Cluster)
    median(isi(cluster))
end

function medianisi(experiment::PhyOutput)
    clusters = clustervector(experiment)
    @. median(isi(clusters))
end

function medianisi!(experiment::PhyOutput)
    clusters = clustervector(experiment)
    experiment.info.median_isi = @. median(isi(clusters))
end

# function isi_distr_exp_diff(arguments)
#     
# end
