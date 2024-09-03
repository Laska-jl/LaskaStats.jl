# Average frequency
function meanfrequency(spikes::Vector{T}, samplerate) where {T <: Real}
    (length(spikes) / spikes[end]) * samplerate
end

function meanfrequency(spikes::Vector{T}, samplerate::Real, t_start::Real) where {T <: Real}
    (length(spikes) / (spikes[end] - t_start)) * samplerate
end

function meanfrequency(spikes::SpikeVector{T}) where {T <: Real}
    (length(spikes) / spikes[end]) * LaskaCore.samplerate(spikes)
end

function meanfrequency(spikes::SpikeVector{T}, t_init::Real) where {T <: Real}
    (length(spikes) / (spikes[end] - t_init)) * LaskaCore.samplerate(spikes)
end

# Binned frequency

function __frequency!(out, spikes::Vector{<:Real}, binsize, samplerate, t_min)
    for s in spikes
        out[Int64((s - t_min) ÷ binsize + 1)] += 1.0
    end
    conv = binsize / samplerate
    out ./= conv
end

function frequency(spikes::Vector{T}, binsize::Real, samplerate::Real) where {T <: Real}
    lowerbound = LaskaCore.rounddown(spikes[begin], binsize)
    upperbound = LaskaCore.rounddown(spikes[end], binsize)
    steps = lowerbound:binsize:upperbound
    out = zeros(Float64, length(steps))
    __frequency!(out, spikes, binsize, samplerate, lowerbound)
    return out
end

function frequency(spikes::Vector{T}, binsize::LaskaCore.TUnit, samplerate::Real) where {T}
    binsizesec = ustrip(Float64, u"s", binsize)
    binsizesamp = round(Int64, binsizesec * samplerate)
    frequency(spikes, binsizesamp, samplerate)
end

function frequency(spikes::SpikeVector, binsize::Real)
    lowerbound = LaskaCore.rounddown(spikes[begin], binsize)
    upperbound = LaskaCore.rounddown(spikes[end], binsize)
    steps = lowerbound:binsize:upperbound
    out = zeros(Float64, length(steps))
    __frequency!(out, spiketimes(spikes), binsize, LaskaCore.samplerate(spikes), lowerbound)
    return out
end

# Binned frequency with Unitful binsize
function frequency(spikes::SpikeVector, binsize::LaskaCore.TUnit)
    binsizesamp = LaskaCore.timetosamplerate(spikes, binsize)
    lowerbound = LaskaCore.rounddown(spikes[begin], binsizesamp)
    upperbound = LaskaCore.rounddown(spikes[end], binsizesamp)
    steps = lowerbound:binsizesamp:upperbound
    out = zeros(Float64, length(steps))
    __frequency!(
        out, spiketimes(spikes), binsizesamp, LaskaCore.samplerate(spikes), lowerbound)
    return out
end

# For Vector{Vector{T}} or RelativeSpikes{T}

function __frequency!(out::Vector{<:Vector{<:AbstractFloat}},
        spikes::Vector{<:Vector{<:Real}}, binsize::Real, samplerate::Real, t_min::Real)
    conv = binsize / samplerate
    for i in eachindex(spikes)
        for s in spikes[i]
            out[i][Int64((s - t_min) ÷ binsize + 1)] += 1.0
        end
        out[i] ./= conv
    end
end

function frequency(spikes::Vector{<:Vector{<:Real}}, binsize::Real, samplerate::Real)
    lowerbound = LaskaCore.rounddown(LaskaCore.minval(spikes), binsize)
    upperbound = LaskaCore.rounddown(LaskaCore.maxval(spikes), binsize)
    steps = lowerbound:binsize:upperbound
    out = [zeros(Float64, length(steps)) for _ in 1:length(spikes)]
    __frequency!(out, spikes, binsize, samplerate, lowerbound)
    return out
end

function frequency(spikes::RelativeSpikeVector, binsize::Real)
    lowerbound = LaskaCore.rounddown(LaskaCore.minval(spikes), binsize)
    upperbound = LaskaCore.rounddown(LaskaCore.maxval(spikes), binsize)
    steps = lowerbound:binsize:upperbound
    out = [zeros(Float64, length(steps)) for _ in 1:length(spikes)]
    __frequency!(out, LaskaCore.spiketimes(spikes), binsize,
        LaskaCore.samplerate(spikes), lowerbound)
    return out
end

function frequency(
        spikes::Vector{<:Vector{<:Real}}, binsize::LaskaCore.TUnit, samplerate::Real)
    binsizesec = ustrip(Float64, u"s", binsize)
    binsizesamp = binsizesec * samplerate
    frequency(spikes, binsizesamp, samplerate)
end

function frequency(
        spikes::RelativeSpikeVector, binsize::LaskaCore.TUnit)
    binsizesamp = LaskaCore.timetosamplerate(spikes, binsize)
    frequency(spikes, binsizesamp)
end
