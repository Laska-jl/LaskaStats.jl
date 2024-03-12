################################
#
# Calculate relative frequencies
#
################################

"""
Calculate relative frequency of a `RelativeCluster`/collection of spikes from a `RelativeSpikes` struct.

    relativefrequency(vec::Vector{Vector{T}}, period::N) where {T<:Real,N<:Real}
    relativefrequency(vec::Vector{Vector{T}}, steps::StepRange{T,T}) where {T<:Real}

A baseline is calculated for *each* trigger session. It is based on the mean number of 
spikes/period in bins in before time=0.

A custom `StepRange` may be provided instead of `period`. In this case, the period will be the `step` of the `StepRange`

# Examples

```julia
# Use 300 time-units as `period`.
relativefrequency(times, 300)

# Use 300 units of time as `period` and have the bins stretch from -1500 to 1500.
relativefrequency(times, -1500:300:1500)
```

"""
function relativefrequency(vec::RelativeSpikeVector, period::T) where {T}
    absolutes = frequency(vec, period)
    abslen = length(absolutes)
    # Determine number of bins before 0
    nbinspre = Int64(floor(abs(LaskaCore.minval(vec)) / period))

    baselines = Vector{Float64}(undef, length(vec))
    @views for n in eachindex(baselines)
        baselines[n] = sum(absolutes[n][1:nbinspre]) / nbinspre
    end
    for n in eachindex(baselines)
        baselineadjust!(absolutes[n], baselines[n])
    end
    out = zeros(Float64, length(absolutes[1]))

    @views for v in eachindex(absolutes)
        @simd for n in eachindex(absolutes[1])
            out[n] += absolutes[v][n]
        end
    end
    for n in eachindex(out)
        @inbounds out[n] /= abslen
    end

    return out
end

# Unitful period
function relativefrequency(times::RelativeSpikeVector,
        period::T) where {T <: LaskaCore.TUnit}
    periodconv = LaskaCore.timetosamplerate(times, period)
    absolutes = frequency(times, period)
    abslen = length(absolutes)
    # Determine number of bins before 0
    nbinspre = Int64(floor(abs(LaskaCore.minval(times)) / periodconv))

    baselines = Vector{Float64}(undef, length(times))
    @views for n in eachindex(baselines)
        baselines[n] = sum(absolutes[n][1:nbinspre]) / nbinspre
    end
    for n in eachindex(baselines)
        baselineadjust!(absolutes[n], baselines[n])
    end
    out = zeros(Float64, length(absolutes[1]))

    @views for v in eachindex(absolutes)
        @simd for n in eachindex(absolutes[1])
            out[n] += absolutes[v][n]
        end
    end
    for n in eachindex(out)
        @inbounds out[n] /= abslen
    end

    return out
end

function relativefrequency(vec::RelativeSpikeVector, steps::StepRange{T}) where {T}
    vfilt = LaskaCore.spikes_in_timerange(vec, steps)
    period = steps.step
    absolutes = frequency(vfilt, steps)
    abslen = length(absolutes)
    # Determine number of bins before 0
    nbinspre::Int64 = floor(abs(LaskaCore.minval(vfilt)) / period)

    baselines::Vector{Float64} = Vector{Float64}(undef, length(vfilt))
    @views for n in eachindex(baselines)
        baselines[n] = sum(absolutes[n][1:nbinspre]) / nbinspre
    end

    for n in eachindex(baselines)
        baselineadjust!(absolutes[n], baselines[n])
    end
    out::Vector{Float64} = zeros(length(absolutes[1]))

    @views for v in eachindex(absolutes)
        @simd for n in eachindex(absolutes[1])
            out[n] += absolutes[v][n]
        end
    end
    for n in eachindex(out)
        @inbounds out[n] /= abslen
    end

    return out
end

function baselineadjust(vec::Vector{T}, baseline) where {T <: Real}
    out = Vector{Float64}(undef, length(vec))
    if !iszero(baseline)
        for n in eachindex(vec)
            out[n] = vec[n] / baseline
        end
    end
    return out
end

function baselineadjust!(vec::Vector{T}, baseline::Float64) where {T <: AbstractFloat}
    if !iszero(baseline)
        for n in eachindex(vec)
            vec[n] /= baseline
        end
    end
end
