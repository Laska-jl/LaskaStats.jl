################################
#
# Calculate relative frequencies
#
################################

"""
    relativefrequency(vec::RelativeSpikeVector, period::T) where {T}


A baseline is calculated for *each* trigger session. It is based on the mean number of 
spikes/bins in bins before time=0.



A custom `StepRange` may be provided instead of `period`. In this case, the period will be the `step` of the `StepRange`

# Examples

```julia
# Use 300 time-units as `period`.
relativefrequency(times, 300)


# Period in the form of a Unitful time
relativefrequency(times, 10u"ms")

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
