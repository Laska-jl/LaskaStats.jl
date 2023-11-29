################################
#
# Calculate relative frequencies
#
################################

"""
Calculate relative frequency of a `RelativeCluster`/collection of spikes from a `RelativeSpikes` struct.

    relativefrequency(vec::Vector{Vector{T}}, period::N) where {T<:Real,N<:Real}

A baseline is calculated for *each* trigger session. It is based on the mean number of 
spikes/period in bins in before time=0.

"""
function relativefrequency(vec::Vector{Vector{T}}, period::N) where {T<:Real,N<:Real}
    absolutes::Vector{Vector{Float64}} = frequency(vec, period, true)
    abslen = length(absolutes)
    # Determine number of bins before 0
    nbinspre::Int64 = floor(abs(LaskaCore.minval(vec)) / period)

    baselines::Vector{Float64} = Vector{Float64}(undef, length(vec))
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

function baselineadjust(vec::Vector{T}, baseline::Float64) where {T<:Real}
    out::Vector{Float64} = Vector{Float64}(undef, length(vec))
    if !iszero(baseline)
        for n in eachindex(vec)
            out[n] = vec[n] / baseline
        end
    end
    return out
end

function baselineadjust!(vec::Vector{T}, baseline::Float64) where {T<:AbstractFloat}
    if !iszero(baseline)
        for n in eachindex(vec)
            vec[n] /= baseline
        end
    end
end
