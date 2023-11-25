##################################################################
# 
# Calculate the CV2 CV2 (CV2= 2 * |ISIn+1 - ISIn|/(ISIn+1 + ISIn)) 
#
##################################################################

"""

    cv2(cluster::Cluster)

Returns CV2 values of `cluster` as a vector.

CV2 is calculated according to:

```math
CV2 = \\frac{2|ISI_{n+1} - ISI_n|}{(ISI_{n+1} + ISI_n)}
```

"""
function cv2(cluster::Cluster)
    out::Vector{Float64} = Vector{Float64}(undef, nspikes(cluster) - 2)
    spikes = spiketimes(cluster)
    @simd for n in 1:length(spikes)-2
        isi1 = spikes[n+1] - spikes[n]
        isi2 = spikes[n+2] - spikes[n+1]
        out[n] = @inline calculatecv2(isi1, isi2)
    end
    return out
end

"""

    cv2mean(cluster::Cluster)

Calculates the mean CV2 of `cluster`.

"""
function cv2mean(cluster::Cluster)
    out::Float64 = 0
    nspik = nspikes(cluster) - 2
    spikes = spiketimes(cluster)
    @views @simd for n in 1:length(spikes)-2
        isi1 = spikes[n+1] - spikes[n]
        isi2 = spikes[n+2] - spikes[n+1]
        out += @inline calculatecv2(isi1, isi2)
    end
    return out / nspik
end

"""

    calculatecv2(n::T, nplusone::T)

Helper function for calculating CV2

"""
function calculatecv2(n::T, nplusone::T) where {T<:Real}
    return 2 * abs(nplusone - n) / (nplusone + n)
end
