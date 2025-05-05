# Detects ISI-violations, ie ISIs below a certain threshold

"""
    isiviolations(spikes, threshold) where {T<:Real}

Finds and the indices of ISI violations.
Returns an empty vector if no violations are found.

`spikes` may be a: 

- `Vector{<:Real}`
- `Vector{<:Vector{<:Real}}`
- `SpikeVector`
- `RelativeSpikeVector`

If `spikes` is a `SpikeVector` or `RelativeSpikeVector` `threshold` may be a `Unitful.jl` unit of time.

A violation is defined as an ISI less than or equal to `threshold`.
Indices returned are the second spike in the pairs that give rise to a violation.

For example, if threshold = 1

```console
             *  *
[0, 2, 4, 6, 7, 8, 10]
```

The indices marked by `*` (`[5, 6]`) will be returned.
"""
function isiviolations(spikes::Vector{<:Real}, threshold)
    inds = Int64[]
    for i in Iterators.drop(eachindex(spikes), 1)
        spikes[i] - spikes[i-1] <= threshold && push!(inds, i)
    end
    return inds
end

function isiviolations(spikes::Vector{<:Vector{<:Real}}, threshold)
    inds = [Int64[] for _ in 1:length(spikes)]
    for n in eachindex(spikes)
        for i in Iterators.drop(eachindex(spikes[n]), 1)
            spikes[n][i] - spikes[n][i-1] <= threshold && push!(inds[n], i)
        end
    end
    return inds
end

function isiviolations(spikes::RelativeSpikeVector, threshold)
    isiviolations(spiketimes(spikes), threshold)
end

function isiviolations(spikes::RelativeSpikeVector, threshold::LaskaCore.TUnit)
    thresholdsamp = LaskaCore.timetosamplerate(spikes, threshold)
    isiviolations(spiketimes(spikes), thresholdsamp)
end

function isiviolations(spikes::AbstractSpikeVector, threshold::LaskaCore.TUnit)
    thresholdsamp = LaskaCore.timetosamplerate(spikes, threshold)
    isiviolations(spikes, thresholdsamp)
end


function isiviolations(spikes::AbstractSpikeVector, threshold)
    isiviolations(spiketimes(spikes), threshold)
end
