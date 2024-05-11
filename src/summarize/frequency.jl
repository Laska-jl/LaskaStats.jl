#=========================================================================================
Calculate the frequency of a Cluster.
If a "whole" Cluster or their (Relative-)SpikeVector is passed the output should be in Hz.
If a normal <:AbstractVector is passed the output will be in spikes/bin.
=========================================================================================#

# TODO: Add methods with Unitful units for function using step(?)
# TODO: Change the names of the "frequency" functions that actually computes some for of psth-isch thing
# TODO: Clean up these docs
"""
    frequency(spiketimes, period)


Returns a Vector containing the frequency of `spiketimes` in the form of spikes/second binned at each multiple of `period`.            
Spiketimes are binned to the nearest *smaller* multiple of `period`. Ie a spike happening at time = 59999 will be in the 30000 bin when period = 30000.

Spiketimes may be provided in the form of:
- An `AbstractVector`
- A `SpikeVector`
- A `RelativeSpikeVector`
- A `Cluster`



# Example

For a cluster sampled at 30 000Hz...
```julia
LaskaStats.frequency(spikes, 30000)
```
...will return spikes/second.

Indexing into the result as:        
```julia
result[n]
```
Will return the spikes/s between `time = period * n` and `time < period * n+1`.


"""
function frequency end

"""

    frequency(cluster::Cluster, period::T) where {T<:Real}

Returns a Vector containing the frequency of the cluster in the form of spikes/second binned at each multiple of `period`.            
Spiketimes are binned to the nearest *smaller* multiple of `period`. Ie a spike happening at time = 59999 will be in the 30000 bin when period = 30000.

# Example

For a cluster sampled at 30 000Hz...
```julia
LaskaStats.frequency(cluster, 30000)
```
...will return spikes/second.

Indexing into the result as:        
```julia
result[n]
```
Will return the spikes/s between `time = period * n` and `time < period * n+1`.


"""
function frequency(cluster::Cluster, period::T) where {T}
    times = LaskaCore.spiketimes(cluster)
    return frequency(times, period)
end

# Frequency for SpikeVector

"""
    frequency(spikes::SpikeVector)

Returns the average frequency in spikes/second for the time period first spiketime--last spiketime.
"""
function frequency(spikes::SpikeVector)
    length(spikes) / ((spikes[end] - spikes[begin]) / LaskaCore.samplerate(spikes))
end

"""
    frequency(times::SpikeVector{T}, period::T) where {T}

Returns a Vector containing the frequency of the `SpikeVector` in the form of spikes/second binned at each multiple of `period`.            
Spiketimes are binned to the nearest *smaller* multiple of `period`. Ie a spike happening at time = 59999 will be in the 30000 bin when period = 30000.

# Example

For a cluster sampled at 30 000Hz...
```julia
LaskaStats.frequency(spikes, 30000)
```
...will return spikes/second.

Indexing into the result as:        
```julia
result[n]
```
Will return the spikes/s between `time = period * n` and `time < period * n+1`.


"""
function frequency(times::SpikeVector{T}, period::T) where {T}
    # The number by which to multiply each bin to convert it to Hz
    tconv = LaskaCore.samplerate(times) / period

    # NOTE: Should the binning be different? Use Laska.arbitraryround instead?
    lowerbound = LaskaCore.rounddown(minimum(times, init = 0), period)
    upperbound = LaskaCore.rounddown(maximum(times, init = 0), period)
    accumulator = Dict{T, Float64}(t => 0.0 for t in lowerbound:period:upperbound)

    @inbounds for n in eachindex(times)
        accumulator[LaskaCore.rounddown(times[n], period)] += 1.0
    end

    sorter = sortperm(collect(keys(accumulator)))
    out = collect(values(accumulator))[sorter]
    for i in eachindex(out)
        out[i] *= tconv
    end
    return out
end

# Version for SpikeVector with Unitful period
"""
    frequency(times::SpikeVector{T, U},
        period::TimeUnit) where {T, U, TimeUnit <: LaskaCore.TUnit}


Returns a Vector containing the frequency of the `SpikeVector` in the form of spikes/second binned at each multiple of `period`.            
Spiketimes are binned to the closest *smaller* multiple of `period`. Ie a spike happening at time = 2.99s will be in the 2s bin when period = `1u"s"`. 

`period` in this method should be a `Unitful.jl`-based unit of time.

# Example

```julia
LaskaStats.frequency(spikes, 1u"s")
```
...will return spikes/second.

Indexing into the result as:        
```julia
result[n]
```
Will return the spikes/s between `time = period * n` and `time < period * n+1`.


"""
function frequency(times::SpikeVector{T, U},
        period::TimeUnit) where {T, U, TimeUnit <: LaskaCore.TUnit}
    # The number by which to multiply each bin to convert it to Hz
    periodconv = LaskaCore.timetosamplerate(times, period)
    tconv = LaskaCore.samplerate(times) / periodconv

    # NOTE: Should the binning be different? Use Laska.arbitraryround instead?
    lowerbound = LaskaCore.rounddown(minimum(times, init = 0), periodconv)
    upperbound = LaskaCore.rounddown(maximum(times, init = 0), periodconv)
    accumulator = Dict{T, Float64}(t => 0.0 for t in lowerbound:periodconv:upperbound)

    @inbounds for n in eachindex(times)
        accumulator[LaskaCore.rounddown(times[n], periodconv)] += 1.0
    end

    sorter = sortperm(collect(keys(accumulator)))
    out = collect(values(accumulator))[sorter]
    @inbounds for i in eachindex(out)
        out[i] *= tconv
    end
    return out
end

"""
    frequency(cluster::Cluster)

Returns the average frequency in spikes/second for the time period first spiketime--last spiketime.
"""
function frequency(cluster::Cluster)
    spikes = spiketimes(cluster)
    length(spikes) / (spikes[end] / LaskaCore.samplerate(spikes))
end

"""
    frequency(cluster::RelativeCluster, period::T) where {T <: Real}

Returns a `Vector` of `Vector`s containing the frequency of the cluster in the form of spikes/second binned at each multiple of `period`.            
Spiketimes are binned to the closest *smaller* multiple of `period`. Ie a spike happening at time = 59999 will be in the 30000 bin.

# Example

For a cluster sampled at 30 000Hz...
```julia
LaskaStats.frequency(cluster, 30000)
```
...will return spikes/second.

Indexing into the result as:        

```julia
result[i][n]
```

Will return the n:th bin of the i:th trigger event which describes the number of spikes occuring between `time = period * n` and `time < period * n+1`.

"""
function frequency(cluster::RelativeCluster, period::T) where {T}
    times = spiketimes(cluster)
    return frequency(times, period)
end

# frequency of RelativeSpikes

function frequency(times::RelativeSpikeVector{T, U}, period::P) where {T, U, P}
    tconv = LaskaCore.samplerate(times) / period
    out = Vector{Vector{Float64}}(undef, length(times))
    lowerbound = LaskaCore.rounddown(LaskaCore.minval(times), period)
    upperbound = LaskaCore.rounddown(LaskaCore.maxval(times), period)
    len = lowerbound:period:upperbound
    for n in eachindex(out)
        out[n] = iszero(length(times[n])) ? zeros(Float64, length(len)) :
                 frequency(times[n], len) .* tconv
    end
    return out
end

function frequency(times::RelativeSpikeVector{T, U},
        period::P) where {T, U, P <: LaskaCore.TUnit}
    periodconv = Float64(LaskaCore.timetosamplerate(times, period))
    tconv = LaskaCore.samplerate(times) / periodconv
    out = Vector{Vector{Float64}}(undef, length(times))
    lowerbound = LaskaCore.rounddown(LaskaCore.minval(times), periodconv)
    upperbound = LaskaCore.rounddown(LaskaCore.maxval(times), periodconv)
    len = lowerbound:periodconv:upperbound
    for n in eachindex(out)
        out[n] = iszero(length(times[n])) ? zeros(Float64, length(len)) :
                 frequency(times[n], len) .* tconv
    end
    return out
end
# With steprange
function frequency(times::RelativeSpikeVector, steps::AbstractRange)
    tconv = LaskaCore.samplerate(times) / steps.step
    out = Vector{Vector{Float64}}(undef, length(times))
    tmp = LaskaCore.spikes_in_timerange(times, steps[begin], steps[end])
    for n in eachindex(tmp)
        out[n] = iszero(length(tmp[n])) ? zeros(Float64, length(steps)) :
                 frequency(tmp[n], steps) .* tconv
    end
    return out
end

function frequency(times::RelativeSpikeVector,
        steps::StepRange{T}) where {T <: LaskaCore.TUnit}
    lowerbound = timetosamplerate(times, steps[begin])
    upperbound = timetosamplerate(times, steps[end])
    period = timetosamplerate(times, steps.step)
    frequency(times, period, lowerbound, upperbound)
end

# Frequency with begin, end and step
function frequency(times::RelativeSpikeVector, period, t_lowerbound, t_upperbound)
    tmp = LaskaCore.spikes_in_timerange(times, t_lowerbound, t_upperbound)
    frequency(tmp, period)
end

# Versions without units/samplerates, just plain old binnin'

function frequency(times::AbstractVector{T}, period::P) where {T, P}
    # NOTE: Should the binning be different? Use Laska.arbitraryround instead?
    lowerbound = LaskaCore.rounddown(minimum(times, init = 0), period)
    upperbound = LaskaCore.rounddown(maximum(times, init = 0), period)
    accumulator = Dict{T, Int64}(t => 0 for t in lowerbound:period:upperbound)

    @inbounds for n in eachindex(times)
        accumulator[LaskaCore.rounddown(times[n], period)] += 1
    end

    sorter = sortperm(collect(keys(accumulator)))

    return collect(values(accumulator))[sorter]
end

function frequency(times::AbstractVector{T}, steps::AbstractRange) where {T}
    period = steps.step
    # NOTE: Should the binning be different? Use Laska.arbitraryround instead?
    accumulator = Dict{T, Int64}(t => 0 for t in steps)

    filtered = LaskaCore.spikes_in_timerange(times, steps[begin], steps[end])

    @inbounds for n in eachindex(filtered)
        accumulator[LaskaCore.rounddown(filtered[n], period)] += 1
    end

    sorter = sortperm(collect(keys(accumulator)))

    return collect(values(accumulator))[sorter]
end

# Version with StepRangeLen
function frequency(times::AbstractVector{T}, steps::StepRangeLen) where {T}
    period = Float64(steps.step)
    # NOTE: Should the binning be different? Use Laska.arbitraryround instead?
    accumulator = Dict{T, Int64}(t => 0 for t in steps)

    filtered = LaskaCore.spikes_in_timerange(times, steps[begin], steps[end])

    @inbounds for n in eachindex(filtered)
        accumulator[LaskaCore.rounddown(filtered[n], period)] += 1
    end

    sorter = sortperm(collect(keys(accumulator)))

    return collect(values(accumulator))[sorter]
end

# Frequency of Vector{Vector{T}}
function frequency(times::Vector{Vector{T}}, period::P) where {T, P}
    out = Vector{Vector{Int64}}(undef, length(times))
    lowerbound = LaskaCore.rounddown(LaskaCore.minval(times), period)
    upperbound = LaskaCore.rounddown(LaskaCore.maxval(times), period)
    len = lowerbound:period:upperbound
    for n in eachindex(out)
        out[n] = iszero(length(times[n])) ? zeros(Float64, length(len)) :
                 frequency(times[n], len)
    end
    return out
end

# TODO: Rename this
"""
    discretize(v::RelativeSpikeVector, min_t, max_t)
    discretize(::Type{T}, v::RelativeSpikeVector, min_t, max_t) where {T}

Returns a `Vector{Vector{Bool}}` where each inner vector is of length `max_t - min_t`.
Each entry of the inner vectors is `1`/`true` if there is a spike at the time matching the index of the entry. Index 1 represents `min_t` while the last index represents `max_t`.

If another type of elements in the inner `Vector` is desired (such as Int), one may pass this type as the first argument.

"""
function discretize(v::RelativeSpikeVector, min_t, max_t)
    index = abs(min_t) + 1
    out = [zeros(Bool, max_t - min_t) for _ in eachindex(v)]
    if max_t < LaskaCore.maxval(v) || LaskaCore.minval(v) < min_t
        spikes = LaskaCore.spikes_in_timerange(v, min_t, max_t)
    else
        spikes = v
    end
    for i in eachindex(out)
        @simd for j in eachindex(spikes[i])
            out[i][spikes[i][j] + index] = 1
        end
    end
    return out
end

function discretize(::Type{T}, v::RelativeSpikeVector, min_t, max_t) where {T}
    index = abs(min_t) + 1
    out = [zeros(T, max_t - min_t) for _ in eachindex(v)]
    if max_t < LaskaCore.maxval(v) || LaskaCore.minval(v) < min_t
        spikes = LaskaCore.spikes_in_timerange(v, min_t, max_t)
    else
        spikes = v
    end
    for i in eachindex(out)
        @simd for j in eachindex(spikes[i])
            out[i][spikes[i][j] + index] = oneunit(T)
        end
    end
    return out
end
