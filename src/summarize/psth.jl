"""
    psthbins(spikes, binsize, [samplerate])

Used for determining the bins to be included in a peristimulus time histogram as 
returned from [`LaskaStats.psth`](@ref).

Finds the first and last bins that includes a spike and returns a range: `first:binsize:last`
that represents the time of each bin returned by [`LaskaStats.psth`](@ref) when called with 
the same arguments. The returned range will be in the same unit as `spikes` no matter the unit of other arguments.

`binsize` may be a `Unitful.jl` time. If `spikes` are not a `RelativeSpikeVector` or `SpikeVector`
`samplerate` will need to be provided *if* binsize is a `Unitful.jl` time.

"""
function psthbins end


function psthbins(spikes::Union{Vector{<:Real}, SpikeVector}, binsize::Real)
    lowerbound = LaskaCore.rounddown(spikes[begin], binsize)
    upperbound = LaskaCore.rounddown(spikes[end], binsize)
    lowerbound:binsize:upperbound
end

function psthbins(spikes::Vector{<:Real}, binsize::LaskaCore.TUnit, samplerate::Real)
    binsizesec = ustrip(Float64, u"s", binsize)
    binsizesamp = binsizesec * samplerate
    psthbins(spikes, binsizesamp)
end

function psthbins(spikes::SpikeVector, binsize::LaskaCore.TUnit)
    binsizesamp = LaskaCore.timetosamplerate(spikes, binsize)
    psthbins(spikes, binsizesamp)
end

function psthbins(spikes::Vector{<:Vector{<:Real}}, binsize::Real)
    lowerbound = LaskaCore.rounddown(LaskaCore.minval(spikes), binsize)
    upperbound = LaskaCore.rounddown(LaskaCore.maxval(spikes), binsize)
    lowerbound:binsize:upperbound
end

function psthbins(spikes::Vector{<:Vector{<:Real}}, binsize::LaskaCore.TUnit, samplerate::Real)
    binsizesec = ustrip(Float64, u"s", binsize)
    binsizesamp = binsizesec * samplerate
    psthbins(spikes, binsizesamp)
end

function psthbins(spikes::RelativeSpikeVector, binsize::LaskaCore.TUnit)
    psthbins(LaskaCore.spiketimes(spikes), binsize, LaskaCore.samplerate(spikes))
end

function __psth!(out::Vector{T}, spikes::Vector{<:Real},
        binsize::Real, t_min::Real) where {T <: Integer}
    for s in spikes
        out[Int64((s - t_min) ÷ binsize + 1)] += one(T)
    end
end

"""
    psth(spikes, binsize, [samplerate])

Calculate the values of a peri stimulus time histogram. Returns a `Vector` with the absolute
number of spikes in each bin as specified in `binsize`.

`binsize` may be supplied as a "normal" number assumed to be of the same unit as the spikes
or as a `Unitful.jl` time. If `spikes` are a normal `Vector` instead of a `RelativeSpikeVector` 
or `SpikeVector` the `samplerate` needs to be specified *if* `binsize` is a `Unitful.jl` time.

The included bins are not guaranteed to start at 0. Rather, the first bin will be the first one
that actually holds a spike. For example, if binsize = `300` and the first spike occurs at t = `1598`, 
the first bin will represent t = `1500:1799`. In order to easily inspect or use the actual times of each bin
one may use the helper function [`LaskaStats.psthbins`](@ref).

"""
function psth end

function psth(spikes::Vector{<:Real}, binsize::Real)
    steps = psthbins(spikes, binsize)
    out = zeros(Int64, length(steps))
    __psth!(out, spikes, binsize, steps[begin])
    return out
end

function psth(spikes::Vector{<:Real}, binsize::LaskaCore.TUnit, samplerate::Real)
    binsizesec = ustrip(Float64, u"s", binsize)
    binsizesamp = binsizesec * samplerate
    psth(spikes, binsizesamp)
end

# With SpikeVector

function psth(spikes::SpikeVector, binsize::Real)
    steps = psthbins(spikes, binsize)
    out = zeros(Int64, length(steps))
    __psth!(out, spiketimes(spikes), binsize, steps[begin])
    return out
end

function psth(spikes::SpikeVector, binsize::LaskaCore.TUnit)
    binsizesamp = LaskaCore.timetosamplerate(spikes, binsize)
    psth(spikes, binsizesamp)
end

# Actual psths

function __psth!(out::Vector{T}, spikes::Vector{<:Vector{<:Real}}, binsize, t_min) where {T<:Integer}
    for v in spikes
        for s in v
            out[Int64((s - t_min) ÷ binsize + 1)] += one(T)
        end
    end
end

function psth(spikes::Vector{<:Vector{<:Real}}, binsize::Real)
    steps = psthbins(spikes, binsize)
    out = zeros(Int64, length(steps))
    __psth!(out, spikes, binsize, steps[begin])
    return out
end

function psth(spikes::Vector{<:Vector{<:Real}}, binsize::LaskaCore.TUnit, samplerate::Real)
    binsizesec = ustrip(Float64, u"s", binsize)
    binsizesamp = binsizesec * samplerate
    psth(spikes, binsizesamp)
end

function psth(spikes::RelativeSpikeVector, binsize::Real)
    psth(spiketimes(spikes), binsize)
end

function psth(spikes::RelativeSpikeVector, binsize::LaskaCore.TUnit)
    psth(LaskaCore.spiketimes(spikes), binsize, LaskaCore.samplerate(spikes))
end
