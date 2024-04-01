#=
Calculates the Fano Factor
As described in "Principles of Spike Train Analysis" by Fabrizio Gabbiani & Christof Koch
=#

#function fanofactor(spikes::SpikeVector, n::Integer)
#    fr = frequency(spikes)
#    ((fr^n) * exp(-fr)) / factorial(n)
#end

function fanofactor(spikes::SpikeVector, window::AbstractRange)
    sp = LaskaCore.spikes_in_timerange(spikes, window)
    fr = frequency(sp)
    time = ustrip(LaskaCore.sampleratetounit(LaskaCore.samplerate(sp),
        window[end] - window[begin],
        u"s"))
    0.5 + (fr / (8 * time)) * (1 - exp(-4 * fr / time))
end

"""
    fanofactor(spikes::RelativeSpikeVector)

Calculates the Fano factor for the trials in a `RelativeSpikeVector`.
"""
function fanofactor(spikes::RelativeSpikeVector)
    ns = length.(spikes)
    Statistics.var(ns) / mean(ns)
end
