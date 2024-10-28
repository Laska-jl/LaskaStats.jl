

"""
    frstability(spikes::SpikeVector, binsize, treshold)

Find stability violations in `spikes`.

Defines stability violations as periods where the binned frequency deviates from the mean
binned frequency by more than the standard deviation multiplied by `threshold`.

Returns a vector of
"""
function frstability(spikes::SpikeVector, binsize, threshold; verbose = false)
	frq = frequency(spikes, binsize)
    tresh = std(frq) * threshold
    meanfrq = mean(frq)
    verbose && println("Thresholds: $meanfrq +/- $tresh")
    l = meanfrq - tresh
    u = meanfrq + threshold

    findall(x -> u < x || x < l, frq)

end

function frstability(spikes::Vector, binsize, samplerate, threshold; verbose = false)
	frq = frequency(spikes, binsize, samplerate)

    tresh = std(frq) * threshold
    meanfrq = mean(frq)
    verbose && println("Thresholds: $meanfrq +/- $tresh")
    l = meanfrq - tresh
    u = meanfrq + threshold

    findall(x -> u < x || x < l, frq)
end

function __violation_bins_to_spike_inds(spikes::Vector, vbins::Vector{<:Integer}, binsize::Real)
	spikesbinned = LaskaCore.rounddownvec(spikes, binsize)
    findall(Base.Fix2(in, vbins), spikesbinned)
end
