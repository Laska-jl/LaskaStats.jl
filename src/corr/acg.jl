#=
https://github.com/cortex-lab/phylib/blob/master/phylib/stats/ccg.py
=#

"""
    acg(cluster::Cluster, binsize, winsize, symmetrize=true)
    acg(cluster::Cluster, binsize::LaskaCore.TUnit, winsize::LaskaCore.TUnit, symmetrize=true)
    acg(spikes::SpikeVector, binsize::LaskaCore.TUnit, winsize::LaskaCore.TUnit, symmetrize=true)

Calculate the autocorrelogram of `cluster` or `spikes`. Bin size and window size may be provided in the form of units from `Unitful.jl`.
If a specific unit is not provided the bin-/windowsize is assumed to be in the same unit as the sample rate of the `cluster`/`spikes`.

Code for the function inspired by https://github.com/cortex-lab/phylib.
"""
function acg end

function acg(cluster::Cluster, binsize, winsize, symmetrize=true)
	return acg(spiketimes(cluster), binsize, winsize, symmetrize)
end

function acg(spikes::SpikeVector, binsize::LaskaCore.TUnit, winsize::LaskaCore.TUnit, symmetrize=true)
	binsamp = LaskaCore.timetosamplerate(spikes, binsize)
    winsamp = LaskaCore.timetosamplerate(spikes, winsize)
    return acg(spikes, binsamp, winsamp, symmetrize)
end


function acg(cluster::Cluster, binsize::LaskaCore.TUnit, winsize::LaskaCore.TUnit, symmetrize=true)
	binsamp = LaskaCore.timetosamplerate(cluster, binsize)
    winsamp = LaskaCore.timetosamplerate(cluster, winsize)
    acg(cluster, binsamp, winsamp, symmetrize)
end

function acg(spikes::AbstractVector{T}, binsize, winsize, symmetrize=true) where {T}
    winsize_bins = 2 * Int64(floor((0.5 * winsize / binsize) + 1))

    shift = 1

    mask = ones(Bool, length(spikes))

    corrarr = zeros(Int32, fld(winsize_bins, 2) + 1)

    while any(mask[begin:end-shift])
	    spike_diff = LaskaStats.diff_shifted(spikes, shift)
        spike_diff_b = Int64.(fld.(spike_diff, binsize))

        updatemask!(mask, shift, spike_diff_b, winsize_bins)
        
        @views d = spike_diff_b[mask[begin:end-shift]]
        LaskaStats.increment!(corrarr, d)
        shift += 1
    end
    if symmetrize
        corrarr = vcat(reverse(corrarr), corrarr)
    end
    return corrarr
end

# Update mask, hiding spikes without a match
function updatemask!(mask, shift, spikediff, winsizebins)
    va = @views mask[begin:end-shift]
    vb = @views va[spikediff .> fld(winsizebins, 2)]
    vb .= zero(Bool)
end

function diff_shifted(v, steps)
	return @views v[steps + 1:end] .- v[begin:length(v) - steps]
end

function noccurences(v::AbstractVector{T}) where {T}
	out = zeros(T, maximum(v, init=-1) + 1)
    for i in v
        out[i + 1] += one(T)
    end
    return out
end

function increment!(v, indices)
	bbins = noccurences(indices)
    for i in eachindex(bbins)
        v[i] += bbins[i]
    end
end

