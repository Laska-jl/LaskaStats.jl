#=
Rhythm index as defined in:
"Serotonin modulation of inferior olivary oscillations and synchronicity: a multiple-electrode study in the rat cerebellum"
DOI: 10.1111/j.1460-9568.1995.tb00657.x
=#

# TODO: Write docstrings for these

"""
    function rhythmindex(
            spikes,
            winsize,
            binsize,
            recordingtime = nothing,
            first_peak_time = (10:40)u"ms",
            next_latency = 10u"ms",
            baseline_std_min = 1,
            diff_std_min = 2
    )

Calculates the rhythm index (RI) similarily to [lang_differential_1997](@cite).

Operates in the following manner:


1. Calculate the baseline: ``\\frac{(total\\ number\\ of\\ spikes)^2 \\times bin\\ width}{recording\\ time}``
2. Finds the maximum within the time frame specified in `first_peak_time`.
3. Searches for the next valley within the time frame ``t_{previous} + \\frac{t_{first}}{2} - bin\\_latency`` to ``t_{previous} + t_{first}  \\times 1.5 + bin\\_latency``.
4. Repeat step 2 until the whole ACG has been searched.
5. Calculate the RI as shown below.

Peaks/valleys are included by default if they are one standard deviation above/below the baseline *or* if they are two standard deviations greater/lesser than the last peak/valley.
These limits can be controlled with `baseline_std_min` and `diff_std_min` respectively.

Once peaks and valleys have been found the RI is caclulated according to:

``\\frac{a_1}{z} + \\frac{b_1}{z} + ... + \\frac{a_n}{z} + \\frac{b_n}{z}``

Where ``z`` is the total number of spikes and ``a_n``/``b_n`` is the absolute difference of the n:th peak/valley from the baseline in the ACG.

If no initial peak is found the function returns `0.0`.

# Arguments

- `spikes`: May be a [`LaskaCore.SpikeVector`](@ref), a [`LaskaCore.Cluster`](@ref) or a [`LaskaCore.PhyOutput`](@ref). In the last case, the output will be a `Vector` with the RI for all `Cluster`s in the `PhyOutput`.
- `winsize`: Window size for calculating the autocorrelogram (ACG).
- `binsize`: Bin size for caclulating the ACG.
- `recordingtime`: Total recording time of the experiment for calculating the baseline. Defaults to `nothing` in which the last spike time is used.
- `first_peak_time`: The time range in which the first peak should be sought. Defaults to 10-40ms.
- `next_latency`: Influences the length of the time segment in which to search for the next peak/valley. See step 3 above. Defaults to 10ms.
- `baseline_std_min`: Controls how many standard deviations above/below the baseline a peak must reach to be included. Defaults to 1.
- `diff_std_min`: Controls how many standard deviations greater/lesser a potential peak/valley must be compared to the last peak/valley to be included. Defaults to 2.

"""
function rhythmindex end

function rhythmindex(
        experiment::PhyOutput,
        winsize::LaskaCore.TUnit,
        binsize::LaskaCore.TUnit,
        diff_std_baseline::Real = 1,
        diff_std_next::Real = 2
)
    spikes = spiketimes.(clustervector(experiment))
    out = zeros(length(spikes))
    for i in eachindex(spikes)
        out[i] = rhythmindex(spikes[i], winsize, binsize, diff_std_baseline, diff_std_next)
    end
    return out
end

function rhythmindex(
        cluster::LaskaCore.Cluster,
        winsize::LaskaCore.TUnit,
        binsize::LaskaCore.TUnit,
        diff_std_baseline::Real = 1,
        diff_std_next::Real = 2
)
    rhythmindex(spiketimes(cluster), winsize, binsize, diff_std_baseline, diff_std_next)
end

function rhythmindex(
        spikes::SpikeVector,
        winsize::LaskaCore.TUnit,
        binsize::LaskaCore.TUnit,
        diff_std_baseline::Real = 1,
        diff_std_next::Real = 2
)
    winsize_samp = LaskaCore.timetosamplerate(spikes, winsize)
    binsize_samp = LaskaCore.timetosamplerate(spikes, binsize)

    rhythmindex(spikes, winsize_samp, binsize_samp, diff_std_baseline, diff_std_next)
end

# Main function
function rhythmindex(
        spikes::AbstractVector, winsize::Real, binsize::Real,
        diff_std_baseline::Real = 1, diff_std_next::Real = 2, w = 10)
    autocorr = acg(spikes, binsize, winsize, false)

    n_spikes = length(spikes)

    baseline, _ = SingularSpectrumAnalysis.analyze(
        autocorr, length(autocorr) ÷ 5, robust = false)

    peaks, valleys = acgpeaksvalleys(
        autocorr, w, baseline, diff_std_baseline, diff_std_next)

    isempty(peaks) && return 0.0

    __calculaterhythmindex(autocorr, peaks, valleys, n_spikes, baseline)
end

function acgpeaksvalleys(autocorr, w, baseline, diff_std_baseline = 1, diff_std_next = 2)
    peaks = __findpeaks(autocorr, w)
    valleys = __findvalleys(autocorr, w)
    __filterpeaksvalleys!(
        peaks, valleys, autocorr, baseline, diff_std_baseline, diff_std_next)
    return peaks, valleys
end

function __calculaterhythmindex(autocorrelogram, peakinds, valleyinds, n_spikes, baseline)
    result = 0.0
    # Add peaks
    result += sum(@. abs((@view(autocorrelogram[peakinds]) - baseline[peakinds]) /
                         n_spikes))
    # Add valleys
    result += sum(@. abs((@view(autocorrelogram[valleyinds]) - baseline[valleyinds]) /
                         n_spikes))
    return result
end

function __filterpeaksvalleys!(peaks::Vector{T}, valleys::Vector{T}, autocorr, baseline,
        diff_std_baseline = 1, diff_std_next = 2) where {T <: Integer}
    peakmask = ones(Bool, length(peaks))
    valleymask = ones(Bool, length(valleys))

    if @views autocorr[peaks[1]] > baseline[peaks[1]] * diff_std_baseline
        peakmask[1] = false
    else
        deleteat!(peaks, peakmask)
        deleteat!(valleys, valleymask)
        return false
    end

    stdd = std(autocorr)

    for i in Iterators.drop(eachindex(peaks), 1)
        current = autocorr[peaks[i]]
        next_valley = Base.Sort.searchsortedfirst(valleys, peaks[i])
        next_valley = next_valley > length(valleys) ? length(valleys) : next_valley
        if abs(current - baseline[peaks[i]]) > stdd * diff_std_baseline ||
           abs(current - autocorr[valleys[next_valley]]) > stdd * diff_std_next
            peakmask[i] = false
        else # Break if current is too small to be included
            break
        end
    end

    for i in eachindex(valleys)
        current = autocorr[valleys[i]]
        next_peak = Base.Sort.searchsortedfirst(peaks, valleys[i])
        next_peak = next_peak > length(peaks) ? length(peaks) : next_peak
        if abs(current - baseline[valleys[i]]) > stdd * diff_std_baseline ||
           abs(autocorr[peaks[next_peak]] - current) > stdd * diff_std_next
            valleymask[i] = false
        else
            break
        end
    end

    deleteat!(peaks, peakmask)
    deleteat!(valleys, valleymask)

    return true
end

function binstotime(bin, binsize)
    bin * binsize
end

function timetobin(time, binsize)
    time / binsize |> floor |> Int64
end

function timetobins(trange, binsize)
    first = trange[begin] / binsize |> floor |> Int64
    last = trange[end] / binsize |> floor |> Int64
    return first:last
end

function __findpeaks(v::AbstractVector, w)
    w < 1 && throw(ArgumentError("w must be > 0"))
    peaks = Vector{Int64}(undef, 0)
    i = w + 1
    while i < length(v) - w
        if all(@views v[(i - w):(i - 1)] .< v[i] .> v[(i + 1):(i + w)])
            push!(peaks, i)
            i += w
        else
            i += 1
        end
    end

    # for i in Iterators.drop(Iterators.take(eachindex(v), length(v) - w), w)
    #     @views all(v[(i - w):(i - 1)] .< v[i] .> v[(i + 1):(i + w)]) && push!(peaks, i)
    # end
    return peaks
end

function __findvalleys(v::AbstractVector, w)
    w < 1 && throw(ArgumentError("w must be > 0"))
    valleys = Vector{Int64}(undef, 0)

    i = w + 1
    while i < length(v) - w
        if all(@views v[(i - w):(i - 1)] .> v[i] .< v[(i + 1):(i + w)])
            push!(valleys, i)
            i += w
        else
            i += 1
        end
    end
    return valleys
end

# Oscillation frequency as reciprocal first peak delay

"""
    oscillationfrequency(
        spikes::SpikeVector,
        binsize,
        winsize,
        first_peak_time,
        diff_std_baseline = 1
    )

Oscillation frequency as described in [`lang_differential_1997`](@cite).

Returns the reciprocal of the latency of the first peak in the autocorrelogram (ACG).
Peaks are recognized as the maximum value within the time span provided in `first_peak_time`
that exceed `baseline + standard_deviation * diff_std_baseline`.

# Arguments

- `spikes`:

"""
function oscillationfrequency end

function oscillationfrequency(
        spikes::SpikeVector, binsize::Real, winsize::Real, first_peak_time::AbstractRange{<:Real},
        recordingtime::Union{Real, Nothing} = nothing, diff_std_baseline = 1)
    autocorr = LaskaStats.acg(spikes, binsize, winsize, false)

    inds = timetobins(first_peak_time, binsize)

    first_peak = argmax(autocorr[inds]) + inds[begin] - 1
    if isnothing(recordingtime)
        recordingtime = spikes[end]
    end

    baseline = length(spikes)^2 / (recordingtime / binsize)

    if autocorr[first_peak] > baseline + std(autocorr) * diff_std_baseline
        return 1 / ustrip(LaskaCore.sampleratetounit(
            LaskaCore.samplerate(spikes), binstotime(first_peak, binsize), u"s")) |> Float64
    else
        return 0.0
    end
end

function oscillationfrequency(
        spikes::SpikeVector,
        binsize::LaskaCore.TUnit,
        winsize::LaskaCore.TUnit,
        first_peak_time::AbstractRange{<:LaskaCore.TUnit},
        recordingtime::Union{Nothing, LaskaCore.TUnit},
        diff_std_baseline::Real = 1
)
    binsize_samp = LaskaCore.timetosamplerate(spikes, binsize)
    winsize_samp = LaskaCore.timetosamplerate(spikes, winsize)
    first_peak_time_samp = LaskaCore.timetosamplerate(spikes, first_peak_time)

    if isnothing(recordingtime)
        recordingtime_samp = spikes[end]
    else
        recordingtime_samp = LaskaCore.timetosamplerate(spikes, recordingtime)
    end

    oscillationfrequency(
        spikes, binsize_samp, winsize_samp, first_peak_time_samp,
        recordingtime_samp, diff_std_baseline)
end

# Method for passing cluster
function oscillationfrequency(
        cluster::LaskaCore.Cluster,
        binsize::LaskaCore.TUnit,
        winsize::LaskaCore.TUnit,
        first_peak_time::AbstractRange{<:LaskaCore.TUnit},
        recordingtime::Union{Nothing, LaskaCore.TUnit},
        diff_std_baseline::Real = 1
)
    oscillationfrequency(
        spiketimes(cluster), binsize, winsize, first_peak_time, recordingtime, diff_std_baseline)
end

# Method for passing entire PhyOutput
function oscillationfrequency(
        experiment::LaskaCore.PhyOutput,
        binsize::LaskaCore.TUnit,
        winsize::LaskaCore.TUnit,
        first_peak_time::AbstractRange{<:LaskaCore.TUnit},
        recordingtime::Union{Nothing, LaskaCore.TUnit} = nothing,
        diff_std_baseline::Real = 1
)
    clusters = clustervector(experiment)
    out = zeros(LaskaCore.nclusters(experiment))
    for i in eachindex(out)
        out[i] = oscillationfrequency(clusters[i], binsize, winsize, first_peak_time,
            recordingtime, diff_std_baseline)
    end
    return out
end
