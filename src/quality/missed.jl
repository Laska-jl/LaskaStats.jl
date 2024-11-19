"""
    missedspikes(amplitudes::AbstractVector; kernelwidth=3, histbins::Integer=500)

Calculate the percentage of missing spikes based on the distribution of their amplitude.

Attempts to fit a gaussian to the distribution of amplitude of amplitudess and calculate the number of 
spikes that fell below the detection limit.
"""
function missedspikes(amplitudes::AbstractVector; kernelwidth = 3, histbins::Integer = 500)
    h = fit(StatsBase.Histogram, amplitudes, nbins = histbins)
    gau = imfilter(h.weights, Kernel.gaussian((kernelwidth,)))

    maxind = argmax(gau)

    @views begin
        diffind = findfirst(
            Base.Fix2(isless, zero(eltype(gau))), @. gau[maxind:end] - gau[begin]) + maxind
    end

    isnothing(diffind) && return 0.0

    nmissed = sum(@view(gau[diffind:end]))

    return nmissed / (sum(gau) + nmissed)
end

"""
    function missedspikespartition(amplitudes::AbstractVector, spikes_per_partition::Integer; kernelwidth = 3, histbins::Integer = 500)


Estimate the percentage of missed spikes `spikes_per_partition` spikes at a time.
Estimation is carried out in the same way as [`LaskaCore.missedspikes`](@ref).
"""
function missedspikespartition(amplitudes::AbstractVector, spikes_per_partition::Integer;
        kernelwidth = 3, histbins::Integer = 500)
    n_partitions = ceil(length(amplitudes) / spikes_per_partition) |> Int64
    out = Vector{Float64}(undef, n_partitions)

    for (i, amps) in enumerate(Iterators.partition(amplitudes, spikes_per_partition))
        out[i] = missedspikes(amps; kernelwidth = kernelwidth, histbins = histbins)
    end

    return out
end
