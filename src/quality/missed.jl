# Estimate number of missed spikes

"""
    missedspikes(amplitudes::AbstractVector; kernelwidth=3, histbins::Integer=500)

Calculate the percentage of missing spikes based on the distribution of their amplitude.

Attempts to fit a gaussian to the distribution of amplitude of amplitudess and calculate the number of 
spikes that fell below the detection limit.
"""
function missedspikes(amplitudes::AbstractVector; kernelwidth=3, histbins::Integer=500)
    h = fit(Histogram, amplitudes, nbins=histbins)
    gau = imfilter(h.weights, Kernel.gaussian((kernelwidth,)))

    maxind = argmax(gau)

    @views begin
        diffind = findfirst(Base.Fix2(isless, zero(eltype(gau))), @. gau[maxind:end] - gau[begin]) + maxind
    end

    nmissed = sum(@view(gau[diffind:end]))

    return nmissed / (sum(gau) + nmissed)
end
