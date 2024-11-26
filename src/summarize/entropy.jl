function isientropy(spikes::Vector{<:Real})
    logisi = log.(isi(spikes))
    __isientropy(logisi)
end

function __isientropy(logisi)
    minisi, maxisi = extrema(logisi)
    binsize = 0.02 * (maxisi - minisi)
    h = fit(StatsBase.Histogram, logisi, minisi:binsize:maxisi)
    filtered = imfilter(h.weights, Kernel.gaussian((3,)))
    out = 0.0
    filtered ./= sum(filtered)
    for i in eachindex(filtered)
        out += filtered[i] * log2(filtered[i])
    end
    return -out
end
