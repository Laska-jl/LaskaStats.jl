function isientropy(spikes::Vector{<:Real})
    logisi = log.(isi(spikes))
    __isientropy(logisi)
end

function __isientropy(logisi)
    minisi, maxisi = extrema(logisi)
    @show minisi maxisi
    binsize = 0.02 * (maxisi - minisi)
    @show binsize
    h = fit(StatsBase.Histogram, logisi, minisi:binsize:maxisi)
    filtered = imfilter(h.weights, Kernel.gaussian((3,)))
    out = 0.0
    filtered ./= sum(filtered)
    @show sum(filtered)
    for i in eachindex(filtered)
        out += filtered[i] * log2(filtered[i])
    end

    return -out
end
