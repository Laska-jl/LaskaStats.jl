#=
Spike train smoothing using an FFT
=#

function fouriersmooth(V::AbstractVector{T}, σ, m) where {T}
    isodd(m) && throw(ArgumentError("`m` must be even."))
    normcentered = rangenormalize(V)
    premean = mean(normcentered)
    normcentered .-= mean(normcentered)
    normcentered = vcat(normcentered, reverse(normcentered))
    plan = plan_fft(normcentered)
    ft = plan * normcentered
    winf = gaussianfilter(length(V) / 2, σ, m)
    for i in eachindex(ft)
        ft[i] *= (1 - winf(i))
    end
    rea = real.(ifft(ft))[begin:length(V)]
    rea .+= premean
    derangenormalize!(rea, minimum(V), maximum(V))
    return rea
end
