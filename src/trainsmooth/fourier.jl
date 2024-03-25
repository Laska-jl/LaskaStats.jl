#=
Spike train smoothing using an FFT
=#

function fouriersmooth(V::AbstractVector{T}, σ, m) where {T}
    isodd(m) && throw(ArgumentError("`m` must be even."))
    normcentered = rangenormalize(V)
    normcentered .-= mean(normcentered)
    plan = plan_fft(normcentered)
    ft = plan * normcentered
    winf = gaussianfilter(length(V) / 2, σ, m)
    for i in eachindex(ft)
        ft[i] *= (1 - winf(i))
    end
    return real.(ifft(ft))
end
