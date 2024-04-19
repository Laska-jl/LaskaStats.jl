#=
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120777/
=#


# Silence sensitive function

function LF_detectsilence(v::SpikeVector, τ::LaskaCore.TUnit)
	LF_detectsilence(v, Int64(LaskaCore.timetosamplerate(v, τ)))
end

function LF_detectsilence(v::SpikeVector, τ)
    t0 = minimum(v)
	out = zeros(Float64, (maximum(v) - t0))
    for i in Iterators.take(eachindex(v), length(v) - 1)
        for t in (v[i] + τ):v[i+1]
            out[t] = t - (v[i] + τ)
        end
    end
    return out
end

function LF_silence_similarity(vec1::SpikeVector{T}, vec2::SpikeVector{T}, τ) where {T}
    v1 = deepcopy(vec1)
    v2 = deepcopy(vec2)
    v1 .-= minimum(vec1)
    v2 .-= minimum(vec2)
    f1, f2 = LF_detectsilence.((v1, v2), τ)
    lendiff = length(f1) - length(f2)
    # Pad the shorter vector with 0:s if they are not of equal length
    if lendiff > 0
        f2 = vcat(f2, zeros(eltype(f2), lendiff))
    elseif lendiff < 0
        f1 = vcat(f1, zeros(eltype(f1), abs(lendiff)))
    end
    numerator = sum(f1 .* f2)
    denominator = sqrt(sum(f1 .^ 2)) * sqrt(sum(f2 .^ 2))
    return 1 - (numerator / denominator)
end

# Sensitive to bursts

"""
    LF_detectburst(v::SpikeVector, b::LaskaCore.TUnit, n::Integer, η, sigma::LaskaCore.TUnit)

Function sensitive to bursts in a spike-train.

# Arguments
- `v::SpikeVector`: A vector of spiketimes.
- `b`: The maximum ISI for spikes in a burst.
- `n::Integer`: The minimum number of spikes in a burst.
- `η`: Scaling factor between 0--1 which controls the sensitivity to spikes outside of bursts defined by `b` & `n`. When set to 1 all other spikes are completely ignored.
- `sigma`: Width of the gaussian kernel.

Described in: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120777/
"""
function LF_detectburst(v::SpikeVector, b::LaskaCore.TUnit, n::Integer, η, sigma::LaskaCore.TUnit)
	b_conv = Float64(LaskaCore.timetosamplerate(v, b))
    sigma_conv = Float64(LaskaCore.timetosamplerate(v, sigma))
    LF_detectburst(v, b_conv, n, η, sigma_conv)
end

function LF_detectburst(v::AbstractVector, b, n::Integer, η, sigma=1)
    if η < 0 || η > 1
        throw(ArgumentError("η should be between 0-1"))
    end
	gauss = gaussianfilter(length(v), sigma, 1)
    veclen = v[end] - v[begin] + 1
    spiketrain = zeros(Int64, veclen)
    min_ind = minimum(v)
    for i in v
        spiketrain[i-min_ind+1] = 1
    end
    conv_vec = gauss.(0:length(v)*2)
    convolved = conv(conv_vec, spiketrain)
    tresh = treshold(b, n, sigma)
    transformburst!(convolved, η, tresh)
    convolved
end

function heaviside(x::T) where T
    x <= zero(T) ? zero(T) : one(T)
end

function transformburst!(v::AbstractVector, η, t)
    etaT = η * t
    @. v = heaviside(v - etaT) * (v - etaT)
end

function treshold(b, n, sigma)
    p_1 = p1(b, n)
    p_2 = p2(b, n)
	T1 = 0.0
    T2 = 0.0
    for k in 1:n
    T1 += exp(-((p_1 - k * b)^2) / sigma^2)
    T2 += exp(-((p_2 - k * b)^2) / sigma^2)
    end
    return max(T1, T2)
end

p1(b, n) = b * (n+1) * 0.5
p2(b, n) = b * (n+2) * 0.5

"""
    LF_burstsimilarity(v1::T, v2::T, b::LaskaCore.TUnit, n::Integer, η, sigma::LaskaCore.TUnit) where {T<:SpikeVector}

Measure sensitive to bursts in a spike train.

# Arguments
- `v::SpikeVector`: A vector of spiketimes.
- `b`: The maximum ISI for spikes in a burst.
- `n::Integer`: The minimum number of spikes in a burst.
- `η`: Scaling factor between 0--1 which controls the sensitivity to spikes outside of bursts defined by `b` & `n`. When set to 1 all other spikes are completely ignored.
- `sigma`: Width of the gaussian kernel.

Described in: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120777/
"""
function LF_burstsimilarity(v1::T, v2::T, b::LaskaCore.TUnit, n::Integer, η, sigma::LaskaCore.TUnit) where {T<:SpikeVector}
	v1_bursts = LF_detectburst(v1, b, n, η, sigma)
	v2_bursts = LF_detectburst(v2, b, n, η, sigma)
    lendiff = length(v1_bursts) - length(v2_bursts)
    # Pad the shorter vector with 0:s if they are not of equal length
    if lendiff > 0
        v2_bursts = vcat(v2_bursts, zeros(eltype(v2_bursts), lendiff))
    elseif lendiff < 0
        v1_bursts = vcat(v1_bursts, zeros(eltype(v1_bursts), abs(lendiff)))
    end
    numerator = sum(v1_bursts .* v2_bursts)
    denominator = sqrt(sum(v1_bursts .^ 2)) * sqrt(sum(v2_bursts))
    return 1 - (numerator / denominator)    
end

# Combined measure

function LF_difference(v1::T, v2::T, b::LaskaCore.TUnit, n::Integer, η, sigma::LaskaCore.TUnit, τ::LaskaCore.TUnit) where {T <: SpikeVector}
	burstsim = LF_burstsimilarity(v1, v2, b, n, η, sigma)
    silencesim = LF_silence_similarity(v1, v2, τ)
    return (burstsim + silencesim) * 0.5
end
