#=
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4120777/
=#


# Silence sensitive function

function maptrain(v::SpikeVector, τ)
    t0 = minimum(v)
	out = zeros(Float64, (maximum(v) - t0))
    for i in Iterators.take(eachindex(v), length(v) - 1)
        for t in (v[i] + τ):v[i+1]
            out[t] = t - (v[i] + τ)
        end
    end
    return out
end

function silence_similarity(vec1::SpikeVector{T}, vec2::SpikeVector{T}, τ, pad, ) where {T}
    v1 = deepcopy(vec1)
    v2 = deepcopy(vec2)
    maxt = max(v1[end], v2[end])
    push!(v1, maxt + pad)
    push!(v2, maxt + pad)
    f1, f2 = maptrain.((v1, v2), τ)
    numerator = sum(f1 .* f2)
    denominator = sqrt(sum(f1 .^ 2)) * sqrt(sum(f2 .^ 2))
    return 1 - (numerator / denominator)
end
