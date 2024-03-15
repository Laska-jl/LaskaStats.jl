#======================================
Functions for obtaining spike-densities
using Base: @propagate_inbounds
======================================#

"""
    

TBW
"""
function psth end

function psth(v::Vector{Vector{T}}, bin) where {T}
    fs = frequency(v, bin)
    sum = zeros(Int64, length(fs[1]))
    @views for v in eachindex(fs)
        for n in eachindex(sum)
            sum[n] += fs[v][n]
        end
    end
    lowerbound, upperbound = LaskaCore.rounddown.(LaskaCore.extremevals(v), bin)
    return sum, lowerbound:bin:upperbound
end

function psth(v::RelativeSpikeVector{T, U}, bin) where {T, U}
    fs = frequency(v, bin)
    sum = zeros(Float64, length(fs[1]))
    @views for v in eachindex(fs)
        for n in eachindex(sum)
            sum[n] += fs[v][n]
        end
    end
    lowerbound, upperbound = LaskaCore.rounddown.(LaskaCore.extremevals(v), bin)
    return sum ./ length(v), lowerbound:bin:upperbound
end

function psth(v::RelativeSpikeVector{T, U}, bin::V) where {T, U, V <: LaskaCore.TUnit}
    binconv = LaskaCore.timetosamplerate(v, bin)
    fs = frequency(v, binconv)
    sum = zeros(Float64, length(fs[1]))
    @views for v in eachindex(fs)
        for n in eachindex(sum)
            sum[n] += fs[v][n]
        end
    end
    u = LaskaCore.unit(bin)
    lowerbound, upperbound = LaskaCore.rounddown.(LaskaCore.extremevals(v), binconv)
    l = LaskaCore.sampleratetounit(LaskaCore.samplerate(v), lowerbound, u)
    u = LaskaCore.sampleratetounit(LaskaCore.samplerate(v), upperbound, u)
    return sum ./ length(v), l:bin:u
end
