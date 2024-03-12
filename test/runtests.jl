using LaskaStats
using Test

spikevec = SpikeVector(vcat([i for i in 1:3000:3000000], [i for i in 3000001:1500:6000000]),
    30000)

@testset "Frequency" begin end
