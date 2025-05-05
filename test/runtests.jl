using LaskaStats
using Test

# Aqua
include("Aqua.jl")

# Frequency
include("frequencytest.jl")


@testset "isiviolations" begin
    testvec1 = [0, 2, 4, 6, 7, 8, 10, 11]
    @test LaskaStats.isiviolations(testvec1, 1) == [5, 6, 8]
    @test LaskaStats.isiviolations(testvec1, 2) == 2:length(testvec1) |> collect
    @test LaskaStats.isiviolations(testvec1, 0.5) == Int64[]

    testvecspike = SpikeVector(testvec1, 1)
    @test LaskaStats.isiviolations(testvecspike, 1) == [5, 6, 8]
    @test LaskaStats.isiviolations(testvecspike, 2) == 2:length(testvec1) |> collect
    @test LaskaStats.isiviolations(testvecspike, 0.5) == Int64[]

    testvecrel = RelativeSpikeVector([deepcopy(testvec1) for _ in 1:10], 1)
    @test all([LaskaStats.isiviolations(testvecrel, 1)[i] == [5, 6, 8] for i in 1:10])
    @test all([LaskaStats.isiviolations(testvecrel, 2)[i] == 2:length(testvec1) |> collect for i in 1:10])
    @test all([LaskaStats.isiviolations(testvecrel, 0.5)[i] == Int64[] for i in 1:10])
end
