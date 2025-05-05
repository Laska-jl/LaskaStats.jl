using Unitful

@testset "Frequency tests" begin
    v15000 = [i for i in 0:29999 if iseven(i)]

    @test all(LaskaStats.frequency(v15000, 300, 30000) .== 15000.0)
    @test all(LaskaStats.frequency(v15000, 10u"ms", 30000) .== 15000.0)
end
