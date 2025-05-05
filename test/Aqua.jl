using Aqua


@testset "aqua deps compat" begin
    Aqua.test_deps_compat(LaskaStats)
end


@testset "aqua unbound_args" begin
    Aqua.test_unbound_args(LaskaStats)
end

@testset "aqua undefined exports" begin
    Aqua.test_undefined_exports(LaskaStats)
end


# Gives false positives from dependencies
#
# @testset "aqua test ambiguities" begin
#     Aqua.test_ambiguities([MyModule, Core, Base])
# end

@testset "aqua piracy" begin
    Aqua.test_piracies(LaskaStats)
end

@testset "aqua project extras" begin
    Aqua.test_project_extras(LaskaStats)
end

@testset "aqua state deps" begin
    Aqua.test_stale_deps(LaskaStats)
end
