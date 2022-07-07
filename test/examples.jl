@testset "Examples" begin
    @testset "HeatEquation" begin
        include("../examples/HeatEquation/HeatEquation.jl")
        U = HeatEquation("../examples/HeatEquation/model2d.msh")
        @test norm(U) ≈ 3.563526379518352
        U = HeatEquation("../examples/HeatEquation/model3d.msh")
        @test norm(U) ≈ 2.477799194320351
    end
    @testset "WaveEquation" begin
        include("../examples/WaveEquation/WaveEquation.jl")
        U = WaveEquation()
        @test norm(U) ≈ 0.07100524583703295
    end
end
