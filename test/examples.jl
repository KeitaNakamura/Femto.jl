@testset "Examples" begin
    @testset "HeatEquation" begin
        include("../examples/HeatEquation/HeatEquation.jl")
        @testset "Structured mesh" begin
            # 2D
            gridset = generate_gridset(Quad4(), -1:0.1:1, -1:0.1:1)
            U = HeatEquation(gridset)
            @test norm(U) ≈ 3.3077439126413033
            gridset = generate_gridset(Quad9(), -1:0.2:1, -1:0.2:1)
            U = HeatEquation(gridset)
            @test norm(U) ≈ 3.3008393640952183
            # 3D
            gridset = generate_gridset(Hex8(), -1:0.1:1, -1:0.1:1, -1:0.1:1)
            U = HeatEquation(gridset)
            @test norm(U) ≈ 8.977227889242897
            gridset = generate_gridset(Hex27(), -1:0.2:1, -1:0.2:1, -1:0.2:1)
            U = HeatEquation(gridset)
            @test norm(U) ≈ 8.938916880390302
        end
        @testset "Gmsh" begin
            U = HeatEquation("../examples/HeatEquation/model2d_1.msh")
            @test norm(U) ≈ 3.563526379518352
            U = HeatEquation("../examples/HeatEquation/model2d_2.msh")
            @test norm(U) ≈ 3.590764105754447
            U = HeatEquation("../examples/HeatEquation/model3d_1.msh")
            @test norm(U) ≈ 2.477799194320351
            U = HeatEquation("../examples/HeatEquation/model3d_2.msh")
            @test norm(U) ≈ 7.277445118641519
        end
    end
    @testset "WaveEquation" begin
        include("../examples/WaveEquation/WaveEquation.jl")
        U = WaveEquation()
        @test norm(U) ≈ 0.07100524583703295
    end
end
