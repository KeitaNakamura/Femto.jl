@testset "Examples" begin
    @testset "Gridset" begin
        for mshfile in ("../examples/HeatEquation/model2d_1.msh",
                        "../examples/HeatEquation/model2d_2.msh",
                        "../examples/HeatEquation/model3d_1.msh",
                        "../examples/HeatEquation/model3d_2.msh",)
            for (name, grid) in readgmsh(mshfile)
                for (i, shape) in enumerate(Femto.get_lower_shapes(get_shape(grid)))
                    n = Femto.num_nodes(shape)
                    inds = unique!(sort!(reduce(vcat, [conn[1:n] for conn in Femto.get_connectivities(grid)])))
                    @test get_allnodes(grid)[inds] ≈ get_allnodes(grid, i)[Femto.get_nodeindices(grid, i)]
                end
            end
        end
    end
    @testset "HeatEquation" begin
        include("../examples/HeatEquation/HeatEquation.jl")
        @testset "Structured mesh" begin
            # 2D
            @test norm(HeatEquation(generate_gridset(Quad4(), -1:0.1:1, -1:0.1:1))) ≈ 3.3077439126413033
            @test norm(HeatEquation(generate_gridset(Quad9(), -1:0.2:1, -1:0.2:1))) ≈ 3.3008393640952183
            @test norm(HeatEquation(ScalarField(1), generate_gridset(Quad9(), -1:0.1:1, -1:0.1:1))) ≈ 3.3077439126413033
            # 3D
            @test norm(HeatEquation(generate_gridset(Hex8(),  -1:0.1:1, -1:0.1:1, -1:0.1:1))) ≈ 8.977227889242897
            @test norm(HeatEquation(generate_gridset(Hex27(), -1:0.2:1, -1:0.2:1, -1:0.2:1))) ≈ 8.938916880390302
            @test norm(HeatEquation(ScalarField(1), generate_gridset(Hex27(), -1:0.1:1, -1:0.1:1, -1:0.1:1))) ≈ 8.977227889242897
        end
        @testset "Gmsh" begin
            @test norm(HeatEquation("../examples/HeatEquation/model2d_1.msh")) ≈ 3.563526379518352
            @test norm(HeatEquation("../examples/HeatEquation/model2d_2.msh")) ≈ 3.590764105754447
            @test norm(HeatEquation("../examples/HeatEquation/model3d_1.msh")) ≈ 2.477799194320351
            @test norm(HeatEquation("../examples/HeatEquation/model3d_2.msh")) ≈ 7.277445118641519
        end
    end
    @testset "WaveEquation" begin
        include("../examples/WaveEquation/WaveEquation.jl")
        @test norm(WaveEquation()) ≈ 0.07100524583703295
    end
    @testset "StokesEquation" begin
        # currently use only nodal velocity values for test
        include("../examples/StokesEquation/StokesEquation.jl")
        gridset = readgmsh("../examples/StokesEquation/model.msh")
        n = num_allnodes(gridset["main"])
        @test norm(StokesEquation(gridset)[1:end-n+1]) ≈ 23.42769930149981
        gridset = generate_gridset(Quad9(), 0:0.01:1, 0:0.01:1)
        n = num_allnodes(gridset["main"])
        @test norm(StokesEquation(gridset)[1:end-n+1]) ≈ 29.7990808415312
    end
end
