@testset "Grid" begin
    @testset "generate_grid" begin
        for ftype in (ScalarField(), VectorField())
            # dim 2
            axs = (0:1.0:3.0, 0:0.5:2.0)
            grid = @inferred generate_grid(ScalarField(), axs...)
            @test Femto.get_nodes(grid) ≈ vec(map(Vec, Iterators.product(axs...)))
            @test Femto.get_shape(grid) === Quad4()
            # dim 3
            axs = (0:1.0:3.0, 0:0.5:2.0, 0:0.2:3.0)
            grid = @inferred generate_grid(ScalarField(), axs...)
            @test Femto.get_nodes(grid) ≈ vec(map(Vec, Iterators.product(axs...)))
            @test Femto.get_shape(grid) === Hex8()
            # connectivities
            grid = @inferred generate_grid(ScalarField(), 0:0.5:1, 0:1:2)
            @test grid.connectivities == [[1,2,5,4], [2,3,6,5], [4,5,8,7], [5,6,9,8]]
            grid = @inferred generate_grid(ScalarField(), 0:0.5:1, 0:1:2, 1:3)
            @test grid.connectivities == [[1,2,5,4,10,11,14,13], [2,3,6,5,11,12,15,14], [4,5,8,7,13,14,17,16], [5,6,9,8,14,15,18,17],
                                          [10,11,14,13,19,20,23,22], [11,12,15,14,20,21,24,23], [13,14,17,16,22,23,26,25], [14,15,18,17,23,24,27,26]]
        end
    end
    @testset "integrate" begin
        # test with one element
        # dim 2
        grid = @inferred generate_grid(ScalarField(), 0:2:2, 1:2:3)
        element = Element(ScalarField(), Quad4())
        inds = only(grid.connectivities)
        @test Femto.sparse(@inferred integrate((index,u,v,dΩ)->v*u*dΩ, grid)) ≈ integrate((u,v,dΩ)->v*u*dΩ, element)[inds, inds]
        # dim 3
        grid = @inferred generate_grid(ScalarField(), 0:2:2, 1:2:3, 2:2:4)
        element = Element(ScalarField(), Hex8())
        inds = only(grid.connectivities)
        @test Femto.sparse(@inferred integrate((index,u,v,dΩ)->v*u*dΩ, grid)) ≈ integrate((u,v,dΩ)->v*u*dΩ, element)[inds, inds]
    end
end
