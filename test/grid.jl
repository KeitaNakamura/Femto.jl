@testset "Grid" begin
    @testset "generate_grid" begin
        # dim 1
        axs = (0:1.0:3.0,)
        grid = @inferred generate_grid(axs...)
        @test Femto.get_nodes(grid) ≈ vec(map(Vec, Iterators.product(axs...)))
        @test Femto.get_shape(grid) === Line2()
        # dim 2
        axs = (0:1.0:3.0, 0:0.5:2.0)
        grid = @inferred generate_grid(axs...)
        @test Femto.get_nodes(grid) ≈ vec(map(Vec, Iterators.product(axs...)))
        @test Femto.get_shape(grid) === Quad4()
        # dim 3
        axs = (0:1.0:3.0, 0:0.5:2.0, 0:0.2:3.0)
        grid = @inferred generate_grid(axs...)
        @test Femto.get_nodes(grid) ≈ vec(map(Vec, Iterators.product(axs...)))
        @test Femto.get_shape(grid) === Hex8()
        # connectivities
        grid = @inferred generate_grid(0:0.5:1, 0:1:2)
        @test grid.connectivities == [[1,2,5,4], [2,3,6,5], [4,5,8,7], [5,6,9,8]]
        grid = @inferred generate_grid(0:0.5:1, 0:1:2, 1:3)
        @test grid.connectivities == [[1,2,5,4,10,11,14,13], [2,3,6,5,11,12,15,14], [4,5,8,7,13,14,17,16], [5,6,9,8,14,15,18,17],
                                      [10,11,14,13,19,20,23,22], [11,12,15,14,20,21,24,23], [13,14,17,16,22,23,26,25], [14,15,18,17,23,24,27,26]]
    end
    @testset "integrate" begin
        # test with one element
        fieldtype = ScalarField()
        # dim 1
        grid = @inferred generate_grid(0:2:2)
        element = Element(Line2())
        inds = only(grid.connectivities)
        @test Femto.sparse(@inferred integrate((index,u,v)->v*u, fieldtype, grid)) ≈ integrate((qp,u,v)->v*u, fieldtype, element)[inds, inds]
        # dim 2
        grid = @inferred generate_grid(0:2:2, 1:2:3)
        element = Element(Quad4())
        inds = only(grid.connectivities)
        @test Femto.sparse(@inferred integrate((index,u,v)->v*u, fieldtype, grid)) ≈ integrate((qp,u,v)->v*u, fieldtype, element)[inds, inds]
        # dim 3
        grid = @inferred generate_grid(0:2:2, 1:2:3, 2:2:4)
        element = Element(Hex8())
        inds = only(grid.connectivities)
        @test Femto.sparse(@inferred integrate((index,u,v)->v*u, fieldtype, grid)) ≈ integrate((qp,u,v)->v*u, fieldtype, element)[inds, inds]
    end
end

@testset "Generating element state" begin
    ElementState = @NamedTuple begin
        x::Vec{2, Float64}
        σ::SymmetricSecondOrderTensor{3, Float64}
        index::Int
    end
    grid = generate_grid(0:2, 0:3)
    shape = Femto.get_shape(grid)
    element = Element(shape)
    eltstate = @inferred generate_elementstate(ElementState, grid)
    @test size(eltstate) == (Femto.num_quadpoints(shape), Femto.num_elements(grid))
    X = @inferred interpolate(grid, Femto.get_nodes(grid))
    for I in CartesianIndices(eltstate)
        qp, eltindex = Tuple(I)
        conn = Femto.get_connectivities(grid)[eltindex]
        q = eltstate[I]
        @test q isa ElementState
        @test q.x ≈ interpolate(element, Femto.get_nodes(grid)[conn], qp)
        @test q.x ≈ X[I]
        @test q.σ == zero(q.σ)
        @test q.index == zero(q.index)
    end
end
