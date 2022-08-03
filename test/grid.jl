@testset "Grid" begin
    @testset "generate_grid" begin
        ## dim 1
        axs = (0:1.0:3.0,)
        @test (@inferred generate_grid(axs...)) isa Grid{Float64}
        @test (@inferred generate_grid(Float32, axs...)) isa Grid{Float32}
        # 1st order
        grid = @inferred generate_grid(Float64, axs...)
        @test Femto.get_allnodes(grid) ≈ vec(map(Vec, Iterators.product(axs...)))
        @test Femto.get_shape(grid) === Line2()
        # 2nd order
        grid = @inferred generate_grid(Float64, Line3(), axs...)
        @test sort(Femto.get_allnodes(grid)) ≈ vec(map(Vec, 0:0.5:3.0))
        @test Femto.get_shape(grid) === Line3()

        ## dim 2
        axs = (0:1.0:3.0, 0:0.5:2.0)
        @test (@inferred generate_grid(axs...)) isa Grid{Float64}
        @test (@inferred generate_grid(Float32, axs...)) isa Grid{Float32}
        # 1st order
        grid = @inferred generate_grid(Float64, axs...)
        @test Femto.get_allnodes(grid) ≈ vec(map(Vec, Iterators.product(axs...)))
        @test Femto.get_shape(grid) === Quad4()
        # 2nd order
        grid = @inferred generate_grid(Float64, Quad9(), axs...)
        @test sort(sort(Femto.get_allnodes(grid), by=x->x[1]), by=x->x[2]) ≈ vec(map(Vec, Iterators.product(0:0.5:3.0, 0:0.25:2.0)))
        @test Femto.get_shape(grid) === Quad9()

        ## dim 3
        axs = (0:1.0:3.0, 0:0.5:2.0, 0:0.2:3.0)
        @test (@inferred generate_grid(axs...)) isa Grid{Float64}
        @test (@inferred generate_grid(Float32, axs...)) isa Grid{Float32}
        # 1st order
        grid = @inferred generate_grid(Float64, axs...)
        @test Femto.get_allnodes(grid) ≈ vec(map(Vec, Iterators.product(axs...)))
        @test Femto.get_shape(grid) === Hex8()
        # 2nd order
        grid = @inferred generate_grid(Float64, Hex27(), axs...)
        @test sort(sort(sort(Femto.get_allnodes(grid), by=x->x[1]), by=x->x[2]), by=x->x[3]) ≈ vec(map(Vec, Iterators.product(0:0.5:3.0, 0:0.25:2.0, 0:0.1:3.0)))
        @test Femto.get_shape(grid) === Hex27()

        ## connectivities
        # 1D
        conns = [[1,2,5,4], [2,3,6,5], [4,5,8,7], [5,6,9,8]]
        grid = @inferred generate_grid(0:0.5:1, 0:1:2)
        @test grid.connectivities == conns
        grid = @inferred generate_grid(Quad9(), 0:0.5:1, 0:1:2)
        @test getindex.(grid.connectivities, (1:4,)) == conns
        # 2D
        conns = [[1,2,5,4,10,11,14,13], [2,3,6,5,11,12,15,14], [4,5,8,7,13,14,17,16], [5,6,9,8,14,15,18,17],
                 [10,11,14,13,19,20,23,22], [11,12,15,14,20,21,24,23], [13,14,17,16,22,23,26,25], [14,15,18,17,23,24,27,26]]
        grid = @inferred generate_grid(0:0.5:1, 0:1:2, 1:3)
        @test grid.connectivities == conns
        grid = @inferred generate_grid(Hex27(), 0:0.5:1, 0:1:2, 1:3)
        @test getindex.(grid.connectivities, (1:8,)) == conns
    end
    @testset "generate_gridset" begin
        a = rand()
        f(i,n,v) = a * v * n
        for shape in (Quad4(), Quad9())
            gridset = generate_gridset(0:2, 0:3) # 2, 3
            @test sum(integrate(f, ScalarField(), gridset["left"])) ≈ [-3a, 0]
            @test sum(integrate(f, ScalarField(), gridset["right"])) ≈ [3a, 0]
            @test sum(integrate(f, ScalarField(), gridset["bottom"])) ≈ [0, -2a]
            @test sum(integrate(f, ScalarField(), gridset["top"])) ≈ [0, 2a]
        end
        for shape in (Hex8(), Hex27())
            gridset = generate_gridset(0:2, 0:3, -2:3) # 2, 3, 5
            @test sum(integrate(f, ScalarField(), gridset["left"])) ≈ [-15a, 0, 0]
            @test sum(integrate(f, ScalarField(), gridset["right"])) ≈ [15a, 0, 0]
            @test sum(integrate(f, ScalarField(), gridset["bottom"])) ≈ [0, -10a, 0]
            @test sum(integrate(f, ScalarField(), gridset["top"])) ≈ [0, 10a, 0]
            @test sum(integrate(f, ScalarField(), gridset["front"])) ≈ [0, 0, 6a]
            @test sum(integrate(f, ScalarField(), gridset["back"])) ≈ [0, 0, -6a]
        end
    end
    @testset "integrate" begin
        @testset "SingleField" begin
            field = ScalarField()
            # test with one element
            # dim 1
            grid = @inferred generate_grid(0:2:2)
            element = Element(Line2())
            inds = get_elementdofs(field, grid, 1)
            @test Array(@inferred integrate((index,v,u)->v*u, field, grid)) ≈ integrate((qp,v,u)->v*u, field, element)[inds, inds]
            # dim 2
            grid = @inferred generate_grid(0:2:2, 1:2:3)
            element = Element(Quad4())
            inds = get_elementdofs(field, grid, 1)
            @test Array(@inferred integrate((index,v,u)->v*u, field, grid)) ≈ integrate((qp,v,u)->v*u, field, element)[inds, inds]
            # dim 3
            grid = @inferred generate_grid(0:2:2, 1:2:3, 2:2:4)
            element = Element(Hex8())
            inds = get_elementdofs(field, grid, 1)
            @test Array(@inferred integrate((index,v,u)->v*u, field, grid)) ≈ integrate((qp,v,u)->v*u, field, element)[inds, inds]
        end
        @testset "MixedField" begin
            # dim 2
            field = mixed(Vf(2), Sf(1))
            grid = @inferred generate_grid(Quad9(), 0:2:2, 1:2:3)
            element = Element((Quad9(), Quad4()))
            inds = get_elementdofs(field, grid, 1)
            f = (index, (v,q), (u,p)) -> ∇(v) ⊡ ∇(u) - (∇⋅v)*p + q*(∇⋅u)
            @test Array(@inferred integrate(f, field, grid)) ≈ integrate(f, field, element)[inds, inds]
        end
    end
end

@testset "Generating element state" begin
    ElementState = @NamedTuple begin
        x::Vec{2, Float64}
        σ::SymmetricSecondOrderTensor{3, Float64}
        index::Int
    end
    grid = generate_grid(0:2, 0:3)
    element = Femto.create_element(grid)
    eltstate = @inferred generate_elementstate(ElementState, grid)
    @test size(eltstate) == (Femto.num_quadpoints(element), Femto.num_elements(grid))
    X = @inferred interpolate(VectorField(), grid, reinterpret(Float64, Femto.get_allnodes(grid)))
    for I in CartesianIndices(eltstate)
        qp, eltindex = Tuple(I)
        conn = get_connectivity(grid, eltindex)
        q = eltstate[I]
        @test q isa ElementState
        @test q.x ≈ interpolate(element, get_allnodes(grid)[conn], qp)
        @test q.x ≈ X[I]
        @test q.σ == zero(q.σ)
        @test q.index == zero(q.index)
    end
end
