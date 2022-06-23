@testset "Element" begin
    @testset "interpolate" begin
        @testset "$shape" for shape in map(S->S(), subtypes(Femto.Shape))
            for T in (Float64, Float32)
                TOL = sqrt(eps(T))
                dim = Femto.get_dimension(shape)
                element = Element{T}(shape)
                X = @inferred interpolate(VectorField(), element, reinterpret(T, Femto.get_local_node_coordinates(T, shape)))
                for qp in Femto.num_quadpoints(element)
                    ## shape values and gradients
                    @test sum(element.N[qp]) ≈ 1
                    @test norm(sum(element.dNdx[qp])) < TOL
                    ## interpolate
                    # qp
                    x = @inferred interpolate(element, Femto.get_local_node_coordinates(T, shape), qp)
                    dxdx = ∇(x)
                    @test x ≈ Femto.quadpoints(T, shape)[qp] atol=TOL
                    @test x ≈ X[qp] atol=TOL
                    @test dxdx ≈ one(dxdx) atol=TOL
                    @test dxdx ≈ ∇(X[qp]) atol=TOL
                    # ξ
                    x′ = rand(Vec{dim, T})
                    x = @inferred interpolate(element, Femto.get_local_node_coordinates(T, shape), x′)
                    dxdx = ∇(x)
                    @test x ≈ x′ atol=TOL
                    @test dxdx ≈ one(dxdx) atol=TOL
                end
            end
        end
    end
    @testset "dofindices" begin
        for shape in (Line2(), Quad4(), Hex8(), Tri6())
            dim = Femto.get_dimension(shape)
            conn = Femto.Index(1,3,6)
            element = Element(shape)
            # scalar field
            @test (@inferred Femto.dofindices(ScalarField(), element, conn)) == Femto.Index(1,3,6)
            # vector field
            if dim == 1
                @test (@inferred Femto.dofindices(VectorField(), element, conn)) == Femto.Index(1,3,6)
            elseif dim == 2
                @test (@inferred Femto.dofindices(VectorField(), element, conn)) == Femto.Index(1,2,5,6,11,12)
            elseif dim == 3
                @test (@inferred Femto.dofindices(VectorField(), element, conn)) == Femto.Index(1,2,3,7,8,9,16,17,18)
            else
                error("unreachable")
            end
        end
    end
end
