@testset "Element" begin
    @testset "interpolate" begin
        for T in (Float64, Float32)
            TOL = sqrt(eps(T))
            for shape in (Quad4(), Tri6())
                dim = Femto.get_dimension(shape)
                element = Element{T}(ScalarField(), shape)
                for qp in Femto.num_quadpoints(element)
                    # shape values and gradients
                    @test sum(element.N[qp]) ≈ 1
                    @test norm(sum(element.dNdx[qp])) < TOL
                    # interpolate
                    x, dxdx = @inferred Femto.interpolate(element, Femto.get_local_node_coordinates(shape), qp)
                    @test x ≈ Femto.quadpoints(shape)[qp] atol=TOL
                    @test dxdx ≈ one(dxdx) atol=TOL
                    x′ = rand(Vec{dim, T})
                    x, dxdx = @inferred Femto.interpolate(element, Femto.get_local_node_coordinates(shape), x′)
                    @test x ≈ x′ atol=TOL
                    @test dxdx ≈ one(dxdx) atol=TOL
                end
            end
        end
    end
    @testset "dofindices" begin
        for shape in (Quad4(), Tri6())
            dim = Femto.get_dimension(shape)
            conn = Femto.Index(1,3,6)
            # scalar field
            element = Element(ScalarField(), shape)
            @test (@inferred Femto.dofindices(element, conn)) == Femto.Index(1,3,6)
            # vector field
            element = Element(VectorField(), shape)
            if dim == 2
                @test (@inferred Femto.dofindices(element, conn)) == Femto.Index(1,2,5,6,11,12)
            end
        end
    end
    @testset "integrate" begin
        @testset "ScalarField" begin
            element = Element(ScalarField(), Quad4())
            update!(element, [Vec(0.0,0.0), Vec(1.0,0.5), Vec(2.0,1.0), Vec(0.5,0.8)])
            # mass matrix
            M = sum(1:Femto.num_quadpoints(element)) do qp
                N = element.N[qp]'
                dΩ = element.detJdΩ[qp]
                N' * N * dΩ
            end
            @test integrate((u,∇u,v,∇v,dΩ) -> (u * v)*dΩ, element) ≈ M
            # stiffness matrix
            K = sum(1:Femto.num_quadpoints(element)) do qp
                B = reduce(hcat, element.dNdx[qp])
                dΩ = element.detJdΩ[qp]
                B' * B * dΩ
            end
            @test integrate((u,∇u,v,∇v,dΩ) -> (∇u ⋅ ∇v)*dΩ, element) ≈ K
            # element vector
            F = sum(1:Femto.num_quadpoints(element)) do qp
                N = element.N[qp]
                dΩ = element.detJdΩ[qp]
                N * dΩ
            end
            @test integrate((v,∇v,dΩ) -> v*dΩ, element) ≈ F
        end
    end
end
