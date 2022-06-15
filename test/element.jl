@testset "Element" begin
    @testset "interpolate" begin
        for T in (Float64, Float32)
            TOL = sqrt(eps(T))
            for shape in (Line2(), Quad4(), Hex8(), Tri6())
                dim = Femto.get_dimension(shape)
                element = Element{T}(ScalarField(), shape)
                for qp in Femto.num_quadpoints(element)
                    # shape values and gradients
                    @test sum(element.N[qp]) ≈ 1
                    @test norm(sum(element.dNdx[qp])) < TOL
                    # interpolate
                    x = @inferred Femto.interpolate(element, Femto.get_local_node_coordinates(shape), qp)
                    dxdx = ∇(x)
                    @test x ≈ Femto.quadpoints(shape)[qp] atol=TOL
                    @test dxdx ≈ one(dxdx) atol=TOL
                    x′ = rand(Vec{dim, T})
                    x = @inferred Femto.interpolate(element, Femto.get_local_node_coordinates(shape), x′)
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
            # scalar field
            element = Element(ScalarField(), shape)
            @test (@inferred Femto.dofindices(element, conn)) == Femto.Index(1,3,6)
            # vector field
            element = Element(VectorField(), shape)
            if dim == 1
                @test (@inferred Femto.dofindices(element, conn)) == Femto.Index(1,3,6)
            elseif dim == 2
                @test (@inferred Femto.dofindices(element, conn)) == Femto.Index(1,2,5,6,11,12)
            elseif dim == 3
                @test (@inferred Femto.dofindices(element, conn)) == Femto.Index(1,2,3,7,8,9,16,17,18)
            else
                error("unreachable")
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
            @test integrate((u,v,dΩ) -> (v * u)*dΩ, element) ≈ M
            # stiffness matrix
            K = sum(1:Femto.num_quadpoints(element)) do qp
                B = reduce(hcat, element.dNdx[qp])
                dΩ = element.detJdΩ[qp]
                B' * B * dΩ
            end
            @test integrate((u,v,dΩ) -> (∇(v) ⋅ ∇(u))*dΩ, element) ≈ K
            # element vector
            F = sum(1:Femto.num_quadpoints(element)) do qp
                N = element.N[qp]
                dΩ = element.detJdΩ[qp]
                N * dΩ
            end
            @test integrate((v,dΩ) -> v*dΩ, element) ≈ F
        end
        @testset "VectorField" begin
            element = Element(VectorField(), Quad4())
            update!(element, [Vec(0.0,0.0), Vec(1.0,0.5), Vec(2.0,1.0), Vec(0.5,0.8)])
            # mass matrix
            M = sum(1:Femto.num_quadpoints(element)) do qp
                N1, N2, N3, N4 = element.N[qp]
                N = [N1 0  N2 0  N3 0  N4 0
                     0  N1 0  N2 0  N3 0  N4]
                dΩ = element.detJdΩ[qp]
                N' * N * dΩ
            end
            @test integrate((u,v,dΩ) -> (v ⋅ u)*dΩ, element) ≈ M
            # stiffness matrix
            ke = rand(SymmetricFourthOrderTensor{2})
            K = sum(1:Femto.num_quadpoints(element)) do qp
                (dNdx1,dNdy1), (dNdx2,dNdy2), (dNdx3,dNdy3), (dNdx4,dNdy4) = element.dNdx[qp]
                B = [dNdx1 0     dNdx2 0     dNdx3 0     dNdx4 0
                     0     dNdy1 0     dNdy2 0     dNdy3 0     dNdy4
                     dNdy1 dNdx1 dNdy2 dNdx2 dNdy3 dNdx3 dNdy4 dNdx4]
                dΩ = element.detJdΩ[qp]
                B' * tovoigt(ke) *  B * dΩ
            end
            @test integrate((u,v,dΩ) -> (symmetric(∇(v)) ⊡ ke ⊡ symmetric(∇(u)))*dΩ, element) ≈ K
            # element vector
            σ = rand(SymmetricSecondOrderTensor{2})
            F = sum(1:Femto.num_quadpoints(element)) do qp
                (dNdx1,dNdy1), (dNdx2,dNdy2), (dNdx3,dNdy3), (dNdx4,dNdy4) = element.dNdx[qp]
                B = [dNdx1 0     dNdx2 0     dNdx3 0     dNdx4 0
                     0     dNdy1 0     dNdy2 0     dNdy3 0     dNdy4
                     dNdy1 dNdx1 dNdy2 dNdx2 dNdy3 dNdx3 dNdy4 dNdx4]
                dΩ = element.detJdΩ[qp]
                B' * tovoigt(σ) * dΩ
            end
            @test integrate((v,dΩ) -> (σ ⊡ symmetric(∇(v)))*dΩ, element) ≈ F
        end
    end
end
