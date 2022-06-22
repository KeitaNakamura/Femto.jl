using Femto: TensorStyle, MatrixStyle

@testset "Element" begin
    @testset "interpolate" begin
        for T in (Float64, Float32)
            TOL = sqrt(eps(T))
            for shape in (Line2(), Quad4(), Hex8(), Tri6())
                dim = Femto.get_dimension(shape)
                element = Element{T}(shape)
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
    @testset "integrate BodyElement" begin
        @testset "ScalarField" begin
            fieldtype = ScalarField()
            element = Element(Quad4())
            update!(element, [Vec(0.0,0.0), Vec(1.0,0.5), Vec(2.0,1.0), Vec(0.5,0.8)])
            # mass matrix
            M = sum(1:Femto.num_quadpoints(element)) do qp
                N = element.N[qp]'
                dΩ = element.detJdΩ[qp]
                N' * N * dΩ
            end
            @test integrate((qp,u,v) -> v*u, fieldtype, element) ≈ M
            @test integrate((qp,u,v) -> v*u, TensorStyle{:matrix}(), fieldtype, element) ≈ M
            @test integrate((qp,Nu,Nv) -> Nv'*Nu, MatrixStyle{:matrix}(), fieldtype, element) ≈ M
            # stiffness matrix
            K = sum(1:Femto.num_quadpoints(element)) do qp
                B = reduce(hcat, element.dNdx[qp])
                dΩ = element.detJdΩ[qp]
                B' * B * dΩ
            end
            @test integrate((qp,u,v) -> ∇(v)⋅∇(u), fieldtype, element) ≈ K
            @test integrate((qp,u,v) -> ∇(v)⋅∇(u), TensorStyle{:matrix}(), fieldtype, element) ≈ K
            @test integrate((qp,Nu,Nv) -> ∇(Nv)'*∇(Nu), MatrixStyle{:matrix}(), fieldtype, element) ≈ K
            # element vector
            F = sum(1:Femto.num_quadpoints(element)) do qp
                N = element.N[qp]
                dΩ = element.detJdΩ[qp]
                N * dΩ
            end
            @test integrate((qp,v) -> v, fieldtype, element) ≈ F
            @test integrate((qp,v) -> v, TensorStyle{:vector}(), fieldtype, element) ≈ F
            @test integrate((qp,Nv) -> Nv', MatrixStyle{:vector}(), fieldtype, element) |> vec ≈ F
        end
        @testset "VectorField" begin
            fieldtype = VectorField()
            element = Element(Quad4())
            update!(element, [Vec(0.0,0.0), Vec(1.0,0.5), Vec(2.0,1.0), Vec(0.5,0.8)])
            # mass matrix
            M = sum(1:Femto.num_quadpoints(element)) do qp
                N1, N2, N3, N4 = element.N[qp]
                N = [N1 0  N2 0  N3 0  N4 0
                     0  N1 0  N2 0  N3 0  N4]
                dΩ = element.detJdΩ[qp]
                N' * N * dΩ
            end
            @test integrate((qp,u,v) -> v⋅u, fieldtype, element) ≈ M
            @test integrate((qp,u,v) -> v⋅u, TensorStyle{:matrix}(), fieldtype, element) ≈ M
            @test integrate((qp,Nu,Nv) -> Nv'*Nu, MatrixStyle{:matrix}(), fieldtype, element) ≈ M
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
            @test integrate((qp,u,v) -> symmetric(∇(v)) ⊡ ke ⊡ symmetric(∇(u)), fieldtype, element) ≈ K
            @test integrate((qp,u,v) -> symmetric(∇(v)) ⊡ ke ⊡ symmetric(∇(u)), TensorStyle{:matrix}(), fieldtype, element) ≈ K
            @test integrate((qp,Nu,Nv) -> symmetric(∇(Nv))' * tovoigt(ke) * symmetric(∇(Nu)), MatrixStyle{:matrix}(), fieldtype, element) ≈ K
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
            @test integrate((qp,v) -> σ ⊡ symmetric(∇(v)), fieldtype, element) ≈ F
            @test integrate((qp,v) -> σ ⊡ symmetric(∇(v)), TensorStyle{:vector}(), fieldtype, element) ≈ F
            @test integrate((qp,Nv) -> symmetric(∇(Nv))' * tovoigt(σ), MatrixStyle{:vector}(), fieldtype, element) ≈ F
        end
    end
    @testset "integrate FaceElement" begin
        @testset "ScalarField" begin
            fieldtype = ScalarField()
            # dim 2
            element = FaceElement(Line2())
            p = rand()
            normal = normalize(Vec(1,-1))
            update!(element, [Vec(0.0,0.0), Vec(1.0,1.0)])
            @test all(≈(normal) , element.normal)
            F = sum(1:Femto.num_quadpoints(element)) do qp
                N1, N2 = element.N[qp]
                N = [N1, N2]
                dΩ = element.detJdΩ[qp]
                p * N * dΩ
            end
            @test integrate((qp,v,n) -> (p * v), fieldtype, element) ≈ F
            @test integrate((qp,v,n) -> (p * v), TensorStyle{:vector}(), fieldtype, element) ≈ F
            @test integrate((qp,Nv,n) -> (p * Nv'), MatrixStyle{:vector}(), fieldtype, element) |> vec ≈ F
        end
        @testset "VectorField" begin
            fieldtype = VectorField()
            # dim 2
            element = FaceElement(Line2())
            p = rand()
            normal = normalize(Vec(1,-1))
            update!(element, [Vec(0.0,0.0), Vec(1.0,1.0)])
            @test all(≈(normal) , element.normal)
            F = sum(1:Femto.num_quadpoints(element)) do qp
                N1, N2 = element.N[qp]
                N = [N1 0  N2 0 
                     0  N1 0  N2]
                dΩ = element.detJdΩ[qp]
                p * N' * normal * dΩ
            end
            @test integrate((qp,v,n) -> p*n ⋅ v, fieldtype, element) ≈ F
            @test integrate((qp,v,n) -> p*n ⋅ v, TensorStyle{:vector}(), fieldtype, element) ≈ F
            @test integrate((qp,Nv,n) -> p * Nv' * n, MatrixStyle{:vector}(), fieldtype, element) ≈ F
            # dim 3
            element = FaceElement(Quad4())
            p = rand()
            normal = normalize(Vec(1,0,1))
            update!(element, [Vec(0.0,0.0,0.0), Vec(1.0,0.0,-1.0), Vec(1.0,1.0,-1.0), Vec(0.0,1.0,0.0)])
            @test all(≈(normal) , element.normal)
            F = sum(1:Femto.num_quadpoints(element)) do qp
                N1, N2, N3, N4 = element.N[qp]
                N = [N1 0  0  N2 0  0  N3 0  0  N4 0  0
                     0  N1 0  0  N2 0  0  N3 0  0  N4 0
                     0  0  N1 0  0  N2 0  0  N3 0  0  N4]
                dΩ = element.detJdΩ[qp]
                p * N' * normal * dΩ
            end
            @test integrate((qp,v,n) -> p*n ⋅ v, fieldtype, element) ≈ F
            @test integrate((qp,v,n) -> p*n ⋅ v, TensorStyle{:vector}(), fieldtype, element) ≈ F
            @test integrate((qp,Nv,n) -> p * Nv' * n, MatrixStyle{:vector}(), fieldtype, element) |> vec ≈ F
        end
    end
end
