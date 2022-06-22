using Femto: TensorStyle, MatrixStyle

@testset "B matrix" begin
    # dim 1
    element = Element(Line2())
    for qp in 1:Femto.num_quadpoints(element)
        dNdx1, dNdx2 = element.dNdx[qp]
        dNdx = Femto.shape_gradients(MatrixStyle{:matrix}(), VectorField(), element, qp)
        @test dNdx ≈ [only(dNdx1) only(dNdx2)]
        @test symmetric(dNdx) ≈ dNdx
    end
    # dim 2
    element = Element(Quad4())
    for qp in 1:Femto.num_quadpoints(element)
        (dNdx1, dNdy1), (dNdx2, dNdy2), (dNdx3, dNdy3), (dNdx4, dNdy4) = element.dNdx[qp]
        dNdx = Femto.shape_gradients(MatrixStyle{:matrix}(), VectorField(), element, qp)
        @test dNdx ≈ [dNdx1 0     dNdx2 0       dNdx3 0     dNdx4 0
                      0     dNdy1 0     dNdy2 0       dNdy3 0     dNdy4]
        @test symmetric(dNdx) ≈ [dNdx1 0     dNdx2 0     dNdx3 0     dNdx4 0
                                 0     dNdy1 0     dNdy2 0     dNdy3 0     dNdy4
                                 dNdy1 dNdx1 dNdy2 dNdx2 dNdy3 dNdx3 dNdy4 dNdx4]
    end
    # dim 3
    element = Element(Hex8())
    for qp in 1:Femto.num_quadpoints(element)
        (dNdx1, dNdy1, dNdz1), (dNdx2, dNdy2, dNdz2), (dNdx3, dNdy3, dNdz3), (dNdx4, dNdy4, dNdz4), (dNdx5, dNdy5, dNdz5), (dNdx6, dNdy6, dNdz6), (dNdx7, dNdy7, dNdz7), (dNdx8, dNdy8, dNdz8) = element.dNdx[qp]
        dNdx = Femto.shape_gradients(MatrixStyle{:matrix}(), VectorField(), element, qp)
        @test dNdx ≈ [dNdx1 0     0     dNdx2 0     0     dNdx3 0     0     dNdx4 0     0     dNdx5 0     0     dNdx6 0     0     dNdx7 0     0     dNdx8 0     0
                      0     dNdy1 0     0     dNdy2 0     0     dNdy3 0     0     dNdy4 0     0     dNdy5 0     0     dNdy6 0     0     dNdy7 0     0     dNdy8 0
                      0     0     dNdz1 0     0     dNdz2 0     0     dNdz3 0     0     dNdz4 0     0     dNdz5 0     0     dNdz6 0     0     dNdz7 0     0     dNdz8]
        @test symmetric(dNdx) ≈ [dNdx1 0     0     dNdx2 0     0     dNdx3 0     0     dNdx4 0     0     dNdx5 0     0     dNdx6 0     0     dNdx7 0     0     dNdx8 0     0
                                 0     dNdy1 0     0     dNdy2 0     0     dNdy3 0     0     dNdy4 0     0     dNdy5 0     0     dNdy6 0     0     dNdy7 0     0     dNdy8 0
                                 0     0     dNdz1 0     0     dNdz2 0     0     dNdz3 0     0     dNdz4 0     0     dNdz5 0     0     dNdz6 0     0     dNdz7 0     0     dNdz8
                                 dNdy1 dNdx1 0     dNdy2 dNdx2 0     dNdy3 dNdx3 0     dNdy4 dNdx4 0     dNdy5 dNdx5 0     dNdy6 dNdx6 0     dNdy7 dNdx7 0     dNdy8 dNdx8 0
                                 dNdz1 0     dNdx1 dNdz2 0     dNdx2 dNdz3 0     dNdx3 dNdz4 0     dNdx4 dNdz5 0     dNdx5 dNdz6 0     dNdx6 dNdz7 0     dNdx7 dNdz8 0     dNdx8
                                 0     dNdz1 dNdy1 0     dNdz2 dNdy2 0     dNdz3 dNdy3 0     dNdz4 dNdy4 0     dNdz5 dNdy5 0     dNdz6 dNdy6 0     dNdz7 dNdy7 0     dNdz8 dNdy8]
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
