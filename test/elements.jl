get_volume(::Line2) = 2
get_volume(::Line3) = 2
get_volume(::Quad4) = 4
get_volume(::Quad9) = 4
get_volume(::Hex8) = 8
get_volume(::Hex27) = 8
get_volume(::Tri3) = 1/2
get_volume(::Tri6) = 1/2
get_volume(::Tet4) = 1/6
get_volume(::Tet10) = 1/6

highorder(::Line2) = Line3()
highorder(::Line3) = nothing
highorder(::Quad4) = Quad9()
highorder(::Quad9) = nothing
highorder(::Hex8)  = Hex27()
highorder(::Hex27) = nothing
highorder(::Tri3)  = Tri6()
highorder(::Tri6)  = nothing
highorder(::Tet4)  = Tet10()
highorder(::Tet10) = nothing

@testset "Element" begin
    @testset "interpolate" begin
        @testset "$shape" for shape in allshapes()
            for T in (Float64, Float32)
                TOL = sqrt(eps(T))
                dim = Femto.get_dimension(shape)
                element = Element{T}(shape)
                X = @inferred interpolate(VectorField(), element, reinterpret(T, Femto.get_local_coordinates(element)))
                for qp in Femto.num_quadpoints(element)
                    ## shape values and gradients
                    @test sum(element.N[qp]) ≈ 1
                    @test norm(sum(element.dNdx[qp])) < TOL
                    ## interpolate
                    # qp
                    x = @inferred interpolate(element, Femto.get_local_coordinates(element), qp)
                    dxdx = ∇(x)
                    @test x ≈ Femto.quadpoints(T, shape)[qp] atol=TOL
                    @test x ≈ X[qp] atol=TOL
                    @test dxdx ≈ one(dxdx) atol=TOL
                    @test dxdx ≈ ∇(X[qp]) atol=TOL
                    # ξ
                    x′ = rand(Vec{dim, T})
                    x = @inferred interpolate(element, Femto.get_local_coordinates(element), x′)
                    dxdx = ∇(x)
                    @test x ≈ x′ atol=TOL
                    @test dxdx ≈ one(dxdx) atol=TOL
                end
            end
        end
    end
    @testset "gauss quadrature" begin
        @testset "$shape" for shape in allshapes()
            element = Element(shape)
            V = sum(integrate((qp,u,v)->v*u, ScalarField(), element))
            @test V ≈ get_volume(shape)
            # high order quadrature
            shape_qr = highorder(shape)
            shape_qr === nothing && continue
            element = Element(shape, shape_qr)
            V = sum(integrate((qp,u,v)->v*u, ScalarField(), element))
            @test V ≈ get_volume(shape)
        end
    end
end

@testset "BodyElement integration" begin
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
        # stiffness matrix
        K = sum(1:Femto.num_quadpoints(element)) do qp
            B = reduce(hcat, element.dNdx[qp])
            dΩ = element.detJdΩ[qp]
            B' * B * dΩ
        end
        @test integrate((qp,u,v) -> ∇(v)⋅∇(u), fieldtype, element) ≈ K
        # element vector
        F = sum(1:Femto.num_quadpoints(element)) do qp
            N = element.N[qp]
            dΩ = element.detJdΩ[qp]
            N * dΩ
        end
        @test integrate((qp,v) -> v, fieldtype, element) ≈ F
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
    end
end

@testset "FaceElement integration" begin
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
    end
    @testset "VectorField" begin
        fieldtype = VectorField()
        @testset "dim = 2" begin
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
        end
        @testset "dim = 3" begin
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
        end
    end
end
