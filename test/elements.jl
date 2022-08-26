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
                element = Element(T, shape)
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
            V = sum(integrate(qp->1, element))
            @test V ≈ get_volume(shape)
            # high order quadrature
            shape_qr = highorder(shape)
            shape_qr === nothing && continue
            element = Element(shape, shape_qr)
            V = sum(integrate(qp->1, element))
            @test V ≈ get_volume(shape)
        end
    end
end

@testset "SingleBodyElement integration" begin
    @testset "ScalarField" begin
        element = Element(Quad4())
        update!(element, [Vec(0.0,0.0), Vec(1.0,0.5), Vec(2.0,1.0), Vec(0.5,0.8)])
        # mass matrix
        M = sum(1:Femto.num_quadpoints(element)) do qp
            N = element.N[qp]'
            dΩ = element.detJdΩ[qp]
            N' * N * dΩ
        end
        @test integrate((qp,v,u) -> v*u, Sf(), element) ≈ M
        @test integrate((qp,v,u) -> v*u, Sf(1), element) ≈ M
        @test_throws Exception integrate((qp,v,u) -> v*u, Sf(2), element)
        # stiffness matrix
        K = sum(1:Femto.num_quadpoints(element)) do qp
            B = reduce(hcat, element.dNdx[qp])
            dΩ = element.detJdΩ[qp]
            B' * B * dΩ
        end
        @test integrate((qp,v,u) -> ∇(v)⋅∇(u), Sf(), element) ≈ K
        @test integrate((qp,v,u) -> ∇(v)⋅∇(u), Sf(1), element) ≈ K
        @test_throws Exception integrate((qp,v,u) -> ∇(v)⋅∇(u), Sf(2), element)
        # element vector
        F = sum(1:Femto.num_quadpoints(element)) do qp
            N = element.N[qp]
            dΩ = element.detJdΩ[qp]
            N * dΩ
        end
        @test integrate((qp,v) -> v, Sf(), element) ≈ F
        @test integrate((qp,v) -> v, Sf(1), element) ≈ F
        @test_throws Exception integrate((qp,v) -> v, Sf(2), element)
    end
    @testset "VectorField" begin
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
        @test integrate((qp,v,u) -> v⋅u, Vf(), element) ≈ M
        @test integrate((qp,v,u) -> v⋅u, Vf(1), element) ≈ M
        @test_throws Exception integrate((qp,v,u) -> v⋅u, Vf(2), element)
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
        @test integrate((qp,v,u) -> symmetric(∇(v)) ⊡ ke ⊡ symmetric(∇(u)), Vf(), element) ≈ K
        @test integrate((qp,v,u) -> symmetric(∇(v)) ⊡ ke ⊡ symmetric(∇(u)), Vf(1), element) ≈ K
        @test_throws Exception integrate((qp,v,u) -> symmetric(∇(v)) ⊡ ke ⊡ symmetric(∇(u)), Vf(2), element)
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
        @test integrate((qp,v) -> σ ⊡ symmetric(∇(v)), Vf(), element) ≈ F
        @test integrate((qp,v) -> σ ⊡ symmetric(∇(v)), Vf(1), element) ≈ F
        @test_throws Exception integrate((qp,v) -> σ ⊡ symmetric(∇(v)), Vf(2), element)
    end
end

@testset "SingleFaceElement integration" begin
    @testset "ScalarField" begin
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
        @test integrate((qp,n,v) -> (p * v), Sf(), element) ≈ F
        @test integrate((qp,n,v) -> (p * v), Sf(1), element) ≈ F
        @test_throws Exception integrate((qp,n,v) -> (p * v), Sf(2), element)
    end
    @testset "VectorField" begin
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
            @test integrate((qp,n,v) -> p*n ⋅ v, Vf(), element) ≈ F
            @test integrate((qp,n,v) -> p*n ⋅ v, Vf(1), element) ≈ F
            @test_throws Exception integrate((qp,n,v) -> p*n ⋅ v, Vf(2), element)
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
            @test integrate((qp,n,v) -> p*n ⋅ v, Vf(), element) ≈ F
            @test integrate((qp,n,v) -> p*n ⋅ v, Vf(1), element) ≈ F
            @test_throws Exception integrate((qp,n,v) -> p*n ⋅ v, Vf(2), element)
        end
    end
end

@testset "MixedElement" begin
    @testset "MixedBodyElement integration" begin
        field = mixed(VectorField(), ScalarField())
        element = Element(mixed(Quad9(), Quad4()))
        f = (index, (v,q), (u,p)) -> ∇(v) ⊡ ∇(u) - (∇⋅v)*p + q*(∇⋅u) + q*p
        # classical way
        elt1 = Element(Quad9(), Quad9())
        elt2 = Element(Quad4(), Quad9())
        K = sum(1:Femto.num_quadpoints(elt1)) do qp
            (dUdx1,dUdy1), (dUdx2,dUdy2), (dUdx3,dUdy3), (dUdx4,dUdy4), (dUdx5,dUdy5), (dUdx6,dUdy6), (dUdx7,dUdy7), (dUdx8,dUdy8), (dUdx9,dUdy9) = elt1.dNdx[qp]
            P1, P2, P3, P4 = elt2.N[qp]
            dudx = [dUdx1 0     dUdx2 0     dUdx3 0     dUdx4 0     dUdx5 0     dUdx6 0     dUdx7 0     dUdx8 0     dUdx9 0
                    0     dUdy1 0     dUdy2 0     dUdy3 0     dUdy4 0     dUdy5 0     dUdy6 0     dUdy7 0     dUdy8 0     dUdy9
                    dUdy1 0     dUdy2 0     dUdy3 0     dUdy4 0     dUdy5 0     dUdy6 0     dUdy7 0     dUdy8 0     dUdy9 0
                    0     dUdx1 0     dUdx2 0     dUdx3 0     dUdx4 0     dUdx5 0     dUdx6 0     dUdx7 0     dUdx8 0     dUdx9]
            dudx_v = [dUdx1 dUdy1 dUdx2 dUdy2 dUdx3 dUdy3 dUdx4 dUdy4 dUdx5 dUdy5 dUdx6 dUdy6 dUdx7 dUdy7 dUdx8 dUdy8 dUdx9 dUdy9]
            p = [P1 P2 P3 P4]
            dΩ = elt1.detJdΩ[qp]
            [dudx'*dudx -dudx_v'*p
             p'*dudx_v  p'*p] * dΩ
        end
        @test integrate(f, field, element) ≈ K
        @test integrate(f, mixed(Vf(2), Sf(1)), element) ≈ K
        @test_throws Exception integrate(f, mixed(Vf(1), Sf(1)), element)
        @test_throws Exception integrate(f, mixed(Vf(2), Sf(2)), element)
    end
    @testset "MixedFaceElement integration" begin
        field = mixed(VectorField(), ScalarField())
        element = FaceElement(mixed(Quad9(), Quad4()))
        a = rand(Vec{3})
        b = rand()
        f = (index, normal, (v,q)) -> v⋅a + b*q
        # classical way
        elt1 = FaceElement(Quad9(), Quad9())
        elt2 = FaceElement(Quad4(), Quad9())
        F = sum(1:Femto.num_quadpoints(elt1)) do qp
            U1, U2, U3, U4, U5, U6, U7, U8, U9 = elt1.N[qp]
            P1, P2, P3, P4 = elt2.N[qp]
            u = [U1 0  0  U2 0  0  U3 0  0  U4 0  0  U5 0  0  U6 0  0  U7 0  0  U8 0  0  U9 0  0
                 0  U1 0  0  U2 0  0  U3 0  0  U4 0  0  U5 0  0  U6 0  0  U7 0  0  U8 0  0  U9 0
                 0  0  U1 0  0  U2 0  0  U3 0  0  U4 0  0  U5 0  0  U6 0  0  U7 0  0  U8 0  0  U9]
            p = [P1 P2 P3 P4]
            dΩ = elt1.detJdΩ[qp]
            vec(vcat(u' * Vector(a), p' * b)) * dΩ
        end
        @test integrate(f, field, element) ≈ F
        @test integrate(f, mixed(Vf(2), Sf(1)), element) ≈ F
        @test_throws Exception integrate(f, mixed(Vf(1), Sf(1)), element)
        @test_throws Exception integrate(f, mixed(Vf(2), Sf(2)), element)
    end
    @testset "interpolate" begin
        # field
        fld1 = VectorField(2)
        fld2 = ScalarField(1)
        mixed_fld = MixedField(fld1, fld2)
        # element
        elt1 = Element(Quad9(), Quad9())
        elt2 = Element(Quad4(), Quad9())
        mixed_elt = MixedElement(elt1, elt2)
        # num_dofs
        n1 = num_dofs(fld1, elt1)
        n2 = num_dofs(fld2, elt2)
        n = num_dofs(mixed_fld, mixed_elt)
        @test n1 + n2 == n
        # interpolate
        Ui = rand(n)
        mixed_U = interpolate(mixed_fld, mixed_elt, Ui)
        @test interpolate(fld1, elt1, Ui[1:n1]) ≈ map(u->u[1], mixed_U)
        @test interpolate(fld2, elt2, Ui[n1+1:n]) ≈ map(u->u[2], mixed_U)
        # wrong interpolation order
        @test_throws Exception interpolate(fld2, elt1, Ui[1:n1])
        @test_throws Exception interpolate(fld1, elt2, Ui[n1+1:n])
    end
end
