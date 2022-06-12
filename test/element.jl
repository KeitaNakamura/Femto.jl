@testset "Element" begin
    for T in (Float64, Float32)
        TOL = sqrt(eps(T))
        for shape in (Quad4(), Tri6())
            element = Element{T}(ScalarField(), shape)
            for qp in Femto.num_quadpoints(element)
                # shape values and gradients
                @test sum(element.N[qp]) ≈ 1
                @test norm(sum(element.dNdx[qp])) < TOL
                # interpolate
                x, dxdx = Femto.interpolate(element, Femto.get_local_node_coordinates(shape), qp)
                @test x ≈ Femto.quadpoints(shape)[qp] atol=TOL
                @test dxdx ≈ one(dxdx) atol=TOL
            end
        end
    end
end
