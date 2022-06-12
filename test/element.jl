@testset "Element" begin
    for T in (Float64, Float32)
        TOL = sqrt(eps(T))
        for shape in (Quad4(), Tri6())
            element = Element{T}(ScalarField(), shape)
            for qp in Femto.num_quadpoints(element)
                @test sum(element.N[qp]) ≈ 1
                @test norm(sum(element.dNdx[qp])) < TOL
            end
        end
    end
end
