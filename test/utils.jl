@testset "Utilities" begin
    @testset "dofindices" begin
        for dim in (1, 2, 3)
            conn = Femto.Index(1,3,6)
            # scalar field
            @test (@inferred Femto.dofindices(ScalarField(), Val(dim), conn)) == Femto.Index(1,3,6)
            # vector field
            if dim == 1
                @test (@inferred Femto.dofindices(VectorField(), Val(dim), conn)) == Femto.Index(1,3,6)
            elseif dim == 2
                @test (@inferred Femto.dofindices(VectorField(), Val(dim), conn)) == Femto.Index(1,2,5,6,11,12)
            elseif dim == 3
                @test (@inferred Femto.dofindices(VectorField(), Val(dim), conn)) == Femto.Index(1,2,3,7,8,9,16,17,18)
            else
                error("unreachable")
            end
        end
    end
end
