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
    @testset "add!" begin
        @testset "matrix" begin
            A = rand(10, 10)
            I = [1,3,8]
            J = [2,9,3,5]
            K = rand(3, 4)
            A′ = copy(A)
            A′[I, J] += K
            @test Femto.add!(A, I, J, K) ≈ A′
        end
        @testset "vector" begin
            A = rand(10)
            I = [2,9,3,5]
            F = rand(4)
            A′ = copy(A)
            A′[I] += F
            @test Femto.add!(A, I, F) ≈ A′
        end
    end
end
