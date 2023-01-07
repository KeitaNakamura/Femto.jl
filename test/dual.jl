@testset "Dual values" begin
    @testset "ScalarGradient" begin
        for T in (Float32, Float64)
            s1 = rand(T)
            s2 = rand(T)
            x1 = Femto.ScalarGradient(s1, rand(T))
            x2 = Femto.ScalarGradient(s2, rand(T))
            for op in (+, -, *, /)
                @test (@inferred op(x1, x2))::T == op(s1, s2)
                @test (@inferred op(x1, s2))::T == op(s1, s2)
                @test (@inferred op(s1, x2))::T == op(s1, s2)
            end
            for op in (+, -)
                @test (@inferred op(x1))::T == op(s1)
            end
        end
    end
    @testset "Automatic differentiation" begin
        for T in (Float32, Float64)
            # ScalarGradient
            s = rand(T)
            x = Femto.ScalarGradient(s, rand(T))
            f(x) = 2x^2 - 3x + 5
            g(x) = 4x - 3
            @test (@inferred gradient(f, x))::T == g(s)
            @test (@inferred gradient(f, x, :all))::Tuple{T,T} == (g(s), f(s))
            # TensorGradient
            v = rand(Vec{3,T})
            x = Femto.TensorGradient(v, rand(Mat{3,3,T}))
            @test (@inferred gradient(mean, x))::Vec{3,T} ≈ [1,1,1]/3
            @test ((@inferred gradient(mean, x, :all))::Tuple{Vec{3,T}, T})[1] ≈ [1,1,1]/3
            @test ((@inferred gradient(mean, x, :all))::Tuple{Vec{3,T}, T})[2] ≈ mean(v)
        end
    end
end
