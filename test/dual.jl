@testset "Dual values" begin
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
