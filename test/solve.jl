@testset "Solve" begin
    @testset "linsolve" begin
        Random.seed!(1234)
        ## not symmetric
        K = sprand(50, 50, 0.2)
        F = rand(50)
        dirichlet = rand(Bool, 50)
        # SparseMatrixCSC
        U1 = zeros(50)
        linsolve!(U1, K, F, dirichlet)
        # Matrix
        U2 = zeros(50)
        linsolve!(U2, Array(K), F, dirichlet)
        @test U1 ≈ U2

        ## symmetric
        Us1 = zeros(50)
        # SparseMatrixCSC
        linsolve!(U1, K+K', F, dirichlet)
        linsolve!(Us1, Symmetric(K+K'), F, dirichlet)
        # Matrix
        Us2 = zeros(50)
        linsolve!(U2, Array(K+K'), F, dirichlet)
        linsolve!(Us2, Symmetric(Array(K+K')), F, dirichlet)
        @test U1 ≈ U2 ≈ Us1 ≈ Us2

        ## diagonal
        V = rand(50)
        Ud = zeros(50)
        linsolve!(U1, Array(Diagonal(V)), F, dirichlet)
        linsolve!(Ud, Diagonal(V), F, dirichlet)
        @test U1 ≈ Ud
    end
    @testset "nlsolve" begin
        Random.seed!(1234)
        U = [0.0]
        dirichlet = [false]
        a = rand()
        b = rand()
        c = rand()
        history = nlsolve!(U, dirichlet) do R, J, U
            @. R = a*U^2 + b*U + c
            @. J = 2a*U + b
        end
        @test U[1] ≈ (-b+sqrt(b^2-4a*c)) / 2a
        @test norm(history) ≈ 1.028666021302189
    end
end
