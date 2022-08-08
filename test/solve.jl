@testset "Solve" begin
    Random.seed!(1234)
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
end
