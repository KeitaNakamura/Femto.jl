@testset "Solve" begin
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
    linsolve!(U1, K+K', F, dirichlet; symmetric=false)
    linsolve!(Us1, K+K', F, dirichlet; symmetric=true)
    # Matrix
    Us2 = zeros(50)
    linsolve!(U2, Array(K+K'), F, dirichlet; symmetric=false)
    linsolve!(Us2, Array(K+K'), F, dirichlet; symmetric=true)
    @test U1 ≈ U2 ≈ Us1 ≈ Us2
end
