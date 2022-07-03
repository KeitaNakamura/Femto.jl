function solve!(U::AbstractVector, K::AbstractMatrix, F::AbstractVector, dirichlet::AbstractVector{Bool})
    @assert length(U) == size(K, 1) == length(F) == length(dirichlet)
    @assert size(K, 1) == size(K, 2)
    for i in Iterators.filter(i -> dirichlet[i], eachindex(dirichlet))
        @inbounds F .-= U[i] * K[:, i]
    end
    fdofs = findall(.!dirichlet)
    @inbounds U[fdofs] = K[fdofs, fdofs] \ F[fdofs]
    U
end
solve!(U::AbstractVector, K::SparseMatrixIJV, F::AbstractVector, dirichlet::AbstractVector{Bool}) = solve!(U, sparse(K), F, dirichlet)
