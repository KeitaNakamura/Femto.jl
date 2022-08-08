function dirichlet_to_neumann!(F::AbstractVector, U::AbstractVector, K::AbstractMatrix, dirichlet::AbstractVector{Bool})
    @assert length(U) == size(K, 1) == size(K, 2) == length(F) == length(dirichlet)
    @inbounds for j in Iterators.filter(j -> dirichlet[j], eachindex(dirichlet))
        for i in 1:length(F)
            F[i] -= U[j] * K[i, j]
        end
    end
end
function dirichlet_to_neumann!(F::AbstractVector, U::AbstractVector, K::SparseMatrixCSC, dirichlet::AbstractVector{Bool})
    @assert length(U) == size(K, 1) == size(K, 2) == length(F) == length(dirichlet)
    rows = rowvals(K)
    vals = nonzeros(K)
    @inbounds for j in Iterators.filter(j -> dirichlet[j], eachindex(dirichlet))
        for i in nzrange(K, j)
            row = rows[i]
            val = vals[i]
            F[row] -= U[j] * val
        end
    end
end

function linsolve!(U::AbstractVector, K::AbstractMatrix, F::AbstractVector, dirichlet::AbstractVector{Bool})
    @assert length(U) == size(K, 1) == size(K, 2) == length(F) == length(dirichlet)
    dirichlet_to_neumann!(F, U, K, dirichlet)
    fdofs = findall(.!dirichlet)
    @inbounds U[fdofs] = K[fdofs, fdofs] \ F[fdofs]
    U
end
linsolve!(U::AbstractVector, K::SparseMatrixCOO, F::AbstractVector, dirichlet::AbstractVector{Bool}) = linsolve!(U, sparse(K), F, dirichlet)

function nlsolve!(f!, U::AbstractVector, dirichlet::AbstractVector{Bool}, args...; maxiter::Int=20, tol::Real=1e-8)
    @assert length(U) == length(dirichlet)
    n = length(U)
    R = zeros(n)
    dU = zeros(n)
    J = spzeros(n, n)
    for step in 1:maxiter
        f!(R, J, U, args...)
        fdofs = findall(.!dirichlet)
        @inbounds dU[fdofs] = J[fdofs, fdofs] \ -R[fdofs]
        @. U += dU
        norm(dU)/norm(U) < tol && return
    end
    error("not converged")
end
