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

function linsolve!(U::AbstractVector, K::AbstractMatrix, F::AbstractVector)
    @assert length(U) == size(K, 1) == size(K, 2) == length(F)
    @inbounds U .= K \ F
    U
end

get_K_fdofs(K::AbstractMatrix, fdofs::Vector{Int}) = (@_propagate_inbounds_meta; K[fdofs,fdofs])
get_K_fdofs(K::Symmetric, fdofs::Vector{Int}) = (@_propagate_inbounds_meta; Symmetric(K.data[fdofs,fdofs], Symbol(K.uplo)))
get_K_fdofs(K::Diagonal, fdofs::Vector{Int}) = (@_propagate_inbounds_meta; Diagonal(view(diag(K), fdofs)))

function linsolve!(U::AbstractVector, K::AbstractMatrix, F::AbstractVector, dirichlet::AbstractVector{Bool})
    @assert length(U) == size(K, 1) == size(K, 2) == length(F) == length(dirichlet)
    dirichlet_to_neumann!(F, U, K, dirichlet)
    fdofs = findall(.!dirichlet)
    @inbounds linsolve!(view(U, fdofs), get_K_fdofs(K, fdofs), F[fdofs])
    U
end
linsolve!(U::AbstractVector, K::SparseMatrixCOO, F::AbstractVector, dirichlet::AbstractVector{Bool}) = linsolve!(U, sparse(K), F, dirichlet)

function nlsolve!(f!, U::AbstractVector{T}, dirichlet::AbstractVector{Bool}, args...; maxiter::Int=20, tol::Real=1e-8, symmetric::Bool=false) where {T <: Real}
    @assert length(U) == length(dirichlet)
    n = length(U)
    R = zeros(T, n)
    dU = zeros(T, n)
    J = spzeros(T, n, n)
    fdofs = findall(.!dirichlet)
    history = Float64[]
    for step in 1:maxiter
        f!(R, J, U, args...)
        if symmetric
            @inbounds linsolve!(view(dU, fdofs), get_K_fdofs(Symmetric(J), fdofs), R[fdofs])
        else
            @inbounds linsolve!(view(dU, fdofs), get_K_fdofs(J, fdofs), R[fdofs])
        end
        dU_norm = norm(dU)
        U_norm = norm(U)
        if dU_norm<tol && U_norm<tol
            push!(history, 0)
        else
            push!(history, dU_norm/U_norm)
        end
        history[end] < tol && return history
        @. U -= dU
    end
    error("not converged")
end
