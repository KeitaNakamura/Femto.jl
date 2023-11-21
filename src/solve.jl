using NewtonSolvers

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
    U .= K \ F
    U
end

get_K_fdofs(K::AbstractMatrix, fdofs::Vector{Int}) = (@_propagate_inbounds_meta; K[fdofs,fdofs])
get_K_fdofs(K::Symmetric, fdofs::Vector{Int}) = (@_propagate_inbounds_meta; Symmetric(get_K_fdofs(K.data, fdofs), Symbol(K.uplo)))
get_K_fdofs(K::Diagonal, fdofs::Vector{Int}) = (@_propagate_inbounds_meta; Diagonal(view(diag(K), fdofs)))

function linsolve!(U::AbstractVector, K::AbstractMatrix, F::AbstractVector, dirichlet::AbstractVector{Bool})
    @assert length(U) == size(K, 1) == size(K, 2) == length(F) == length(dirichlet)
    dirichlet_to_neumann!(F, U, K, dirichlet)
    fdofs = findall(.!dirichlet)
    @inbounds linsolve!(view(U, fdofs), get_K_fdofs(K, fdofs), F[fdofs])
    U
end

function nlsolve!(
        R!,
        J!,
        U::AbstractVector{T},
        dirichlet::AbstractVector{Bool};
        maxiter::Int = 20,
        f_tol::Real = convert(T, 1e-8),
        x_tol::Real = zero(T),
        backtracking::Bool = true,
        showtrace::Bool = false,
        symmetric::Bool = false,
    ) where {T <: Real}
    @assert length(U) == length(dirichlet)

    ndofs = length(U)
    fdofs = findall(.!dirichlet)
    R = zeros(T, ndofs)
    J = spzeros(T, ndofs, ndofs)

    function R_mod!(R, U)
        R!(R, U)
        R[dirichlet] .= 0
    end

    function linsolve(x, A, b)
        x′ = view(x, fdofs)
        A′ = get_K_fdofs(symmetric ? Symmetric(A) : A, fdofs)
        b′ = b[fdofs]
        linsolve!(x′, A′, b′)
    end

    converged = NewtonSolvers.solve!(R_mod!, J!, R, J, U;
                                     linsolve,
                                     backtracking,
                                     f_tol,
                                     x_tol,
                                     maxiter,
                                     showtrace)
    converged || @warn "not converged in Newton iteration"

    converged
end
