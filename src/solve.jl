using NewtonSolvers

function dirichlet_to_neumann!(F::AbstractVector, U::AbstractVector, K::AbstractMatrix, ddofs::AbstractVector{Int})
    @boundscheck checkbounds(F, ddofs)
    @assert length(U) == size(K, 1) == size(K, 2) == length(F)
    @inbounds for j in ddofs
        for i in 1:length(F)
            F[i] -= U[j] * K[i, j]
        end
    end
    @. F[ddofs] = U[ddofs]
    _modify_stencil!(K, ddofs)
end

function dirichlet_to_neumann!(F::AbstractVector, U::AbstractVector, K::SparseMatrixCSC, ddofs::AbstractVector{Int})
    @boundscheck checkbounds(F, ddofs)
    @assert length(U) == size(K, 1) == size(K, 2) == length(F)
    rows = rowvals(K)
    vals = nonzeros(K)
    @inbounds for j in ddofs
        for i in nzrange(K, j)
            row = rows[i]
            val = vals[i]
            F[row] -= U[j] * val
        end
    end
    @. F[ddofs] = U[ddofs]
    _modify_stencil!(K, ddofs)
end

function _modify_stencil!(K::AbstractMatrix, ddofs::AbstractVector{Int})
    @inbounds for j in ddofs
        K[:,j] .= 0
        K[j,:] .= 0
        K[j,j] = 1
    end
end
function _modify_stencil!(K::SparseMatrixCSC, ddofs::AbstractVector{Int})
    rows = rowvals(K)
    vals = nonzeros(K)
    jᵈ = 1
    @inbounds for j in 1:size(K, 2)
        if jᵈ ≤ length(ddofs) && j == ddofs[jᵈ]
            for i in nzrange(K, j)
                row = rows[i]
                vals[i] = ifelse(row==j, 1, 0)
            end
            jᵈ += 1
        else
            for i in nzrange(K, j)
                row = rows[i]
                idx = searchsortedfirst(ddofs, row)
                if idx ≤ length(ddofs) && row == ddofs[idx]
                    vals[i] = 0
                end
            end
        end
    end
end
_modify_stencil!(K::Symmetric, ddofs::AbstractVector{Int}) = _modify_stencil!(parent(K), ddofs)
_modify_stencil!(K::Diagonal, ddofs::AbstractVector{Int}) = fill!(view(diag(K), ddofs), 1)

function linsolve!(U::AbstractVector, K::AbstractMatrix, F::AbstractVector, dirichlet::AbstractVector{Bool})
    linsolve!(U, K, F, findall(dirichlet))
end

function linsolve!(U::AbstractVector, K::AbstractMatrix, F::AbstractVector, ddofs::AbstractVector{Int})
    dirichlet_to_neumann!(F, U, K, ddofs)
    U .= K \ F
    U
end

function nlsolve!(
        R!,
        J!,
        U::AbstractVector{T},
        dirichlet::AbstractVector{Bool};
        iterations::Int = 1000,
        f_tol::Real = sqrt(eps(T)),
        x_tol::Real = zero(T),
        dx_tol::Real = zero(T),
        backtracking::Bool = true,
        showtrace::Bool = false,
        logall::Bool = false,
        symmetric::Bool = false,
    ) where {T <: Real}
    @assert length(U) == length(dirichlet)

    ndofs = length(U)
    ddofs = findall(dirichlet)
    R = zeros(T, ndofs)
    J = spzeros(T, ndofs, ndofs)

    function R_mod!(R, U)
        R!(R, U)
        R[ddofs] .= 0
    end

    ch = NewtonSolvers.solve!(R_mod!, J!, R, J, U;
                              linsolve = (x,A,b) -> linsolve!(x,ifelse(symmetric,Symmetric(A),A),b,ddofs),
                              backtracking,
                              f_tol,
                              x_tol,
                              iterations,
                              showtrace)
    ch.isconverged || @warn "not converged in Newton iteration"

    ch
end
