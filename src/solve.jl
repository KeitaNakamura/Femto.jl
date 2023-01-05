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
        f!,
        U::AbstractVector{T},
        dirichlet::AbstractVector{Bool},
        parameter = nothing;
        maxiter::Int = 20,
        tol::Real = 1e-8,
        symmetric::Bool = false,
    ) where {T <: Real}
    @assert length(U) == length(dirichlet)

    ndofs = length(U)
    R = zeros(T, ndofs)
    dU = zeros(T, ndofs)
    J = spzeros(T, ndofs, ndofs)
    fdofs = findall(.!dirichlet)

    history = Float64[]
    local r0::Float64

    for step in 1:maxiter
        parameter===nothing ? f!(R, J, U) : f!(R, J, U, parameter)

        R′ = R[fdofs]
        if step == 1
            r0 = norm(R′)
            r0<tol && return history
        end
        push!(history, norm(R′)/r0)
        history[end]<tol && return history

        if symmetric
            @inbounds linsolve!(view(dU, fdofs), get_K_fdofs(Symmetric(J), fdofs), R′)
        else
            @inbounds linsolve!(view(dU, fdofs), get_K_fdofs(J, fdofs), R′)
        end
        @. U -= dU
    end
    error("not converged")
end
