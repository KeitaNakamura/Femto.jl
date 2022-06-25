##################
# SolutionVector #
##################

struct SolutionVector{T} <: AbstractVector{T}
    data::Vector{T}
    mask::BitVector
end
SolutionVector{T}(n::Int) where {T} = SolutionVector(zeros(T, n), falses(n))

function create_solutionvector(::Type{T}, fieldtype::FieldType, grid::Grid) where {T}
    n = num_dofs(fieldtype, grid)
    SolutionVector{T}(n)
end
create_solutionvector(fieldtype::FieldType, grid::Grid{T}) where {T} = create_solutionvector(T, fieldtype, grid)

Base.size(x::SolutionVector) = size(x.data)
Base.getindex(x::SolutionVector, i::Int) = (@_propagate_inbounds_meta; x.data[i])
function Base.setindex!(x::SolutionVector, v, i::Int)
    @_propagate_inbounds_meta
    x.data[i] = v
    x.mask[i] = true
    x
end
function fillzero!(x::SolutionVector)
    fillzero!(x.data)
    fillzero!(x.mask)
    x
end

##########
# solve! #
##########

function solve!(U::SolutionVector, K::AbstractMatrix, F::AbstractVector)
    @assert length(U) == size(K, 1) == length(F)
    mask = U.mask
    for i in Iterators.filter(i -> mask[i], eachindex(mask))
        @inbounds F .-= U[i] * K[:, i]
    end
    fdofs = findall(.!mask)
    @inbounds U[fdofs] = K[fdofs, fdofs] \ F[fdofs]
    U
end
solve!(U::SolutionVector, K::SparseMatrixIJV, F::AbstractVector) = solve!(U, sparse(K), F)

