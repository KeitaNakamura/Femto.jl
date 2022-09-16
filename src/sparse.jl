###################
# SparseMatrixCOO #
###################

struct SparseMatrixCOO{T} <: AbstractMatrix{T}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}
    m::Int
    n::Int
end

function SparseMatrixCOO{T}(m::Int, n::Int; sizehint::Int = 0) where T
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, sizehint)
    sizehint!(J, sizehint)
    sizehint!(V, sizehint)
    SparseMatrixCOO(I, J, V, m, n)
end
SparseMatrixCOO(m::Int, n::Int; sizehint::Int = round(Int, 0.01*m*n)) = SparseMatrixCOO{Float64}(m, n; sizehint)

Base.size(A::SparseMatrixCOO) = (A.m, A.n)

SparseArrays.sparse(A::SparseMatrixCOO) = sparse(A, size(A)...)
SparseArrays.sparse(A::SparseMatrixCOO, m::Int, n::Int, args...) = sparse(A.I, A.J, A.V, m, n, args...)
Base.copyto!(dest::AbstractMatrix, src::SparseMatrixCOO) = (copyto!(dest, sparse(src)); dest)

function add!(A::SparseMatrixCOO, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix)
    m, n = map(length, (I, J))
    @assert size(K) == (m, n)
    append!(A.V, K)
    @inbounds for j in 1:n
        append!(A.I, I)
        for i in 1:m
            push!(A.J, J[j])
        end
    end
    A
end

fillzero!(A::SparseMatrixCOO) = (map(empty!, findnz(A)); A)
fillzero!(A::SparseMatrixCSC) = fill!(A, 0)
SparseArrays.findnz(A::SparseMatrixCOO) = A.I, A.J, A.V

Base.:\(A::SparseMatrixCOO, b::AbstractVector) = sparse(A) \ b

function Base.show(io::IO, mime::MIME"text/plain", A::SparseMatrixCOO)
    if !haskey(io, :compact) && length(axes(A, 2)) > 1
        io = IOContext(io, :compact => true)
    end
    S = sparse(A)
    m, n = size(S)
    k = nnz(S)
    println(io, m, "×", n, " ", typeof(A), " with ", k, " stored ", k == 1 ? "entry" : "entries", ":")
    isempty(A.I) || Base.print_array(io, S)
end
Base.show(io::IO, A::SparseMatrixCOO) = show(io, sparse(A))

###################
# SparseMatrixCSC #
###################

function add!(A::SparseMatrixCSC, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix)
    # `I` must be sorted
    @boundscheck checkbounds(A, I, J)
    @assert size(K) == map(length, (I, J))
    rows = rowvals(A)
    vals = nonzeros(A)
    perm = sortperm(I)
    @inbounds for j in 1:length(J)
        i = 1
        for k in nzrange(A, J[j])
            row = rows[k] # row candidate
            i′ = perm[i]
            if I[i′] == row
                vals[k] += K[i′,j]
                i += 1
            end
            i > length(I) && break
        end
        if i <= length(I)
            error("wrong sparsity pattern")
        end
    end
    A
end
