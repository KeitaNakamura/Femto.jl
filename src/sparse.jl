###################
# SparseMatrixIJV #
###################

struct SparseMatrixIJV{T} <: AbstractMatrix{T}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{T}
    m::Int
    n::Int
end

function SparseMatrixIJV{T}(m::Int, n::Int; sizehint::Int = 0) where T
    I = Int[]
    J = Int[]
    V = T[]
    sizehint!(I, sizehint)
    sizehint!(J, sizehint)
    sizehint!(V, sizehint)
    SparseMatrixIJV(I, J, V, m, n)
end
SparseMatrixIJV(m::Int, n::Int; sizehint::Int = 0) = SparseMatrixIJV{Float64}(m, n; sizehint)

Base.size(A::SparseMatrixIJV) = (A.m, A.n)

SparseArrays.sparse(A::SparseMatrixIJV) = sparse(A, size(A)...)
SparseArrays.sparse(A::SparseMatrixIJV, m::Int, n::Int, args...) = sparse(A.I, A.J, A.V, m, n, args...)

function add!(A::SparseMatrixIJV, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix)
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
function add!(A::SparseMatrixIJV, I::AbstractVector{Int}, K::AbstractMatrix)
    add!(A, I, I, K)
end

fillzero!(A::SparseMatrixIJV) = (map(empty!, findnz(A)); A)
fillzero!(A::SparseMatrixCSC) = dropzeros!(fill!(A, 0))
SparseArrays.findnz(A::SparseMatrixIJV) = A.I, A.J, A.V

Base.:\(A::SparseMatrixIJV, b::AbstractVector) = sparse(A) \ b

function Base.show(io::IO, mime::MIME"text/plain", A::SparseMatrixIJV)
    if !haskey(io, :compact) && length(axes(A, 2)) > 1
        io = IOContext(io, :compact => true)
    end
    m, n = size(A)
    k = length(A.I)
    println(io, m, "×", m, " ", typeof(A), " with ", k, " stored ", k == 1 ? "entry" : "entries", ":")
    isempty(A.I) || Base.print_array(io, sparse(A))
end
Base.show(io::IO, A::SparseMatrixIJV) = show(io, sparse(A))
