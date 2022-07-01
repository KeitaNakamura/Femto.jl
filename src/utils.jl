## dofindices
# ScalarField
dofindices(::ScalarField, ::Val, nodeindex::Int) = nodeindex
dofindices(::ScalarField, ::Val, conn::Index) = conn
# VectorField
dofindices(::VectorField, ::Val{dim}, nodeindex::Int) where {dim} = Index(ntuple(d -> dim*(nodeindex-1) + d, Val(dim)))
function dofindices(fieldtype::VectorField, ::Val{dim}, conn::Index{L}) where {dim, L}
    Index{dim*L}(Iterators.flatten(map(i -> dofindices(fieldtype, Val(dim), i), conn))...)
end

# zero_recursive
zero_recursive(::Type{Array{T, N}}) where {T, N} = Array{T, N}(undef, nfill(0, Val(N)))
@generated function zero_recursive(::Type{T}) where {T}
    if Base._return_type(zero, Tuple{T}) == Union{}
        exps = [:(zero_recursive($t)) for t in fieldtypes(T)]
        :(@_inline_meta; T($(exps...)))
    else
        :(@_inline_meta; zero(T))
    end
end
@generated function zero_recursive(::Type{T}) where {T <: Union{Tuple, NamedTuple}}
    exps = [:(zero_recursive($t)) for t in fieldtypes(T)]
    :(@_inline_meta; T(($(exps...),)))
end
zero_recursive(x) = zero_recursive(typeof(x))

# fillzero!
function fillzero!(x::AbstractArray)
    @simd for i in eachindex(x)
        @inbounds x[i] = zero_recursive(eltype(x))
    end
    x
end

# add!
function add!(A::AbstractMatrix, I::AbstractVector{Int}, J::AbstractVector{Int}, K::AbstractMatrix)
    @assert size(K) == (length(I), length(J))
    @boundscheck checkbounds(A, I, J)
    for j in eachindex(J)
        @simd for i in eachindex(I)
            @inbounds A[I[i], J[j]] += K[i,j]
        end
    end
    A
end
add!(A::AbstractMatrix, I::AbstractVector{Int}, K::AbstractMatrix) = add!(A, I, I, K)

function add!(A::AbstractVector, I::AbstractVector{Int}, F::AbstractVector)
    @assert length(F) == length(I)
    @boundscheck checkbounds(A, I)
    @simd for i in eachindex(I)
        @inbounds A[I[i]] += F[i]
    end
    A
end

# map_tuple
promote_tuple_length() = -1
promote_tuple_length(xs::Type{<: NTuple{N, Any}}...) where {N} = N
# apply map calculations only for tuples
# if one of the argunents is not tuple, treat it as scalar
@generated function map_tuple(f, xs::Vararg{Any, N}) where {N}
    L = promote_tuple_length([x for x in xs if x <: Tuple]...)
    if L == -1 # no tuples
        quote
            @_inline_meta
            @_propagate_inbounds_meta
            f(xs...)
        end
    else
        exps = map(1:L) do i
            args = [xs[j] <: Tuple ? :(xs[$j][$i]) : :(xs[$j]) for j in 1:N]
            :(f($(args...)))
        end
        quote
            @_inline_meta
            @_propagate_inbounds_meta
            tuple($(exps...))
        end
    end
end
