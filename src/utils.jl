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
