###########
# RealVec #
###########

struct RealVec{T, dim} <: Real
    scalar::T
    vector::Vec{dim, T}
end

Base.promote_rule(::Type{<: RealVec{T}}, ::Type{U}) where {T <: Real, U <: Real} = promote_type(T, U)
Base.convert(::Type{T}, x::RealVec) where {T <: Real} = convert(T, x.scalar)
Base.zero(::Type{RealVec{T, dim}}) where {T, dim} = RealVec(zero(T), zero(Vec{dim, T}))

for op in (:+, :-, :*, :/)
    @eval Base.$op(a::RealVec, b::RealVec) = $op(a.scalar, b.scalar)
end

∇(x::RealVec) = x.vector

function Base.show(io::IO, x::RealVec)
    print(io, x.scalar, " (∇x = ", x.vector, ")")
end

#######################
# ValueGradientTensor #
#######################

struct ValueGradientTensor{S, T, N, L, G} <: AbstractTensor{S, T, N}
    val::Tensor{S, T, N, L}
    grad::G
end
Base.Tuple(x::ValueGradientTensor) = Tuple(x.val)
@inline function Base.getindex(x::ValueGradientTensor, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds x.val[i]
end
Base.zero(::Type{ValueGradientTensor{S, T, N, L, G}}) where {S, T, N, L, G} = ValueGradientTensor(zero(Tensor{S, T, N, L}), zero(G))

∇(x::ValueGradientTensor) = x.grad

function Base.show(io::IO, ::MIME"text/plain", x::ValueGradientTensor)
    print(io, x.val, " (∇x = ", x.grad, ")")
end

########
# dual #
########

dual(u::Real, dudx::Vec) = RealVec(u, dudx)
dual(u::Tensor, dudx::Tensor) = ValueGradientTensor(u, dudx)
