###########
# RealVec #
###########

struct RealVec{T, dim} <: Real
    scalar::T
    vector::Vec{dim, T}
end

Base.promote_rule(::Type{<: RealVec{T}}, ::Type{U}) where {T <: Real, U <: Real} = promote_type(T, U)
Base.convert(::Type{T}, x::RealVec) where {T <: Real} = convert(T, x.scalar)

for op in (:+, :-, :*, :/)
    @eval Base.$op(a::RealVec, b::RealVec) = $op(a.scalar, b.scalar)
end

∇(x::RealVec) = x.vector

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

∇(x::ValueGradientTensor) = x.grad

########
# dual #
########

dual(u::Real, dudx::Vec) = RealVec(u, dudx)
dual(u::Tensor, dudx::Tensor) = ValueGradientTensor(u, dudx)
