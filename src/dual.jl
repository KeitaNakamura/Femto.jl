###########
# RealVec #
###########

struct RealVec{T, dim} <: Real
    scalar::T
    vector::Vec{dim, T}
end

Base.promote_rule(::Type{<: RealVec{T}}, ::Type{U}) where {T <: Real, U <: Real} = promote_type(T, U)
Base.promote_rule(::Type{T}, ::Type{<: RealVec{U}}) where {T <: Real, U <: Real} = promote_type(T, U)
Base.promote_rule(::Type{<: RealVec{T}}, ::Type{<: RealVec{U}}) where {T <: Real, U <: Real} = promote_type(T, U)

Base.convert(::Type{T}, x::RealVec) where {T <: Real} = convert(T, x.scalar)

for op in (:+, :-, :*, :/)
    @eval Base.$op(a::RealVec, b::RealVec) = $op(a.scalar, b.scalar)
end

∇(x::RealVec) = x.vector

##########
# VecMat #
##########

struct VecMat{dim, T, dim²} <: AbstractVec{dim, T}
    vector::Vec{dim, T}
    matrix::Mat{dim, dim, T, dim²}
end
Base.Tuple(x::VecMat) = Tuple(x.vector)
@inline function Base.getindex(x::VecMat, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds x.vector[i]
end

∇(x::VecMat) = x.matrix

########
# dual #
########

dual(u::Real, dudx::Vec) = RealVec(u, dudx)
dual(u::Vec, dudx::Mat) = VecMat(u, dudx)
