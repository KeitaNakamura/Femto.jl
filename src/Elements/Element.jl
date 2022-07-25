abstract type Element{T, dim} end

# core constructors
Element{T}(shape::Shape{dim}, shape_qr::Shape{dim} = shape) where {T, dim} = SingleBodyElement{T}(shape, shape_qr)
Element{T, dim}(shape::Shape{dim}, shape_qr::Shape{dim} = shape) where {T, dim} = SingleBodyElement{T}(shape, shape_qr)
Element{T, dim}(shape::Shape{shape_dim}, shape_qr::Shape{shape_dim} = shape) where {T, dim, shape_dim} = SingleFaceElement{T, dim}(shape, shape_qr)

Element(::Type{T}, shape::Shape, shape_qr::Shape = shape) where {T} = Element{T}(shape, shape_qr)
Element(shape::Shape, shape_qr::Shape = shape) = Element(Float64, shape, shape_qr)

FaceElement(::Type{T}, shape::Shape{shape_dim}, shape_qr::Shape{shape_dim} = shape) where {T, shape_dim} = Element{T, shape_dim+1}(shape, shape_qr)
FaceElement(shape::Shape, shape_qr::Shape = shape) = FaceElement(Float64, shape, shape_qr)
