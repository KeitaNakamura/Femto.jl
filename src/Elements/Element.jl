abstract type Element{T, dim} end

## core constructors
# single
Element{T}(shape::Shape{dim}, shape_qr::Shape{dim} = shape) where {T, dim} = SingleBodyElement{T}(shape, shape_qr)
Element{T, dim}(shape::Shape{dim}, shape_qr::Shape{dim} = shape) where {T, dim} = SingleBodyElement{T}(shape, shape_qr)
Element{T, dim}(shape::Shape{shape_dim}, shape_qr::Shape{shape_dim} = shape) where {T, dim, shape_dim} = SingleFaceElement{T, dim}(shape, shape_qr)
# mixed
Element{T}(shapes::Tuple{Vararg{Shape{dim}}}, shape_qr::Shape{dim} = default_quadrature_shape(shapes)) where {T, dim} = MixedBodyElement{T}(shapes, shape_qr)
Element{T, dim}(shapes::Tuple{Vararg{Shape{dim}}}, shape_qr::Shape{dim} = default_quadrature_shape(shapes)) where {T, dim} = MixedBodyElement{T}(shapes, shape_qr)
Element{T, dim}(shapes::Tuple{Vararg{Shape{shape_dim}}}, shape_qr::Shape{shape_dim} = default_quadrature_shape(shapes)) where {T, dim, shape_dim} = MixedFaceElement{T, dim}(shapes, shape_qr)

# single
Element(::Type{T}, shape::Shape, shape_qr::Shape = shape) where {T} = Element{T}(shape, shape_qr)
Element(shape::Shape, shape_qr::Shape = shape) = Element(Float64, shape, shape_qr)
# mixed
Element(::Type{T}, shapes::Tuple{Vararg{Shape}}, shape_qr::Shape = default_quadrature_shape(shapes)) where {T} = Element{T}(shapes, shape_qr)
Element(shapes::Tuple{Vararg{Shape}}, shape_qr::Shape = default_quadrature_shape(shapes)) = Element(Float64, shapes, shape_qr)

# single
FaceElement(::Type{T}, shape::Shape{shape_dim}, shape_qr::Shape{shape_dim} = shape) where {T, shape_dim} = Element{T, shape_dim+1}(shape, shape_qr)
FaceElement(shape::Shape, shape_qr::Shape = shape) = FaceElement(Float64, shape, shape_qr)
# mixed
FaceElement(::Type{T}, shapes::Tuple{Vararg{Shape{shape_dim}}}, shape_qr::Shape{shape_dim} = default_quadrature_shape(shapes)) where {T, shape_dim} = Element{T, shape_dim+1}(shapes, shape_qr)
FaceElement(shapes::Tuple{Vararg{Shape}}, shape_qr::Shape = default_quadrature_shape(shapes)) = FaceElement(Float64, shapes, shape_qr)
