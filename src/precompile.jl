function _precompile_()
    for S in (Line2, Line3, Quad4, Quad9, Hex8, Hex27, Tri3, Tri6, Tet4, Tet10)
        @assert precompile(Tuple{typeof(Element), S})
    end
end
