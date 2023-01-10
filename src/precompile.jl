using SnoopPrecompile

@precompile_all_calls begin
    for shape in (Line2(), Line3(), Quad4(), Quad9(), Hex8(), Hex27(), Tri3(), Tri6(), Tet4(), Tet10())
        Element(shape)
        get_dimension(shape) != 3 && FaceElement(shape)
    end
end
