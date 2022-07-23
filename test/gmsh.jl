import GmshReader

@testset "Gmsh" begin
    @testset "Shape $shape" for (shape, prop) in (
            Line2() => GmshReader.element_properties("Line", 1),
            Line3() => GmshReader.element_properties("Line", 2),
            Quad4() => GmshReader.element_properties("Quadrangle", 1),
            Quad9() => GmshReader.element_properties("Quadrangle", 2),
            Hex8()  => GmshReader.element_properties("Hexahedron", 1),
            Hex27() => GmshReader.element_properties("Hexahedron", 2),
            Tri3()  => GmshReader.element_properties("Triangle", 1),
            Tri6()  => GmshReader.element_properties("Triangle", 2),
            Tet4()  => GmshReader.element_properties("Tetrahedron", 1),
            Tet10() => GmshReader.element_properties("Tetrahedron", 2),
        )
        @test shape == Femto.from_gmsh_shape(prop.elementname)
        @test Femto.get_dimension(shape) == prop.dim
        @test Femto.num_nodes(shape) == prop.numnodes
        @test Femto.get_local_coordinates(shape) == prop.localnodecoord[Femto.from_gmsh_connectivity(shape)]
        @test Femto.get_order(shape) == prop.order
    end
    @testset "readgmsh" begin
        data = readgmsh("square.msh")
        @test sort(collect(keys(data))) == sort(["left_right", "top_bottom", "main"])
        @test data["main"] isa Grid{Float64, 2}
        @test data["left_right"] isa Grid{Float64, 2}
        @test data["top_bottom"] isa Grid{Float64, 2}
        @test Femto.get_shape(Femto.get_element(data["main"])) === Tri3()
        @test Femto.get_shape(Femto.get_element(data["left_right"])) === Line2()
        @test Femto.get_shape(Femto.get_element(data["top_bottom"])) === Line2()
        ## nodes
        @test data["main"].nodes === data["left_right"].nodes === data["top_bottom"].nodes
        ## nodal indices
        # left_right
        grid = data["left_right"]
        nodes = grid.nodes[grid.nodeindices]
        @test all(x -> x[1] ≈ 0 || x[1] ≈ 1, nodes)
        # top_bottom
        grid = data["top_bottom"]
        nodes = grid.nodes[grid.nodeindices]
        @test all(x -> x[2] ≈ 0 || x[2] ≈ 1, nodes)
        # main
        grid = data["main"]
        @test allunique(grid.nodeindices)
        @test sort(grid.nodeindices) == 1:length(grid.nodes)
    end
end
