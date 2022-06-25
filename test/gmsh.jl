@testset "Gmsh" begin
    data = readgmsh("square.msh")
    @test sort(collect(keys(data))) == sort(["left_right", "top_bottom", "main"])
    @test data["main"] isa Grid{Float64, 2, 2, Tri3, 3}
    @test data["left_right"] isa Grid{Float64, 2, 1, Line2, 2}
    @test data["top_bottom"] isa Grid{Float64, 2, 1, Line2, 2}
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
