@testset "Shapes" begin
    @testset "$shape" for shape in (Line2(), Quad4(), Hex8(), Tri6())
        for T in (Float64, Float32)
            dim = @inferred Femto.get_dimension(shape)
            n_nodes = @inferred Femto.num_nodes(shape)
            n_quadpoints = @inferred Femto.num_quadpoints(shape)
            @test (@inferred Femto.get_local_node_coordinates(T, shape)) isa SVector{n_nodes, Vec{dim, T}}
            @test (@inferred values(shape, rand(Vec{dim, T}))) isa SVector{n_nodes, T}
            @test (@inferred Femto.quadpoints(T, shape)) isa NTuple{n_quadpoints, Vec{dim, T}}
            @test (@inferred Femto.quadweights(T, shape)) isa NTuple{n_quadpoints, T}
        end
    end
end
