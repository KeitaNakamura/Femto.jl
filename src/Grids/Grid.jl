abstract type Grid{T, dim} end

function Grid(nodes::Vector{<: Vec}, shape::Shape, connectivities::Vector{<: Index}, nodeindices::Vector{Int} = collect(1:length(nodes)))
    SingleGrid(nodes, shape, connectivities, nodeindices)
end
