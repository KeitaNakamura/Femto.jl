using Femto

function WaveEquation(filename = joinpath(@__DIR__, "model.msh"))
    field = ScalarField()

    gridset = readgmsh(filename)
    grid = gridset["main"]
    source = gridset["source"]

    ndofs = num_dofs(field, grid)
    K = sparse(integrate((i,v,u)->∇(v)⋅∇(u), field, grid))
    M = Diagonal(integrate((i,v)->v, field, grid)) # lumped mass matrix
    F = zeros(ndofs)

    Uₙ = zeros(ndofs)
    Uₙ₋₁ = zeros(ndofs)
    Uₙ₊₁ = zeros(ndofs)

    dirichlet = falses(ndofs)
    dirichlet[get_nodedofs(field, gridset["boundary"])] .= true

    outdir = joinpath(dirname(filename), "Output")
    mkpath(outdir)
    pvd = openpvd(joinpath(outdir, "WaveEquation"))
    mesh = Grid(map(x -> [x; 0], get_allnodes(grid)), get_shape(grid), get_connectivities(grid))

    timespan = 0.0:0.01:5.0
    dt = step(timespan)

    for (step, t) in enumerate(timespan)
        if t < 0.2
            integrate!((i,v)->v, F, field, source)
        else
            F .= 0
        end
        linsolve!(Uₙ₊₁, M, M*(2Uₙ - Uₙ₋₁) - (K*Uₙ - F)*dt^2, dirichlet)
        @. Uₙ₋₁ = Uₙ
        @. Uₙ = Uₙ₊₁

        reshape(reinterpret(Float64, get_allnodes(mesh)), 3, num_allnodes(mesh))[3,:] .= 50*Uₙ

        if step % 5 == 0
            openvtk(joinpath(outdir, "WaveEquation_$step"), mesh) do vtk
                vtk["Displacement"] = Uₙ
                pvd[t] = vtk
            end
        end
    end
    closepvd(pvd)

    Uₙ
end
