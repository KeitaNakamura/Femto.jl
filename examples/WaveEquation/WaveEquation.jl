using Femto

function WaveEquation(filename = joinpath(@__DIR__, "model.msh"))
    gridset = readgmsh(filename)

    grid = gridset["main"]
    source = gridset["source"]
    fieldtype = ScalarField()

    ndofs = num_dofs(fieldtype, grid)
    K = SparseMatrixCOO(ndofs, ndofs)
    M = zeros(ndofs)
    F = zeros(ndofs)

    K̃(index,v,δv) = ∇(δv) ⋅ ∇(v)
    F̃(index,δv) = δv

    Uₙ = zeros(ndofs)
    Uₙ₋₁ = zeros(ndofs)
    Uₙ₊₁ = zeros(ndofs)

    dirichlet = falses(ndofs)
    dirichlet[eachnode(fieldtype, gridset["boundary"])] .= true

    outdir = joinpath(@__DIR__, "Output")
    mkpath(outdir)
    pvd = openpvd(joinpath(outdir, "WaveEquation"))
    mesh = Grid{Float64, 3}(grid)

    timespan = 0.0:0.01:5.0
    dt = step(timespan)

    for (step, t) in enumerate(timespan)
        integrate!((K̃, F̃), (K, M), fieldtype, grid)
        if t < 0.2
            integrate!(F̃, F, fieldtype, source)
        else
            F .= 0
        end
        solve!(Uₙ₊₁, spdiagm(M), M.*(2Uₙ - Uₙ₋₁) - (sparse(K)*Uₙ - F)*dt^2, dirichlet)
        Uₙ₋₁ .= Uₙ
        Uₙ .= Uₙ₊₁

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
