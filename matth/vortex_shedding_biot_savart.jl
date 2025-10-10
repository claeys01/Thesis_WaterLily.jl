using WaterLily
using ..BiotSavartBCs

# includet("../BiotSavartBCs.jl")


# @inline function dynamicSpline(::Type{T}=Float32; new_cps_list,D=2^7,Re=302,U=1,ϵ=0.5,thk=2ϵ+√3,mem=Array, use_biotsavart=false) where {T<:AbstractFloat}
#     cps = new_cps_list[1] .* 3 .* D .+ SA{T}[D,3D]
#     degree = 2
#     n_ctrl = size(cps, 2)
#     weights = ones(T, n_ctrl)
#     knots = T.(clamped_uniform_knots(degree, n_ctrl))    curve = NurbsCurve(cps, knots, weights)    body = DynamicNurbsBody(curve; thk=thk, boundary=true)
#     ν = U*D/Re
#     return use_biotsavart ?
#     BiotSimulation((10D,8D),(0,0),D; U, ν, body, T, mem, ϵ) :
#     Simulation((10D,8D),(0,0),D; U, ν, body, T, mem,ϵ,
#     # exitBC=true  
#     )
# end
function circle_shedding_biot(Re=250, U=1; mem=Array)
    n = 3*2^7
    m = 2^7
    radius = Float32(m / 16) # radius of the circle relative to the height of the domain
    center = Float32(m / 2) # location of the circle relative to the height of the domain

    f = 2.5
    St = 0.2
    visc = St * Re / (f * (2*radius)^2)

    sdf(x,t) = √sum(abs2, x .- center) - radius
    sim = BiotSimulation(
        (4L,2L,2L), Ut, L; 
        body=AutoBody(sdf,map), ν=U*L/Re, T, 
        mem=mem)
    perturb!(sim; noise=0.1)
    return sim
end

sim = circle_shedding_biot(mem=Array)
t_end = 50

sim_step!(sim, t_end)