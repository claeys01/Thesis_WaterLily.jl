using WaterLily
using CUDA
using Plots

# include("vortex_shedding.jl")

@assert CUDA.functional()
# simulation of a flow around a cilinder that exhibits vortex shedding

start = time()

# Define simulation size, geometry dimensions, viscosity
function circle(n, m, Re=250, U=1; mem=Array)
    radius = Float32(m / 16) # radius of the circle relative to the height of the domain
    center = Float32(m / 2) # location of the circle relative to the height of the domain

    f = 2.5
    St = 0.2
    visc = St * Re / (f * (2*radius)^2)

    sdf(x,t) = √sum(abs2, x .- center) - radius
    Simulation(
        (n, m),          # domain size
        (U, 0),          # flow velocity
        2f0radius; ν=visc, # defining viscosity
        body=AutoBody(sdf),
        mem
    )
end

x=3*2^7
y=2^7

sim = circle(x, y; mem=CuArray)
t_end = 10

perturb!(sim; noise=0.1)

sim_gif!(sim,duration=t_end,clims=(-5,5),xlims=(0,x),ylims=(0,y),plotbody=true)
# sim_step!(sim)

# u = sim.flow.u[:,:,1] # x velocity
# ω = zeros(size(u));

# @inside ω[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U

# flood(ω, clims=(-5,5), border=:none)