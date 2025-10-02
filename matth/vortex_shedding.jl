using WaterLily
using CUDA

# Define simulation size, geometry dimensions, viscosity
function circle_shedding(Re=250, U=1; mem=Array)
    n = 3*2^7
    m = 2^7
    radius = Float32(m / 16) # radius of the circle relative to the height of the domain
    center = Float32(m / 2) # location of the circle relative to the height of the domain

    f = 2.5
    St = 0.2
    visc = St * Re / (f * (2*radius)^2)

    sdf(x,t) = √sum(abs2, x .- center) - radius
    sim = Simulation(
            (n, m),          # domain size
            (U, 0),          # flow velocity
            2f0radius; ν=visc, # defining viscosity
            body=AutoBody(sdf),
            mem
        )
    perturb!(sim; noise=0.1)
    return sim
end
# sim = circle_shedding(mem=CUDA.CuArray)

# t_end = 50
# sim_gif!(sim,duration=t_end,clims=(-5,5),plotbody=true)
# sim_step!(sim)

# u = sim.flow.u[:,:,1] # x velocity
# ω = zeros(size(u));

# @inside ω[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U

# flood(ω, clims=(-5,5), border=:none)