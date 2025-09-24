using WaterLily
using TimerOutputs
using Plots

to = TimerOutput()
# Define simulation size, geometry dimensions, viscosity
function basic(n, m, U=1; mem=Array)
    Simulation(
            (n, m),          # domain size
            (U, 0),          # flow velocity
            1
        )
end

# sim = basic(3*2^7, 2^7)

# sim_step!(sim; verbose=true, timer=to)

# u = sim.flow.u[:,:,1] # x velocity
# ω = zeros(size(u));

# @inside ω[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U

# println(sum(abs, ω))
# println(sim.U)

# plt = flood(u, clims=(-5,5), border=:none)
# display(plt)