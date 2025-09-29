using WaterLily
using Plots
using TimerOutputs
using Revise


to = TimerOutput()

includet("../../vortex_shedding.jl")


sim_shedding = circle_shedding(mem=Array)
t_end = 50.0

sim_step!(sim_shedding)
flow = sim_shedding.flow
jemoeder = WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,WaterLily.quick;ν=flow.ν,perdir=flow.perdir)

println(size(jemoeder))

# function RHS(flow::Flow{N};λ=WaterLily.quick,udf=nothing,kwargs...)
#     # Suppose p is a scalar field (pressure field from a simulation)
#     grad_p = zeros(size(flow.p,1), size(flow.p,2), 2)  # 2D gradient

#     @inside grad_p[I,1] = ∂(1, I, flow.p)   # ∂p/∂x
#     @inside grad_p[I,2] = ∂(2, I, flow.p)   # ∂p/∂y

#     RHS = conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir) - grad_p

# end

# function run(time_max; sample_instance=25)
#     while sim_time(sim) < time_max
#         sim_step!(sim)
#         if sim_time(sim) < sample_instance


#     end
# end