using WaterLily
import WaterLily: ∂, @loop
using Plots
using TimerOutputs
using Revise


to = TimerOutput()

includet("../../vortex_shedding.jl")


sim_shedding = circle_shedding(mem=Array)
t_end = 50.0

# sim_step!(sim_shedding)
# flow = sim_shedding.flow
# jemoeder = WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,WaterLily.quick;ν=flow.ν,perdir=flow.perdir)

# println(size(jemoeder))

function grad(field::AbstractArray)
    T = eltype(field)
    sz = size(field)
    N = ndims(field)
    grad = zeros(T, sz..., N)  # e.g., zeros(Float32, Nx, Ny, 2)
    for n in 1:N
        @loop grad[Tuple(I)..., n] = ∂(n, I, field) over I ∈ inside(field)
    end
    return grad
end

function RHS(flow::Flow{N};λ=WaterLily.quick,udf=nothing,kwargs...) where N
    RHS = inside(WaterLily.conv_diff!(flow.f,flow.u⁰,flow.σ,λ;ν=flow.ν,perdir=flow.perdir) - grad(flow.p))
    return RHS
end

test = RHS(sim_shedding.flow)
println(size(test), typeof(test))

# function run(time_max; sample_instance=25)
#     while sim_time(sim) < time_max
#         sim_step!(sim)
#         if sim_time(sim) < sample_instance


#     end
# end