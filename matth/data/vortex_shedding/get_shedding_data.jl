using WaterLily
import WaterLily: ∂, @loop
using Revise

includet("../../vortex_shedding.jl")
includet("../../custom.jl")


sim_shedding = circle_shedding(mem=Array)
t_end = 100.0

function data_run(sim::AbstractSimulation, time_max; sample_instance=50, verbose=false)
    RHS_arr = zeros(T, sz..., N)
    while sim_time(sim) < time_max
        sim_step!(sim)
        verbose && sim_info(sim)
        if sim_time(sim) > sample_instance
            print("Sampling RHS - ")
            push!(RHS_arr, RHS(sim.flow))
        end
    end
    return RHS_arr
end

RHS_arr = data_run(sim_shedding, t_end; verbose=true)
println(size(RHS_arr[2]))
println(size(RHS_arr), typeof(RHS_arr))