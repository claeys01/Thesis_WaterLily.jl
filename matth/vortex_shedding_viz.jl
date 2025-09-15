using Plots, WaterLily

import WaterLily: flood,addbody,body_plot!,plot_logger


function vortex_gif(sim;duration=1,step=0.1,verbose=true,R=inside(sim.flow.p),
                    remeasure=false,plotbody=false,kv...)
    t₀ = round(WaterLily.sim_time(sim))
    @time @gif for tᵢ in range(t₀,t₀+duration;step)
        WaterLily.sim_step!(sim,tᵢ;remeasure)
        @WaterLily.inside sim.flow.σ[I] = WaterLily.curl(3,I,sim.flow.u)*sim.L/sim.U
        flood(sim.flow.σ[R]; kv...)
        plotbody && body_plot!(sim)
        verbose && println("tU/L=",round(tᵢ,digits=4),
                           ", Δt=",round(sim.flow.Δt[end],digits=3))
    end
end