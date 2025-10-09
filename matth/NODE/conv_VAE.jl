using Flux
using Flux: glorot_uniform
using JLD2

@load "matth/data/RHS_shedding_data_arr.jld2" RHS_data


# Select indices corresponding to time 50 to 75
time_indices = findall(t -> t ≥ 50 && t ≤ 75, RHS_data["time"])
selected_indices = time_indices

# Downsample to ~50 entries
n_samples = 50
downsampled_indices = round.(Int, range(1, length(selected_indices), length=n_samples))
final_indices = selected_indices[downsampled_indices]

# Downsample all relevant entries in RHS_data
RHS_data["time"] = RHS_data["time"][final_indices]
RHS_data["Δt"] = RHS_data["Δt"][final_indices]
RHS_data["RHS"] = RHS_data["RHS"][final_indices]

println("Downsampled to ", length(RHS_data["time"]), " time steps.")
println("Input data size: ", size(RHS_data["RHS"][1]))

# Convolutional encoder

filter_size = (5, 5)

conv_layers = Chain(
    Conv((3, 3), 2 => 16, relu; pad=(1, 1)),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, relu; pad=(1, 1)),
    MaxPool((2, 2)),
    Conv((3, 3), 32 => 64, relu; pad=(1, 1)),
    MaxPool((2, 2)),
    x -> reshape(x, :, size(x, 4)),  # Flatten
)

# Run a single snapshot through the chain
snapshot = cat(RHS_data["RHS"][1:10]...; dims=4)  # shape: (386, 130, 2, 10)
println("Input snapshot size: ", size(snapshot))
output = conv_layers(snapshot)
println("Output size: ", size(output))

CR = (386 * 130 * 2) / size(output, 1) 
println("Compression ratio: ", CR)

