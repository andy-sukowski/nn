# See LICENSE file for copyright and license details.

using Printf
using MLDatasets: MNIST
using ProgressMeter

include("../cnn.jl")

layers = [Conv(1 => 2, (28, 28), (5, 5)),
          Pool(2, 2),
          Conv(2 => 3, (12, 12), (5, 5)),
          Pool(2, 2),
          Flatten(3, (4, 4)),
          Dense(48 => 24),
          Dense(24 => 10)]

# load MNIST dataset from MLDatasets
train_x, train_y = MNIST(split=:train)[:]
inputs = [[Float64.(train_x[:, :, i])] for i in 1:size(train_x, 3)]
expected = one_hot.(train_y .+ 1, 10)
data = collect(zip(inputs, expected))::Data

# number of batches: 60000 / batch_size
batch_size = 10
batches = copy.(eachcol(reshape(data, batch_size, :)))

# average loss for each batch
Σlosses = Vector{Float64}(undef, length(batches))

p = Progress(length(batches); desc="Training:", dt=0.1, barlen=16)
for i in eachindex(batches)
	Σlosses[i] = train!(layers, batches[i], η=1.5)
	next!(p; showvalues = [(:batch, i),
		(:loss, @sprintf("%0.16f", Σlosses[i]))])
end
finish!(p)

# load MNIST test dataset
test_x, test_y = MNIST(split=:test)[:]
test_inputs = [[Float64.(test_x[:, :, i])] for i in 1:size(test_x, 3)]

matches = argmax.(forward!.((layers,), test_inputs)) .- 1 .== test_y
accuracy = sum(matches) / length(test_y)
println("Accuracy on test dataset: ", accuracy)
