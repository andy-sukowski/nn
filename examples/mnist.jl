# See LICENSE file for copyright and license details.

using Printf
using MLDatasets

include("../network.jl")

layers = [Conv(1 => 2, (28, 28), (5, 5)),
          Pool(2, 2),
          Conv(2 => 3, (12, 12), (5, 5)),
          Pool(2, 2),
          Flatten(3, (4, 4)),
          Dense(48 => 24),
          Dense(24 => 10)]

# number of batches: 60000 / batch_size
batch_size = 10

# loading MNIST dataset from MLDatasets
train_x, train_y = MNIST(split=:train)[:]
inputs = [[Float64.(train_x[:, :, i])] for i in 1:size(train_x, 3)]
expected = one_hot.(train_y .+ 1, 10)
data = collect(zip(inputs, expected)) :: Data
batches = copy.(eachcol(reshape(data, batch_size, :)))

Σloss = Vector{Float64}(undef, length(batches))
@time for i in eachindex(batches)
	Σloss[i] = train!(layers, batches[i], η=2.0)
	@printf "Σloss[%d] = %.12f\n" i Σloss[i]
end
