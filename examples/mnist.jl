# See LICENSE file for copyright and license details.

# This example definitely needs some tweaking, as it currently
# performs very badly (it stagnates at a loss of about 1)...

using Printf
using MLDatasets

include("../network.jl")

layers = [Conv(1 => 4, (28, 28), (5, 5)),
          Pool((2, 2)),
          Conv(4 => 2, (12, 12), (3, 3)),
          Pool((2, 2)),
          Flatten((2, 5, 5)),
          Dense(50 => 25),
          Dense(25 => 10)]

batch_count = 6000
batch_size = 10

# Loading MNIST dataset from MLDatasets
train_x, train_y = MNIST(split=:train)[:]
inputs = [[Float64.(train_x[:, :, i])] for i in 1:size(train_x, 3)]
expected = one_hot.(train_y .+ 1, 10)
data = collect(zip(inputs, expected)) :: Data
batches = [reshape(data, (batch_size, batch_count))[:, i] for i in 1:batch_count]

for i in eachindex(batches)
	@printf "Σcost[%d] = %.12f\n" i train!(layers, batches[i], η=100.0)
end
