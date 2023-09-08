# See LICENSE file for copyright and license details.

include("layers/conv.jl")
include("layers/dense.jl")
include("layers/flatten.jl")
include("layers/pool.jl")

# needed for some layer types
Σ∇clear!(l :: Layer)  = nothing
Σ∇update!(l :: Layer, data_len :: Int) = nothing
Σ∇apply!(l :: Layer, η :: Float64)  = nothing

loss(x, y)  = sum((x - y) .^ 2)
loss′(x, y) = 2 .* (x - y)

# batch of data points
Data = Vector{<:Tuple}

function forward!(layers :: Vector{<:Layer}, input :: Vector) :: Vector
	return foldl((inp, l) -> forward!(l, inp), [input, layers...])
end

function backprop!(layers :: Vector{<:Layer}, ∇output :: Vector) :: Vector
	return foldr(backprop!, [layers..., ∇output])
end

# data: [(input, expected)], only one batch!
function train!(layers :: Vector{<:Layer}, data :: Data; η=1 :: Float64) :: Float64
	Σ∇clear!.(layers)

	Σloss = 0
	for d in data
		output = forward!(layers, d[1])
		Σloss += loss(output, d[2]) / length(data)
		backprop!(layers, loss′(output, d[2]))

		Σ∇update!.(layers, length(data))
	end

	# play around with learning rate η
	Σ∇apply!.(layers, η)

	return Σloss
end
