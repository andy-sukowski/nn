# See LICENSE file for copyright and license details.

abstract type Layer end

# batch of data points
Data = Vector{<:Tuple}

include("utils.jl")
include("layers/conv.jl")
include("layers/dense.jl")
include("layers/flatten.jl")
include("layers/lstm.jl")
include("layers/pool.jl")

# needed for Conv and Dense layers
Σ∇clear!(_::Layer) = nothing
Σ∇update!(_::Layer, _::Int) = nothing
Σ∇apply!(_::Layer, _::Float64) = nothing

loss(x, y)  = sum((x - y) .^ 2)
loss′(x, y) = 2 .* (x - y)

function forward!(layers::Vector{<:Layer}, input::Vector; t=1::Int)::Vector
	return foldl((inp, l) -> forward!(l, inp; t=t), [input, layers...])
end

function backprop!(layers::Vector{<:Layer}, ∇output::Vector; t=1::Int)::Vector
	return foldr((l, ∇out) -> backprop!(l, ∇out; t=t), [layers..., ∇output])
end

function forward_seq!(layers::Vector{<:Layer}, seq::Vector{<:Vector})::Vector
	out_seq = Vector{Vector}(undef, length(seq))
	for t in 1:length(seq)
		out_seq[t] = forward!(layers, seq[t]; t=t)
	end
	return out_seq
end

function backprop_seq!(layers::Vector{<:Layer}, ∇out_seq::Vector{<:Vector})::Vector
	∇inp_seq = Vector{Vector}(undef, length(∇out_seq))
	for t in length(∇out_seq):-1:1
		∇inp_seq[t] = backprop!(layers, ∇out_seq[t]; t=t)
	end
	return ∇inp_seq
end

# data: [(input, expected)], only one batch!
function train!(layers::Vector{<:Layer}, data::Data; η=1.0::Float64)::Float64
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

# sequential data: [(input_seq, expected_seq)], only one batch!
function train_seq!(layers::Vector{<:Layer}, data::Data; η=1.0::Float64)::Float64
	Σ∇clear!.(layers)

	Σloss = 0
	for d in data
		output = forward_seq!(layers, d[1])
		Σloss += sum(loss.(output, d[2])) / length(data)
		backprop_seq!(layers, loss′.(output, d[2]))

		Σ∇update!.(layers, length(data))
	end

	# play around with learning rate η
	Σ∇apply!.(layers, η)

	return Σloss
end
