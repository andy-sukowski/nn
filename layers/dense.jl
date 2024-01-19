# See LICENSE file for copyright and license details.

# dense layer with BPTT support: input, gradient
mutable struct Dense <: Layer
	act      ::Function
	act′     ::Function

	input    ::Vector{Vector{Float64}} # BPTT
	weights  ::Matrix{Float64}
	biases   ::Vector{Float64}
	zs       ::Vector{Vector{Float64}} # BPTT

	∇weights ::Vector{Matrix{Float64}} # BPTT
	∇biases  ::Vector{Vector{Float64}} # BPTT

	Σ∇weights::Matrix{Float64}
	Σ∇biases ::Vector{Float64}
end

function Dense(dims::Pair{Int, Int}; act=σ, act′=σ′, t=1::Int)::Dense
	Dense(
		act,
		act′,
		[Vector{Float64}(undef, dims[1])          for _ in 1:t],
		randn(dims[2], dims[1]),
		zeros(dims[2]),
		[Vector{Float64}(undef, dims[2])          for _ in 1:t],
		[Matrix{Float64}(undef, dims[2], dims[1]) for _ in 1:t],
		[Vector{Float64}(undef, dims[2])          for _ in 1:t],
		Matrix{Float64}(undef, dims[2], dims[1]),
		Vector{Float64}(undef, dims[2])
	)
end

# forward pass, return output
function forward!(l::Dense, input::Vector{Float64}; t=1::Int)::Vector{Float64}
	if length(l.input[t]) != length(input)
		throw(DimensionMismatch("dimensions of l.input[t] and input must match"))
	end

	# l.input and l.zs used by backprop!()
	l.input[t] = input
	l.zs[t]    = l.weights * l.input[t] + l.biases
	return l.act.(l.zs[t])
end

# l.input and l.zs set by forward!()
function backprop!(l::Dense, ∇output::Vector{Float64}; t=1::Int)::Vector{Float64}
	if length(l.zs[t]) != length(∇output)
		throw(DimensionMismatch("dimensions of l.zs[t] and ∇output must match"))
	end

	l.∇biases[t]  = l.act′.(l.zs[t]) .* ∇output
	l.∇weights[t] = l.input[t]' .* l.∇biases[t]
	return l.weights' * l.∇biases[t]
end

# clear average gradient
function Σ∇clear!(l::Dense)
	l.Σ∇weights .= 0
	l.Σ∇biases  .= 0
	return nothing
end

# update average gradient
function Σ∇update!(l::Dense, data_len::Int)
	l.Σ∇weights += sum(l.∇weights) / data_len
	l.Σ∇biases  += sum(l.∇biases)  / data_len
	return nothing
end

# apply average gradient
function Σ∇apply!(l::Dense, η::Float64)
	l.weights -= η * l.Σ∇weights
	l.biases  -= η * l.Σ∇biases
	return nothing
end
