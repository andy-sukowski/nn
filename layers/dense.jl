# See LICENSE file for copyright and license details.

# dense layer: input, gradient
mutable struct Dense <: Layer
	act      ::Function
	act′     ::Function

	input    ::Vector{Float64}
	weights  ::Matrix{Float64}
	biases   ::Vector{Float64}
	zs       ::Vector{Float64}

	∇weights ::Matrix{Float64}
	∇biases  ::Vector{Float64}

	Σ∇weights::Matrix{Float64}
	Σ∇biases ::Vector{Float64}
end

function Dense(dims::Pair{Int, Int}; act = σ, act′ = σ′)::Dense
	Dense(
		act,
		act′,
		Vector{Float64}(undef, dims[1]),
		randn(dims[2], dims[1]),
		zeros(dims[2]),
		Vector{Float64}(undef, dims[2]),
		Matrix{Float64}(undef, dims[2], dims[1]),
		Vector{Float64}(undef, dims[2]),
		Matrix{Float64}(undef, dims[2], dims[1]),
		Vector{Float64}(undef, dims[2])
	)
end

# forward pass, return output
function forward!(l::Dense, input::Vector{Float64})::Vector{Float64}
	if length(l.input) != length(input)
		throw(DimensionMismatch("dimensions of l.input and input must match"))
	end

	l.input = input # used by backprop!()
	l.zs = l.weights * l.input + l.biases
	return l.act.(l.zs)
end

# l.input is set by forward!()
function backprop!(l::Dense, ∇output::Vector{Float64})::Vector{Float64}
	if length(l.zs) != length(∇output)
		throw(DimensionMismatch("dimensions of l.zs and ∇output must match"))
	end

	l.∇biases = l.act′.(l.zs) .* ∇output
	l.∇weights = l.input' .* l.∇biases
	return l.weights' * l.∇biases
end

# clear average gradient
function Σ∇clear!(l::Dense)
	l.Σ∇weights .= 0
	l.Σ∇biases  .= 0
	return nothing
end

# update average gradient
function Σ∇update!(l::Dense, data_len::Int)
	l.Σ∇weights += l.∇weights / data_len
	l.Σ∇biases  += l.∇biases  / data_len
	return nothing
end

# apply average gradient
function Σ∇apply!(l::Dense, η::Float64)
	l.weights -= η * l.Σ∇weights
	l.biases  -= η * l.Σ∇biases
	return nothing
end
