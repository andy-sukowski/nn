# See LICENSE file for copyright and license details.

mutable struct Flatten <: Layer
	len :: Int
	input_size :: Tuple{Vararg{Int}}
end

Flatten(len, input_size...) = Flatten(len, input_size)

# no side-effects, but named forward!() for consistency
function forward!(l :: Flatten, input :: Vector{<:Array{Float64}}) :: Vector{Float64}
	if l.len != length(input) || l.input_size != size(input[1])
		throw(DimensionMismatch("dimensions of the layer and input must match"))
	end

	a = vcat(reshape.(input, 1, size(input[1])...)...)
	return vec(a)
end

# no side-effects, but named backprop!() for consistency
function backprop!(l :: Flatten, ∇output :: Vector{Float64}) :: Vector{Array{Float64}}
	if l.len * prod(l.input_size) != length(∇output)
		throw(DimensionMismatch("dimensions of the layer and ∇output must match"))
	end

	a = reshape(∇output', l.len, l.input_size...)
	return [copy.(eachslice(a, dims=1))...]
end
