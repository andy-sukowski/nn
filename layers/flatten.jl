# See LICENSE file for copyright and license details.

mutable struct Flatten <: Layer
	len :: Int
	input_size :: Tuple{Vararg{Int}}
end

Flatten(len, input_size...) = Flatten(len, input_size)

function forward!(l :: Flatten, input :: Vector{<:Array{Float64}}) :: Vector{Float64}
	a = vcat(reshape.(input, 1, size(input[1])...)...)
	return vec(a)
end

function backprop!(l :: Flatten, ∇output :: Vector{Float64}) :: Vector{Array{Float64}}
	a = reshape(∇output', l.len, l.input_size...)
	return [copy.(eachslice(a, dims=1))...]
end
