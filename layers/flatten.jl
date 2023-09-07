# See LICENSE file for copyright and license details.

mutable struct Flatten <: Layer
	input_size :: Tuple{Vararg{Int}}
end

# kinda hacky, more elegant solution welcome
function forward!(l :: Flatten, input :: Vector{<:Array{Float64}}) Vector{Float64}
	a = vcat([reshape(input[i], 1, size(input[1])...) for i in 1:length(input)]...)
	return vec(a)
end

function backprop!(l :: Flatten, ∇output :: Vector{Float64}) :: Vector{Array{Float64}}
	a = reshape(∇output', l.input_size...)
	return [copy.(eachslice(a, dims=1))...]
end
