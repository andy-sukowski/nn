# See LICENSE file for copyright and license details.

include("../utils.jl")

# pooling layer: pool size
mutable struct Pool <: Layer
	pool_size :: Tuple{Vararg{Int}}
end

Pool(pool_size...) = Pool(pool_size)

# mean pooling: combine neuron clusters, discard border
function pool(a :: Matrix{Float64}, pool_size :: Tuple{Vararg{Int}})
	indices = cart([1:step:stop for (step, stop) in zip(pool_size, size(a) .- pool_size .+ 1)]...)
	return [sum(a[range.(i, i .+ pool_size .- 1)...]) / prod(pool_size) for i in indices]
end

# no side-effects, but named forward!() for consistency
function forward!(l :: Pool, input :: Vector{<:Array{Float64}}) :: Vector{<:Array{Float64}}
	return pool.(input, (l.pool_size,))
end

# no side-effects, but named backprop!() for consistency, also kinda ugly
function backprop!(l :: Pool, ∇output :: Vector{<:Array{Float64}}) :: Vector{<:Array{Float64}}
	∇input = [zeros(size(∇output[1]) .* l.pool_size) for i in eachindex(∇output)]
	indices = cart(range.(1, size(∇input[1]))...)
	for i in eachindex(∇input)
		for j in indices
			∇input[i][j...] = ∇output[i][Int.(ceil.(j ./ l.pool_size))...] / prod(l.pool_size)
		end
	end
	return ∇input
end
