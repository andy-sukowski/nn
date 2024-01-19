# See LICENSE file for copyright and license details.

# pooling layer: pool size
mutable struct Pool <: Layer
	pool_size::Tuple{Vararg{Int}}
end

Pool(pool_size...) = Pool(pool_size)

# mean pooling: combine neuron clusters, discard border
function pool(a::Matrix{Float64}, pool_size::Tuple{Vararg{Int}})::Matrix{Float64}
	indices = cart((:).(1, pool_size, size(a) .- pool_size .+ 1)...)
	sums = [sum(a[range.(i, i .+ pool_size .- 1)...]) for i in indices]
	return sums ./ prod(pool_size)
end

function forward(l::Pool, input::Vector{<:Array{Float64}})::Vector{<:Array{Float64}}
	return pool.(input, (l.pool_size,))
end

function backprop(l::Pool, ∇output::Vector{<:Array{Float64}})::Vector{<:Array{Float64}}
	∇input = fill(zeros(size(∇output[1]) .* l.pool_size), length(∇output))
	indices = cart(range.(1, size(∇input[1]))...)
	for i in eachindex(∇input), j in indices
			∇input[i][j...] = ∇output[i][ceil.(Int, j ./ l.pool_size)...] / prod(l.pool_size)
	end
	return ∇input
end

# for multiple dispatch, no side-effects, but named forward!(), backprop!() for consistency
forward!( l::Pool, input::Vector{<:Array{Float64}};   t=1::Int)::Vector{Array{Float64}} = forward( l, input)
backprop!(l::Pool, ∇output::Vector{<:Array{Float64}}; t=1::Int)::Vector{Array{Float64}} = backprop(l, ∇output)
