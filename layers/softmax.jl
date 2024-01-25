# See LICENSE file for copyright and license details.

mutable struct Softmax <: Layer
	output::Vector{Vector{Float64}} # BPTT
end

Softmax(;t=1) = Softmax(Vector{Vector{Float64}}(undef, t))

function forward!(l::Softmax, input::Vector{Float64}; t=1)::Vector{Float64}
	exps = exp.(input .- maximum(input)) # numerical stability
	l.output[t] = exps ./ sum(exps)
	return l.output[t]
end

# # 1st iteration
# function backprop!(l::Softmax, ∇output::Vector{Float64}; t=1::Int)::Vector{Float64}
# 	n = length(l.output[t])
# 	jacobian = zeros(n, n)
#
# 	for i in 1:n
# 		for j in 1:n
# 			jacobian[i, j] = l.output[t][i] * ((i == j) - l.output[t][j])
# 		end
# 	end
#
# 	return jacobian * ∇output
# end

# no side-effects, but named backprop!() for multiple dispatch
function backprop!(l::Softmax, ∇output::Vector{Float64}; t=1::Int)::Vector{Float64}
	n = length(l.output[t])

	diagonal = 1:n .== (1:n)'
	jacobian = l.output[t] .* (diagonal .- l.output[t]')

	return jacobian * ∇output
end
