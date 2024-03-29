# See LICENSE file for copyright and license details.

import DSP # hopefully just temporary

# convolutional layer with BPTT support: kernels, biases, gradient
mutable struct Conv{N} <: Layer
	act        ::Function
	act′       ::Function

	input      ::Vector{Vector{Array{Float64, N}}} # BPTT
	kernels    ::Matrix{Array{Float64, N}}
	biases     ::Vector{Array{Float64, N}}
	z_maps     ::Vector{Vector{Array{Float64, N}}} # BPTT

	∇kernels   ::Vector{Matrix{Array{Float64, N}}} # BPTT
	∇biases    ::Vector{Vector{Array{Float64, N}}} # BPTT

	avg∇kernels::Matrix{Array{Float64, N}}
	avg∇biases ::Vector{Array{Float64, N}}
end

function Conv(dims::Pair{Int, Int}, input_size::NTuple{N, Int}, kernel_size::NTuple{N, Int}; act=σ, act′=σ′, t=1::Int)::Conv where {N}
	output_size = input_size .- kernel_size .+ 1
	return Conv{N}(
		act,
		act′,
		[[Array{Float64}(undef, input_size...) for _ in 1:dims[1]] for _ in 1:t],
		[randn(kernel_size...) for _ in 1:dims[2], _ in 1:dims[1]],
		[zeros(output_size...) for _ in 1:dims[2]],
		[Vector{Array{Float64}}(undef, dims[2])                    for _ in 1:t],
		[Matrix{Array{Float64}}(undef, dims[2], dims[1])           for _ in 1:t],
		[Vector{Array{Float64}}(undef, dims[2])                    for _ in 1:t],
		[Array{Float64}(undef, kernel_size...) for _ in 1:dims[2], _ in 1:dims[1]],
		[Array{Float64}(undef, output_size...) for _ in 1:dims[2]]
	)
end

# # Because my custom xcorr() and full_conv() functions are currently
# # comparatively slow, I will use DSP.conv(), until I've optimized them.
#
# # return sum of product of kernel and region at certain index
# function apply_kernel(a::Array{Float64}, k::Array{Float64}, i::Tuple{Vararg{Int}})
# 	region = range.(i, i .+ size(k) .- 1)
# 	return sum(a[region...] .* k)
# end
#
# # cross-correlation: sliding dot product
# function xcorr(a::Array{Float64}, k::Array{Float64})
# 	indices = cart(range.(1, size(a) .- size(k) .+ 1)...)
# 	return apply_kernel.((a,), (k,), indices)
# end
#
# # full convolution: input padding and 180° kernel rotation
# function full_conv(a::Array{Float64}, k::Array{Float64})
# 	pad_a = zeros(size(a) .+ 2 .* size(k) .- 2)
# 	pad_a[range.(size(k), size(a) .+ size(k) .- 1)...] .= a
# 	return xcorr(pad_a, reverse(k))
# end

# full convolution: input padding and 180° kernel rotation
full_conv = DSP.conv

# n-dimensional cross-correlation using DSP.conv
function xcorr(a::Array{Float64}, k::Array{Float64})
	region = range.(size(k), size(a))
	return DSP.conv(a, reverse(k))[region...]
end

# forward pass, return output
function forward!(l::Conv, input::Vector{<:Array{Float64}}; t=1::Int)::Vector{<:Array{Float64}}
	if size.(l.input[t]) != size.(input)
		throw(DimensionMismatch("dimensions of l.input[t] and input must match"))
	end

	# l.input and l.z_maps used by backprop!()
	l.input[t]  .= input
	l.z_maps[t] .= l.biases + sum.(eachrow(xcorr.(permutedims(l.input[t]), l.kernels)))
	return (z_map -> l.act.(z_map)).(l.z_maps[t])
end

# l.input and l.z_maps set by forward!()
function backprop!(l::Conv, ∇output::Vector{<:Array{Float64}}; t=1::Int)::Vector{<:Array{Float64}}
	if size.(l.z_maps[t]) != size.(∇output)
		throw(DimensionMismatch("dimensions of l.z_maps[t] and ∇output must match"))
	end

	∇input  = zeros.(size.(l.input[t]))
	∇z_maps = (z_map -> l.act′.(z_map)).(l.z_maps[t])
	for k in eachindex(∇output)
		l.∇biases[t][k] = ∇z_maps[k] .* ∇output[k]
		for j in eachindex(l.input[t])
			l.∇kernels[t][k, j] = xcorr(l.input[t][j], ∇z_maps[k] .* ∇output[k])
			∇input[j] += full_conv(∇z_maps[k] .* ∇output[k], l.kernels[k, j])
		end
	end
	return ∇input
end

# clear average gradient
function avg∇clear!(l::Conv)
	fill!.(l.avg∇kernels, 0)
	fill!.(l.avg∇biases, 0)
	return nothing
end

# update average gradient
function avg∇update!(l::Conv, data_len::Int)
	l.avg∇kernels += sum(l.∇kernels) / data_len
	l.avg∇biases  += sum(l.∇biases)  / data_len
	return nothing
end

# apply average gradient
function avg∇apply!(l::Conv, η::Float64)
	l.kernels -= η * l.avg∇kernels
	l.biases  -= η * l.avg∇biases
	return nothing
end
