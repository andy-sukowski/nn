# See LICENSE file for copyright and license details.

include("../utils.jl")

# convolutional layer: kernels, biases, gradient
mutable struct Conv <: Layer
	act       :: Function
	act′      :: Function

	input     :: Vector{<:Array{Float64}}
	kernels   :: Matrix{Array{Float64}}
	biases    :: Vector{Array{Float64}}
	z_maps    :: Vector{Array{Float64}}

	∇kernels  :: Matrix{Array{Float64}}
	∇biases   :: Vector{Array{Float64}}

	Σ∇kernels :: Matrix{Array{Float64}}
	Σ∇biases  :: Vector{Array{Float64}}
end

function Conv(dims :: Pair{Int, Int}, input_size :: Tuple{Vararg{Int}}, kernel_size :: Tuple{Vararg{Int}}, act = σ, act′ = σ′) :: Conv
	if length(input_size) != length(kernel_size)
		throw(DimensionMismatch("input_size and kernel_size must be the same length"))
	end

	output_size = input_size .- kernel_size .+ 1
	Conv(
		act,
		act′,
		[Array{Float64}(undef, input_size...) for _ in 1:dims[1]],
		[randn(kernel_size...) for _ in 1:dims[2], _ in 1:dims[1]],
		[zeros(output_size...) for _ in 1:dims[2]],
		Vector{Array{Float64}}(undef, dims[2]),
		Matrix{Array{Float64}}(undef, dims[2], dims[1]),
		Vector{Array{Float64}}(undef, dims[2]),
		[Array{Float64}(undef, kernel_size...) for _ in 1:dims[2], _ in 1:dims[1]],
		[Array{Float64}(undef, output_size...) for _ in 1:dims[2]]
	)
end

# return sum of product of kernel and region at certain index
function apply_kernel(a :: Array{Float64}, k :: Array{Float64}, i :: Tuple{Vararg{Int}})
	region = range.(i, i .+ size(k) .- 1)
	return sum(a[region...] .* k)
end

# cross-correlation: sliding dot product
function xcorr(a :: Array{Float64}, k :: Array{Float64})
	indices = cart(range.(1, size(a) .- size(k) .+ 1)...)
	return apply_kernel.((a,), (k,), indices)
end

# full convolution: input padding and 180° kernel rotation
function full_conv(a :: Array{Float64}, k :: Array{Float64})
	pad_a = zeros(size(a) .+ 2 .* size(k) .- 2)
	pad_a[range.(size(k), size(a) .+ size(k) .- 1)...] .= a
	return xcorr(pad_a, reverse(k))
end

# forward pass, return output
function forward!(l :: Conv, input :: Vector{<:Array{Float64}}) :: Vector{<:Array{Float64}}
	if size.(l.input) != size.(input)
		throw(DimensionMismatch("dimensions of l.input and input must match"))
	end

	l.input .= input
	l.z_maps .= l.biases + sum.(eachrow(xcorr.(permutedims(l.input), l.kernels)))
	return (z_map -> l.act.(z_map)).(l.z_maps)
end

# l.input is set by forward!()
function backprop!(l :: Conv, ∇output :: Vector{<:Array{Float64}}) :: Vector{<:Array{Float64}}
	if size.(l.z_maps) != size.(∇output)
		throw(DimensionMismatch("dimensions of l.z_maps and ∇output must match"))
	end

	∇input = zeros.(size.(l.input))
	∇z_maps = (z_map -> l.act′.(z_map)).(l.z_maps)
	for k in eachindex(∇output)
		l.∇biases[k] = ∇z_maps[k] .* ∇output[k]
		for j in eachindex(l.input)
			l.∇kernels[k, j] = xcorr(l.input[j], ∇z_maps[k] .* ∇output[k])
			∇input[j] += full_conv(∇z_maps[k] .* ∇output[k], l.kernels[k, j])
		end
	end
	return ∇input
end

# clear average gradient
function Σ∇clear!(l :: Conv)
	fill!.(l.Σ∇kernels, 0)
	fill!.(l.Σ∇biases, 0)
end

# update average gradient
function Σ∇update!(l :: Conv, data_len :: Int)
	l.Σ∇kernels += l.∇kernels / data_len
	l.Σ∇biases  += l.∇biases  / data_len
	return nothing
end

# apply average gradient
function Σ∇apply!(l :: Conv, η :: Float64)
	l.kernels -= η * l.Σ∇kernels
	l.biases  -= η * l.Σ∇biases
	return nothing
end
