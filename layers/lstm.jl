# See LICENSE file for copyright and license details.

# long short-term memory cell
mutable struct LSTM <: Layer
	forget_gate  ::Dense
	input_gate   ::Dense
	output_gate  ::Dense
	new_candidate::Dense

	cell_state   ::Vector{Float64}
	hidden_state ::Vector{Float64}
end

function LSTM(dims::Pair{Int, Int})
	LSTM(
		Dense(sum(dims) => dims[2]; act=σ, act′=σ′),
		Dense(sum(dims) => dims[2]; act=σ, act′=σ′),
		Dense(sum(dims) => dims[2]; act=σ, act′=σ′),
		Dense(sum(dims) => dims[2]; act=tanh, act′=tanh′),
		Vector{Float64}(undef, dims[2]),
		Vector{Float64}(undef, dims[2])
	)
end
