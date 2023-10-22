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

# forward pass of LSTM cell, return hidden state as output
function forward!(lstm::LSTM, input::Vector{Float64})::Vector{Float64}
	input_concat = [input; lstm.hidden_state]

	f = forward!(lstm.forget_gate,   input_concat)
	i = forward!(lstm.input_gate,    input_concat)
	o = forward!(lstm.output_gate,   input_concat)
	n = forward!(lstm.new_candidate, input_concat)

	lstm.cell_state = f .* lstm.cell_state + i .* n
	lstm.hidden_state = o .* tanh(lstm.cell_state)
	return lstm.hidden_state
end
