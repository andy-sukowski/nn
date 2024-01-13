# See LICENSE file for copyright and license details.

# long short-term memory cell
mutable struct LSTM <: Layer
	forget_gate      ::Dense
	input_gate       ::Dense
	output_gate      ::Dense
	new_candidate    ::Dense

	forget_gate_out  ::Vector{Vector{Float64}}
	input_gate_out   ::Vector{Vector{Float64}}
	output_gate_out  ::Vector{Vector{Float64}}
	new_candidate_out::Vector{Vector{Float64}}

	cell_state       ::Vector{Vector{Float64}}
	hidden_state     ::Vector{Vector{Float64}}
end

function LSTM(dims::Pair{Int, Int}; t::Int)::LSTM
	return LSTM(
		Dense(sum(dims) => dims[2]; act=σ, act′=σ′),
		Dense(sum(dims) => dims[2]; act=σ, act′=σ′),
		Dense(sum(dims) => dims[2]; act=σ, act′=σ′),
		Dense(sum(dims) => dims[2]; act=tanh, act′=tanh′),
		[Vector{Float64}(undef, dims[2]) for _ in 1:t],
		[Vector{Float64}(undef, dims[2]) for _ in 1:t],
		[Vector{Float64}(undef, dims[2]) for _ in 1:t],
		[Vector{Float64}(undef, dims[2]) for _ in 1:t],
		[zeros(dims[2]) for _ in 1:t+1],
		[zeros(dims[2]) for _ in 1:t+1]
	)
end

# forward pass of LSTM cell, return hidden state as output
function forward!(l::LSTM, input::Vector{Float64}; t::Int)::Vector{Float64}
	input_concat = [input; l.hidden_state[t]]

	l.forget_gate_out[t]   = forward!(l.forget_gate,   input_concat)
	l.input_gate_out[t]    = forward!(l.input_gate,    input_concat)
	l.output_gate_out[t]   = forward!(l.output_gate,   input_concat)
	l.new_candidate_out[t] = forward!(l.new_candidate, input_concat)

	l.cell_state[t+1]   = l.forget_gate_out[t] .* l.cell_state[t] + l.input_gate_out[t] .* l.new_candidate_out[t]
	l.hidden_state[t+1] = l.output_gate_out[t] .* tanh.(l.cell_state[t+1])
	return l.hidden_state[t+1]
end
