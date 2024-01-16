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

	∇cell_state      ::Vector{Vector{Float64}}
	∇hidden_state    ::Vector{Vector{Float64}}
end

function LSTM(dims::Pair{Int, Int}; t::Int)::LSTM
	return LSTM(
		Dense(sum(dims) => dims[2]; act=σ,    act′=σ′,    t=t),
		Dense(sum(dims) => dims[2]; act=σ,    act′=σ′,    t=t),
		Dense(sum(dims) => dims[2]; act=σ,    act′=σ′,    t=t),
		Dense(sum(dims) => dims[2]; act=tanh, act′=tanh′, t=t),
		[Vector{Float64}(undef, dims[2]) for _ in 1:t],
		[Vector{Float64}(undef, dims[2]) for _ in 1:t],
		[Vector{Float64}(undef, dims[2]) for _ in 1:t],
		[Vector{Float64}(undef, dims[2]) for _ in 1:t],
		[zeros(dims[2]) for _ in 1:t+1],
		[zeros(dims[2]) for _ in 1:t+1],
		[zeros(dims[2]) for _ in 1:t+1],
		[zeros(dims[2]) for _ in 1:t+1]
	)
end

# forward pass of LSTM cell, return hidden state as output
function forward!(l::LSTM, input::Vector{Float64}; t::Int)::Vector{Float64}
	if t == 1 # reset derivatives
		l.∇cell_state[end]   .= 0
		l.∇hidden_state[end] .= 0
	end

	input_concat = [l.hidden_state[t]; input]

	l.forget_gate_out[t]   = forward!(l.forget_gate,   input_concat; t=t)
	l.input_gate_out[t]    = forward!(l.input_gate,    input_concat; t=t)
	l.output_gate_out[t]   = forward!(l.output_gate,   input_concat; t=t)
	l.new_candidate_out[t] = forward!(l.new_candidate, input_concat; t=t)

	l.cell_state[t+1]   = l.forget_gate_out[t] .* l.cell_state[t] + l.input_gate_out[t] .* l.new_candidate_out[t]
	l.hidden_state[t+1] = l.output_gate_out[t] .* tanh.(l.cell_state[t+1])
	return l.hidden_state[t+1]
end

# backpropagation of LSTM cell, return input gradient
function backprop!(l::LSTM, ∇output::Vector{Float64}; t::Int)::Vector{Float64}
	l.∇hidden_state[t+1] += ∇output
	l.∇cell_state[t+1]   += l.∇hidden_state[t+1] .* l.output_gate_out[t] .* tanh′.(l.cell_state[t+1])
	l.∇cell_state[t]      = l.∇cell_state[t+1]   .* l.forget_gate_out[t]

	∇forget_gate_out   = l.∇cell_state[t+1]   .* l.cell_state[t]
	∇input_gate_out    = l.∇cell_state[t+1]   .* l.new_candidate_out[t]
	∇output_gate_out   = l.∇hidden_state[t+1] .* tanh.(l.cell_state[t+1])
	∇new_candidate_out = l.∇cell_state[t+1]   .* l.input_gate_out[t]

	∇input_concat = backprop!(l.forget_gate,   ∇forget_gate_out;   t=t)
	              + backprop!(l.input_gate,    ∇input_gate_out;    t=t)
	              + backprop!(l.output_gate,   ∇output_gate_out;   t=t)
	              + backprop!(l.new_candidate, ∇new_candidate_out; t=t)

	l.∇hidden_state[t] = ∇input_concat[1:length(l.∇hidden_state[t])]
	return ∇input_concat[length(l.∇hidden_state[t])+1:end]
end
