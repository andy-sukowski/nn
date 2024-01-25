# See LICENSE file for copyright and license details.

# Rectified Linear Unit (ReLU)
relu(x)  = max(0, x)
relu′(x) = x < 0 ? 0 : 1

# leaky ReLU to avoid dead neurons
lrelu(x)  = max(0.01, x)
lrelu′(x) = x < 0 ? 0.01 : 1

# sigmoid
σ(x)  = 1 / (1 + exp(-x))
σ′(x) = σ(x) * (1 - σ(x))

# hyperbolic tangent
tanh(x)  = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
tanh′(x) = 1 - tanh(x)^2

# mean squared error
mse(x, y)  = sum((x - y) .^ 2) / length(x)
mse′(x, y) = 2 .* (x - y) / length(x)

# multi-class cross-entropy
xent(x, y)  = -sum(y .* log.(x .+ 1e-8))
xent′(x, y) = -y ./ (x .+ 1e-8)

# recursive n-fold Cartesian product
# cart(x) = Tuple.(x) # base case
# cart(xs...) = [(i, j...) for i in xs[1], j in cart(xs[2:end]...)]
# alternatively, use Iterators.product():
cart = Iterators.product

function one_hot(n::Int, d::Int)::Vector{Int}
	out = zeros(n)
	out[d] = 1
	return out
end
