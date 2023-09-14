# See LICENSE file for copyright and license details.

ReLU(x)  = max(0, x)
ReLU′(x) = x >= 0 ? 1 : 0

σ(x)  = 1 / (1 + exp(-x))
σ′(x) = σ(x) * (1 - σ(x))

# recursive n-fold Cartesian product
# cart(x) = Tuple.(x) # base case
# cart(xs...) = [(i, j...) for i in xs[1], j in cart(xs[2:end]...)]
# alternatively, use Iterators.product():
cart = collect ∘ Iterators.product

function one_hot(d :: Int, n :: Int) :: Vector{Int}
	out = zeros(n)
	out[d] = 1
	return out
end
