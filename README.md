# cnn - convolutional neural network

This is an n-dimensional convolutional neural network, written from
scratch in Julia. It implements the following [layer types][1]:

* [`Conv(dims, input_size, kernel_size)`][2]: convolutional layer
* [`Pool(pool_size)`][3]: mean pooling, combine neuron clusters
* [`Flatten(input_size)`][4]: flatten output of `Conv` to vector
* [`Dense(dims)`][5]: dense/fully connected layer

## Usage

First, initialize the neural network by chaining different layers and
storing them in a vector.

```julia
include("network.jl")

layers = [Conv(1 => 2, (28, 28), (5, 5)),
          Pool((2, 2)),
          Flatten((2, 12, 12)),
          Dense(144 => 40),
          Dense(40 => 10)]
```

Then train the network on a data batch of type `Data` (defined in
[network.jl][6]. The `train!()` function modifies the networks
parameters based on the average gradient across all data points.
Optionally, the learning rate `η` can be passed (default `η=1`). The
function returns the average loss of the network.

```julia
train!(layers, batch, η=10.0)
```

In order to achieve stochastic gradient descent, the `train!()` function
can be called from a `for`-loop. The `forward!()` and `loss()` function
can also be called manually. Have a look at the [examples][7].

## Gradient equations

![forward propagation equation](./forward.svg)

Based on the above equation, one can infer the partial derivatives of
the biases, kernels and activations with respect to the loss / cost
using the chain rule.

![derivatives of biases, kernels and activations](./gradient.svg)

[1]: ./layers/
[2]: ./layers/conv.jl
[3]: ./layers/pool.jl
[4]: ./layers/flatten.jl
[5]: ./layers/dense.jl
[6]: ./network.jl
[7]: ./examples/
