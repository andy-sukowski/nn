# Neural Network Library

A small, extensible neural network library, written from scratch in
Julia, that makes it very easy to define and train models consisting of
the following [layer types][1]:

* [`Conv`][2]: n-dimensional convolutional layer
* [`Pool`][3]: mean pooling, combine neuron clusters
* [`Flatten`][4]: flatten output of `Conv` to vector
* [`Dense`][5]: dense/fully connected layer
* [`LSTM`][6]: long short-term memory cell

> :warning: **Warning**\
> Further optimization and testing of the convolutional layer type with
> higher-dimensional datasets is needed. Until then, the network is
> rather slow and unexpected errors might occur.

## Usage

First, initialize the neural network by chaining different layers and
storing them in a vector.

```julia
include("nn.jl")

layers = [Conv(1 => 2, (28, 28), (5, 5)),
          Pool(2, 2),
          Conv(2 => 3, (12, 12), (5, 5)),
          Pool(2, 2),
          Flatten(3, (4, 4)),
          Dense(48 => 24),
          Dense(24 => 10)]
```

Then train the network on a data batch of type `Data` (defined in
[nn.jl][7]). The `train!()` function modifies the networks parameters
based on the average gradient across all data points. Optionally, the
learning rate `η` can be passed (default `η=1.0`). The function returns
the average loss of the network.

```julia
train!(layers, batch, η=1.5)
```

In order to achieve stochastic gradient descent, the `train!()` function
can be called from a `for`-loop. The `forward!()` and `loss()` function
can also be called manually. Have a look at the [examples][8].

> **Note**\
> `train_seq!()`, `forward_seq!()` and `backprop_seq!()` are currently
> used for sequential datasets. However, I plan to improve the
> interface, as `train()!` and `train_seq!()` appear almost identical.

## Convolutional forward pass and gradient

The forward pass and gradient equations of fully connected (dense)
layers are available in my [Multilayer Perceptron (MLP) repository][9].
And the forward pass of a convolutional layer is defined by this equation:

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./images/forward.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./images/forward_inv.svg">
  <img alt="forward propagation equation" src="./images/forward.svg">
</picture>

Based on the above equation, one can infer the partial derivatives of
the biases, kernels and activations in a convolutional layer with
respect to the loss / cost using the chain rule.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="./images/gradient.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./images/gradient_inv.svg">
  <img alt="derivatives of biases, kernels and activations" src="./images/gradient.svg">
</picture>

[1]: ./layers/
[2]: ./layers/conv.jl
[3]: ./layers/pool.jl
[4]: ./layers/flatten.jl
[5]: ./layers/dense.jl
[6]: ./layers/lstm.jl
[7]: ./nn.jl
[8]: ./examples/
[9]: https://git.andy-sb.com/mlp
