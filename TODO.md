Activation fucntions

Each step of the network: output = dot(weights, input) + b where weights are matrices, inputs are vectors and bs are vector offsets.

Without an activation function the outputs can only ever be linear and thus a nn couldn't learn non linear relationships.

Optimizer fucntions

Network topologies
When there's little tarining data it's preferable to use a small network and kfolds to avoid overfitting.

loss funcitons:
regression:
mean average error

binary classification:
cross-entropy

Cross entropy:
Entropy is the measure of uncertainty, calculated with the base 2 logarithm it tells how many bits are needed to represent the information.

$E = -\sum_{i=0}^{N} p_i\log_2(1/p_i)$  

Cross entropy is the entropy across two prbability distributions, if the distribution are the same(ideal) the entropy to encode their difference is 0. This is therefore a function to minimize, a loss function.

CUDA