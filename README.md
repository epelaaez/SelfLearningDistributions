# SelfLearningDistributions

This Python module implements the paper [*Quantum self-learning Monte Carlo with quantum Fourier transform sampler*](https://arxiv.org/abs/2005.14075) by Katsuhiro Endo, Taichi Nakamura, Keisuke Fujii and Naoki Yamamoto using Qiskit to create and simulate the quantum circuits. The mentioned paper presents an adaptation of the self-learning [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm) algorithm using a quantum sampler inspired by the quantum Fourier transform to approximate a hard to sample distribution. This algorithm is a Monte-Carlo method that uses machine learning to generate an easy to sample approximation of a hard to sample target distribution. The resulting sampler that approximates the target distribution ends up being efficiently simulable, which gives rise to a quantum inspired algorithm that offers an advantage over other classical approaches to the same problem. After testing Qiskit's available methods within the [`AerSimulator`](https://qiskit.org/documentation/stubs/qiskit.providers.aer.AerSimulator.html) class, `statevector` was found to be the most efficient for the amount of qubits used here. When running the models with a larger amount of qubits, other simulation methods *may* turn out more efficient. 

A walk-through of the implementation of the paper is available in the [derivation notebook](https://github.com/epelaaez/SelfLearningDistributions/tree/main/notebooks). This notebook goes through the parts of the paper that concern this paper the most and gives a Python and Qiskit implementation for them. Since the purpose of this notebook is to give an idea of how the paper is implemented, some things are not implemented as efficiently as they could be. But all of that is resolved in the actual module, where the code may be a bit harder to read but faster to run. At the end of the derivation notebook is an example of the algorithm for a normal distribution using four qubits. 

The resulting algorithm that this module implements is able to train an easy-to-sample quantum sampler to approximate a hard-to-sample distribution, which, as the original paper mentions, is classically simulable while mantaining advantage over conventional alternatives to the same task. One very interesting application of this algorithm is in the problem of molecular simulation. When talking about the stochastic dynamics of two atoms obeying the Lennard-Jones potential field, we are required to sample the Boltzmann distribution, which depends on two three-dimensional vectors. Thus, we can use our self-learning algorithm to approximate this distribution using a quantum circuit. The time and computing power to demonstrate this was not available to me during QHack, but the algorithm implemented **is** able to accomplish this as demonstrated by the original authors. 

## Usage

### Initialize sampler

This module consists of a single class [`Sampler`](https://github.com/epelaaez/SelfLearningDistributions/blob/main/selflearning/sampler.py#L3) that deals with the whole algorithm. The simplest way to initialize a `Sampler` object is initialized as:

```python
s = Sampler(D, n, m)
```

`D` is the dimension of the sampler, `n` the total number of qubits in each circuit of the sampler (only one cirucit for the one-dimensional case but `D` circuits otherwise), and `m` the number of qubits used to inject learning parameters. This means that the complex vector paramter to be learned is of size $2^m$, corresponding to the state vector of `m` qubits.

If the `Sampler` is initialized like this, it will have initial paramters $[1 + 1i, 0, 0, \dots, 0]^T$ (normalized), which corresponds to an equal superposition of $2^n$ states after going through the QFT. The initial paramters can be modified using the optional parameter `params`.

Likewise, the function that the `Sampler` will approximate by default is a uniform distribution over the domain $[0, 2^n)$, which it already approximates exactly (see paragraph just above). This can be changed by passing a callable object to the optional parameter `target` when initializing the `Sampler`. Make sure said callable takes an `int` (one-dimensional case) or `list[int]` of size `D` (multi-dimensional case) as argument and outputs a `float`. This function should be of unit norm over the $[0, 2^n)$ range for the algorithm to work properly (i.e., $\sum_{i=0}^{2^n-1}t(i)=1$, where $t$ is the target function); there is some tolerance to this, which can be tested using `np.isclose(sum_of_your_fun, 1)` with the default tolerance values.

Finally, as discussed earlier, the default backend is the `AerSimulator` using the `statevector` method. This can be modified using the `backend` parameter. And the default number of shots is $10000$, which can be changed using the `shots` parameter.

### Train sampler

Once you initialized your `Sampler`, you can train it by calling the following function:

```python
final_params = s.train()
```

You can keep it as simple as that! However, of course, there are optional arguments which you can modify that alter the leraning procedure of the algorithm. First, there are the two arguments that control the change we make to our paramters in each step: `mu` and `alpha`. This are set to `0.9` and `0.01` by default, respectively. 

You can also choose the function to approximate if you don't want to train it for the function stored in `s.target`, to do this pass the new function to the optional argument `target`. Then, you can set the number of samples to use to calculate the gradient of the sampler with the argument `sample_size`, by default this is set to $2^{n-1}$. Also regarding the samples, the parameter `all_domain` can be set to `True` in order to use the whole domain of the circuit to train it (however, this can get expensive quickly and is therefore set to `False` as default).

The number of training steps can be controlled with the paramter `steps`, which is set to ten thousand by default. Finally, we got the argument `callback`, which is a function that will be called after each training step with the following arguments: `callback(step, diff, params, flag)`, where `diff` is the difference between the old parameters and the new ones, `params` are the new parameters, and `flag` is set to `True` when the training converged at step `step` and is `False` otherwise. By default, this function is set to print `Step {step}: change with difference {diff}. New params: {params}`

## Examples and results

To see how the `Sampler` works first-hand, go into [`demo_one_dimension.py`](https://github.com/epelaaez/SelfLearningDistributions/blob/main/demo_one_dimension.py) and run it yourself. After the learning process finished, there should be a file called `demo_one_dimension.csv` in the `data` folder containing the parameters each one hundred steps and after the training is done or converges. If everything goes correctly, the final parameters should look somewhat similar to that in [here](https://github.com/epelaaez/SelfLearningDistributions/blob/main/data/three_qubit_normal.csv). Then, you can use the [`create_figure.py`](https://github.com/epelaaez/SelfLearningDistributions/blob/main/create_figure.py) script to plot the final approximate distribution, just make sure to set the starting variables to the correct values before running it.

Previous results are included in the [`data`](https://github.com/epelaaez/SelfLearningDistributions/tree/main/data) folder. Each has a CSV and a PNG file of the same name. The CSV file contains the training process until convergence of the sampler and the PNG file contains the resulting distribution plotted against the target distribution. For reference, the function used for the target distribution for `three_qubit_normal` and `four_qubit_normal` was the following (with `n=3` and `n=4`, respectively):

```python
std  = (2**(n-1)-1) / 6
mean = (2**(n-1)-1) / 2

def target(x):
    x    = x - (2 ** (n-2))
    frac = 1 / (std * np.sqrt(2 * np.pi))
    exp  = (- 1 / 2) * (((x - mean) / std)**2)
    return frac * np.exp(exp)
```

And the target distribution for `three_qubit_normal_two` was the following with `n=3`:

```python
std  = (2**n-1) / 6
mean = (2**n-1) / 2

def target(x):
    frac = 1 / (std * np.sqrt(2 * np.pi))
    exp  = (- 1 / 2) * (((x - mean) / std)**2)
    return frac * np.exp(exp)
```

An example of a 2-dimensional sampler is also given in [`demo_two_dimension.py`](https://github.com/epelaaez/SelfLearningDistributions/blob/main/demo_one_dimension.py). The sampler that is trained here uses only two qubits for each circuit and tries to approximate a uniform distribution, so you can say it is very simple. However, this demo works really good to illustrate how the paramterer processing functions and the derivatives of said functions need to be passed into the sampler. In this case, the sampler uses a linear-basis regression model, but it is easy to generalize to a non-linear model by simply applying the chain rule to the derivative, which in code translates to changing the function `lbrm_deriv` to use `phi(var[i])` instead of just `var[i]` in the loop, where `phi` corresponds the functions $\phi_i$ as introduced in the [derivation notebook](https://github.com/epelaaez/SelfLearningDistributions/blob/main/notebooks/derivation.ipynb).

## Bug

At the moment, there is a minor bug that makes it neccesary to `.reverse_bits()` the circuits after they are finished training in order to get the correct approximation of the target distribution. This is probably due to some discrepancy in endianness between the original paper and Qiskit, but fortunately this is completely solved with `.reverse_bits()`, so it poses no immediate difficulties to the algorithm.

<sub>
Developed for QHack 2022 by Emilio Peláez.
</sub>
