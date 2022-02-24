# SelfLearningDistributions

This Python module implements the paper [*Quantum self-learning Monte Carlo with quantum Fourier transform sampler*](https://arxiv.org/abs/2005.14075) by Katsuhiro Endo, Taichi Nakamura, Keisuke Fujii and Naoki Yamamoto using Qiskit to create and simulate the quantum circuits. The mentioned paper presents an adaptation of the self-learning [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm) algorithm using a quantum sampler inspired by the quantum Fourier transform to approximate a hard to sample distribution. This algorithm is a Monte-Carlo method that uses machine learning to generate an easy to sample approximation of a hard to sample target distribution. The resulting sampler that approximates the target distribution ends up being efficiently simulable, which gives rise to a quantum inspired algorithm that offers an advantage over other classical approaches to the same problem. After testing Qiskit's available methods within the [`AerSimulator`](https://qiskit.org/documentation/stubs/qiskit.providers.aer.AerSimulator.html) class, `statevector` was found to be the most efficient for the amount of qubits used here. When running the models with a larger amount of qubits, other simulation methods *may* turn out more efficient. 

A walk-through of the implementation of the paper is available in the [derivation notebook](https://github.com/epelaaez/SelfLearningDistributions/tree/main/notebooks). This notebook goes through the parts of the paper that concern this paper the most and gives a Python and Qiskit implementation for them. Since the purpose of this notebook is to give an idea of how the paper is implemented, some things are not implemented as efficiently as they could be. But all of that is resolved in the actual module, where the code may be a bit harder to read but faster to run. At the end of the derivation notebook is an example of the algorithm for a normal distribution using four qubits. 

## Usage

This module consists of a single class [`Sampler`](https://github.com/epelaaez/SelfLearningDistributions/blob/main/selflearning/sampler.py#L3) that deals with the whole algorithm. The simplest way to initialize a `Sampler` object is initialized as:

```python
s = Sampler(D, n, m)
```

`D` is the dimension of the sampler, `n` the total number of qubits in each circuit of the sampler (only one cirucit for the one-dimensional case but `D` circuits otherwise), and `m` the number of qubits used to inject learning parameters. This means that the complex vector paramter to be learned is of size $2^m$, corresponding to the state vector of `m` qubits.

If the `Sampler` is initialized like this, it will have initial paramters $[1 + 1i, 0, 0, \dots, 0]^T$ (normalized), which corresponds to an equal superposition of $2^n$ states after going through the QFT. The initial paramters can be modified using the optional parameter `params`.

Likewise, the function that the `Sampler` will approximate by default is a uniform distribution over the domain $[0, 2^n)$, which it already approximates exactly (see paragraph just above). This can be changed by passing a callable object to the optional parameter `target` when initializing the `Sampler`. Make sure said callable takes an `int` (one-dimensional case) or `list[int]` of size `D` (multi-dimensional case) as argument and outputs a `float`. This function should be of unit norm over the $[0, 2^n)$ range for the algorithm to work properly (i.e., $\sum_{i=0}^{2^n-1}t(i)=1$, where $t$ is the target function); there is some tolerance to this, which can be tested using `np.isclose(sum_of_your_fun, 1)` with the default tolerance values.

Finally, as discussed earlier, the default backend is the `AerSimulator` using the `statevector` method. This can be modified using the `backend` parameter. And the default number of shots is $10000$, which can be changed using the `shots` parameter.

<sub>
Developed for QHack 2022 by Emilio Peláez.
</sub>
