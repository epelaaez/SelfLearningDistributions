from qiskit.quantum_info import Statevector
import selflearning as sl
import matplotlib.pyplot as plt
import numpy as np

## Remember to change all the variables to match the sampler you want to plot
## and the target distribution you are trying to approximate. Also, this visualization
## only works for one-dimensional samplers since it is very simple. 

D = 1
n = 4
m = 2

std  = (2**(n-1)-1) / 6
mean = (2**(n-1)-1) / 2

def target(x):
    x    = x - (2 ** (n-2))
    frac = 1 / (std * np.sqrt(2 * np.pi))
    exp  = (- 1 / 2) * (((x - mean) / std)**2)
    return frac * np.exp(exp)

fig, ax = plt.subplots(1, 1)

## Here I use params from an already trained model, for which you can find the 
## data in the csv file called four_qubit_normal. Replace this with your trained params!

params  = [0.34611681+0.34611681j, -0.28350761-0.42429912j, 0.19528344+0.47145592j, -0.09549341-0.4800778j]
params  = params / np.linalg.norm(params)
sampler = sl.Sampler(D, n, m, params=[params])
circuit = sampler.create_circuits()
s       = Statevector(circuit.reverse_bits()).data
x       = np.array([i for i in range(2**n)])
y1      = [np.real(np.dot(s[i], s[i].conj())) for i in x]
y2      = [target(i) for i in x]

ax.plot(x, y1, 'r-', lw=5, alpha=0.6, label='sampler pdf')
ax.plot(x, y2, 'b-', lw=5, alpha=0.6, label='target pdf')
ax.legend()
fig.savefig("data/four_qubit_normal.png", dpi=fig.dpi) # change where you want to save the image