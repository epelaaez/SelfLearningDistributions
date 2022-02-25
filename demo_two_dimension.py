import selflearning as sl
import numpy as np
import csv

D = 2
n = 2
m = 1

params = []
params.append(np.random.random(2**m) + np.random.random(2**m) * 1j)
params.append([np.random.random(2**m) + np.random.random(2**m) * 1j for _ in range(2)])

def lbrm(var, theta):
    vec = np.array(theta[-1])
    for i in range(len(var)):
        vec += var[i] * theta[i]
    return vec / np.sqrt(np.dot(vec, vec.conj()))

funcs = [
    lambda var, theta: theta / (np.sqrt(np.dot(theta, theta.conj()))),
    lbrm
]

def id_deriv(var, theta):
    mat = []
    n   = np.linalg.norm(theta)
    for i in range(len(theta)):
        ones    = np.zeros(len(theta))
        ones[i] = 1
        mat.append((ones / n) - ((theta[i] / (n**3)) * theta))
    return np.array(mat)

def lbrm_deriv(var, theta):
    norm_deriv = id_deriv(var, theta)
    mat        = []
    for i in range(len(var)):
        mat.append(np.multiply(norm_deriv, var[i]))
    mat.append(norm_deriv)
    return np.array(mat)

derivs = [
    id_deriv,
    lbrm_deriv,
]

s = sl.Sampler(D, n, m, params=params, funcs=funcs, derivs=derivs)

d_steps  = []
d_params = []
filename = "demo_two_dimension"

def callback(step, diff, params, flag):
    print(f"Step {step}.")
    print(f"Difference between parameters: {diff}.")
    if flag:
        print(f"Converged at step {step}.")
    if step % 100 == 0 or flag:
        d_steps.append(step)
        d_params.append(params)

sampler = sl.Sampler(D, n, m)
sampler.train(all_domain=True, callback=callback)

with open('data/'+filename+'.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["step", "params"])
    for i in range(len(d_steps)):
        writer.writerow([d_steps[i], d_params[i]])