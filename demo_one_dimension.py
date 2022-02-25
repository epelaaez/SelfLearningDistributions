import selflearning as sl
import numpy as np
import csv

D = 1
n = 3
m = 2

std  = (2**(n-1)-1) / 6
mean = (2**(n-1)-1) / 2

def target(x):
    frac = 1 / (std * np.sqrt(2 * np.pi))
    exp  = (- 1 / 2) * (((x - mean) / std)**2)
    return frac * np.exp(exp)

d_steps  = []
d_params = []
filename = "demo_one_dimension"

def callback(step, diff, params, flag):
    print(f"Step {step}.")
    print(f"Difference between parameters: {diff}.")
    if flag:
        print(f"Converged at step {step}.")
    if step % 100 == 0 or flag:
        d_steps.append(step)
        d_params.append(params)

sampler = sl.Sampler(D, n, m, target=target)
sampler.train(all_domain=True, callback=callback)

with open('data/'+filename+'.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["step", "params"])
    for i in range(len(d_steps)):
        writer.writerow([d_steps[i], d_params[i]])
