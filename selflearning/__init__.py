from qiskit import QuantumCircuit, QuantumRegister, transpile, Aer
from qiskit.algorithms import FasterAmplitudeEstimation, EstimationProblem
from itertools import product

import numpy as np

from .sampler import Sampler