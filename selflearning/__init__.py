from qiskit import QuantumCircuit, QuantumRegister, transpile, Aer
from qiskit.algorithms import FasterAmplitudeEstimation, EstimationProblem
import numpy as np
from .sampler import Sampler