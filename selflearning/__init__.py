from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms import FasterAmplitudeEstimation, EstimationProblem
from itertools import product

import numpy as np
import random

from .sampler import Sampler