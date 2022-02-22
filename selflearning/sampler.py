from qiskit import QuantumCircuit, QuantumRegister, transpile
import numpy as np

class Sampler:
    """
    Create a QFT-based distribution sampler.
    """
    def __init__(self, D, n, m, params=None, funcs=None) -> None:
        self.D = D
        self.n = n
        self.m = m
        self.r = None

        if params == None:
            params = []
            for _ in range(D):
                p    = np.zeros(2**m, dtype="complex")
                p[0] = (1/np.sqrt(2)) * (1 + 1j) 
                params.append(p)
        self.params = params

        if funcs == None:
            if D > 1:
                norm       = lambda var, theta: theta / (np.sqrt(np.dot(theta, theta.conj())))
                self.funcs = [norm for _ in range(D)]

        qft = QuantumCircuit(n)
        for i in range(n -1, -1, -1):
            qft.h(i)
            for j in range(i -1, -1, -1):
                x = 2 ** (j - i)
                qft.cp(np.pi * x, i, j)
        self._qft = qft

    def create_circuits(self):
        """
        Create the sampler circuit(s) using initialization data.
        """
        D      = self.D
        params = self.params

        # One-dimensional sampler
        if D == 1:
            self.circuits = self._one_dimension_sampler(params[0])
        # Multi-dimensional sampler
        else:
            if self.r == None:
                raise TypeError("Random variable (`self.r`) must be specified to construct multi-dimensional circuit.")
            circuits         = []
            processed_params = self._process_params()
            for theta in processed_params:
                circuits.append(self._one_dimension_sampler(theta))
            self.circuits = circuits
                

    def _one_dimension_sampler(self, theta):
        """
        Construct a one-dimensional sampler
        """
        n   = self.n
        m   = self.m
        qft = self._qft

        p  = QuantumRegister(m, "in")
        g  = QuantumRegister(n - m, "g")
        qc = QuantumCircuit(p, g)
        qc.initialize(theta, p) # encode parameters
        qc.compose(qft, inplace=True)

        return transpile(qc, basis_gates=["cx", "u"], optimization_level=3)

    def _process_params(self):
        """
        Process params for multi-dimensional sampler
        """
        params = []
        for k in range(len(self.params)):
            params.append(self.funcs[k](self.r[0:k], self.params[k]))
        return params