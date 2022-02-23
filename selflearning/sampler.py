from . import *

class Sampler:
    """
    Create a QFT-based distribution sampler.
    """
    def __init__(self, D, n, m, params=None, funcs=None, target=None, backend=None, shots=10000) -> None:
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

        if target == None:
            self.target = lambda r: 1 / (2 ** n)

        if backend == None:
            self.backend = Aer.get_backend("statevector_simulator")
        self.shots = shots

        qft = QuantumCircuit(n)
        for i in range(n -1, -1, -1):
            qft.h(i)
            for j in range(i -1, -1, -1):
                x = 2 ** (j - i)
                qft.cp(np.pi * x, i, j)
        self._qft = qft

    def create_circuits(self, D=None, r=None, params=None, process=True):
        """
        Create the sampler circuit(s) using initialization data.
        """
        if D == None:
            D = self.D
        if r == None and self.r == None:
            raise TypeError("Random variable must be set in Sampler object or passed to this function.")
        elif r == None and self.r != None:
            r = self.r
        if params == None:
            params = self.params

        # One-dimensional sampler
        if D == 1:
            circuits = self._one_dimension_sampler(params[0])
        # Multi-dimensional sampler
        else:
            if self.r == None:
                raise TypeError("Random variable (`self.r`) must be specified to construct multi-dimensional circuit.")
            circuits = []
            if process:
                params = self._process_params(r)
            for theta in params:
                circuits.append(self._one_dimension_sampler(theta))
        return circuits

    def _one_dimension_sampler(self, theta):
        """
        Construct a one-dimensional sampler.
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

    def _process_params(self, r=None):
        """
        Process params for multi-dimensional sampler.
        """
        if r == None and self.r == None:
            raise TypeError("Random variable must be set in Sampler object or passed to this function.")
        elif r == None and self.r != None:
            r = self.r 

        params = []
        for k in range(len(self.params)):
            params.append(self.funcs[k](r[0:k], self.params[k]))
        return params

    def prob_from_sampler(self, r=None):
        """
        Given a random variable, give its probability using the sampler's distribution.
        """
        if r == None and self.r == None:
            raise TypeError("Random variable must be set in Sampler object or passed to this function.")
        elif r == None and self.r != None:
            r = self.r
        n = self.n

        if isinstance(r, int):
            # One-dimensional case
            b  = format(r, "b").zfill(n)[::-1]
            qc = self.create_circuits(D=1)

    def qft_derivative(self, r=None, theta=None):
        """
        Compute the derivative of sampler(s) when a random variable is given or set.
        """
        if r == None and self.r == None:
            raise TypeError("Random variable must be set in Sampler object or passed to this function.")
        elif r == None and self.r != None:
            r = self.r
        if theta == None:
            theta = self._process_params()
        N = self.n

        if isinstance(r, int):
            # One-dimensional case
            m   = len(theta)
            u_x = np.array(self._u_qft_x_j(r, j, N) for j in range(m))
            dot = np.dot(u_x, theta)
            return dot * u_x.conj()

        elif isinstance(r, np.ndarray) or isinstance(r, list):
            # Multi-dimensional case
            derivs = []
            for i, r_i in enumerate(r):
                derivs.append(self.qft_derivative(r_i, theta[i]))
            return derivs

        else:
            raise TypeError("Random variable `r` must be an int for one-dimensional case or np.ndarray or list for multi-dimensional case.")

    def _u_qft_x_j(self, r, j):
        """
        Return j-th element of the r-th row vector of QFT.
        """
        N = self.n
        return (1 / np.sqrt(2 ** N)) * np.exp(1j * 2 * np.pi * r * j / (2 ** N))