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

        if isinstance(params, type(None)):
            params = []
            for _ in range(D):
                p    = np.zeros(2**m, dtype="complex")
                p[0] = (1/np.sqrt(2)) * (1 + 1j) 
                params.append(p)
        self.params = np.array(params)

        if isinstance(funcs, type(None)):
            if D > 1:
                norm       = lambda var, theta: theta / (np.sqrt(np.dot(theta, theta.conj())))
                self.funcs = [norm for _ in range(D)]

        if isinstance(target, type(None)):
            self.target = lambda r: 1 / (2 ** n)

        if isinstance(backend, type(None)):
            self.backend = Aer.get_backend("aer_simulator_statevector")
        self.shots = shots

        qft = QuantumCircuit(n)
        for i in range(n -1, -1, -1):
            qft.h(i)
            for j in range(i -1, -1, -1):
                x = 2 ** (j - i)
                qft.cp(np.pi * x, i, j)
        for i in range(n // 2):
            qft.swap(i, n - i - 1)
        self._qft = qft

    def create_circuits(self, D=None, r=None, params=None, process=True):
        """
        Create the sampler circuit(s) using initialization data.
        """
        if isinstance(D, type(None)):
            D = self.D
        if isinstance(r, type(None)) and isinstance(self.r, type(None)) and D != 1:
            raise TypeError("Random variable must be set in Sampler object or passed to this function.")
        elif isinstance(r, type(None)) and not isinstance(self.r, type(None)):
            r = self.r
        if isinstance(params, type(None)):
            params = self.params

        # One-dimensional sampler
        if D == 1:
            circuits = self._one_dimension_sampler(params[0])
        # Multi-dimensional sampler
        else:
            circuits = []
            if process:
                params = self._process_params(r)
            for theta in params:
                circuits.append(self._one_dimension_sampler(theta))
        return circuits

    def _process_params(self, r=None):
        """
        Process params for multi-dimensional sampler.
        """
        if self.D == 1:
            return self.params

        if isinstance(r, type(None)) and isinstance(self.r, type(None)):
            raise TypeError("Random variable must be set in Sampler object or passed to this function.")
        elif isinstance(r, type(None)) and not isinstance(self.r, type(None)):
            r = self.r 

        params = []
        for k in range(len(self.params)):
            params.append(self.funcs[k](r[0:k], self.params[k]))
        return params

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

    def prob_from_sampler(self, r=None, qc=None, delta=0.01, maxiter=3):
        """
        Given a random variable, give its probability using the sampler's distribution.
        """
        if isinstance(r, type(None)) and isinstance(self.r, type(None)):
            raise TypeError("Random variable must be set in Sampler object or passed to this function.")
        elif isinstance(r, type(None)) and not isinstance(self.r, type(None)):
            r = self.r
        n = self.n

        if isinstance(r, int):
            # One-dimensional case
            b  = format(r, "b").zfill(n)[::-1]
            if qc == None:
                qc = self.create_circuits(D=1)
            for i in range(n):
                if b[i] == "0":
                    qc.x(i)

            problem = EstimationProblem(
                state_preparation = qc,
                objective_qubits  = [i for i in range(n)],
            )

            fae = FasterAmplitudeEstimation(
                delta            = delta,
                maxiter          = maxiter,
                quantum_instance = self.backend,
            )

            result = fae.estimate(problem)
            return result.estimation

        elif isinstance(r, np.ndarray) or isinstance(r, list):
            # Multi-dimensional case
            prob = 1
            if qc == None:
                qc = self.create_circuits(r=r)
            for i in range(len(r)):
                prob *= self.prob_from_sampler(qc=qc[i], r=r[i])
            return prob

        else:
            raise TypeError("Random variable `r` must be an int for one-dimensional case or np.ndarray or list for multi-dimensional case.")

    def cross_entropy(self, target=None, r=None):
        """
        Calculate cross entropy of sampler with target distribution.
        """
        D = self.D
        n = self.n

        if isinstance(target, type(None)):
            target = self.target
        elif not callable(target):
            raise TypeError("Target distribution passed to the function is not callable.")
        if isinstance(r, type(None)):
            if D == 1:
                r = [i for i in range(2**n)]
            else:
                r = list(product([i for i in range(2**n)], repeat=D))
                r = [list(r_i) for r_i in r]

        entropy = 0
        q_x     = []
        for var in r:
            q_x.append(self.prob_from_sampler(r=var))
        for i in range(len(r)):
            if np.isclose(q_x[i], 0):
                continue
            entropy += target(r[i]) * np.log2(q_x[i])
        return -entropy

    def qft_derivative(self, r=None, theta=None):
        """
        Compute the derivative of sampler(s) when a random variable is given or set.
        """
        if isinstance(r, type(None)) and isinstance(self.r, type(None)):
            raise TypeError("Random variable must be set in Sampler object or passed to this function.")
        elif isinstance(r, type(None)) and not isinstance(self.r, type(None)):
            r = self.r
        if isinstance(theta, type(None)):
            theta = self._process_params()
        N = self.n

        if isinstance(r, int):
            # One-dimensional case
            m   = len(theta)
            u_x = np.array([self._u_qft_x_j(r, j) for j in range(m)])
            dot = np.dot(u_x, theta)
            return np.array(dot * u_x.conj())

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

    def loss_gradient(self, r=None, target=None):
        """
        Compute gradient of loss function.
        """
        D      = self.D
        n      = self.n
        params = self._process_params()

        if isinstance(target, type(None)):
            target = self.target
        if isinstance(r, type(None)):
            if D == 1:
                r = [i for i in range(2**n)]
            else:
                r = list(product([i for i in range(2**n)], repeat=D))
                r = [list(r_i) for r_i in r]
        B = len(r)

        if D == 1:
            # One-dimensional case
            grad = np.zeros(len(params), dtype="complex")
            for i in range(B):
                num  = target(r[i])
                den  = self.prob_from_sampler(r=r[i])
                frac = num / 1e-8 if den == 0 else num / den
                grad = grad + (frac * self.qft_derivative(r=r[i]))
            return - (1 / B) * grad
        else:
            # Multi-dimensional case
            # TODO: add derivative of processing function
            grad = []
            for k in range(len(params)):
                grad_k = np.zeros(len(params[k]), dtype="complex")
                for i in range(B):
                    circs   = self.create_circuits(r=r[i])
                    frac_1  = self.qft_derivative(r=r[i][k], theta=params[k]) / self.prob_from_sampler(r=r[i][k], qc=circs[k])
                    frac_2  = target(r[i]) / self.prob_from_sampler(r=r[i])
                    grad_k += frac_1 * frac_2
                grad.append(- (1 / B) * grad_k)
            return grad

    def accept(self, r_hat, r=None, target=None):
        if not (isinstance(r_hat, int) or isinstance(r_hat, list) or isinstance(r_hat, np.ndarray)):
            raise TypeError("Random variable `r_hat` must be passed to this function as int, list, or ndarray.")
        if isinstance(r, type(None)) and isinstance(self.r, type(None)):
            raise TypeError("Random variable `r` must be set in Sampler object or passed to this function.")
        elif isinstance(r, type(None)) and not isinstance(self.r, type(None)):
            r = self.r
        if isinstance(target, type(None)):
            target = self.target
        
        num  = target(r_hat) * self.prob_from_sampler(r=r)
        den  = target(r) * self.prob_from_sampler(r=r_hat)
        prob = num / den if den != 0 else 1 
        return min([prob, 1])