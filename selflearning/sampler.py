from . import *

class Sampler:
    """
    Create a QFT-based distribution sampler.
    """
    def __init__(self, D, n, m, params=None, funcs=None, derivs=None, target=None, backend=None, shots=10000) -> None:
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
        self.params = np.array(params, dtype="object")

        if isinstance(funcs, type(None)):
            if D > 1:
                norm  = lambda var, theta: theta / (np.sqrt(np.dot(theta, theta.conj())))
                funcs = [norm for _ in range(D)]
        self.funcs = funcs

        if isinstance(derivs, type(None)):
            if D > 1:
                def n_deriv(var, theta):
                    mat = []
                    n   = np.linalg.norm(theta)
                    for i in range(len(theta)):
                        ones    = np.zeros(len(theta))
                        ones[i] = 1
                        mat.append((ones / n) - ((theta[i] / (n ** 3)) * theta))
                    return np.array(mat)
                derivs = [n_deriv for _ in range(D)]
        self.derivs = derivs

        if isinstance(target, type(None)):
            target = lambda r: 1 / ((2 ** n) ** D)
        self.target = target

        if isinstance(backend, type(None)):
            backend = AerSimulator(method="statevector")
        self.backend = backend
        self.shots   = shots

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
        D = self.D
        n = self.n
        m = self.m

        if isinstance(target, type(None)):
            target = self.target
        if isinstance(r, type(None)):
            if D == 1:
                r = [i for i in range(2**n)]
            else:
                r = list(product([i for i in range(2**n)], repeat=D))
                r = [list(r_i) for r_i in r]
        B      = len(r)
        params = self._process_params(r=r)

        if D == 1:
            # One-dimensional case
            grad = np.zeros(len(params), dtype="complex")
            for i in range(B):
                num  = target(r[i])
                den  = self.prob_from_sampler(r=r[i])
                frac = num / 1e-20 if den == 0 else num / den
                grad = grad + (frac * self.qft_derivative(r=r[i]))
            return [- (1 / B) * grad]
        else:
            # Multi-dimensional case
            grad = []
            for k in range(len(params)):
                grad_k = [0 for _ in range(len(params[k]))]
                for i in range(B):
                    circs   = self.create_circuits(r=r[i])
                    mat     = self.derivs[k](r[i][0:k], params[k])
                    frac_1  = self.qft_derivative(r=r[i][k], theta=params[k]) / self.prob_from_sampler(r=r[i][k], qc=circs[k])
                    frac_2  = target(r[i]) / self.prob_from_sampler(r=r[i])
                    if len(r[0][0:k]) > 0 and np.array(grad_k).shape != mat.shape:
                        for j in range(len(grad_k)):
                            if grad_k[j] == 0:
                                grad_k[j]  = np.dot(frac_1 * frac_2, mat[j])
                            else: 
                                grad_k[j] += np.dot(frac_1 * frac_2, mat[j])
                    else:
                        grad_k += np.dot(frac_1 * frac_2, mat)
                grad.append(np.multiply(- (1 / B), grad_k))
            return grad

    def accept(self, r_hat, r=None, target=None):
        """
        Compute probability of accepting `r_hat` given `r` was accepted last. 
        """
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

    def train(self, mu=0.9, alpha=0.01, target=None, sample_size=None, all_domain=False, steps=10000, callback=None):
        """
        Train sampler to approximate target distribution.
        """
        D      = self.D
        n      = self.n

        if isinstance(target, type(None)):
            target = self.target
        if sample_size == None and all_domain == False:
            sample_size = 2 ** (n - 1)
        if isinstance(callback, type(None)):
            callback = lambda a, b, c: print(f"Step {a}: change with difference {b}.\nNew params: {c}")

        if D == 1:
            change = [0]
            for step in range(0, steps):
                if all_domain:
                    samples = [i for i in range(2**n)]
                else:
                    possible = [i for i in range(2**n)]
                    probs    = [self.prob_from_sampler(r) for r in possible]
                    samples  = []
                    r        = np.random.randint(2**n)
                    possible.remove(r)
                    for _ in range(sample_size - 1):
                        weights = []
                        p_q_r   = probs[r]
                        for r_hat in possible:
                            p_q_r_hat = probs[r_hat]
                            weights.append(min([1, (target(r_hat) * p_q_r) / (target(r) * p_q_r_hat)]))
                        r = random.choices(possible, weights=weights)[0]
                        samples.append(r)
                        possible.remove(r)
                grad   = self.loss_gradient(r=samples)
                change = self._gradient_change(mu, grad, change)
                new_p  = np.subtract(self.params, np.multiply(alpha, change))
                new_p  = new_p / np.linalg.norm(new_p)
                diff   = np.linalg.norm(np.subtract(self.params, new_p))
                if diff < 1e-20:
                    return self.params
                self.params = new_p
                callback(step, diff, self.params)
            return self.params
        else:
            change = [0 for _ in range(D)]
            for step in range(0, steps):
                if all_domain:
                    samples = list(product([i for i in range(2**n)], repeat=D))
                    samples = [list(r_i) for r_i in samples]
                else:
                    possible = list(product([i for i in range(2**n)], repeat=D))
                    possible = [list(r_i) for r_i in possible]
                    probs    = [self.prob_from_sampler(r) for r in possible]
                    samples  = []
                    r        = list(np.random.randint(2**n, size=D))
                    r_idx    = possible.index(r)
                    possible.remove(r)
                    for _ in range(sample_size - 1):
                        weights = []
                        p_q_r   = probs[r_idx]
                        for i, r_hat in enumerate(possible):
                            p_q_r_hat = probs[i]
                            weights.append(min([1, (target(r_hat) * p_q_r) / (target(r) * p_q_r_hat)]))
                        r     = list(random.choices(possible, weights=weights)[0])
                        r_idx = possible.index(r)
                        samples.append(r)
                        possible.remove(r)
                grad   = self.loss_gradient(r=samples)
                change = self._gradient_change(mu, grad, change)
                new_p  = []
                diffs  = []
                for i, c in enumerate(change):
                    new_p_1 = []
                    for k in range(len(c)):
                        new_p_1.append(self.params[i][k] - alpha * c[k])
                        diffs.append(np.linalg.norm(self.params[i][k] - new_p_1[k]))
                    new_p.append(new_p_1 / np.linalg.norm(new_p_1))
                if np.allclose(diffs, 1e-20):
                    return self.params
                self.params = new_p
                callback(step, np.linalg.norm(diffs), self.params)
            return self.params
                
    def _gradient_change(self, mu, grad, prev_grad):
        m = []
        for i in range(len(grad)):
            m.append(np.multiply(mu, prev_grad[i]) + np.multiply(1-mu, grad[i]))
        return m