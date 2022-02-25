import unittest
import selflearning as sl
from qiskit.quantum_info import Statevector, Operator
import numpy as np

class TestSampler(unittest.TestCase):

    def test_single_circuit_probs(self):
        params = np.random.rand(2**2) + np.random.rand(2**2) * 1j
        params = params / np.linalg.norm(params)
        s      = sl.Sampler(1, 4, 2, params=[params])
        vec    = np.asarray(Statevector(s.create_circuits()))
        tot    = 0
        for i in range(2**4):
            s.r  = i
            p    = s.prob_from_sampler()
            v    = vec[i]
            tot += p
            self.assertTrue(np.isclose(p, np.dot(v, v.conj())), f"Probability for {s.r} gives {p}, should be {np.dot(v, v.conj())}")
        self.assertTrue(np.isclose(tot, 1), f"Total probability is {tot}, i.e., not close to 1.")

    def test_multiple_circuit_probs(self):
        params = []
        for _ in range(2):
            p_i = np.random.rand(2) + np.random.rand(2) * 1j
            params.append(p_i / np.linalg.norm(p_i))
        s   = sl.Sampler(2, 2, 1, params=params)
        tot = 0
        for i in range(4):
            for j in range(4):
                s.r  = [i, j]
                tot += s.prob_from_sampler()
        self.assertTrue(np.isclose(tot, 1), f"Total probability is {tot}, i.e., not close to 1.")

    def test_cross_entropy(self):
        s = sl.Sampler(1, 3, 2)
        e = s.cross_entropy()
        self.assertTrue(np.isclose(e, 3), f"Entropy is {e}, but should be closer to 3.")

    def test_qft_derivative(self):
        r    = 3
        m    = 3
        s    = sl.Sampler(1, 4, m)
        u_x1 = np.array([s._u_qft_x_j(r, j) for j in range(2 ** m)])
        u_x2 = np.asarray(Operator(s._qft))[r][0:2**3]
        self.assertTrue(np.allclose(u_x1, u_x2), f"QFT derivative is computed incorrectly.")

    def test_one_loss_gradient(self):        
        s   = sl.Sampler(1, 3, 2)
        p_i = s.params
        g   = s.loss_gradient()
        p_f = p_i - 0.01 * g
        p_f = p_f / np.linalg.norm(p_f)
        self.assertTrue(np.isclose(np.linalg.norm(p_f - p_i), 0), "Change in params for optimal distribution is not zero.")

    def test_multi_loss_gradient(self):
        s = sl.Sampler(2, 3, 2)
        p = s.params
        g = s.loss_gradient([[2, 1]])
        try:
            a = p[0] + g[0]
            b = p[1] + g[1]
        except Exception:
            self.fail("Shape of two-dimensional gradient is not equal to paramter shape.")

    def test_uniform_accept(self):
        s = sl.Sampler(1, 5, 2)
        p = []
        r = 0
        for r_hat in range(1, 32):
            p.append(s.accept(r_hat, r=r))
            r = r_hat
        self.assertTrue(np.allclose(p, [1 for _ in range(31)]), "Did not accept all random variables of uniform distribution.") 

if __name__ == "__main__":
    unittest.main()