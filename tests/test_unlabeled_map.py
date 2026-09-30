import unittest
import numpy as np
from scipy.special import log_ndtr

from models.unlabeled_map import apply_update, conditional_factor, prepare_update, reconstruct
from scripts.evaluate_scm_tobit import tobit_match
from scripts.evaluate_transfer_map import calibrate


class UpdateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.p = rng.normal(size=(16, 2)) * 20
        self.f = rng.uniform(-110, -40, (16, 8))
        self.x = rng.uniform(-110, -40, (25, 8))
        self.h = rng.dirichlet(np.ones(16), size=25)
        self.l = conditional_factor(self.p, self.p[:3])

    def test_stationarity_and_descent(self):
        _, d = reconstruct(self.f, self.x, self.h, self.l, 1, 3)
        self.assertLess(d['normal_equation_relative_error'], 1e-10)
        self.assertLess(d['objective_after_unclipped'], d['objective_before'])

    def test_batch_duplication_and_permutation(self):
        v, _ = reconstruct(self.f, self.x, self.h, self.l, 1, 3)
        v2, _ = reconstruct(self.f, np.tile(self.x, (2, 1)), np.tile(self.h, (2, 1)), self.l, 1, 3)
        v3, _ = reconstruct(self.f, self.x[::-1], self.h[::-1], self.l, 1, 3)
        np.testing.assert_allclose(v, v2, atol=1e-10)
        np.testing.assert_allclose(v, v3, atol=1e-10)

    def test_exact_tpm_fallback(self):
        base = (self.f, self.p)
        s = self.f[:3].copy()
        for u, strength in [(self.x, 0), (self.x[:0], 1)]:
            prepared = prepare_update(base, s, self.p[:3], u)
            updated, _ = apply_update(prepared, strength)
            expected = calibrate(base, s, self.p[:3], (40., .3))
            np.testing.assert_array_equal(updated[0], expected[0])
            np.testing.assert_array_equal(updated[1], expected[1])

    def test_tobit_refactor_equivalence(self):
        q = self.x.copy()
        q[::2, ::2] = 100
        det = (q != 100).astype(float)
        x = np.where(q == 100, -110., q) * det
        d2 = (x*x).sum(1, keepdims=True) + det @ (self.f*self.f).T - 2*x @ self.f.T
        score = -.5 / 12**2 * d2 + (1-det) @ log_ndtr((-85-self.f)/12).T
        w = np.exp(score-score.max(1, keepdims=True))
        original = (w/w.sum(1, keepdims=True)) @ self.p
        np.testing.assert_array_equal(tobit_match(q, self.f, self.p, 12, -85), original)


if __name__ == '__main__':
    unittest.main()
