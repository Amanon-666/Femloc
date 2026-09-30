import unittest

import numpy as np
import torch

from models.learned_map import MapContext, MapEncoder, adapt, kernels, match, tensor
from scripts.evaluate_scm_tobit import tobit_match
from scripts.evaluate_signal_calibrated_map import build_map
from scripts.run_learned_map import source_names


class LearnedMapTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        rng = np.random.default_rng(3)
        n, a = 12, 6
        xy = rng.uniform(0, 80, (n, 2)) + [1e5, 4.8e6]
        counts = rng.integers(0, 9, (n, a))
        mean = rng.uniform(-95, -45, (n, a))
        self.old = {'xy': xy, 'n': np.full(n, 8.), 'k': counts,
                    's1': counts * mean, 's2': counts * mean**2}
        self.prior = np.clip(mean - 15, -110, 0)
        self.xy = np.repeat(xy[:4] + [.2, -.3], 3, axis=0)
        self.rssi = rng.uniform(-95, -50, (12, a))
        self.rssi[rng.random(self.rssi.shape) < .3] = 100
        self.query = rng.uniform(-90, -50, (5, a))
        self.query[rng.random(self.query.shape) < .3] = 100
        self.c = MapContext(self.old, self.prior)
        self.s = self.c.support(self.rssi, self.xy)
        self.encoder = MapEncoder().double()
        self.field = (40., .3)

    def test_original_tpm_and_empty_fallback(self):
        cand, info = adapt(self.c, self.s, self.field, diagnostics=True)
        expected = build_map('scm', (self.prior, self.old['xy']), self.rssi, self.xy,
                             self.field, {'anchor_lookup_neighbors': 3})
        np.testing.assert_allclose(cand[0].numpy(), expected[0], atol=2e-10, rtol=0)
        pred = match(tensor(self.query, 'cpu'), cand, 12, -85).numpy() + self.c.origin
        np.testing.assert_allclose(pred, tobit_match(self.query, *expected, 12, -85), atol=2e-8, rtol=0)
        self.assertLess(info['normal_residual'], 1e-12)
        empty = self.c.support(self.rssi[:0], self.xy[:0])
        mapped, _ = adapt(self.c, empty, self.field, self.encoder)
        self.assertTrue(torch.equal(mapped[0], self.c.f))

    def test_psd_and_parameter_gradient_through_localization(self):
        _, kss, _ = kernels(self.c, self.s, self.field[0], self.encoder)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(kss.detach()).min()), -1e-10)
        fixed0 = self.encoder.net[0]
        bias = self.encoder.net[2].bias.detach()
        def loss(w):
            def encode(z):
                return torch.tanh(z @ fixed0.weight.detach().T + fixed0.bias.detach()) @ w.T + bias
            mapped, _ = adapt(self.c, self.s, self.field, encode)
            pred = match(tensor(self.query, 'cpu'), mapped, 12, -85)
            return (pred / 20).square().mean()
        w = self.encoder.net[2].weight.detach().clone().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(loss, (w,), eps=1e-5, atol=2e-6, rtol=1e-4))
        self.assertGreater(float(torch.autograd.grad(loss(w), w)[0].norm()), 1e-8)

    def test_support_order_map_order_ap_order_and_translation(self):
        with torch.no_grad():
            original, _ = adapt(self.c, self.s, self.field, self.encoder)
            q = tensor(self.query, 'cpu')
            expected = match(q, original, 12, -85)
            rng = np.random.default_rng(42)
            perm = rng.permutation(len(self.xy))
            candidate, _ = adapt(self.c, self.c.support(self.rssi[perm], self.xy[perm]), self.field, self.encoder)
            torch.testing.assert_close(match(q, candidate, 12, -85), expected, atol=1e-9, rtol=0)

            rows = rng.permutation(len(self.old['xy']))
            cols = rng.permutation(self.prior.shape[1])
            old = {k: (v[rows][:, cols] if v.ndim == 2 and k != 'xy' else v[rows]) for k, v in self.old.items()}
            other = MapContext(old, self.prior[rows][:, cols])
            s = other.support(self.rssi[:, cols], self.xy)
            candidate, _ = adapt(other, s, self.field, self.encoder)
            absolute = match(tensor(self.query[:, cols], 'cpu'), candidate, 12, -85).numpy() + other.origin
            np.testing.assert_allclose(absolute, expected.numpy() + self.c.origin, atol=2e-8, rtol=0)

            shift = np.array([270., -93.])
            shifted_old = {**self.old, 'xy': self.old['xy'] + shift}
            other = MapContext(shifted_old, self.prior)
            candidate, _ = adapt(other, other.support(self.rssi, self.xy + shift), self.field, self.encoder)
            absolute = match(q, candidate, 12, -85).numpy() + other.origin
            np.testing.assert_allclose(absolute, expected.numpy() + self.c.origin + shift, atol=2e-8, rtol=0)

    def test_source_building_isolation(self):
        for held in range(3):
            names = source_names(held)
            self.assertTrue(names)
            self.assertFalse(any(f.startswith(f'B{held}F') for f in names))


if __name__ == '__main__':
    unittest.main()
