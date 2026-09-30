import unittest
import numpy as np
from scipy.special import ndtr
from data.uji import load_floors
from models.cross_floor import prior_map
from models.icl_context import latent_prior, sample_prior_scans, encode
from models.radio_map import cells
from scripts.evaluate_scm_tobit import PAIRS
from scripts.evaluate_transfer_map import transfer_prior

TRAIN = "/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv"


class PriorContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tr = load_floors(TRAIN)
        floors = sorted({f for p in PAIRS for f in p})
        cls.C = {f: cells(tr[f]) for f in floors}
        cls.tf = transfer_prior(cls.C, floors, "B1F2")
        cls.lc = cls.C["B1F1"]

    def test_latent_prior_matches_tpm_prior_map(self):
        """合成上下文与 TPM 用的是同一个先验：由 latent_prior 重算 E[x_imp] 必须等于 prior_map。"""
        mu, v, present = latent_prior(self.lc, self.tf)
        tf = self.tf
        sd = np.sqrt(tf["s"] ** 2 + v)
        z = (mu - tf["theta"]) / sd
        x = tf["h"] * (ndtr(z) * mu + v * np.exp(-0.5 * z * z) / (np.sqrt(2 * np.pi) * sd)) + (1 - tf["h"] * ndtr(z)) * -110.0
        x[:, ~present] = -110.0
        np.testing.assert_allclose(np.clip(x, -110, 0), prior_map(self.lc, tf)[0], atol=1e-10)

    def test_samples_are_valid_uji_scans(self):
        rssi, xy = sample_prior_scans(self.lc, self.tf, np.random.default_rng(0))
        n = len(self.lc["xy"])
        self.assertEqual(rssi.shape, (3 * n, 520))
        np.testing.assert_array_equal(xy, np.repeat(self.lc["xy"], 3, 0))
        det = rssi != 100
        self.assertTrue(((rssi[det] >= -104) & (rssi[det] <= 0)).all())
        absent = self.lc["k"].sum(0) == 0
        self.assertFalse(det[:, absent].any())
        e = encode(rssi)
        self.assertTrue(((e >= 0) & (e <= 105 / 105)).all())
        self.assertTrue((e[~det] == 0).all())

    def test_sampling_is_seeded(self):
        a = sample_prior_scans(self.lc, self.tf, np.random.default_rng(3))[0]
        b = sample_prior_scans(self.lc, self.tf, np.random.default_rng(3))[0]
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
