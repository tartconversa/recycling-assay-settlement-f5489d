"""结算口径：公式、暂付抵扣、负差额追偿与暂付上限核对。"""
import unittest
from pathlib import Path

import numpy as np

from recsettle.data import load
from recsettle.estimate import LotDraws, build_calibration
from recsettle.settle import settlement_draws, statement
from tiny import make_bundle

ROOT = Path(__file__).resolve().parents[1]
SIMS = 2000


def fake_draws(dry, nickel, lithium, contaminant, sims=100):
    qty = {
        "nickel": np.full(sims, nickel),
        "lithium": np.full(sims, lithium),
        "contaminant": np.full(sims, contaminant),
    }
    return LotDraws(
        "L1", sims, ("C1",), np.array([dry]), np.array([dry]),
        np.zeros((sims, 1)), {}, np.full(sims, dry), np.full(sims, dry), qty, qty,
    )


class FormulaTests(unittest.TestCase):
    def test_settlement_formula(self):
        contract = make_bundle().contracts["K1"]
        draws = fake_draws(1000.0, 100.0, 10.0, 1.0)
        values = settlement_draws(draws, contract, {"nickel": 120.0, "lithium": 220.0})
        # 0.85×(100×120 + 10×220) − 4×1000 − 0 = 12070 − 4000 = 8070
        self.assertAlmostEqual(float(values[0]), 8070.0, places=6)

    def test_penalty_only_above_soft_limit(self):
        contract = make_bundle().contracts["K1"]
        draws = fake_draws(1000.0, 100.0, 10.0, 3.0)  # 污染物比例 0.003 > 软界 0.002
        values = settlement_draws(draws, contract, {"nickel": 120.0, "lithium": 220.0})
        # 罚项 = (0.003 − 0.002) × 1000 × 800 = 800
        self.assertAlmostEqual(float(values[0]), 8070.0 - 800.0, places=6)

    def test_paid_deducted_once_and_negative_is_recover(self):
        bundle = make_bundle(payments=(9000.0,))
        cal = build_calibration(bundle)
        st = statement(bundle, cal, "L1", None, SIMS, 19)
        self.assertAlmostEqual(st["due_mean"], st["value"]["mean"] - 9000.0, places=6)
        self.assertGreater(st["recover"], 0.0)  # 多付部分为待追偿，不再支付

    def test_provisional_cap_passes_on_repo_data(self):
        bundle = load(ROOT)
        cal = build_calibration(bundle)
        for lot_id in bundle.lots:
            st = statement(bundle, cal, lot_id, None, SIMS, 23)
            self.assertTrue(st["provisional_cap"]["ok"], lot_id)


if __name__ == "__main__":
    unittest.main()
