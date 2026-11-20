"""估值核心：容器加权、未检出处理、贡献分解、复检证据差异与可知时刻纪律。"""
import datetime as dt
import unittest
from pathlib import Path

import numpy as np

from recsettle.data import load
from recsettle.estimate import attribute, build_calibration, build_lot_model, simulate
from recsettle.plan import evaluate_action, candidate_actions
from tiny import make_bundle

ROOT = Path(__file__).resolve().parents[1]
SIMS = 2000


def tiny_draws(bundle, sims=SIMS, seed=7):
    cal = build_calibration(bundle)
    model = build_lot_model(bundle, cal, "L1")
    return simulate(model, cal, sims, seed)


class WeightingTests(unittest.TestCase):
    def test_dry_mass_uses_container_weights_not_sample_count(self):
        # 容器 100 kg 水分 0.10、容器 200 kg 水分 0.20：干基 = 90 + 160 = 250；
        # 若按样本数平均（水分 0.15）则会得到 255。
        bundle = make_bundle()
        draws = tiny_draws(bundle)
        self.assertAlmostEqual(float(np.mean(draws.dry_full)), 250.0, delta=3.0)

    def test_quantities_follow_dry_basis(self):
        bundle = make_bundle()
        draws = tiny_draws(bundle)
        # 镍量 ≈ 90×0.10 + 160×0.12 = 9 + 19.2 = 28.2 kg（收缩允许小幅偏差）
        self.assertAlmostEqual(float(np.mean(draws.qty_full["nickel"])), 28.2, delta=1.5)


class CensoringTests(unittest.TestCase):
    def test_nondetect_is_interval_not_zero(self):
        bundle = make_bundle(contaminant_lt=True)
        draws = tiny_draws(bundle)
        fraction = draws.contaminant_fraction()
        mean_fraction = float(np.mean(fraction))
        self.assertGreater(mean_fraction, 0.0003)  # 不为零
        self.assertLess(mean_fraction, 0.0017)  # 区间中点附近
        self.assertLessEqual(float(np.percentile(fraction, 97.5)), 0.0021)  # 上界受定量限约束

    def test_upper_mode_uses_limit(self):
        bundle = make_bundle(contaminant_lt=True)
        cal = build_calibration(bundle)
        model = build_lot_model(bundle, cal, "L1")
        upper = simulate(model, cal, SIMS, 7, censor_mode="upper")
        self.assertAlmostEqual(float(np.mean(upper.contaminant_fraction())), 0.002, delta=0.0002)


class AttributionTests(unittest.TestCase):
    def test_shares_sum_to_one_and_dry_is_moisture(self):
        bundle = load(ROOT)
        cal = build_calibration(bundle)
        result = attribute(bundle, cal, "L6", sims=SIMS, seed=11)
        for name, entry in result.items():
            total = sum(entry["share"].values())
            self.assertAlmostEqual(total, 1.0, places=6, msg=name)
        self.assertGreater(result["dry"]["share"]["moisture"], 0.95)


class RetestEvidenceTests(unittest.TestCase):
    def test_resample_adds_more_evidence_than_retest(self):
        bundle = load(ROOT)
        cal = build_calibration(bundle)
        actions = {a.target: a for a in candidate_actions(bundle, "L6")}
        resample = evaluate_action(bundle, cal, "L6", actions["CT6-0"], SIMS, 13)
        retest = evaluate_action(bundle, cal, "L6", actions["S6-0"], SIMS, 13)
        self.assertGreater(resample.width_gain, retest.width_gain)
        self.assertIn("不能宣称解决容器间不均匀", actions["S6-0"].note)


class AsOfTests(unittest.TestCase):
    def test_future_evidence_invisible(self):
        bundle = load(ROOT)
        before = bundle.as_of(dt.datetime(2025, 1, 14, 8, tzinfo=dt.timezone(dt.timedelta(hours=8))))
        after = bundle.as_of(dt.datetime(2025, 1, 16, 8, tzinfo=dt.timezone(dt.timedelta(hours=8))))
        self.assertNotIn("AS6-0R-moisture", before.assays)
        self.assertIn("AS6-0R-moisture", after.assays)

    def test_retest_narrows_dry_interval(self):
        bundle = load(ROOT)
        tz = dt.timezone(dt.timedelta(hours=8))
        widths = []
        for day in (14, 16):
            view = bundle.as_of(dt.datetime(2025, 1, day, 8, tzinfo=tz))
            cal = build_calibration(view)
            model = build_lot_model(view, cal, "L6")
            draws = simulate(model, cal, SIMS, 17)
            widths.append(float(np.percentile(draws.dry_full, 95) - np.percentile(draws.dry_full, 5)))
        self.assertLess(widths[1], widths[0])


if __name__ == "__main__":
    unittest.main()
