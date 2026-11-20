"""复检规划：预算、期限、样品量约束与证据类型差异。"""
import datetime as dt
import unittest
from pathlib import Path

from recsettle.data import load
from recsettle.estimate import build_calibration
from recsettle.plan import candidate_actions, recommend
from tiny import make_bundle

ROOT = Path(__file__).resolve().parents[1]
SIMS = 1000
TZ = dt.timezone(dt.timedelta(hours=8))
AS_OF = dt.datetime(2025, 2, 10, 8, 0, tzinfo=TZ)


class ConstraintTests(unittest.TestCase):
    def test_budget_limits_choice(self):
        bundle = load(ROOT)
        cal = build_calibration(bundle)
        result = recommend(bundle, cal, AS_OF, budget=350.0, deadline_days=None,
                           lot_ids=["L6"], sims=SIMS, seed=41)
        self.assertLessEqual(result["spent"], 350.0)

    def test_deadline_marks_infeasible(self):
        bundle = load(ROOT)
        cal = build_calibration(bundle)
        result = recommend(bundle, cal, AS_OF, budget=10000.0, deadline_days=2,
                           lot_ids=["L6"], sims=SIMS, seed=41)
        kinds = {e.action.kind for e in result["chosen"]}
        self.assertNotIn("resample", kinds)  # 重取耗时 4 天，超 2 天期限
        self.assertTrue(any(e.action.kind == "resample" for e in result["infeasible"]))

    def test_grams_limit_blocks_retest(self):
        bundle = make_bundle()
        # 把留样余量改为 50 g，低于留样复测所需 100 g
        for sid, sample in bundle.samples.items():
            bundle.samples[sid] = type(sample)(sample.sample_id, sample.container_id, sample.position,
                                               sample.sampled, 50.0, sample.retest_of)
        actions = candidate_actions(bundle, "L1")
        self.assertFalse(any(a.kind == "retained" for a in actions))
        self.assertTrue(any(a.kind == "resample" for a in actions))

    def test_unsampled_container_is_candidate(self):
        bundle = load(ROOT)
        targets = {a.target for a in candidate_actions(bundle, "L1") if a.kind == "resample"}
        self.assertIn("CT1-2", targets)  # 未取样容器应可重取


class ValueTests(unittest.TestCase):
    def test_retest_cannot_fix_heterogeneity_note(self):
        bundle = load(ROOT)
        notes = [a.note for a in candidate_actions(bundle, "L6") if a.kind == "retained"]
        self.assertTrue(all("不能宣称解决容器间不均匀" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
