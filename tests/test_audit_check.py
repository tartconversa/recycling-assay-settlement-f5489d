"""审计与检查：差额追溯、可调整性划分与一致性核对。"""
import datetime as dt
import unittest
from pathlib import Path

from recsettle.audit import adjustability, difference_decomposition, timeline
from recsettle.checks import check_mass_balance, check_payment_caps, check_plan, check_running_hard_limits
from recsettle.data import load

ROOT = Path(__file__).resolve().parents[1]
SIMS = 1000
TZ = dt.timezone(dt.timedelta(hours=8))
NOW = dt.datetime(2025, 2, 20, 8, 0, tzinfo=TZ)


class AuditTests(unittest.TestCase):
    def test_processed_and_remaining_split(self):
        bundle = load(ROOT)
        adj = adjustability(bundle, "L6")
        self.assertAlmostEqual(adj["processed_wet_kg"], 300.0)
        self.assertAlmostEqual(adj["remaining_wet_kg"], 900.0)
        self.assertEqual(adj["adjustable"]["已加工实物"], "不可逆转")
        self.assertEqual(adj["adjustable"]["余料"], "可改配")

    def test_decomposition_sums_to_total(self):
        bundle = load(ROOT)
        decomp = difference_decomposition(bundle, "L6", NOW, SIMS, 43)
        accumulated = decomp["base_value_p50"] + sum(s["delta"] for s in decomp["steps"])
        self.assertAlmostEqual(accumulated, decomp["current_value_p50"], places=6)
        self.assertAlmostEqual(decomp["total_delta"],
                               decomp["current_value_p50"] - decomp["base_value_p50"], places=6)

    def test_timeline_covers_payment_assay_and_allocation(self):
        bundle = load(ROOT)
        kinds = {e["kind"] for e in timeline(bundle, "L6")}
        self.assertIn("暂付款", kinds)
        self.assertIn("化验可知", kinds)
        self.assertIn("投料", kinds)
        self.assertIn("留样复测取样", kinds)


class CheckTests(unittest.TestCase):
    def test_repo_data_consistency(self):
        bundle = load(ROOT)
        self.assertEqual(check_mass_balance(bundle).status, "ok")
        self.assertEqual(check_payment_caps(bundle, SIMS, 47).status, "ok")

    def test_running_hard_limit_flags_L3(self):
        bundle = load(ROOT)
        result = check_running_hard_limits(bundle, SIMS, 47)
        self.assertEqual(result.status, "warn")
        self.assertIn("L3", result.detail)

    def test_plan_double_commit_fails(self):
        bundle = load(ROOT)
        plan = {"name": "t", "batches": [
            {"line_id": "LINE1", "items": [{"lot_id": "L1", "wet_kg": 900.0}]},
            {"line_id": "LINE2", "items": [{"lot_id": "L1", "wet_kg": 900.0}]},
        ]}
        results = check_plan(bundle, plan, SIMS, 47)
        double = next(r for r in results if r.check_id == "plan_double_commit")
        self.assertEqual(double.status, "fail")  # 同一批实物被承诺给两条线且超余料

    def test_plan_over_capacity_fails(self):
        bundle = load(ROOT)
        plan = {"name": "t", "batches": [
            {"line_id": "LINE2", "items": [{"lot_id": "L1", "wet_kg": 1125.0},
                                           {"lot_id": "L2", "wet_kg": 1140.0},
                                           {"lot_id": "L4", "wet_kg": 1170.0}]},
        ]}
        results = check_plan(bundle, plan, SIMS, 47)
        capacity = next(r for r in results if r.check_id == "plan_capacity")
        self.assertEqual(capacity.status, "fail")  # 3435 kg 超 LINE2 容量 2500


if __name__ == "__main__":
    unittest.main()
