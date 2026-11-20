"""加工决策：经验回收、支持范围、硬边界与方案比较。"""
import unittest
from pathlib import Path

import numpy as np

from recsettle.data import load
from recsettle.estimate import build_calibration, build_lot_model, simulate
from recsettle.process import (
    ACCEPT_REJECT,
    ACCEPT_RETEST,
    build_candidates,
    evaluate_batch,
    evaluate_portfolio,
    fit_line_evidence,
    remaining_wet,
)
from recsettle.settle import prices_at, settlement_draws

ROOT = Path(__file__).resolve().parents[1]
SIMS = 2000


def repo_context():
    bundle = load(ROOT)
    cal = build_calibration(bundle)
    draws = {lot: simulate(build_lot_model(bundle, cal, lot), cal, SIMS, 29) for lot in bundle.lots}
    prices = prices_at(bundle)
    due = {}
    for lot_id in bundle.lots:
        contract = bundle.contracts[bundle.lots[lot_id].contract_id]
        paid = sum(p.amount for p in bundle.lot_payments(lot_id))
        due[lot_id] = settlement_draws(draws[lot_id], contract, prices) - paid
    evidence = {lid: fit_line_evidence(bundle, lid) for lid in bundle.lines}
    return bundle, cal, draws, due, evidence, prices


class EvidenceTests(unittest.TestCase):
    def test_recovery_rate_in_observed_range(self):
        bundle = load(ROOT)
        evidence = fit_line_evidence(bundle, "LINE1")
        x = np.array([1.0, 0.13, 0.026, 0.0018])
        rate = float(x @ evidence.beta["nickel"])
        self.assertGreater(rate, 0.75)
        self.assertLess(rate, 0.97)

    def test_out_of_support_is_scenario(self):
        bundle = load(ROOT)
        evidence = fit_line_evidence(bundle, "LINE1")
        self.assertEqual(evidence.supported_dims((0.13, 0.026, 0.0018)), [])
        self.assertTrue(evidence.supported_dims((0.20, 0.026, 0.0018)))  # 镍越界
        self.assertTrue(evidence.supported_dims((0.13, 0.026, 0.0100)))  # 污染物越界


class BatchTests(unittest.TestCase):
    def test_hard_limit_rejects_high_contaminant(self):
        bundle, cal, draws, due, evidence, prices = repo_context()
        rem = remaining_wet(bundle)
        # L3 污染物中位约 0.004，破 LINE2 硬界 0.003
        result = evaluate_batch([("L3", rem["L3"])], bundle.lines["LINE2"], evidence["LINE2"],
                                draws, rem, prices, SIMS, 31)
        self.assertEqual(result.acceptance, ACCEPT_REJECT)
        # 同一批在 LINE1（硬界 0.004）处于需复检或不可投入的边界状态，但不得判为可投入
        borderline = evaluate_batch([("L3", rem["L3"])], bundle.lines["LINE1"], evidence["LINE1"],
                                    draws, rem, prices, SIMS, 31)
        self.assertNotEqual(borderline.acceptance, "可投入")

    def test_cost_includes_fixed_and_variable(self):
        bundle, cal, draws, due, evidence, prices = repo_context()
        rem = remaining_wet(bundle)
        result = evaluate_batch([("L7", rem["L7"])], bundle.lines["LINE1"], evidence["LINE1"],
                                draws, rem, prices, SIMS, 31)
        expected = bundle.lines["LINE1"].cost_fixed + evidence["LINE1"].variable_rate * rem["L7"]
        self.assertAlmostEqual(result.cost, expected, places=6)

    def test_portfolio_totals_add_up(self):
        bundle, cal, draws, due, evidence, prices = repo_context()
        portfolios = build_candidates(bundle, cal, draws, due, evidence, prices, SIMS, 37)
        stockpile = next(p for p in portfolios if p.name == "全部暂存")
        expected = -sum(float(np.mean(d)) for d in due.values())
        self.assertAlmostEqual(float(np.mean(stockpile.total)), expected, places=4)
        # 至少有一个加工方案优于全部暂存（加工批净额为正）
        best = max(portfolios, key=lambda p: float(np.mean(p.total)))
        self.assertGreater(float(np.mean(best.total)), float(np.mean(stockpile.total)))

    def test_grade_first_not_chosen_over_margin_first(self):
        bundle, cal, draws, due, evidence, prices = repo_context()
        portfolios = build_candidates(bundle, cal, draws, due, evidence, prices, SIMS, 37)
        values = {p.name: float(np.mean(p.total)) for p in portfolios}
        self.assertGreaterEqual(values["边际优先（污染物感知）"], values["品位优先（朴素基线）"])


if __name__ == "__main__":
    unittest.main()
