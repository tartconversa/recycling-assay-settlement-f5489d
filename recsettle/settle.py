"""合同结算：最终应付、暂付抵扣、争议状态与资金占用。

结算口径（docs/资料口径.md）：
最终应付 = Σ 干基金属量 × 当日价格 × payable_fraction − 干基处理费 − 合同罚项；
罚项 = max(0, 污染物干基比例 − 软界) × 干基量 × 罚单价；
暂付款在最终应付中抵扣，负差额为待追偿，已付款不在最终价值之外再支付一次。
"""
from __future__ import annotations

import numpy as np

from .data import METALS, Bundle
from .estimate import (
    DEFAULT_SEED,
    DEFAULT_SIMS,
    Calibration,
    LotDraws,
    build_calibration,
    build_lot_model,
    simulate,
    _stats,
)


def settlement_draws(draws: LotDraws, contract, prices: dict) -> np.ndarray:
    """每次模拟的合同最终应付（元）。prices: analyte -> 元/kg。"""
    value = np.zeros(draws.sims)
    for metal in METALS:
        value = value + draws.qty_full[metal] * prices[metal] * contract.payable_fraction
    value = value - contract.processing_fee_kg * draws.dry_full
    excess_fraction = np.maximum(0.0, draws.contaminant_fraction() - contract.soft_contaminant_limit)
    value = value - excess_fraction * draws.dry_full * contract.contaminant_penalty_kg
    return value


def penalty_draws(draws: LotDraws, contract) -> np.ndarray:
    """每次模拟的污染物罚项（元），单列以便复检规划。"""
    excess_fraction = np.maximum(0.0, draws.contaminant_fraction() - contract.soft_contaminant_limit)
    return excess_fraction * draws.dry_full * contract.contaminant_penalty_kg


def prices_at(bundle: Bundle, day=None) -> dict:
    """各金属不晚于 day 的最新价格；缺价时报错而不是按零计。"""
    prices = {}
    for metal in METALS:
        price = bundle.price_on(metal, day)
        if price is None:
            raise ValueError(f"无可用价格：{metal}（{day}）")
        prices[metal] = price.yuan_kg
    return prices


def statement(bundle: Bundle, cal: Calibration, lot_id, as_of=None, sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> dict:
    """一批的结算单：估值区间、已付、应付/待追偿、暂付上限核对与状态说明。"""
    lot = bundle.lots[lot_id]
    contract = bundle.contracts[lot.contract_id]
    prices = prices_at(bundle, as_of.date() if as_of else None)
    model = build_lot_model(bundle, cal, lot_id)
    draws = simulate(model, cal, sims, seed)
    upper_draws = simulate(model, cal, sims, seed, censor_mode="upper")
    values = settlement_draws(draws, contract, prices)
    paid = sum(p.amount for p in bundle.lot_payments(lot_id))
    due = values - paid

    payments = sorted(bundle.lot_payments(lot_id), key=lambda p: p.paid_at)
    cap_check = None
    if payments:
        first = payments[0]
        past = bundle.as_of(first.paid_at)
        past_cal = build_calibration(past, cal.within_share)
        past_model = build_lot_model(past, past_cal, lot_id)
        past_prices = prices_at(past, first.paid_at.date())
        past_values = settlement_draws(simulate(past_model, past_cal, sims, seed), contract, past_prices)
        cap = contract.provisional_fraction * float(np.percentile(past_values, 50))
        cap_check = {
            "paid": paid,
            "cap": cap,
            "ok": paid <= cap * 1.02,
            "basis_time": first.paid_at.isoformat(),
        }

    notes = []
    if lot.state == "disputed":
        notes.append(f"争议中：最终结算须按合同认可依据（{contract.accepted_basis}）完成双方认可复检")
    if lot.state == "arbitration":
        notes.append("仲裁中：结算口径冻结，只可按仲裁结果调整")
    if bundle.lot_allocations(lot_id):
        notes.append("已部分投料：已投实物不可逆转，结算仍按合同口径对全批计算")

    return {
        "lot_id": lot_id,
        "contract_id": lot.contract_id,
        "state": lot.state,
        "prices": prices,
        "dry": _stats(draws.dry_full),
        "quantities": {a: _stats(v) for a, v in draws.qty_full.items()},
        "contaminant_upper": float(np.percentile(upper_draws.contaminant_fraction(), 97.5)),
        "value": _stats(values),
        "penalty": _stats(penalty_draws(draws, contract)),
        "paid": paid,
        "due": _stats(due),
        "due_mean": float(np.mean(due)),
        "recover": max(0.0, -float(np.mean(due))),
        "capital_tied": max(0.0, paid - float(np.percentile(values, 5))),
        "provisional_cap": cap_check,
        "notes": notes,
    }
