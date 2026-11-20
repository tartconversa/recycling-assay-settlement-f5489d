"""一致性检查：覆盖已付款、已投料与新化验相遇的情况。

每条检查返回 CheckResult（ok / warn / fail）：
- 质量平衡：已投湿重不超过容器净重，投入记录与容器台账一致；
- 在投硬界：已投料的批按当前证据的污染物上界是否破线硬界（不可撤销，需关注）；
- 暂付上限：已付款不超过合同暂付比例 × 付款时点可知证据下的估值中位；
- 计划完整性：新计划不重复承诺同一实物、不超线容量与批次数、不违反硬界；
- 复检承诺：争议待复检的批不同时承诺给两条线。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Bundle
from .estimate import DEFAULT_SEED, DEFAULT_SIMS, build_calibration, build_lot_model, simulate
from .process import evaluate_batch, fit_line_evidence, remaining_wet
from .settle import prices_at, settlement_draws


@dataclass
class CheckResult:
    check_id: str
    title: str
    status: str  # ok / warn / fail
    detail: str


def check_mass_balance(bundle: Bundle) -> CheckResult:
    bad = []
    allocated = {}
    for allocation in bundle.allocations.values():
        allocated[allocation.container_id] = allocated.get(allocation.container_id, 0.0) + allocation.wet_kg
    for container in bundle.containers.values():
        summed = allocated.get(container.container_id, 0.0)
        if abs(summed - container.allocated_kg) > 1e-6:
            bad.append(f"{container.container_id} 台账 {container.allocated_kg} ≠ 投入合计 {summed}")
        if container.allocated_kg > container.wet_kg + 1e-9:
            bad.append(f"{container.container_id} 已投 {container.allocated_kg} 超过净重 {container.wet_kg}")
    if bad:
        return CheckResult("mass_balance", "质量平衡", "fail", "；".join(bad))
    return CheckResult("mass_balance", "质量平衡", "ok", f"{len(bundle.containers)} 个容器台账与投入记录一致")


def check_running_hard_limits(bundle: Bundle, sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> CheckResult:
    """已投料的批按当前证据（未检出按定量限计上界）是否破线硬界。"""
    cal = build_calibration(bundle)
    flagged = []
    for lot_id in sorted(bundle.lots):
        allocations = bundle.lot_allocations(lot_id)
        if not allocations:
            continue
        model = build_lot_model(bundle, cal, lot_id)
        upper_draws = simulate(model, cal, sims, seed, censor_mode="upper")
        upper = float(np.percentile(upper_draws.contaminant_fraction(), 97.5))
        for line_id in sorted({a.line_id for a in allocations}):
            hard = bundle.lines[line_id].hard_contaminant_limit
            if upper > hard:
                flagged.append(f"{lot_id} 已投 {line_id}，污染物上界 {upper:.4f} 破硬界 {hard}")
    if flagged:
        return CheckResult("running_hard_limit", "在投硬界", "warn",
                           "；".join(flagged) + "（已投不可撤销，需关注后续实绩）")
    return CheckResult("running_hard_limit", "在投硬界", "ok", "已投料各批污染物上界均未破线硬界")


def check_payment_caps(bundle: Bundle, sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> CheckResult:
    """已付款不超过 合同暂付比例 × 付款时点可知证据下的估值中位（容差 2%）。"""
    flagged = []
    for lot_id in sorted(bundle.lots):
        lot = bundle.lots[lot_id]
        contract = bundle.contracts[lot.contract_id]
        for payment in bundle.lot_payments(lot_id):
            past = bundle.as_of(payment.paid_at)
            cal = build_calibration(past)
            model = build_lot_model(past, cal, lot_id)
            prices = prices_at(past, payment.paid_at.date())
            values = settlement_draws(simulate(model, cal, sims, seed), contract, prices)
            cap = contract.provisional_fraction * float(np.percentile(values, 50))
            if payment.amount > cap * 1.02:
                flagged.append(f"{payment.payment_id} 付 {payment.amount:.0f} 超上限 {cap:.0f}")
    if flagged:
        return CheckResult("payment_cap", "暂付上限", "fail", "；".join(flagged))
    return CheckResult("payment_cap", "暂付上限", "ok", "各笔暂付款均在合同暂付上限内")


def check_plan(bundle: Bundle, plan: dict, sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> list:
    """校验一个待承诺计划：不重复承诺同一实物、不超容量与批次数、不违反硬界。"""
    results = []
    rem = remaining_wet(bundle)
    used = {}
    batches = plan.get("batches", [])
    per_line = {}
    for i, batch in enumerate(batches):
        line_id = batch["line_id"]
        per_line[line_id] = per_line.get(line_id, 0) + 1
        wet = 0.0
        for item in batch["items"]:
            lot_id, wet_kg = item["lot_id"], float(item["wet_kg"])
            used[lot_id] = used.get(lot_id, 0.0) + wet_kg
            wet += wet_kg
        line = bundle.lines[line_id]
        if wet > line.capacity_wet_kg + 1e-9:
            results.append(CheckResult("plan_capacity", "计划容量", "fail",
                                       f"批 {i} 投 {wet:.0f} kg 超 {line_id} 容量 {line.capacity_wet_kg:.0f}"))
    for line_id, count in per_line.items():
        if count > bundle.lines[line_id].free_batches:
            results.append(CheckResult("plan_batches", "计划批次数", "fail",
                                       f"{line_id} 计划 {count} 批超可用 {bundle.lines[line_id].free_batches} 批"))
    double = [f"{lot} 承诺 {wet:.0f} 超余料 {rem.get(lot, 0):.0f}" for lot, wet in used.items() if wet > rem.get(lot, 0.0) + 1e-9]
    if double:
        results.append(CheckResult("plan_double_commit", "重复承诺", "fail", "；".join(double)))
    else:
        results.append(CheckResult("plan_double_commit", "重复承诺", "ok", "各批承诺量均未超余料，同一实物未重复承诺"))

    disputed_committed = []
    for lot_id in used:
        if bundle.lots[lot_id].state == "disputed":
            lines = {b["line_id"] for b in batches for it in b["items"] if it["lot_id"] == lot_id}
            if len(lines) > 1:
                disputed_committed.append(f"{lot_id} 同时承诺给 {sorted(lines)}")
            else:
                disputed_committed.append(f"{lot_id} 争议待复检已承诺 {sorted(lines)}")
    if disputed_committed:
        results.append(CheckResult("plan_disputed", "争议批承诺", "warn", "；".join(disputed_committed)))
    else:
        results.append(CheckResult("plan_disputed", "争议批承诺", "ok", "无争议批被承诺"))

    # 硬界与证据支持：逐批评估
    cal = build_calibration(bundle)
    draws_by_lot = {}
    for lot_id in used:
        model = build_lot_model(bundle, cal, lot_id)
        draws_by_lot[lot_id] = simulate(model, cal, sims, seed)
    prices = prices_at(bundle)
    hard_bad, scenario_note = [], []
    for i, batch in enumerate(batches):
        line = bundle.lines[batch["line_id"]]
        evidence = fit_line_evidence(bundle, batch["line_id"])
        items = [(it["lot_id"], float(it["wet_kg"])) for it in batch["items"]]
        result = evaluate_batch(items, line, evidence, draws_by_lot, rem, prices, sims, seed + i)
        if result.acceptance == "不可投入":
            hard_bad.append(f"批 {i}（{line.line_id}）污染物中位破硬界")
        elif result.acceptance == "需复检":
            hard_bad.append(f"批 {i}（{line.line_id}）污染物上界破硬界，需复检确认")
        if not result.supported:
            scenario_note.append(f"批 {i}（{line.line_id}）" + "；".join(result.scenario_reasons))
    if hard_bad:
        results.append(CheckResult("plan_hard_limit", "计划硬界", "fail", "；".join(hard_bad)))
    else:
        results.append(CheckResult("plan_hard_limit", "计划硬界", "ok", "各加工批污染物上界均未破硬界"))
    if scenario_note:
        results.append(CheckResult("plan_scenario", "证据支持范围", "warn",
                                   "；".join(scenario_note) + "（超出部分按情景处理）"))
    else:
        results.append(CheckResult("plan_scenario", "证据支持范围", "ok", "各加工批配料均在历史支持范围内"))
    if not any(r.check_id == "plan_capacity" for r in results):
        results.append(CheckResult("plan_capacity", "计划容量", "ok", "各加工批未超线容量"))
    if not any(r.check_id == "plan_batches" for r in results):
        results.append(CheckResult("plan_batches", "计划批次数", "ok", "各线计划批次数未超可用批数"))
    return results


def run_checks(bundle: Bundle, plan: dict | None = None, sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> list:
    results = [
        check_mass_balance(bundle),
        check_running_hard_limits(bundle, sims, seed),
        check_payment_caps(bundle, sims, seed),
    ]
    if plan is not None:
        results.extend(check_plan(bundle, plan, sims, seed))
    return results
