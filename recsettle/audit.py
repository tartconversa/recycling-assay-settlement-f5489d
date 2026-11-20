"""审计：事件时间线、差额来源分解与可调整性划分。

业务要求：
- 采购要能查到这批料为什么有这一笔差额；
- 生产要知道还能改动哪部分配料；
- 实验室要知道再取哪份样才有意义；
- 后来的水分复核或参考样偏差修订，要区分尚可改配的余料、已加工不能逆转的实物
  以及可追加或追偿的结算，不得用重算把已发生付款和加工抹去。
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .data import Bundle
from .estimate import DEFAULT_SEED, DEFAULT_SIMS, build_calibration, build_lot_model, simulate
from .settle import prices_at, settlement_draws


def timeline(bundle: Bundle, lot_id) -> list:
    """一批的事件时间线（到货、付款、投料、取样、化验可知、参考样可知）。"""
    lot = bundle.lots[lot_id]
    events = [{"at": lot.received.isoformat(), "kind": "到货", "detail": f"批 {lot_id} 到厂"}]
    for payment in sorted(bundle.lot_payments(lot_id), key=lambda p: p.paid_at):
        events.append({"at": payment.paid_at.isoformat(), "kind": "暂付款",
                       "detail": f"{payment.payment_id} 付 {payment.amount:.0f} 元"})
    for allocation in sorted(bundle.lot_allocations(lot_id), key=lambda a: a.started_at):
        events.append({"at": allocation.started_at.isoformat(), "kind": "投料",
                       "detail": f"{allocation.allocation_id} 投 {allocation.wet_kg:.0f} kg 湿料至 {allocation.line_id}（{allocation.state}）"})
    container_ids = {c.container_id for c in bundle.lot_containers(lot_id)}
    for sample in sorted(bundle.samples.values(), key=lambda s: (s.sampled, s.sample_id)):
        if sample.container_id in container_ids:
            kind = "留样复测取样" if sample.is_retest else "独立取样"
            events.append({"at": sample.sampled.isoformat(), "kind": kind,
                           "detail": f"{sample.sample_id}（{sample.container_id}，{sample.position}）"})
    for assay in sorted(bundle.assays.values(), key=lambda a: a.available_at):
        sample = bundle.samples.get(assay.sample_id)
        if sample and sample.container_id in container_ids:
            value = f"<{assay.limit}" if assay.censored else f"{assay.value}"
            events.append({"at": assay.available_at.isoformat(), "kind": "化验可知",
                           "detail": f"{assay.assay_id} {assay.analyte}={value}（{assay.lab_id}）"})
    events.sort(key=lambda e: e["at"])
    return events


def _value_at(bundle_full: Bundle, lot_id, when, sims, seed):
    """某可知时刻下的合同结算价值中位（元）。"""
    lot = bundle_full.lots[lot_id]
    contract = bundle_full.contracts[lot.contract_id]
    view = bundle_full.as_of(when)
    cal = build_calibration(view)
    model = build_lot_model(view, cal, lot_id)
    prices = prices_at(view, when.date())
    values = settlement_draws(simulate(model, cal, sims, seed), contract, prices)
    return float(np.percentile(values, 50))


def difference_decomposition(bundle_full: Bundle, lot_id, as_of, sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> dict:
    """把“当前估值 − 暂付时估值”的差额按证据到达顺序分解。

    顺序为时间顺序：每个可知时刻加入该时刻新到的全部证据（本批化验、他批化验与
    参考样——后两者经全资料校准参数影响本批估值），因此分解是路径依赖的，
    报告需注明；已发生的付款与投料不被重算抹去。
    """
    lot = bundle_full.lots[lot_id]
    payments = sorted(bundle_full.lot_payments(lot_id), key=lambda p: p.paid_at)
    t0 = payments[0].paid_at if payments else dt.datetime.combine(lot.received, dt.time(8, 0), dt.timezone(dt.timedelta(hours=8)))

    container_ids = {c.container_id for c in bundle_full.lot_containers(lot_id)}
    own_samples = {s.sample_id for s in bundle_full.samples.values() if s.container_id in container_ids}
    arrivals = {}
    for assay in bundle_full.assays.values():
        if t0 < assay.available_at <= as_of:
            label = f"本批化验 {assay.assay_id}" if assay.sample_id in own_samples else "他批化验（校准参数）"
            arrivals.setdefault(assay.available_at, []).append(label)
    for ref in bundle_full.references.values():
        if t0 < ref.available_at <= as_of:
            arrivals.setdefault(ref.available_at, []).append(f"参考样 {ref.reference_id}")

    base = _value_at(bundle_full, lot_id, t0, sims, seed)
    steps = []
    previous = base
    for when in sorted(arrivals):
        value = _value_at(bundle_full, lot_id, when, sims, seed)
        labels = arrivals[when]
        own = [x for x in labels if x.startswith("本批") or x.startswith("参考样")]
        other = sum(1 for x in labels if x.startswith("他批"))
        shown = own + ([f"他批化验×{other}（校准参数）"] if other else [])
        steps.append({"at": when.isoformat(), "evidence": shown,
                      "value_p50": value, "delta": value - previous})
        previous = value
    current = _value_at(bundle_full, lot_id, as_of, sims, seed)
    if abs(current - previous) > 1e-9 or not steps:
        steps.append({"at": as_of.isoformat(), "evidence": ["（截点估值）"],
                      "value_p50": current, "delta": current - previous})
    return {
        "lot_id": lot_id,
        "base_time": t0.isoformat(),
        "base_value_p50": base,
        "steps": steps,
        "current_value_p50": current,
        "total_delta": current - base,
        "note": "分解按证据到达的时间顺序，路径依赖；他批化验经全资料校准参数影响本批；已付款与已投料不被重算抹去",
    }


def adjustability(bundle: Bundle, lot_id, as_of=None) -> dict:
    """可调整性划分：已加工实物（不可逆转）/ 余料（可改配）/ 结算（可追加或追偿）。"""
    lot = bundle.lots[lot_id]
    containers = []
    processed_wet = 0.0
    remaining_wet = 0.0
    for container in bundle.lot_containers(lot_id):
        allocations = [a for a in bundle.allocations.values() if a.container_id == container.container_id]
        containers.append({
            "container_id": container.container_id,
            "wet_kg": container.wet_kg,
            "allocated_kg": container.allocated_kg,
            "remaining_kg": container.remaining_kg,
            "lines": sorted({a.line_id for a in allocations}),
        })
        processed_wet += container.allocated_kg
        remaining_wet += container.remaining_kg
    paid = sum(p.amount for p in bundle.lot_payments(lot_id))

    limits = []
    if processed_wet > 0:
        limits.append("加工已开始：已投实物不可逆转，仅余料可改配")
    if lot.state == "disputed":
        limits.append("争议中：最终结算须按合同认可依据完成双方认可复检后才能定案")
    if lot.state == "accepted":
        limits.append("供应商已接受：结算口径冻结，不能再调整")
    if lot.state == "arbitration":
        limits.append("仲裁中：只可按仲裁结果调整")
    if not limits:
        limits.append("余料可改配，结算未定案")

    return {
        "lot_id": lot_id,
        "state": lot.state,
        "containers": containers,
        "processed_wet_kg": processed_wet,
        "remaining_wet_kg": remaining_wet,
        "paid_yuan": paid,
        "adjustable": {
            "已加工实物": "不可逆转" if processed_wet > 0 else "无",
            "余料": "可改配" if remaining_wet > 0 and lot.state not in ("accepted", "arbitration") else "不可",
            "结算": "可追加或追偿（暂付款只抵扣不重付）" if lot.state not in ("accepted",) else "冻结",
        },
        "limits": limits,
    }
