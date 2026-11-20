"""加工决策：组合加工、独立处理与暂存的净回收价值与下行损失比较。

证据边界（docs/资料口径.md）：
- 加工实绩是观测，不给理论化学回收公式；回收率只按历史配料范围的经验模型给出，
  超出支持范围的配料只给“情景”，不当作证据；
- 超过 hard_contaminant_limit 不可投入对应线，不能用收益抵消；
- 同一批实物不能同时承诺给两条线：计划只使用余料（湿重 − 已投），
  已投部分保持原状，不参与新计划。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import METALS, Bundle
from .estimate import DEFAULT_SEED, DEFAULT_SIMS, Calibration, LotDraws, _stats
from .settle import prices_at, settlement_draws

ACCEPT_OK = "可投入"
ACCEPT_RETEST = "需复检"
ACCEPT_REJECT = "不可投入"


@dataclass(frozen=True)
class LineEvidence:
    """一条线的历史加工经验：回收率经验模型、支持范围与单位变动成本。"""

    line_id: str
    beta: dict  # metal -> 回归系数 [1, ni, li, cont]
    resid_sd: dict  # metal -> 残差标准差
    xtx_inv: np.ndarray
    support: dict  # analyte -> (min, max)
    variable_rate: float  # 元/湿kg（历史成本扣除批固定费后）

    def predict_sd(self, metal, blend):
        x = np.array([1.0, blend[0], blend[1], blend[2]])
        leverage = max(0.0, float(x @ self.xtx_inv @ x))
        return float(self.resid_sd[metal] * math.sqrt(1.0 + leverage))

    def supported_dims(self, blend):
        """逐维检查配料是否落在历史支持范围内，返回越界说明列表。"""
        bad = []
        for analyte, value in zip(("nickel", "lithium", "contaminant"), blend):
            lo, hi = self.support[analyte]
            if value < lo or value > hi:
                bad.append(f"{analyte} {value:.4f} 超出历史支持 [{lo:.4f}, {hi:.4f}]")
        return bad


def fit_line_evidence(bundle: Bundle, line_id) -> LineEvidence | None:
    runs = [r for r in bundle.runs.values() if r.line_id == line_id]
    if len(runs) < 5:
        return None
    line = bundle.lines[line_id]
    x = np.array([[1.0, r.nickel_fraction, r.lithium_fraction, r.contaminant_fraction] for r in runs])
    # 历史配料中镍与污染物完全共线（合成资料），X'X 不可逆；
    # 用伪逆得最小范数解：支持范围内预测有效，越界方向只作情景。
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta, resid_sd = {}, {}
    for metal, recovered in (("nickel", "recovered_nickel_kg"), ("lithium", "recovered_lithium_kg")):
        y = np.array([getattr(r, recovered) / (r.dry_kg * getattr(r, f"{metal}_fraction")) for r in runs])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        beta[metal] = coef
        resid_sd[metal] = float(np.std(resid, ddof=x.shape[1]))
    support = {
        "nickel": (min(r.nickel_fraction for r in runs), max(r.nickel_fraction for r in runs)),
        "lithium": (min(r.lithium_fraction for r in runs), max(r.lithium_fraction for r in runs)),
        "contaminant": (min(r.contaminant_fraction for r in runs), max(r.contaminant_fraction for r in runs)),
    }
    variable_rate = float(np.mean([(r.cost - line.cost_fixed) / r.wet_kg for r in runs]))
    return LineEvidence(line_id, beta, resid_sd, xtx_inv, support, variable_rate)


@dataclass
class BatchResult:
    """一个加工批（一条线上的一次投料组合）的评估结果。"""

    line_id: str
    items: tuple  # ((lot_id, wet_kg), ...)
    wet: float
    acceptance: str
    scenario_reasons: tuple
    blend_p50: dict
    contaminant_p975: float
    revenue: np.ndarray
    cost: float
    net: np.ndarray

    @property
    def supported(self):
        return not self.scenario_reasons

    def summary(self):
        return {
            "line": self.line_id,
            "items": [(lot, round(w, 1)) for lot, w in self.items],
            "wet": round(self.wet, 1),
            "acceptance": self.acceptance,
            "supported": self.supported,
            "scenario": list(self.scenario_reasons),
            "blend_p50": {k: round(v, 5) for k, v in self.blend_p50.items()},
            "contaminant_p975": round(self.contaminant_p975, 5),
            "revenue": _stats(self.revenue),
            "cost": round(self.cost, 1),
            "net": _stats(self.net),
        }


def evaluate_batch(items, line, evidence: LineEvidence, draws_by_lot: dict, rem_wet_by_lot: dict,
                   prices: dict, sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> BatchResult:
    """评估一个加工批：配料分布 → 经验回收 → 收入 − 成本（含批固定费）。"""
    rng = np.random.default_rng(seed)
    dry = np.zeros(sims)
    qty = {a: np.zeros(sims) for a in ("nickel", "lithium", "contaminant")}
    for lot_id, wet_used in items:
        draws: LotDraws = draws_by_lot[lot_id]
        scale = wet_used / rem_wet_by_lot[lot_id]
        dry = dry + draws.dry_rem * scale
        for analyte in qty:
            qty[analyte] = qty[analyte] + draws.qty_rem[analyte] * scale
    fractions = {a: qty[a] / dry for a in qty}
    blend_p50 = {a: float(np.percentile(f, 50)) for a, f in fractions.items()}
    blend_tuple = (blend_p50["nickel"], blend_p50["lithium"], blend_p50["contaminant"])

    x_sim = np.column_stack([np.ones(sims), fractions["nickel"], fractions["lithium"], fractions["contaminant"]])
    revenue = np.zeros(sims)
    for metal in METALS:
        sd = evidence.predict_sd(metal, blend_tuple)
        rate = x_sim @ evidence.beta[metal] + rng.normal(0.0, sd, sims)
        rate = np.clip(rate, 0.0, 0.995)
        revenue = revenue + dry * fractions[metal] * rate * prices[metal]

    wet_total = float(sum(w for _, w in items))
    cost = line.cost_fixed + evidence.variable_rate * wet_total
    net = revenue - cost

    cont_p975 = float(np.percentile(fractions["contaminant"], 97.5))
    hard = line.hard_contaminant_limit
    if blend_p50["contaminant"] > hard:
        acceptance = ACCEPT_REJECT
    elif cont_p975 > hard:
        acceptance = ACCEPT_RETEST
    else:
        acceptance = ACCEPT_OK

    scenario = tuple(evidence.supported_dims(blend_tuple))
    return BatchResult(
        line.line_id, tuple(items), wet_total, acceptance, scenario,
        blend_p50, cont_p975, revenue, cost, net,
    )


@dataclass
class Portfolio:
    """一个完整加工方案：若干加工批 + 未投部分暂存。组合价值已含全部批的结算净额。"""

    name: str
    note: str
    batches: list
    total: np.ndarray  # Σ 加工批净额 − Σ 各批最终应付差额（暂存部分只承担后者）

    def summary(self):
        return {
            "name": self.name,
            "note": self.note,
            "batches": [b.summary() for b in self.batches],
            "total": _stats(self.total),
        }


def evaluate_portfolio(name, note, batches, due_draws_by_lot) -> Portfolio:
    """组合价值 = Σ 加工批净额 − Σ 批最终应付差额（已付抵扣后）。

    结算义务与加工去向无关（合同对全批成立），因此每个批的差额只计一次；
    不同方案在相同批集合上可比。
    """
    first = batches[0].net if batches else next(iter(due_draws_by_lot.values()))
    total = np.zeros_like(first)
    for batch in batches:
        total = total + batch.net
    for due in due_draws_by_lot.values():
        total = total - due
    return Portfolio(name, note, batches, total)


def remaining_wet(bundle: Bundle) -> dict:
    """每批余料湿重（已投部分不可再用）。"""
    out = {}
    for lot_id in bundle.lots:
        out[lot_id] = sum(c.remaining_kg for c in bundle.lot_containers(lot_id))
    return out


def _unit_margin(draws: LotDraws, evidence: LineEvidence, prices: dict, due_mean: float, rem_wet: float) -> float:
    """每湿 kg 余料的期望边际（回收收入 − 变动成本 − 结算差额），用于排序。"""
    if rem_wet <= 0:
        return float("-inf")
    blend = (
        float(np.mean(draws.qty_rem["nickel"] / draws.dry_rem)),
        float(np.mean(draws.qty_rem["lithium"] / draws.dry_rem)),
        float(np.mean(draws.qty_rem["contaminant"] / draws.dry_rem)),
    )
    x = np.array([1.0, *blend])
    revenue = 0.0
    for metal in METALS:
        rate = float(np.clip(x @ evidence.beta[metal], 0.0, 0.995))
        revenue += float(np.mean(draws.qty_rem[metal])) * rate * prices[metal]
    return (revenue - evidence.variable_rate * rem_wet - due_mean) / rem_wet


def build_candidates(bundle: Bundle, cal: Calibration, draws_by_lot: dict, due_by_lot: dict,
                     evidence_by_line: dict, prices: dict, sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> list:
    """构造并评估候选方案：全部暂存、品位优先（朴素）、边际优先（污染物感知）、均衡配料。"""
    rem = remaining_wet(bundle)
    lots = [lot_id for lot_id in bundle.lots if rem[lot_id] > 1e-9]
    lines = {lid: bundle.lines[lid] for lid in evidence_by_line}
    due_mean = {lot: float(np.mean(d)) for lot, d in due_by_lot.items()}
    cont_p50 = {lot: float(np.percentile(draws_by_lot[lot].contaminant_fraction(remaining=True), 50)) for lot in lots}
    ni_grade = {lot: float(np.percentile(draws_by_lot[lot].qty_rem["nickel"] / draws_by_lot[lot].dry_rem, 50)) for lot in lots}

    portfolios = [evaluate_portfolio("全部暂存", "不新投任何加工批，仅承担结算差额", [], due_by_lot)]

    def make_batches(order, respect_contaminant):
        """按顺序把整批余料装入加工批；respect_contaminant 时跳过会破硬界的批。"""
        batches = []
        used = set()
        for line_id, line in lines.items():
            for _ in range(line.free_batches):
                current, wet = [], 0.0
                for lot_id in order:
                    if lot_id in used or rem[lot_id] <= 1e-9:
                        continue
                    if wet + rem[lot_id] > line.capacity_wet_kg:
                        continue
                    if respect_contaminant:
                        trial = current + [lot_id]
                        blend_cont = np.average([cont_p50[t] for t in trial], weights=[rem[t] for t in trial])
                        if blend_cont > line.hard_contaminant_limit:
                            continue
                    current.append(lot_id)
                    wet += rem[lot_id]
                if current:
                    batches.append((line_id, [(lot, rem[lot]) for lot in current]))
                    used.update(current)
        return batches

    def to_results(batches):
        results = []
        for i, (line_id, items) in enumerate(batches):
            results.append(evaluate_batch(items, lines[line_id], evidence_by_line[line_id],
                                          draws_by_lot, rem, prices, sims, seed + 101 + i))
        return results

    # 品位优先（朴素基线）：按镍品位从高到低投料，不看污染物
    naive_order = sorted(lots, key=lambda lot: -ni_grade[lot])
    naive = to_results(make_batches(naive_order, respect_contaminant=False))
    portfolios.append(evaluate_portfolio(
        "品位优先（朴素基线）", "只按镍品位排序投料；超硬界的批不可执行，仅作对照", naive, due_by_lot))

    # 边际优先（污染物感知）：按单位边际排序，装入时保持配料污染物不破硬界
    main_line = max(evidence_by_line, key=lambda lid: lines[lid].capacity_wet_kg)
    margin = {lot: _unit_margin(draws_by_lot[lot], evidence_by_line[main_line], prices, due_mean[lot], rem[lot]) for lot in lots}
    margin_order = sorted(lots, key=lambda lot: -margin[lot])
    greedy = to_results(make_batches(margin_order, respect_contaminant=True))
    portfolios.append(evaluate_portfolio(
        "边际优先（污染物感知）", "按单位边际排序，装批时保持配料污染物中位不破硬界", greedy, due_by_lot))

    # 均衡配料：高低污染物批搭配，使每个加工批配料接近历史支持中部
    by_cont = sorted(lots, key=lambda lot: cont_p50[lot])
    interleaved = []
    lo, hi = 0, len(by_cont) - 1
    while lo <= hi:
        interleaved.append(by_cont[hi])
        if lo != hi:
            interleaved.append(by_cont[lo])
        hi -= 1
        lo += 1
    balanced = to_results(make_batches(interleaved, respect_contaminant=True))
    portfolios.append(evaluate_portfolio(
        "均衡配料", "高低污染物批搭配装批，配料尽量落在历史支持范围内", balanced, due_by_lot))

    return portfolios


def _natural_key(lot_id):
    """L2 排在 L10 前。"""
    head, _, tail = lot_id.partition("-")
    digits = "".join(ch for ch in head if ch.isdigit())
    return (int(digits) if digits else 0, lot_id)


def independent_options(draws_by_lot: dict, due_by_lot: dict, evidence_by_line: dict, lines: dict,
                        rem: dict, prices: dict, sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> list:
    """每批“独立处理 vs 暂存”的对照（结算差额两者相同，比较加工批净额与 0）。"""
    options = []
    for i, lot_id in enumerate(sorted(draws_by_lot, key=_natural_key)):
        if rem[lot_id] <= 1e-9:
            continue
        best = None
        for line_id, evidence in evidence_by_line.items():
            line = lines[line_id]
            if rem[lot_id] > line.capacity_wet_kg:
                continue
            result = evaluate_batch([(lot_id, rem[lot_id])], line, evidence, draws_by_lot, rem,
                                    prices, sims, seed + 701 + i)
            if best is None or float(np.mean(result.net)) > float(np.mean(best.net)):
                best = result
        options.append({
            "lot_id": lot_id,
            "remaining_wet": rem[lot_id],
            "due_mean": float(np.mean(due_by_lot[lot_id])),
            "independent": best.summary() if best else None,
        })
    return options
