"""复检与取样规划：在样品量、检验预算和结算期限下选择实际取样。

原则（docs/资料口径.md 与业务要求）：
- 留样复测只压缩化验重复性方差，不能宣称解决容器间不均匀；
- 原容器重取同时压缩容器内取样误差，独立证据更多，但费用与耗时更高；
- 预 posterior 分析：假设新证据落在当前后验均值处，比较结算区间、期望罚项、
  污染物上界与投入判定的变化，按单位成本收益排序；
- 争议批的最终结算须满足合同认可依据（mutual_retest），仅留样复测不足以结案。
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace

import numpy as np

from .data import Bundle
from .estimate import (
    DEFAULT_SEED,
    DEFAULT_SIMS,
    Calibration,
    Draw,
    LotModel,
    build_lot_model,
    simulate,
)
from .process import ACCEPT_OK, ACCEPT_REJECT, ACCEPT_RETEST
from .settle import penalty_draws, prices_at, settlement_draws


@dataclass(frozen=True)
class RecheckAction:
    """一个候选复检动作。"""

    kind: str  # resample（原容器重取）/ retained（留样复测）
    lot_id: str
    target: str  # container_id 或 sample_id
    option_id: str
    cost: float
    days: int
    grams: float
    note: str


@dataclass
class ActionEffect:
    """一个复检动作的预 posterior 效果。"""

    action: RecheckAction
    width_before: float
    width_after: float
    penalty_before: float
    penalty_after: float
    cont_upper_before: float
    cont_upper_after: float
    acceptance_before: dict
    acceptance_after: dict
    feasible: bool
    reason: str

    @property
    def width_gain(self):
        return self.width_before - self.width_after

    @property
    def penalty_gain(self):
        return self.penalty_before - self.penalty_after

    @property
    def benefit(self):
        """启发式收益（元）：期望罚项下降 + 结算区间收窄；分量并列展示，不混同物理量。"""
        return self.penalty_gain + self.width_gain


def candidate_actions(bundle: Bundle, lot_id) -> list:
    """一批的候选复检动作：有余料的容器可重取；有余样的留样可复测。"""
    actions = []
    resample = bundle.options.get("resample")
    retained = bundle.options.get("retained")
    containers = {c.container_id: c for c in bundle.lot_containers(lot_id)}
    if resample:
        for container in containers.values():
            if container.remaining_kg > 1e-9:
                actions.append(RecheckAction(
                    "resample", lot_id, container.container_id, "resample",
                    resample.cost, resample.days, resample.grams,
                    "原容器重取：同时压缩容器内取样误差与化验误差，增加独立容器证据",
                ))
    if retained:
        for sample in bundle.samples.values():
            if sample.container_id not in containers or sample.is_retest:
                continue
            if sample.remaining_grams >= retained.grams:
                actions.append(RecheckAction(
                    "retained", lot_id, sample.sample_id, "retained",
                    retained.cost, retained.days, retained.grams,
                    "留样复测：只压缩化验重复性方差，不能宣称解决容器间不均匀",
                ))
    return actions


def _posterior_mean(cal: Calibration, model: LotModel, draw: Draw | None, analyte) -> float:
    if draw is None or draw.censored:
        return model.mu_hat[analyte]
    tau2 = max(cal.tau2[analyte], 1e-12)
    like_var = max(draw.likelihood_var(cal), 1e-12)
    post_var = 1.0 / (1.0 / tau2 + 1.0 / like_var)
    return (model.mu_hat[analyte] / tau2 + draw.mean / like_var) * post_var


def _apply_action(cal: Calibration, model: LotModel, action: RecheckAction, bundle: Bundle) -> LotModel:
    """把动作应用到批模型上（预 posterior：新证据取当前后验均值）。"""
    draws = {a: dict(entries) for a, entries in model.draws.items()}
    if action.kind == "resample":
        container_id = action.target
        for analyte in draws:
            existing = draws[analyte].get(container_id)
            within = cal.within_var[analyte]
            post_mean = _posterior_mean(cal, model, existing, analyte)
            meas_new = cal.meas_var(analyte, post_mean, 1, None)
            if existing is not None and not existing.censored:
                like_old = max(existing.likelihood_var(cal), 1e-12)
                like_new = meas_new + within
                combined = 1.0 / (1.0 / like_old + 1.0 / like_new)
                draws[analyte][container_id] = Draw(
                    container_id, analyte, False, post_mean,
                    max(combined - within, 1e-12), 0.0, existing.n_reps + 1, None,
                )
            else:
                draws[analyte][container_id] = Draw(container_id, analyte, False, post_mean, meas_new, 0.0, 1, None)
    else:  # retained：留样复测，只摊薄化验重复性，偏差修正不确定度不摊薄
        sample = bundle.samples[action.target]
        container_id = sample.container_id
        for analyte in draws:
            existing = draws[analyte].get(container_id)
            if existing is None or existing.censored:
                continue
            n_old, n_new = existing.n_reps, existing.n_reps + 1
            if cal.rep_is_relative[analyte]:
                scale = (abs(existing.mean) * cal.rep_sd[analyte]) ** 2
            else:
                scale = cal.rep_sd[analyte] ** 2
            rep_old, rep_new = scale / n_old, scale / n_new
            bias_var = max(existing.meas_var - rep_old, 0.0)
            draws[analyte][container_id] = replace(existing, meas_var=rep_new + bias_var, n_reps=n_new)
    return LotModel(model.lot_id, model.container_ids, model.wet, model.remaining, draws,
                    model.mu_hat, model.mu_var)


def _acceptance_by_line(bundle: Bundle, draws, upper) -> dict:
    """按污染物中位与上界判定该批余料对每条线的投入状态。"""
    p50 = float(np.percentile(draws.contaminant_fraction(remaining=True), 50))
    out = {}
    for line_id, line in bundle.lines.items():
        if p50 > line.hard_contaminant_limit:
            out[line_id] = ACCEPT_REJECT
        elif upper > line.hard_contaminant_limit:
            out[line_id] = ACCEPT_RETEST
        else:
            out[line_id] = ACCEPT_OK
    return out


def evaluate_action(bundle: Bundle, cal: Calibration, lot_id, action: RecheckAction,
                    sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> ActionEffect:
    """评估一个复检动作：结算区间、期望罚项、污染物上界与投入判定的变化。"""
    lot = bundle.lots[lot_id]
    contract = bundle.contracts[lot.contract_id]
    prices = prices_at(bundle)
    model = build_lot_model(bundle, cal, lot_id)

    before = simulate(model, cal, sims, seed)
    before_upper = simulate(model, cal, sims, seed, censor_mode="upper")
    values_before = settlement_draws(before, contract, prices)
    penalty_before = float(np.mean(penalty_draws(before, contract)))

    model_after = _apply_action(cal, model, action, bundle)
    after = simulate(model_after, cal, sims, seed)
    after_upper = simulate(model_after, cal, sims, seed, censor_mode="upper")
    values_after = settlement_draws(after, contract, prices)
    penalty_after = float(np.mean(penalty_draws(after, contract)))

    width = lambda v: float(np.percentile(v, 95) - np.percentile(v, 5))
    return ActionEffect(
        action,
        width(values_before), width(values_after),
        penalty_before, penalty_after,
        float(np.percentile(before_upper.contaminant_fraction(remaining=True), 97.5)),
        float(np.percentile(after_upper.contaminant_fraction(remaining=True), 97.5)),
        _acceptance_by_line(bundle, before, float(np.percentile(before_upper.contaminant_fraction(remaining=True), 97.5))),
        _acceptance_by_line(bundle, after, float(np.percentile(after_upper.contaminant_fraction(remaining=True), 97.5))),
        True, "",
    )


def recommend(bundle: Bundle, cal: Calibration, as_of: dt.datetime | None,
              budget: float, deadline_days: int | None, lot_ids=None,
              sims=DEFAULT_SIMS, seed=DEFAULT_SEED) -> dict:
    """在预算、期限与样品量约束下选择实际复检动作（按单位成本收益贪心）。"""
    lot_ids = lot_ids or sorted(bundle.lots)
    effects = []
    for lot_id in lot_ids:
        lot = bundle.lots[lot_id]
        days_left = (lot.settlement_due - as_of.date()).days if as_of is not None else None
        limit = deadline_days if deadline_days is not None else days_left
        for action in candidate_actions(bundle, lot_id):
            if limit is not None and action.days > limit:
                effects.append(ActionEffect(action, 0, 0, 0, 0, 0, 0, {}, {}, False,
                                            f"耗时 {action.days} 天超出剩余期限 {limit} 天"))
                continue
            effect = evaluate_action(bundle, cal, lot_id, action, sims, seed)
            if lot.state == "disputed" and action.kind == "retained":
                effect.reason = "争议批：仅留样复测不满足双方认可依据，需配合重取"
            effects.append(effect)

    feasible = [e for e in effects if e.feasible]
    ranked = sorted(feasible, key=lambda e: -(e.benefit / max(e.action.cost, 1e-9)))
    chosen, spent = [], 0.0
    for effect in ranked:
        if effect.benefit <= 0:
            continue
        if spent + effect.action.cost > budget:
            continue
        chosen.append(effect)
        spent += effect.action.cost
    return {
        "budget": budget,
        "spent": spent,
        "chosen": chosen,
        "ranked": ranked,
        "infeasible": [e for e in effects if not e.feasible],
    }
