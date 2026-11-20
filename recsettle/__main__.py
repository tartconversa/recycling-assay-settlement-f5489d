"""命令行入口：估值、加工、结算、复检规划、审计与检查。

用法示例：
    python3 -m recsettle overview
    python3 -m recsettle estimate --lot L6
    python3 -m recsettle settle --lot L15
    python3 -m recsettle process --save-plan plan.json
    python3 -m recsettle plan-recheck --budget 1000
    python3 -m recsettle audit --lot L6
    python3 -m recsettle check --plan plan.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from . import audit as audit_mod
from . import checks as checks_mod
from . import plan as plan_mod
from . import process as process_mod
from .data import load
from .estimate import DEFAULT_SEED, DEFAULT_SIMS, attribute, build_calibration, build_lot_model, simulate, summarize
from .report import format_estimate, format_statement, interval, kg, pct, table, yuan
from .settle import prices_at, settlement_draws, statement


def _parse_as_of(text):
    if not text:
        return None
    value = dt.datetime.fromisoformat(text)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone(dt.timedelta(hours=8)))
    return value


class Context:
    """按需构建并缓存每批的校准、模型与模拟。"""

    def __init__(self, args):
        self.root = Path(args.data).resolve()
        self.as_of = _parse_as_of(args.as_of)
        self.sims = args.sims
        self.seed = args.seed
        self.bundle_full = load(self.root)
        self.bundle = self.bundle_full.as_of(self.as_of)
        self.cal = build_calibration(self.bundle)
        self._models = {}
        self._draws = {}

    def model(self, lot_id):
        if lot_id not in self._models:
            self._models[lot_id] = build_lot_model(self.bundle, self.cal, lot_id)
        return self._models[lot_id]

    def draws(self, lot_id):
        if lot_id not in self._draws:
            self._draws[lot_id] = simulate(self.model(lot_id), self.cal, self.sims, self.seed)
        return self._draws[lot_id]

    def due_draws(self, lot_id):
        lot = self.bundle.lots[lot_id]
        contract = self.bundle.contracts[lot.contract_id]
        prices = prices_at(self.bundle, self.as_of.date() if self.as_of else None)
        values = settlement_draws(self.draws(lot_id), contract, prices)
        paid = sum(p.amount for p in self.bundle.lot_payments(lot_id))
        return values - paid


def _print(obj, as_json, text):
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
    else:
        print(text)


def cmd_overview(ctx, args):
    rows = []
    out = []
    for lot_id in sorted(ctx.bundle.lots, key=process_mod._natural_key):
        lot = ctx.bundle.lots[lot_id]
        st = statement(ctx.bundle, ctx.cal, lot_id, ctx.as_of, ctx.sims, ctx.seed)
        flags = []
        if lot.state == "disputed":
            flags.append("争议")
        if ctx.bundle.lot_allocations(lot_id):
            flags.append("已投料")
        hard = min(line.hard_contaminant_limit for line in ctx.bundle.lines.values())
        if st["contaminant_upper"] > hard:
            flags.append("污染物上界破最严硬界")
        rows.append([
            lot_id, lot.state, kg(st["dry"]["p50"]), yuan(st["value"]["p50"]),
            yuan(st["paid"]), yuan(st["due_mean"]), f"{st['contaminant_upper']:.4f}", "、".join(flags) or "-",
        ])
        out.append({"lot_id": lot_id, "state": lot.state, "dry_p50": st["dry"]["p50"],
                    "value_p50": st["value"]["p50"], "paid": st["paid"], "due_mean": st["due_mean"],
                    "contaminant_upper": st["contaminant_upper"], "flags": flags})
    text = "批总览（估值为全批合同口径）\n" + table(
        ["批", "状态", "干基P50", "应付P50", "已付", "应付差额", "污染物上界", "标记"], rows)
    _print(out, args.json, text)


def cmd_estimate(ctx, args):
    lot_id = args.lot
    model = ctx.model(lot_id)
    draws = ctx.draws(lot_id)
    summary = summarize(draws)
    lot = ctx.bundle.lots[lot_id]
    contract = ctx.bundle.contracts[lot.contract_id]
    prices = prices_at(ctx.bundle, ctx.as_of.date() if ctx.as_of else None)
    value_fn = lambda d: settlement_draws(d, contract, prices)
    attrib = attribute(ctx.bundle, ctx.cal, lot_id, ctx.sims, ctx.seed, value_fn)
    result = {"summary": summary, "attribution": attrib,
              "assumption": f"容器内取样误差占比假设 {ctx.cal.within_share}；位置偏移与τ²为全资料矩估计"}
    _print(result, args.json, format_estimate(summary, attrib))


def cmd_settle(ctx, args):
    st = statement(ctx.bundle, ctx.cal, args.lot, ctx.as_of, ctx.sims, ctx.seed)
    _print(st, args.json, format_statement(st))


def cmd_process(ctx, args):
    lot_ids = sorted(ctx.bundle.lots, key=process_mod._natural_key)
    draws_by_lot = {lot: ctx.draws(lot) for lot in lot_ids}
    due_by_lot = {lot: ctx.due_draws(lot) for lot in lot_ids}
    evidence = {}
    for line_id in ctx.bundle.lines:
        ev = process_mod.fit_line_evidence(ctx.bundle, line_id)
        if ev is not None:
            evidence[line_id] = ev
    prices = prices_at(ctx.bundle, ctx.as_of.date() if ctx.as_of else None)
    portfolios = process_mod.build_candidates(ctx.bundle, ctx.cal, draws_by_lot, due_by_lot, evidence, prices, ctx.sims, ctx.seed)
    rem = process_mod.remaining_wet(ctx.bundle)
    options = process_mod.independent_options(draws_by_lot, due_by_lot, evidence, ctx.bundle.lines, rem, prices, ctx.sims, ctx.seed)

    lines = ["加工方案比较（组合价值 = Σ 加工批净额 − Σ 各批应付差额；暂存部分只承担后者）"]
    best = None
    for portfolio in portfolios:
        s = portfolio.summary()
        lines.append(f"\n■ {s['name']}：{s['note']}")
        lines.append(f"  组合价值: {interval(s['total'], '元')}（下行 P5 {yuan(s['total']['p5'])}）")
        for batch in s["batches"]:
            lines.append(f"  批 {batch['line']} 投 {batch['wet']:.0f} kg：{batch['acceptance']}"
                         + ("" if batch["supported"] else "（情景：超出历史支持）")
                         + f"，批净额 {interval(batch['net'], '元')}")
            for item in batch["items"]:
                lines.append(f"    - {item[0]} {item[1]:.0f} kg")
        if best is None or s["total"]["mean"] > best.summary()["total"]["mean"]:
            best = portfolio
    lines.append(f"\n推荐：{best.name}（按组合价值均值）")
    lines.append("\n各批独立处理 vs 暂存（结算差额两者相同，比较加工批净额与 0）")
    rows = []
    for opt in options:
        ind = opt["independent"]
        if ind is None:
            rows.append([opt["lot_id"], kg(opt["remaining_wet"]), "-", "-", "-", "超线容量"])
        else:
            rows.append([opt["lot_id"], kg(opt["remaining_wet"]), ind["line"],
                         yuan(ind["net"]["mean"]), yuan(ind["net"]["p5"]),
                         f"{ind['acceptance']}" + ("" if ind["supported"] else "/情景")])
    lines.append(table(["批", "余料", "最佳线", "独立净额均值", "下行P5", "状态"], rows))

    result = {"portfolios": [p.summary() for p in portfolios], "recommended": best.name,
              "independent_options": options}
    if args.save_plan:
        plan = {"name": best.name, "batches": [
            {"line_id": b.line_id, "items": [{"lot_id": lot, "wet_kg": w} for lot, w in b.items]}
            for b in best.batches]}
        Path(args.save_plan).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        lines.append(f"\n已保存推荐计划：{args.save_plan}")
    _print(result, args.json, "\n".join(lines))


def cmd_plan_recheck(ctx, args):
    lot_ids = [args.lot] if args.lot else None
    result = plan_mod.recommend(ctx.bundle, ctx.cal, ctx.as_of, args.budget, args.deadline_days,
                                lot_ids, ctx.sims, ctx.seed)
    lines = [f"复检规划（预算 {yuan(result['budget'])}，已选花费 {yuan(result['spent'])}）",
             "收益 = 期望罚项下降 + 结算区间收窄（启发式，分量并列）"]
    if result["chosen"]:
        lines.append("选定动作：")
        for effect in result["chosen"]:
            a = effect.action
            lines.append(f"  ■ {a.lot_id} {a.kind} {a.target}（{yuan(a.cost)}，{a.days} 天）：{a.note}")
            lines.append(f"    结算区间 {yuan(effect.width_before)} → {yuan(effect.width_after)}；"
                         f"期望罚项 {yuan(effect.penalty_before)} → {yuan(effect.penalty_after)}；"
                         f"污染物上界 {effect.cont_upper_before:.4f} → {effect.cont_upper_after:.4f}")
            if effect.acceptance_before != effect.acceptance_after:
                lines.append(f"    投入判定变化: {effect.acceptance_before} → {effect.acceptance_after}")
            if effect.reason:
                lines.append(f"    注意: {effect.reason}")
    else:
        lines.append("预算与期限内无正收益动作。")
    if result["infeasible"]:
        lines.append("不可行动作：")
        for effect in result["infeasible"]:
            lines.append(f"  - {effect.action.lot_id} {effect.action.kind} {effect.action.target}: {effect.reason}")
    payload = {
        "budget": result["budget"], "spent": result["spent"],
        "chosen": [_effect_dict(e) for e in result["chosen"]],
        "ranked": [_effect_dict(e) for e in result["ranked"]],
        "infeasible": [_effect_dict(e) for e in result["infeasible"]],
    }
    _print(payload, args.json, "\n".join(lines))


def _effect_dict(effect):
    a = effect.action
    return {"lot_id": a.lot_id, "kind": a.kind, "target": a.target, "cost": a.cost, "days": a.days,
            "width_gain": effect.width_gain, "penalty_gain": effect.penalty_gain,
            "cont_upper_before": effect.cont_upper_before, "cont_upper_after": effect.cont_upper_after,
            "acceptance_before": effect.acceptance_before, "acceptance_after": effect.acceptance_after,
            "feasible": effect.feasible, "reason": effect.reason, "note": a.note}


def cmd_audit(ctx, args):
    lot_id = args.lot
    as_of = ctx.as_of
    if as_of is None:
        known = [a.available_at for a in ctx.bundle_full.assays.values()]
        known += [r.available_at for r in ctx.bundle_full.references.values()]
        known += [p.paid_at for p in ctx.bundle_full.payments.values()]
        known += [a.started_at for a in ctx.bundle_full.allocations.values()]
        as_of = max(known)
    events = audit_mod.timeline(ctx.bundle, lot_id)
    decomp = audit_mod.difference_decomposition(ctx.bundle_full, lot_id, as_of, ctx.sims, ctx.seed)
    adj = audit_mod.adjustability(ctx.bundle, lot_id, ctx.as_of)
    lines = [f"批 {lot_id} 审计"]
    lines.append("事件时间线：")
    for event in events:
        lines.append(f"  {event['at']} {event['kind']}: {event['detail']}")
    lines.append(f"\n差额分解（暂付时 {yuan(decomp['base_value_p50'])} → 当前 {yuan(decomp['current_value_p50'])}，"
                 f"差额 {yuan(decomp['total_delta'])}）：")
    for step in decomp["steps"]:
        lines.append(f"  {step['at']} [{'、'.join(step['evidence'])}] Δ {yuan(step['delta'])} → {yuan(step['value_p50'])}")
    lines.append(f"  注: {decomp['note']}")
    lines.append("\n可调整性：")
    lines.append(f"  已加工实物 {kg(adj['processed_wet_kg'])}（{adj['adjustable']['已加工实物']}）；"
                 f"余料 {kg(adj['remaining_wet_kg'])}（{adj['adjustable']['余料']}）；"
                 f"结算（{adj['adjustable']['结算']}）；已付 {yuan(adj['paid_yuan'])}")
    for container in adj["containers"]:
        lines.append(f"  {container['container_id']}: 净重 {kg(container['wet_kg'])}，"
                     f"已投 {kg(container['allocated_kg'])}，余 {kg(container['remaining_kg'])}"
                     + (f"，线 {container['lines']}" if container["lines"] else ""))
    for limit in adj["limits"]:
        lines.append(f"  限制: {limit}")
    _print({"timeline": events, "decomposition": decomp, "adjustability": adj}, args.json, "\n".join(lines))


def cmd_check(ctx, args):
    plan = None
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    results = checks_mod.run_checks(ctx.bundle, plan, ctx.sims, ctx.seed)
    lines = ["一致性检查"]
    worst = "ok"
    for result in results:
        lines.append(f"  [{result.status.upper()}] {result.title}: {result.detail}")
        if result.status == "fail":
            worst = "fail"
        elif result.status == "warn" and worst == "ok":
            worst = "warn"
    payload = [{"check_id": r.check_id, "title": r.title, "status": r.status, "detail": r.detail} for r in results]
    _print(payload, args.json, "\n".join(lines))
    if worst == "fail":
        sys.exit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="recsettle", description="回收业务取样与加工结算分析")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=str(Path(__file__).resolve().parents[1]), help="资料根目录")
    common.add_argument("--as-of", default=None, help="可知时刻（ISO，如 2025-02-20T08:00:00+08:00）")
    common.add_argument("--sims", type=int, default=DEFAULT_SIMS, help="蒙特卡洛次数")
    common.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    common.add_argument("--json", action="store_true", help="输出 JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("overview", parents=[common], help="各批估值与状态总览")

    p = sub.add_parser("estimate", parents=[common], help="单批估值与不确定性贡献")
    p.add_argument("--lot", required=True)

    p = sub.add_parser("settle", parents=[common], help="单批结算单")
    p.add_argument("--lot", required=True)

    p = sub.add_parser("process", parents=[common], help="加工方案比较")
    p.add_argument("--save-plan", default=None, help="把推荐计划存为 JSON")

    p = sub.add_parser("plan-recheck", parents=[common], help="复检与取样规划")
    p.add_argument("--lot", default=None)
    p.add_argument("--budget", type=float, default=1000.0)
    p.add_argument("--deadline-days", type=int, default=None)

    p = sub.add_parser("audit", parents=[common], help="单批差额追溯与可调整性")
    p.add_argument("--lot", required=True)

    p = sub.add_parser("check", parents=[common], help="一致性检查")
    p.add_argument("--plan", default=None, help="待校验的计划 JSON")

    args = parser.parse_args(argv)
    ctx = Context(args)
    {
        "overview": cmd_overview,
        "estimate": cmd_estimate,
        "settle": cmd_settle,
        "process": cmd_process,
        "plan-recheck": cmd_plan_recheck,
        "audit": cmd_audit,
        "check": cmd_check,
    }[args.command](ctx, args)


if __name__ == "__main__":
    main()
