"""报告格式化：把分析结果整理为中文文本（JSON 由调用方直接序列化）。"""

from .estimate import GROUP_LABELS


def yuan(x):
    return f"{x:,.0f} 元"


def kg(x):
    return f"{x:,.1f} kg"


def pct(x):
    return f"{x * 100:.1f}%"


def interval(stats, unit="kg"):
    fmt = yuan if unit == "元" else kg
    return f"{fmt(stats['p50'])} [{fmt(stats['p5'])} ~ {fmt(stats['p95'])}]"


def table(headers, rows):
    """简单对齐的文本表（按字符数，不处理宽字符）。"""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = [" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("-+-".join("-" * w for w in widths))
    for row in rows:
        lines.append(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def format_estimate(summary, attribution, value_name="结算价值"):
    lines = [f"批 {summary['lot_id']} 估值（模拟 {summary['sims']} 次）"]
    lines.append(f"  干基质量（全批）: {interval(summary['dry'])}")
    lines.append(f"  干基质量（余料）: {interval(summary['dry_remaining'])}")
    for analyte, label in (("nickel", "镍"), ("lithium", "锂"), ("contaminant", "污染物")):
        if analyte in summary:
            lines.append(f"  可结算量（{label}）: {interval(summary[analyte])}")
    cont = summary["contaminant_fraction"]
    lines.append(f"  污染物干基比例: 中位 {cont['p50']:.5f}，97.5% 上界 {cont['p975']:.5f}（未检出按区间处理，不为零）")
    lines.append("  不确定性贡献（方差份额）:")
    for name, result in attribution.items():
        label = {"dry": "干基质量", "nickel": "镍量", "lithium": "锂量", "contaminant": "污染物量", "value": value_name}.get(name, name)
        shares = " / ".join(f"{GROUP_LABELS[g]} {pct(result['share'][g])}" for g in ("moisture", "sampling", "assay"))
        lines.append(f"    {label}: {shares}")
    return "\n".join(lines)


def format_statement(st):
    lines = [f"批 {st['lot_id']} 结算单（合同 {st['contract_id']}，状态 {st['state']}）"]
    lines.append(f"  价格: " + ", ".join(f"{k} {v} 元/kg" for k, v in st["prices"].items()))
    lines.append(f"  干基质量: {interval(st['dry'])}")
    for analyte, label in (("nickel", "镍"), ("lithium", "锂"), ("contaminant", "污染物")):
        lines.append(f"  {label}可结算量: {interval(st['quantities'][analyte])}")
    lines.append(f"  污染物上界（97.5%，未检出按定量限计）: {st['contaminant_upper']:.5f}")
    lines.append(f"  最终应付: {interval(st['value'], '元')}")
    lines.append(f"  其中罚项: {interval(st['penalty'], '元')}")
    lines.append(f"  已暂付: {yuan(st['paid'])}")
    lines.append(f"  应付差额（终付−已付）: {interval(st['due'], '元')}")
    if st["recover"] > 0:
        lines.append(f"  待追偿: {yuan(st['recover'])}")
    if st["capital_tied"] > 0:
        lines.append(f"  资金占用（已付−应付P5）: {yuan(st['capital_tied'])}")
    cap = st.get("provisional_cap")
    if cap:
        status = "符合" if cap["ok"] else "超出"
        lines.append(f"  暂付上限核对: 已付 {yuan(cap['paid'])}，上限 {yuan(cap['cap'])}（{cap['basis_time']} 可知证据），{status}")
    for note in st["notes"]:
        lines.append(f"  注意: {note}")
    return "\n".join(lines)
