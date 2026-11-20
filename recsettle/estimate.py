"""批估值：干基质量与各成分可结算量的联合不确定性及来源分解。

模型要点（与 docs/资料口径.md 对齐）：
- 干基质量 = Σ 容器净重 × (1 − 水分)，按容器净重加权，不以样本数为权重；
- 化验值先按参考样偏差修正（因子 = 参考值/测值），修正不确定度来自参考记录；
- 定量限以下（lt）按区间 [0, 定量限] 处理：模拟中在区间内抽潜伏值，上界按定量限计，
  不把未检出当成确定为零；
- 留样复测（retest_of 非空）只压缩化验重复性方差，不增加容器代表性；
  原容器重取同时压缩容器内取样误差，两者增加的独立证据不同；
- 取样位置 top 相对容器平均（以 mixed 为代表）的偏移由全资料成对差估计，偏移带标准误；
- 容器间总不均匀由容器间散布的矩估计得到，再按 within_share 拆成真容器间差异与
  容器内取样误差。每个容器只有一次独立取样，二者不可分别识别，拆分比例是假设，
  只影响“重取 vs 复测”的相对收益，不改变批均值，报告中需注明。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .data import ANALYTES, METALS, Bundle

DEFAULT_WITHIN_SHARE = 0.5  # 容器内取样误差占总不均匀的假设比例
DEFAULT_SIMS = 4000
DEFAULT_SEED = 20250220

# 物理截断，避免正态抽伸出无意义比例
CLIP = {
    "moisture": (0.0, 0.6),
    "nickel": (0.0, 0.5),
    "lithium": (0.0, 0.5),
    "contaminant": (0.0, 0.2),
}

# 贡献分解的分组：水分 / 取样 / 化验
GROUPS = ("moisture", "sampling", "assay")
GROUP_LABELS = {"moisture": "水分", "sampling": "取样", "assay": "化验"}


@dataclass(frozen=True)
class Calibration:
    """全资料 pooled 的测量与取样参数（随 as_of 过滤后的资料重算）。"""

    bias_factor: dict  # (lab, analyte) -> 修正因子 = 参考值/测值
    bias_rel_unc: dict  # (lab, analyte) -> 修正因子相对标准不确定度
    rep_sd: dict  # analyte -> 重复性标准差（moisture 绝对，其余相对）
    rep_is_relative: dict  # analyte -> bool
    position_offset: dict  # analyte -> (top 相对容器平均的偏移, 标准误)
    tau2: dict  # analyte -> 容器间真差异方差
    within_var: dict  # analyte -> 容器内取样误差方差
    within_share: float
    pin_censored: bool = False  # True 时未检出按 定量限/2 固定（化验来源关闭）

    def meas_var(self, analyte, value, n_reps, lab):
        """单次抽估的化验方差：重复性（随复测次数摊薄）+ 偏差修正不确定度（不摊薄）。

        lab 为 None（如规划中的新取样）时，偏差修正不确定度取该成分各实验室的中位。
        """
        if self.rep_is_relative[analyte]:
            rep = (abs(value) * self.rep_sd[analyte]) ** 2
        else:
            rep = self.rep_sd[analyte] ** 2
        if lab is not None and (lab, analyte) in self.bias_rel_unc:
            unc = self.bias_rel_unc[(lab, analyte)]
        else:
            uncs = [v for k, v in self.bias_rel_unc.items() if k[1] == analyte]
            unc = float(np.median(uncs)) if uncs else 0.0
        bias = (abs(value) * unc) ** 2
        return rep / max(1, n_reps) + bias

    def variant(self, keep):
        """只保留 keep 集合中的不确定度来源（moisture / sampling / assay），用于贡献分解。

        - moisture 关闭：水分全部来源归零；
        - sampling 关闭：金属/污染物的容器间、容器内与位置标准误归零；
        - assay 关闭：金属/污染物的重复性与偏差修正归零，未检出按 定量限/2 固定。
        """
        keep = frozenset(keep)
        result = self
        if "moisture" not in keep:
            result = replace(
                result,
                rep_sd={**result.rep_sd, "moisture": 0.0},
                position_offset={**result.position_offset, "moisture": (result.position_offset["moisture"][0], 0.0)},
                tau2={**result.tau2, "moisture": 0.0},
                within_var={**result.within_var, "moisture": 0.0},
                bias_rel_unc={k: (0.0 if k[1] == "moisture" else v) for k, v in result.bias_rel_unc.items()},
            )
        if "sampling" not in keep:
            result = replace(
                result,
                position_offset={k: (v if k == "moisture" else (v[0], 0.0)) for k, v in result.position_offset.items()},
                tau2={k: (v if k == "moisture" else 0.0) for k, v in result.tau2.items()},
                within_var={k: (v if k == "moisture" else 0.0) for k, v in result.within_var.items()},
            )
        if "assay" not in keep:
            result = replace(
                result,
                rep_sd={k: (v if k == "moisture" else 0.0) for k, v in result.rep_sd.items()},
                bias_rel_unc={k: (0.0 if k[1] != "moisture" else v) for k, v in result.bias_rel_unc.items()},
                pin_censored=True,
            )
        return result


def _independent_samples(bundle, lot_id):
    """批内独立取样（留样复测不算独立）。"""
    container_ids = {c.container_id for c in bundle.lot_containers(lot_id)}
    return [s for s in bundle.samples.values() if s.container_id in container_ids and not s.is_retest]


def _corrected_values(bundle, cal, sample, analyte):
    """一个样本（含其留样复测）某成分的修正后检出值与定量限。"""
    related = [sample] + [s for s in bundle.samples.values() if s.retest_of == sample.sample_id]
    values, limits = [], []
    for rep in related:
        for assay in bundle.assays.values():
            if assay.sample_id != rep.sample_id or assay.analyte != analyte:
                continue
            factor = cal.bias_factor.get((assay.lab_id, analyte), 1.0)
            if assay.censored:
                limits.append(assay.limit * factor)
            else:
                values.append((assay.value * factor, assay.lab_id))
    return values, limits


def build_calibration(bundle: Bundle, within_share=DEFAULT_WITHIN_SHARE) -> Calibration:
    """从参考样、留样复测对和容器间散布估计测量与取样参数。"""
    # 1) 实验室偏差修正：每个 (实验室, 成分) 取最新参考记录
    bias_factor, bias_rel_unc = {}, {}
    latest = {}
    for ref in bundle.references.values():
        key = (ref.lab_id, ref.analyte)
        if key not in latest or ref.available_at > latest[key].available_at:
            latest[key] = ref
    for (lab, analyte), ref in latest.items():
        bias_factor[(lab, analyte)] = ref.assigned / ref.measured
        bias_rel_unc[(lab, analyte)] = ref.uncertainty / ref.measured

    # 2) 水分重复性：留样复测对（同一留样再测，差异即化验重复性）
    shell = _calibration_shell(bias_factor, bias_rel_unc)
    diffs = []
    for sample in bundle.samples.values():
        if sample.is_retest:
            continue
        retests = [s for s in bundle.samples.values() if s.retest_of == sample.sample_id]
        if not retests:
            continue
        orig, _ = _corrected_values(bundle, shell, sample, "moisture")
        for retest in retests:
            re_values, _ = _corrected_values(bundle, shell, retest, "moisture")
            if orig and re_values:
                diffs.append(orig[0][0] - re_values[0][0])
    if diffs:
        moisture_rep = math.sqrt(sum(d * d for d in diffs) / (2 * len(diffs)))
    else:
        moisture_rep = 0.1 * _typical_level(bundle, shell, "moisture")

    rep_sd, rep_is_relative = {}, {}
    for analyte in ANALYTES:
        if analyte == "moisture":
            rep_sd[analyte] = moisture_rep
            rep_is_relative[analyte] = False
        else:
            uncs = [bias_rel_unc[k] for k in bias_rel_unc if k[1] == analyte]
            rep_sd[analyte] = float(np.median(uncs)) if uncs else 0.02
            rep_is_relative[analyte] = True

    # 3) 每个容器每条独立取样的修正抽估值（位置调整前），供位置偏移与 τ² 估计
    raw_draws = {a: {} for a in ANALYTES}  # analyte -> container_id -> (value, meas_var, position, lot_id)
    for lot_id in bundle.lots:
        for sample in _independent_samples(bundle, lot_id):
            for analyte in ANALYTES:
                values, limits = _corrected_values(bundle, shell, sample, analyte)
                if not values:
                    continue
                mean = sum(v for v, _ in values) / len(values)
                var = shell.meas_var(analyte, mean, len(values), values[0][1])
                raw_draws[analyte][sample.container_id] = (mean, var, sample.position, lot_id)

    # 4) 位置偏移：同批 top 与 mixed 的成对差
    position_offset = {}
    for analyte in ANALYTES:
        by_lot = {}
        for container_id, (value, _, position, lot_id) in raw_draws[analyte].items():
            by_lot.setdefault(lot_id, {}).setdefault(position, []).append(value)
        diffs = []
        for positions in by_lot.values():
            tops = positions.get("top", [])
            mixed = positions.get("mixed", [])
            if tops and mixed:
                diffs.append(sum(mixed) / len(mixed) - sum(tops) / len(tops))
        if len(diffs) >= 3:
            offset = float(np.mean(diffs))
            se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
        else:
            offset, se = 0.0, 0.05
        position_offset[analyte] = (offset, se)

    # 5) 容器间总不均匀的矩估计：批内容器散布 − 平均化验方差
    tau2, within_var = {}, {}
    for analyte in ANALYTES:
        numer, denom = 0.0, 0.0
        by_lot = {}
        for container_id, (value, var, position, lot_id) in raw_draws[analyte].items():
            offset, se = position_offset[analyte]
            adj = value + (offset if position == "top" else 0.0)
            v = var + (se * se if position == "top" else 0.0)
            by_lot.setdefault(lot_id, []).append((adj, v))
        for entries in by_lot.values():
            if len(entries) < 2:
                continue
            values = [e[0] for e in entries]
            spread = float(np.var(values, ddof=1)) - sum(e[1] for e in entries) / len(entries)
            weight = len(entries) - 1
            numer += weight * spread
            denom += weight
        total = max(0.0, numer / denom) if denom else 0.0
        tau2[analyte] = (1.0 - within_share) * total
        within_var[analyte] = within_share * total

    return Calibration(bias_factor, bias_rel_unc, rep_sd, rep_is_relative, position_offset, tau2, within_var, within_share)


def _calibration_shell(bias_factor, bias_rel_unc):
    """只有偏差参数的简版校准，供校准过程中的修正计算使用。"""
    return Calibration(
        bias_factor,
        bias_rel_unc,
        {a: 0.0 for a in ANALYTES},
        {a: True for a in ANALYTES},
        {a: (0.0, 0.0) for a in ANALYTES},
        {a: 0.0 for a in ANALYTES},
        {a: 0.0 for a in ANALYTES},
        DEFAULT_WITHIN_SHARE,
    )


def _typical_level(bundle, cal, analyte):
    values = []
    for sample in bundle.samples.values():
        if sample.is_retest:
            continue
        found, _ = _corrected_values(bundle, cal, sample, analyte)
        values.extend(v for v, _ in found)
    return float(np.mean(values)) if values else 0.1


@dataclass(frozen=True)
class Draw:
    """一个容器某成分的一次独立取样证据（留样复测已并入）。"""

    container_id: str
    analyte: str
    censored: bool
    mean: float | None  # 修正+位置调整后
    meas_var: float | None  # 化验方差（含偏差修正，已按复测次数摊薄）
    position_var: float  # 位置偏移标准误平方
    n_reps: int
    limit: float | None  # 修正后定量限（censored 时）

    def likelihood_var(self, cal):
        return (self.meas_var or 0.0) + self.position_var + cal.within_var[self.analyte]


def build_draws(bundle: Bundle, cal: Calibration, lot_id) -> dict:
    """批内每个容器每个成分的独立取样证据。"""
    draws = {a: {} for a in ANALYTES}
    for sample in _independent_samples(bundle, lot_id):
        for analyte in ANALYTES:
            values, limits = _corrected_values(bundle, cal, sample, analyte)
            offset, se = cal.position_offset[analyte]
            position_var = se * se if sample.position == "top" else 0.0
            shift = offset if sample.position == "top" else 0.0
            if values:
                mean = sum(v for v, _ in values) / len(values) + shift
                meas_var = cal.meas_var(analyte, mean, len(values), values[0][1])
                draws[analyte][sample.container_id] = Draw(
                    sample.container_id, analyte, False, mean, meas_var, position_var, len(values), None
                )
            elif limits:
                limit = min(limits)
                draws[analyte][sample.container_id] = Draw(
                    sample.container_id, analyte, True, None, None, position_var, 1, limit
                )
    return draws


@dataclass
class LotModel:
    """一批的估值模型：容器、余料与各成分证据。"""

    lot_id: str
    container_ids: tuple
    wet: np.ndarray
    remaining: np.ndarray
    draws: dict  # analyte -> {container_id: Draw}
    mu_hat: dict
    mu_var: dict


def build_lot_model(bundle: Bundle, cal: Calibration, lot_id) -> LotModel:
    containers = bundle.lot_containers(lot_id)
    container_ids = tuple(c.container_id for c in containers)
    wet = np.array([c.wet_kg for c in containers], dtype=float)
    remaining = np.array([c.remaining_kg for c in containers], dtype=float)
    draws = build_draws(bundle, cal, lot_id)
    mu_hat, mu_var = {}, {}
    for analyte in ANALYTES:
        entries = []
        for cid, draw in draws[analyte].items():
            idx = container_ids.index(cid)
            if draw.censored:
                half = draw.limit / 2.0
                censor_var = 0.0 if cal.pin_censored else draw.limit * draw.limit / 12.0
                entries.append((wet[idx], half, censor_var + cal.within_var[analyte]))
            else:
                entries.append((wet[idx], draw.mean, draw.likelihood_var(cal)))
        if entries:
            w = np.array([e[0] for e in entries])
            m = np.array([e[1] for e in entries])
            v = np.array([e[2] for e in entries])
            mu_hat[analyte] = float(np.sum(w * m) / np.sum(w))
            mu_var[analyte] = float(np.sum(w * w * (v + cal.tau2[analyte])) / np.sum(w) ** 2)
        else:
            mu_hat[analyte], mu_var[analyte] = 0.0, 1.0
    return LotModel(lot_id, container_ids, wet, remaining, draws, mu_hat, mu_var)


@dataclass
class LotDraws:
    """一批的联合模拟结果（各成分相关经由共同的批均值与偏差因子）。"""

    lot_id: str
    sims: int
    container_ids: tuple
    wet: np.ndarray
    remaining: np.ndarray
    moisture: np.ndarray  # [S, C]
    theta: dict  # 非水分成分 -> [S, C]
    dry_full: np.ndarray
    dry_rem: np.ndarray
    qty_full: dict  # analyte -> [S]
    qty_rem: dict

    def quantity(self, analyte, remaining=False):
        return (self.qty_rem if remaining else self.qty_full)[analyte]

    def dry(self, remaining=False):
        return self.dry_rem if remaining else self.dry_full

    def contaminant_fraction(self, remaining=False):
        return self.quantity("contaminant", remaining) / self.dry(remaining)


def simulate(model: LotModel, cal: Calibration, sims=DEFAULT_SIMS, seed=DEFAULT_SEED, censor_mode="interval") -> LotDraws:
    """分层蒙特卡洛：批均值 → 容器真值 → 干基质量与各成分量。

    censor_mode="interval" 时未检出按 [0, 定量限] 抽潜伏值；
    censor_mode="upper" 时未检出按定量限计，用于污染物上界。
    """
    rng = np.random.default_rng(seed)
    n_containers = len(model.container_ids)
    theta = {}
    for analyte in ANALYTES:
        mu = rng.normal(model.mu_hat[analyte], math.sqrt(model.mu_var[analyte]), sims)
        tau2 = max(cal.tau2[analyte], 1e-12)
        columns = []
        for idx, cid in enumerate(model.container_ids):
            draw = model.draws[analyte].get(cid)
            if draw is None:
                column = rng.normal(mu, math.sqrt(tau2), sims)
            elif draw.censored:
                if censor_mode == "upper":
                    # 上界口径：未检出按定量限计，不再向批均值收缩
                    columns.append(np.full(sims, draw.limit))
                    continue
                if cal.pin_censored:
                    latent = np.full(sims, draw.limit / 2.0)
                else:
                    latent = rng.uniform(0.0, draw.limit, sims)
                like_var = cal.meas_var(analyte, float(np.mean(latent)), 1, None) + cal.within_var[analyte]
                like_var = max(like_var, 1e-12)
                post_var = 1.0 / (1.0 / tau2 + 1.0 / like_var)
                post_mean = (mu / tau2 + latent / like_var) * post_var
                column = rng.normal(post_mean, math.sqrt(post_var), sims)
            else:
                like_var = max(draw.likelihood_var(cal), 1e-12)
                post_var = 1.0 / (1.0 / tau2 + 1.0 / like_var)
                post_mean = (mu / tau2 + draw.mean / like_var) * post_var
                column = rng.normal(post_mean, math.sqrt(post_var), sims)
            lo, hi = CLIP[analyte]
            columns.append(np.clip(column, lo, hi))
        theta[analyte] = np.column_stack(columns) if columns else np.zeros((sims, 0))

    moisture = theta["moisture"]
    dry_per_container_full = model.wet[None, :] * (1.0 - moisture)
    dry_per_container_rem = model.remaining[None, :] * (1.0 - moisture)
    dry_full = dry_per_container_full.sum(axis=1)
    dry_rem = dry_per_container_rem.sum(axis=1)
    qty_full, qty_rem = {}, {}
    for analyte in ANALYTES:
        if analyte == "moisture":
            continue
        qty_full[analyte] = (dry_per_container_full * theta[analyte]).sum(axis=1)
        qty_rem[analyte] = (dry_per_container_rem * theta[analyte]).sum(axis=1)
    return LotDraws(
        model.lot_id, sims, model.container_ids, model.wet, model.remaining,
        moisture, {a: t for a, t in theta.items() if a != "moisture"},
        dry_full, dry_rem, qty_full, qty_rem,
    )


def summarize(draws: LotDraws) -> dict:
    """干基质量与各成分可结算量的分布摘要。"""
    out = {"lot_id": draws.lot_id, "sims": draws.sims, "dry": _stats(draws.dry_full), "dry_remaining": _stats(draws.dry_rem)}
    for analyte, values in draws.qty_full.items():
        out[analyte] = _stats(values)
    out["contaminant_fraction"] = _stats(draws.contaminant_fraction())
    return out


def _stats(values):
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "p5": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p975": float(np.percentile(values, 97.5)),
    }


def attribute(bundle: Bundle, cal: Calibration, lot_id, sims=DEFAULT_SIMS, seed=DEFAULT_SEED, value_fn=None) -> dict:
    """方差贡献分解：分别只保留水分、取样、化验来源（一阶分解），看目标量方差占比。

    份额 = 只保留该来源时的方差 / 各来源方差之和（同种子公共随机数）；
    交互作用计入分母但不单列，报告需注明。value_fn 可把 LotDraws 映射为金额等目标量。
    每个分组用对应的校准变体重建批模型（化验方差、位置标准误等已固化在证据中）。
    """
    base_model = build_lot_model(bundle, cal, lot_id)
    base = simulate(base_model, cal, sims, seed)
    targets = {"dry": lambda d: d.dry_full}
    for analyte in base.qty_full:
        targets[analyte] = (lambda a: lambda d: d.qty_full[a])(analyte)
    if value_fn is not None:
        targets["value"] = value_fn

    only_draws = {}
    for group in GROUPS:
        only_cal = cal.variant(keep={group})
        only_model = build_lot_model(bundle, only_cal, lot_id)
        only_draws[group] = simulate(only_model, only_cal, sims, seed)

    result = {}
    for name, fn in targets.items():
        var_base = float(np.var(fn(base), ddof=1))
        only_vars = {g: float(np.var(fn(only_draws[g]), ddof=1)) for g in GROUPS}
        total = sum(only_vars.values())
        shares = {g: (v / total if total > 0 else 0.0) for g, v in only_vars.items()}
        result[name] = {"variance": var_base, "only_variance": only_vars, "share": shares}
    return result
