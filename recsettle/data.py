"""业务资料载入、类型化记录与可知时刻（available_at）过滤。

载入前先运行 sourcepack.catalog.inspect 核对资料完整性；
as_of 过滤保证历史截点看不到更晚才知道的事实。
"""
from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from sourcepack.catalog import inspect

# 资料口径中的成分；moisture 为水分（湿基质量比例），其余为干基比例。
ANALYTES = ("moisture", "nickel", "lithium", "contaminant")
METALS = ("nickel", "lithium")


def _text(row, name):
    value = row[name]
    return value if value != "" else None


def _num(row, name):
    value = row[name]
    return float(value) if value != "" else None


def _int(row, name):
    value = row[name]
    return int(value) if value != "" else None


def _date(row, name):
    value = row[name]
    return dt.date.fromisoformat(value) if value != "" else None


def _stamp(row, name):
    value = row[name]
    return dt.datetime.fromisoformat(value) if value != "" else None


def _bool(row, name):
    return row[name] == "true"


@dataclass(frozen=True)
class Lot:
    lot_id: str
    supplier_id: str
    contract_id: str
    received: dt.date
    settlement_due: dt.date
    state: str  # provisional / disputed / accepted / arbitration / settled


@dataclass(frozen=True)
class Container:
    container_id: str
    lot_id: str
    wet_kg: float
    allocated_kg: float

    @property
    def remaining_kg(self):
        return self.wet_kg - self.allocated_kg


@dataclass(frozen=True)
class Allocation:
    allocation_id: str
    container_id: str
    line_id: str
    wet_kg: float
    started_at: dt.datetime
    state: str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    container_id: str
    position: str  # top / mixed / retained
    sampled: dt.date
    remaining_grams: float
    retest_of: str | None  # 非空表示同一留样复测，不增加容器代表性

    @property
    def is_retest(self):
        return bool(self.retest_of)


@dataclass(frozen=True)
class Assay:
    assay_id: str
    sample_id: str
    lab_id: str
    analyte: str
    qualifier: str  # value / lt
    value: float | None
    limit: float | None
    available_at: dt.datetime

    @property
    def censored(self):
        return self.qualifier == "lt"


@dataclass(frozen=True)
class Reference:
    reference_id: str
    lab_id: str
    analyte: str
    assigned: float
    measured: float
    uncertainty: float
    available_at: dt.datetime


@dataclass(frozen=True)
class Contract:
    contract_id: str
    payable_fraction: float
    processing_fee_kg: float
    contaminant_penalty_kg: float
    soft_contaminant_limit: float
    accepted_basis: str
    provisional_fraction: float


@dataclass(frozen=True)
class Price:
    analyte: str
    yuan_kg: float
    date: dt.date


@dataclass(frozen=True)
class Line:
    line_id: str
    hard_contaminant_limit: float
    capacity_wet_kg: float
    cost_fixed: float
    free_batches: int


@dataclass(frozen=True)
class Run:
    run_id: str
    line_id: str
    wet_kg: float
    dry_kg: float
    nickel_fraction: float
    lithium_fraction: float
    contaminant_fraction: float
    recovered_nickel_kg: float
    recovered_lithium_kg: float
    cost: float
    available_at: dt.datetime


@dataclass(frozen=True)
class Payment:
    payment_id: str
    lot_id: str
    amount: float
    paid_at: dt.datetime


@dataclass(frozen=True)
class SamplingOption:
    option_id: str
    cost: float
    grams: float
    days: int
    new_container_sample: bool


@dataclass(frozen=True)
class Bundle:
    """一批业务资料；字典键为记录键。as_of 已按可知时刻过滤。"""

    lots: dict[str, Lot]
    containers: dict[str, Container]
    allocations: dict[str, Allocation]
    samples: dict[str, Sample]
    assays: dict[str, Assay]
    references: dict[str, Reference]
    contracts: dict[str, Contract]
    prices: tuple[Price, ...]
    lines: dict[str, Line]
    runs: dict[str, Run]
    payments: dict[str, Payment]
    options: dict[str, SamplingOption]

    def as_of(self, when: dt.datetime | None) -> "Bundle":
        """只保留 when 时刻已可知的事实；None 表示不过滤。"""
        if when is None:
            return self
        day = when.date() if isinstance(when, dt.datetime) else when
        return Bundle(
            lots=self.lots,
            containers=self.containers,
            allocations={k: v for k, v in self.allocations.items() if v.started_at <= when},
            samples={k: v for k, v in self.samples.items() if v.sampled <= day},
            assays={k: v for k, v in self.assays.items() if v.available_at <= when},
            references={k: v for k, v in self.references.items() if v.available_at <= when},
            contracts=self.contracts,
            prices=self.prices,  # 价格不按截点过滤；查询时按日取价，早于最早记录时退回最早一条
            lines=self.lines,
            runs={k: v for k, v in self.runs.items() if v.available_at <= when},
            payments={k: v for k, v in self.payments.items() if v.paid_at <= when},
            options=self.options,
        )

    def lot_containers(self, lot_id) -> list[Container]:
        return [c for c in self.containers.values() if c.lot_id == lot_id]

    def lot_allocations(self, lot_id) -> list[Allocation]:
        ids = {c.container_id for c in self.lot_containers(lot_id)}
        return [a for a in self.allocations.values() if a.container_id in ids]

    def lot_payments(self, lot_id) -> list[Payment]:
        return [p for p in self.payments.values() if p.lot_id == lot_id]

    def price_on(self, analyte, day=None) -> Price | None:
        """不晚于 day 的最新价格；若无，则退回最早一条（价格史不在资料内，

        各截点共用同一参考价，价格风险不另建模，报告需注明）。"""
        choices = [p for p in self.prices if p.analyte == analyte and (day is None or p.date <= day)]
        if choices:
            return max(choices, key=lambda p: p.date)
        fallback = [p for p in self.prices if p.analyte == analyte]
        return min(fallback, key=lambda p: p.date) if fallback else None


def _rows(root, name):
    with (Path(root) / "data" / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load(root) -> Bundle:
    """核对资料完整性后载入全部表。"""
    inspect(root)  # 摘要、字段、记录键、引用不符时直接抛错
    root = Path(root)
    lots = {r["lot_id"]: Lot(r["lot_id"], r["supplier_id"], r["contract_id"], _date(r, "received"), _date(r, "settlement_due"), r["state"]) for r in _rows(root, "lots")}
    containers = {r["container_id"]: Container(r["container_id"], r["lot_id"], _num(r, "wet_kg"), _num(r, "allocated_kg")) for r in _rows(root, "containers")}
    allocations = {r["allocation_id"]: Allocation(r["allocation_id"], r["container_id"], r["line_id"], _num(r, "wet_kg"), _stamp(r, "started_at"), r["state"]) for r in _rows(root, "allocations")}
    samples = {r["sample_id"]: Sample(r["sample_id"], r["container_id"], r["position"], _date(r, "sampled"), _num(r, "remaining_grams"), _text(r, "retest_of")) for r in _rows(root, "samples")}
    assays = {r["assay_id"]: Assay(r["assay_id"], r["sample_id"], r["lab_id"], r["analyte"], r["qualifier"], _num(r, "value"), _num(r, "limit"), _stamp(r, "available_at")) for r in _rows(root, "assays")}
    references = {r["reference_id"]: Reference(r["reference_id"], r["lab_id"], r["analyte"], _num(r, "assigned"), _num(r, "measured"), _num(r, "uncertainty"), _stamp(r, "available_at")) for r in _rows(root, "references")}
    contracts = {r["contract_id"]: Contract(r["contract_id"], _num(r, "payable_fraction"), _num(r, "processing_fee_kg"), _num(r, "contaminant_penalty_kg"), _num(r, "soft_contaminant_limit"), r["accepted_basis"], _num(r, "provisional_fraction")) for r in _rows(root, "contracts")}
    prices = tuple(Price(r["analyte"], _num(r, "yuan_kg"), _date(r, "date")) for r in _rows(root, "prices"))
    lines = {r["line_id"]: Line(r["line_id"], _num(r, "hard_contaminant_limit"), _num(r, "capacity_wet_kg"), _num(r, "cost_fixed"), _int(r, "free_batches")) for r in _rows(root, "lines")}
    runs = {r["run_id"]: Run(r["run_id"], r["line_id"], _num(r, "wet_kg"), _num(r, "dry_kg"), _num(r, "nickel_fraction"), _num(r, "lithium_fraction"), _num(r, "contaminant_fraction"), _num(r, "recovered_nickel_kg"), _num(r, "recovered_lithium_kg"), _num(r, "cost"), _stamp(r, "available_at")) for r in _rows(root, "processing_history")}
    payments = {r["payment_id"]: Payment(r["payment_id"], r["lot_id"], _num(r, "amount"), _stamp(r, "paid_at")) for r in _rows(root, "payments")}
    options = {r["option_id"]: SamplingOption(r["option_id"], _num(r, "cost"), _num(r, "grams"), _int(r, "days"), _bool(r, "new_container_sample")) for r in _rows(root, "sampling_options")}
    return Bundle(lots, containers, allocations, samples, assays, references, contracts, prices, lines, runs, payments, options)
