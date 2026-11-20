"""测试用微型资料构造器：字段齐全的最小 Bundle，便于手算核对。"""
import datetime as dt

from recsettle.data import (
    Allocation,
    Assay,
    Bundle,
    Container,
    Contract,
    Line,
    Lot,
    Payment,
    Price,
    Reference,
    Run,
    Sample,
    SamplingOption,
)

TZ = dt.timezone(dt.timedelta(hours=8))
DAY = dt.date(2025, 1, 1)
STAMP = dt.datetime(2025, 1, 2, 8, 0, tzinfo=TZ)


def make_bundle(
    containers=(("C1", 100.0), ("C2", 200.0)),
    moisture=(0.10, 0.20),
    nickel=(0.10, 0.12),
    lithium=(0.02, 0.02),
    contaminant=(0.001, 0.001),
    contaminant_lt=False,
    allocated=None,
    payments=(),
    state="provisional",
):
    """两个容器、每容器一个 mixed 独立取样的最小批；化验值按参数给出。"""
    lots = {"L1": Lot("L1", "SUP", "K1", DAY, dt.date(2025, 3, 1), state)}
    containers_d = {
        cid: Container(cid, "L1", wet, (allocated or {}).get(cid, 0.0))
        for (cid, wet) in containers
    }
    contracts = {"K1": Contract("K1", 0.85, 4.0, 800.0, 0.002, "mutual_retest", 0.7)}
    prices = (
        Price("nickel", 120.0, DAY),
        Price("lithium", 220.0, DAY),
    )
    references = {}
    for lab in ("LAB1",):
        for analyte, level in (("moisture", 0.1), ("nickel", 0.15), ("lithium", 0.03), ("contaminant", 0.003)):
            references[f"R-{analyte}"] = Reference(f"R-{analyte}", lab, analyte, level, level, level * 0.001, STAMP)
    samples, assays = {}, {}
    for i, (cid, _wet) in enumerate(containers):
        sid = f"S{i}"
        samples[sid] = Sample(sid, cid, "mixed", DAY, 500.0, None)
        values = {"moisture": moisture[i], "nickel": nickel[i], "lithium": lithium[i], "contaminant": contaminant[i]}
        for analyte, value in values.items():
            if analyte == "contaminant" and contaminant_lt:
                assays[f"A{i}-{analyte}"] = Assay(f"A{i}-{analyte}", sid, "LAB1", analyte, "lt", None, 0.002, STAMP)
            else:
                assays[f"A{i}-{analyte}"] = Assay(f"A{i}-{analyte}", sid, "LAB1", analyte, "value", value, None, STAMP)
    payments_d = {f"PAY{k}": Payment(f"PAY{k}", "L1", amount, STAMP) for k, amount in enumerate(payments)}
    lines = {
        "LINE1": Line("LINE1", 0.004, 3000.0, 2000.0, 2),
        "LINE2": Line("LINE2", 0.003, 2500.0, 1500.0, 2),
    }
    options = {
        "retained": SamplingOption("retained", 120.0, 100.0, 2, False),
        "resample": SamplingOption("resample", 350.0, 500.0, 4, True),
    }
    return Bundle(
        lots=lots,
        containers=containers_d,
        allocations={},
        samples=samples,
        assays=assays,
        references=references,
        contracts=contracts,
        prices=prices,
        lines=lines,
        runs={},
        payments=payments_d,
        options=options,
    )
