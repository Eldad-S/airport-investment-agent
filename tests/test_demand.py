from src.analytics.demand import REQUIRED_DATA_UNMET_DEMAND, assess_demand
from src.analytics.metrics import build_airport_table
from src.analytics.scoring import score_airports


def test_demand_assessment_never_fabricates_unmet_demand(tiny_dataset):
    table = build_airport_table(tiny_dataset)
    scored = score_airports(table)
    d = assess_demand(tiny_dataset, table, scored, "AAA")
    assert d.unmet_demand_estimate is None
    assert "cannot be quantified" in d.unmet_demand_statement
    assert d.required_data == REQUIRED_DATA_UNMET_DEMAND
    assert any("T-100" in name for name, _ in d.required_data)


def test_demand_layers_are_separate(tiny_dataset):
    table = build_airport_table(tiny_dataset)
    d = assess_demand(tiny_dataset, table, score_airports(table), "AAA")
    assert any("enplanements" in k.lower() for k in d.observed)
    assert any("growth" in k.lower() for k in d.growth)
    assert d.pressure_level in ("Elevated", "Moderate", "Low", "Insufficient evidence")


def test_airport_outside_peer_set_is_insufficient_evidence(tiny_dataset):
    table = build_airport_table(tiny_dataset)
    scored = score_airports(table)
    assert "CCC" not in scored.index or scored.loc["CCC", "status"] == "scored"
    d = assess_demand(tiny_dataset, table, scored.drop(index="CCC", errors="ignore"), "CCC")
    assert d.pressure_level == "Insufficient evidence"
