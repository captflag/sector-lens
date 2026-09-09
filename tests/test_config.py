"""Persona and sector configuration is data; these tests keep it honest."""

from __future__ import annotations

import pytest

from sectorlens.config import (classify_sector, get_persona, load_personas,
                           load_sectors)
from sectorlens.db.metrics import METRICS_BY_CODE


def test_three_personas_are_defined():
    personas = load_personas()
    assert set(personas) == {"mutual_fund_analyst", "equity_analyst", "pe_analyst"}


def test_persona_weights_sum_to_one():
    for persona in load_personas().values():
        total = sum(w.weight for w in persona.screen_weights)
        assert total == pytest.approx(1.0, abs=1e-6), persona.id


def test_every_persona_metric_exists_in_the_registry():
    """A weight naming a metric that cannot be stored would silently do nothing."""
    for persona in load_personas().values():
        for weight in persona.screen_weights:
            assert weight.metric in METRICS_BY_CODE, weight.metric
        for metric in persona.priority_metrics:
            assert metric in METRICS_BY_CODE, metric


def test_personas_disagree_on_direction_somewhere():
    """The whole design rests on the personas not being reskins of each other."""
    mf = get_persona("mutual_fund_analyst")
    pe = get_persona("pe_analyst")
    mf_cap = mf.weight_for("market_cap")
    pe_cap = pe.weight_for("market_cap")
    assert mf_cap and pe_cap
    assert mf_cap.direction != pe_cap.direction, (
        "mutual fund and PE personas must score company size in opposite "
        "directions -- that inversion is what makes their rankings differ")


def test_sector_classification_prefers_sub_industry_over_sector():
    # Air freight sits inside GICS Industrials but must land in logistics,
    # not manufacturing.
    assert classify_sector("Industrials", "Air Freight & Logistics") == "logistics"
    assert classify_sector("Industrials", "Aerospace & Defense") == "manufacturing"
    assert classify_sector("Information Technology", "Systems Software") == "tech"


def test_unmapped_sector_returns_none():
    assert classify_sector("Health Care", "Biotechnology") is None


def test_unknown_persona_raises():
    with pytest.raises(KeyError):
        get_persona("day_trader")
