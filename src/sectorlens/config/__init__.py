"""Loads persona and sector definitions from YAML into typed objects.

Personas and sectors are data, not code: adding a sector or re-weighting a
persona is a YAML edit plus an ingest re-run, with no Python change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ScreenWeight:
    metric: str
    weight: float
    direction: str  # "high" | "low"

    @property
    def higher_is_better(self) -> bool:
        return self.direction == "high"


@dataclass(frozen=True)
class Persona:
    id: str
    label: str
    short_label: str
    lens: str
    priority_metrics: tuple[str, ...]
    screen_weights: tuple[ScreenWeight, ...]
    rationale: str
    answer_sections: tuple[str, ...]
    vocabulary: tuple[str, ...]
    guardrails: str

    def weight_for(self, metric: str) -> ScreenWeight | None:
        return next((w for w in self.screen_weights if w.metric == metric), None)


@dataclass(frozen=True)
class Sector:
    id: str
    label: str
    description: str
    gics_sectors: tuple[str, ...] = field(default_factory=tuple)
    gics_sub_industries: tuple[str, ...] = field(default_factory=tuple)

    def matches(self, gics_sector: str, gics_sub_industry: str) -> bool:
        return (
            gics_sector in self.gics_sectors
            or gics_sub_industry in self.gics_sub_industries
        )


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text())


@lru_cache(maxsize=1)
def load_personas() -> dict[str, Persona]:
    raw = _load_yaml("personas.yaml")["personas"]
    out: dict[str, Persona] = {}
    for pid, body in raw.items():
        weights = tuple(
            ScreenWeight(metric=m, weight=float(w["weight"]), direction=w["direction"])
            for m, w in body["screen_weights"].items()
        )
        out[pid] = Persona(
            id=pid,
            label=body["label"],
            short_label=body["short_label"],
            lens=body["lens"].strip(),
            priority_metrics=tuple(body["priority_metrics"]),
            screen_weights=weights,
            rationale=body["rationale"].strip(),
            answer_sections=tuple(body["answer_sections"]),
            vocabulary=tuple(body["vocabulary"]),
            guardrails=body["guardrails"].strip(),
        )
    return out


@lru_cache(maxsize=1)
def load_sectors() -> dict[str, Sector]:
    raw = _load_yaml("sectors.yaml")["sectors"]
    return {
        sid: Sector(
            id=sid,
            label=body["label"],
            description=body["description"].strip(),
            gics_sectors=tuple(body.get("gics_sectors") or ()),
            gics_sub_industries=tuple(body.get("gics_sub_industries") or ()),
        )
        for sid, body in raw.items()
    }


def get_persona(persona_id: str) -> Persona:
    personas = load_personas()
    if persona_id not in personas:
        raise KeyError(
            f"unknown persona {persona_id!r}; valid: {sorted(personas)}"
        )
    return personas[persona_id]


def get_sector(sector_id: str) -> Sector:
    sectors = load_sectors()
    if sector_id not in sectors:
        raise KeyError(f"unknown sector {sector_id!r}; valid: {sorted(sectors)}")
    return sectors[sector_id]


def classify_sector(gics_sector: str, gics_sub_industry: str) -> str | None:
    """Map a GICS classification onto our canonical sector id, if any.

    Sub-industry rules are checked before whole-sector rules so a narrow
    override (e.g. logistics inside GICS Industrials) wins over a broad one.
    """
    sectors = load_sectors()
    for s in sectors.values():
        if gics_sub_industry in s.gics_sub_industries:
            return s.id
    for s in sectors.values():
        if gics_sector in s.gics_sectors:
            return s.id
    return None
