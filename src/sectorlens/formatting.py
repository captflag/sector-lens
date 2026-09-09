"""Small display helpers shared by the deterministic provider and the scripts.

They live here rather than being duplicated so a formatting bug is fixed once.
"""

from __future__ import annotations


def ordinal(n: int) -> str:
    """1 -> 1st, 11 -> 11th, 82 -> 82nd. The teens are the whole trick."""
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def pct_rank(fraction: float) -> str:
    """Render a 0..1 percentile as '73rd pct'."""
    return f"{ordinal(round(fraction * 100))} pct"


def fmt_value(value: float | None, unit: str | None) -> str:
    """Format a metric for humans, using the unit from the metric registry."""
    if value is None:
        return "n/a"
    if unit == "usd":
        for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(value) >= cutoff:
                return f"${value / cutoff:.1f}{suffix}"
        return f"${value:,.0f}"
    if unit in ("ratio", "percent"):
        return f"{value * 100:.1f}%"
    if unit == "multiple":
        return f"{value:.1f}x"
    if unit == "count":
        return f"{value:,.0f}"
    return f"{value:,.2f}"
