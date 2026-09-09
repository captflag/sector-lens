#!/usr/bin/env python3
"""Render the persona-divergence figure used in the README.

Generated from the live database rather than hand-drawn, so the picture cannot
drift away from what `make demo` prints. Two variants are written -- light and
dark -- and the README picks between them on the reader's GitHub theme.

    python scripts/make_hero_image.py

Regenerate after a rebuild; the upstream data refreshes daily and the ordering
moves with it.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sectorlens.agent.mcp_client import open_toolbox  # noqa: E402
from sectorlens.config import load_personas  # noqa: E402

PERSONAS = ["mutual_fund_analyst", "equity_analyst", "pe_analyst"]
SECTOR = "tech"
TOP_N = 5
OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"

SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    panel: str
    ink: str
    muted: str
    rule: str
    accents: tuple[str, str, str]


LIGHT = Theme("light", "#FFFFFF", "#F5F7F8", "#12171D", "#667586", "#DCE1E6",
              ("#3E5C8A", "#14706A", "#A2552A"))
DARK = Theme("dark", "#0D1117", "#161B22", "#E8EDF3", "#8D9BAB", "#2A313B",
             ("#86A6D8", "#4FC2B3", "#D9945F"))

W, H = 1280, 620
PAD = 48
COL_GAP = 24


def font(path: str, size: int):
    if not Path(path).exists():
        win_map = {SANS: 'C:/Windows/Fonts/arial.ttf', SANS_BOLD: 'C:/Windows/Fonts/arialbd.ttf', MONO: 'C:/Windows/Fonts/consola.ttf', MONO_BOLD: 'C:/Windows/Fonts/consolab.ttf'}
        fb = win_map.get(path, 'C:/Windows/Fonts/arial.ttf')
        if Path(fb).exists(): return ImageFont.truetype(fb, size)
        return ImageFont.load_default()
    return ImageFont.truetype(path, size)


async def fetch() -> tuple[dict[str, list[dict]], int]:
    tables: dict[str, list[dict]] = {}
    universe = 0
    async with open_toolbox() as toolbox:
        for persona in PERSONAS:
            data = await toolbox.call_json(
                "screen_sector",
                {"sector": SECTOR, "persona": persona, "limit": TOP_N})
            tables[persona] = data.get("results", [])
            universe = data.get("universe_size", universe)
    return tables, universe


def render(theme: Theme, tables: dict[str, list[dict]], universe: int) -> Image.Image:
    labels = {p: load_personas()[p].short_label for p in PERSONAS}
    img = Image.new("RGB", (W, H), theme.bg)
    d = ImageDraw.Draw(img)

    f_title = font(SANS_BOLD, 34)
    f_sub = font(SANS, 17)
    f_col = font(SANS_BOLD, 19)
    f_row = font(MONO, 19)
    f_rank = font(MONO_BOLD, 19)
    f_small = font(SANS, 15)
    f_foot = font(SANS_BOLD, 19)

    d.text((PAD, 40), "Same sector. Same rows. Three lenses.",
           font=f_title, fill=theme.ink)
    d.text((PAD, 86),
           f"{universe} technology companies, identical for all three personas "
           f"— ranked by each persona's own weighting, inside the retrieval tool.",
           font=f_sub, fill=theme.muted)

    top = 140
    panel_h = 340
    col_w = (W - 2 * PAD - 2 * COL_GAP) // 3

    for i, persona in enumerate(PERSONAS):
        x = PAD + i * (col_w + COL_GAP)
        accent = theme.accents[i]

        d.rectangle([x, top, x + col_w, top + panel_h], fill=theme.panel)
        d.rectangle([x, top, x + col_w, top + 5], fill=accent)
        d.text((x + 20, top + 26), labels[persona], font=f_col, fill=accent)
        d.line([x + 20, top + 62, x + col_w - 20, top + 62], fill=theme.rule)

        for j, rec in enumerate(tables[persona][:TOP_N]):
            y = top + 84 + j * 46
            d.text((x + 20, y), f"{j + 1}.", font=f_rank, fill=theme.muted)
            d.text((x + 52, y), rec["ticker"], font=f_rank, fill=theme.ink)
            d.text((x + col_w - 20, y), f"{rec['score']:.2f}",
                   font=f_row, fill=theme.muted, anchor="ra")

    shared = set.intersection(*[{r["ticker"] for r in tables[p]} for p in PERSONAS])
    verdict = (f"Companies appearing in all three top-{TOP_N}: "
               f"{', '.join(sorted(shared)) if shared else 'none'}")
    d.text((PAD, top + panel_h + 38), verdict, font=f_foot, fill=theme.ink)
    d.text((PAD, top + panel_h + 68),
           "The persona changes what is retrieved, not just how it is worded. "
           "Reproduce with `make demo` — no API key required.",
           font=f_small, fill=theme.muted)
    return img


def main() -> int:
    tables, universe = asyncio.run(fetch())
    if not all(tables.values()):
        print("no screen results — build the database first", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for theme in (LIGHT, DARK):
        path = OUT_DIR / f"personas-{theme.name}.png"
        render(theme, tables, universe).save(path, optimize=True)
        print(f"wrote {path.relative_to(Path.cwd())} "
              f"({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
