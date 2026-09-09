"""The metric registry.

The facts table is EAV, so this registry is what keeps it self-describing:
every `metric_code` written to `company_metrics` must exist here, which stops
adapters from silently inventing near-duplicate metric names.

`higher_is_better` is None where the answer genuinely depends on the lens --
`ebitda_margin_gap_to_sector` is the clearest case: an equity analyst reads a
negative gap as a quality problem, a buyout shop reads it as headroom.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDef:
    code: str
    label: str
    unit: str  # usd | ratio | percent | count | multiple
    description: str
    higher_is_better: bool | None = None


METRICS: tuple[MetricDef, ...] = (
    # -- market snapshot (reported) -----------------------------------------
    MetricDef("price", "Share price", "usd", "Last close at snapshot date."),
    MetricDef("market_cap", "Market capitalisation", "usd",
              "Equity market value at snapshot date.", None),
    MetricDef("pe_ratio", "P/E ratio", "multiple",
              "Price divided by trailing earnings per share.", False),
    MetricDef("price_to_sales", "Price / sales", "multiple",
              "Market cap divided by trailing revenue.", False),
    MetricDef("price_to_book", "Price / book", "multiple",
              "Market cap divided by book value of equity.", False),
    MetricDef("dividend_yield", "Dividend yield", "ratio",
              "Trailing dividend divided by price.", True),
    MetricDef("eps", "Earnings per share", "usd", "Trailing EPS.", True),
    MetricDef("week52_low", "52-week low", "usd", "Lowest close over 52 weeks."),
    MetricDef("week52_high", "52-week high", "usd", "Highest close over 52 weeks."),

    # -- income statement ----------------------------------------------------
    MetricDef("revenue_ttm", "Revenue (TTM)", "usd",
              "Trailing twelve month revenue.", True),
    MetricDef("ebitda", "EBITDA", "usd",
              "Earnings before interest, tax, depreciation and amortisation.", True),
    MetricDef("gross_profit", "Gross profit", "usd", "Revenue less cost of sales.", True),
    MetricDef("operating_income", "Operating income", "usd", "EBIT.", True),
    MetricDef("net_income_ttm", "Net income (TTM)", "usd",
              "Trailing twelve month net income.", True),
    MetricDef("shares_outstanding", "Shares outstanding", "count",
              "Common shares outstanding.", None),

    # -- balance sheet / cash flow ------------------------------------------
    MetricDef("total_debt", "Total debt", "usd",
              "Short plus long term borrowings.", False),
    MetricDef("cash_and_equivalents", "Cash and equivalents", "usd",
              "Cash, equivalents and short term investments.", True),
    MetricDef("net_debt", "Net debt", "usd", "Total debt less cash.", False),
    MetricDef("free_cash_flow", "Free cash flow", "usd",
              "Operating cash flow less capital expenditure.", True),

    # -- ratios (usually derived) -------------------------------------------
    MetricDef("ebitda_margin", "EBITDA margin", "ratio",
              "EBITDA divided by revenue.", True),
    MetricDef("net_margin", "Net margin", "ratio",
              "Net income divided by revenue.", True),
    MetricDef("gross_margin", "Gross margin", "ratio",
              "Gross profit divided by revenue.", True),
    MetricDef("earnings_yield", "Earnings yield", "ratio",
              "Inverse of P/E; cheapness on a higher-is-better axis.", True),
    MetricDef("ev_to_ebitda_proxy", "EV / EBITDA (proxy)", "multiple",
              "Market cap divided by EBITDA. Excludes net debt when balance "
              "sheet data is unavailable, so it understates leverage-heavy "
              "businesses -- always presented as a proxy, never as true EV.",
              False),
    MetricDef("net_debt_to_ebitda", "Net debt / EBITDA", "multiple",
              "Turns of leverage already on the balance sheet.", False),
    MetricDef("fcf_conversion", "FCF conversion", "ratio",
              "Free cash flow divided by EBITDA.", True),
    MetricDef("revenue_growth_yoy", "Revenue growth (YoY)", "ratio",
              "Year on year revenue growth.", True),
    MetricDef("price_vs_52w_range", "Position in 52-week range", "ratio",
              "0 = at the 52-week low, 1 = at the 52-week high.", None),

    # -- sector-relative (computed once benchmarks exist) --------------------
    MetricDef("ebitda_margin_gap_to_sector", "EBITDA margin gap to sector", "ratio",
              "Sector median EBITDA margin minus the company's. Positive means "
              "the company runs BELOW its peers -- read as a quality problem by "
              "public-market personas and as operational headroom by PE.", None),
    MetricDef("pe_vs_sector_median", "P/E vs sector median", "ratio",
              "Company P/E divided by the sector median P/E. Below 1.0 is a "
              "discount to peers.", False),

    # -- workforce -----------------------------------------------------------
    MetricDef("employees", "Employees", "count",
              "Headcount as disclosed on the most recent annual filing cover.",
              None),
)

METRICS_BY_CODE: dict[str, MetricDef] = {m.code: m for m in METRICS}


def require(code: str) -> MetricDef:
    if code not in METRICS_BY_CODE:
        raise KeyError(
            f"metric {code!r} is not in the registry; add it to metrics.py first"
        )
    return METRICS_BY_CODE[code]
