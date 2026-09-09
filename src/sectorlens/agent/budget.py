"""A daily ceiling on model-backed requests.

A public demo carries the owner's API key, which makes it free credit for
anyone who finds the URL. This caps how many requests per day may reach the
model; past the cap the deterministic provider answers instead, so the demo
keeps working and the bill stops growing.

The counter is process-local and resets on the UTC date. That is deliberate:
persisting it would mean a writable store the rest of the system does not need,
and for a single-process demo the restart-resets-the-budget failure mode is
cheap. Anything with a real cost model would move this behind a shared counter.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone


@dataclass
class BudgetStatus:
    limited: bool
    limit: int
    spent: int
    remaining: int
    day: str

    def as_dict(self) -> dict[str, object]:
        return {
            "limited": self.limited,
            "daily_limit": self.limit,
            "spent_today": self.spent,
            "remaining_today": self.remaining,
            "utc_day": self.day,
        }


class RequestBudget:
    """Counts model-backed requests against a daily allowance."""

    def __init__(self, daily_limit: int) -> None:
        #: 0 or below means no limit, which is the right default for local use.
        self._limit = max(0, int(daily_limit or 0))
        self._lock = threading.Lock()
        self._day: date = self._today()
        self._spent = 0

    @staticmethod
    def _today() -> date:
        return datetime.now(timezone.utc).date()

    def _roll(self) -> None:
        """Reset the counter when the UTC day changes. Caller holds the lock."""
        today = self._today()
        if today != self._day:
            self._day = today
            self._spent = 0

    def try_spend(self) -> bool:
        """Claim one model-backed request. False when the day's budget is gone.

        Counted on claim rather than on success: a request that reaches the
        provider and fails has still cost something, and under-counting is the
        expensive direction to be wrong in.
        """
        if not self._limit:
            return True
        with self._lock:
            self._roll()
            if self._spent >= self._limit:
                return False
            self._spent += 1
            return True

    def status(self) -> BudgetStatus:
        with self._lock:
            self._roll()
            return BudgetStatus(
                limited=bool(self._limit),
                limit=self._limit,
                spent=self._spent,
                remaining=(self._limit - self._spent) if self._limit else -1,
                day=self._day.isoformat(),
            )
