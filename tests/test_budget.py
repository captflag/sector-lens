"""The daily model-request budget.

A public deployment carries the owner's key, so the cap is what stops the URL
being free credit. It has to fail closed -- toward the deterministic provider,
never toward an uncapped model call.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from sectorlens.agent import llm as llm_module
from sectorlens.agent.budget import RequestBudget
from sectorlens.agent.llm import DeterministicProvider, build_provider
from sectorlens.settings import Settings


def test_zero_means_unlimited():
    budget = RequestBudget(0)
    assert all(budget.try_spend() for _ in range(500))
    assert budget.status().limited is False


def test_a_limit_is_enforced_exactly():
    budget = RequestBudget(3)
    assert [budget.try_spend() for _ in range(5)] == [True, True, True, False, False]
    status = budget.status()
    assert (status.spent, status.remaining) == (3, 0)


def test_the_allowance_resets_on_a_new_utc_day():
    budget = RequestBudget(2)
    assert budget.try_spend() and budget.try_spend()
    assert budget.try_spend() is False

    budget._day = date.today() - timedelta(days=1)   # simulate the day rolling
    assert budget.try_spend() is True
    assert budget.status().spent == 1


def test_spend_is_counted_on_claim_not_on_success():
    """A request that reaches the provider and fails has still cost money, so
    under-counting is the expensive direction to be wrong in."""
    budget = RequestBudget(1)
    assert budget.try_spend() is True
    assert budget.try_spend() is False


@pytest.fixture(autouse=True)
def _reset_process_budget():
    llm_module._BUDGET = None
    yield
    llm_module._BUDGET = None


def test_an_exhausted_budget_degrades_instead_of_failing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings = Settings(llm_provider="anthropic", daily_llm_budget=1,
                        anthropic_api_key="sk-ant-test")

    first = build_provider(settings)
    assert first.name == "anthropic", "the first request should reach the model"

    second = build_provider(settings)
    assert isinstance(second, DeterministicProvider)
    assert second.reason == "daily_budget_exhausted"


def test_the_degraded_answer_says_why():
    settings = Settings(llm_provider="deterministic")
    note = DeterministicProvider(settings, reason="daily_budget_exhausted")._provider_note()
    assert "budget" in note.lower()
    assert "00:00 utc" in note.lower(), "a user should know when it comes back"
    # And the honest part: the data layer is untouched.
    assert "unaffected" in note.lower()


def test_no_budget_configured_never_degrades(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    settings = Settings(llm_provider="anthropic", daily_llm_budget=0,
                        anthropic_api_key="sk-ant-test")
    assert all(build_provider(settings).name == "anthropic" for _ in range(20))
