"""A scripted stand-in for the Anthropic Messages API.

There is no API key in CI, so without this the agentic loop in
`AnthropicProvider` would ship unexecuted. This fake reproduces the parts of
the response surface the loop actually touches -- content blocks, `stop_reason`,
`stop_details` -- and records every outbound request, which is what lets the
tests assert on message shape (role alternation, tool_result pairing,
`tool_choice`) rather than only on the returned answer.

It is a test double, not a mock of behaviour: responses are scripted per test,
and errors are scripted the same way so the downgrade paths can be driven.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str = "toolu_test"
    type: str = "tool_use"


@dataclass
class StopDetails:
    type: str = "refusal"
    category: str | None = None
    explanation: str | None = None


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str = "end_turn"
    stop_details: StopDetails | None = None


def answer_message(**overrides: Any) -> FakeMessage:
    """A well-formed final answer matching the structured output contract."""
    payload = {
        "answer": "Scripted analysis.",
        "companies_referenced": ["BIGC"],
        "evidence": [{
            "company": "BIGC", "metric": "ebitda_margin", "value": 0.36,
            "unit": "ratio", "context": "top quartile", "source": "fixture",
            "as_of": None, "is_derived": True,
        }],
        "caveats": ["Scripted."],
        "out_of_scope": [],
        "confidence": "medium",
    }
    payload.update(overrides)
    return FakeMessage(content=[TextBlock(text=json.dumps(payload))])


def tool_message(name: str, arguments: dict[str, Any] | None = None,
                 block_id: str = "toolu_1") -> FakeMessage:
    return FakeMessage(
        content=[ToolUseBlock(name=name, input=arguments or {}, id=block_id)],
        stop_reason="tool_use")


@dataclass
class RecordedRequest:
    kwargs: dict[str, Any]
    used_beta: bool

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.kwargs.get("messages", [])

    @property
    def roles(self) -> list[str]:
        return [m["role"] for m in self.messages]


class _Endpoint:
    def __init__(self, owner: "FakeAnthropic", used_beta: bool) -> None:
        self._owner = owner
        self._used_beta = used_beta

    async def create(self, **kwargs: Any) -> FakeMessage:
        # The beta endpoint carries these; strip them so recorded requests are
        # comparable between the two paths.
        kwargs.pop("betas", None)
        kwargs.pop("fallbacks", None)
        self._owner.requests.append(
            RecordedRequest(kwargs=kwargs, used_beta=self._used_beta))
        return self._owner._next()


class _Beta:
    def __init__(self, owner: "FakeAnthropic") -> None:
        self.messages = _Endpoint(owner, used_beta=True)


class FakeAnthropic:
    """Returns scripted responses in order; a scripted Exception is raised.

    When the script runs out, the last entry repeats -- so a test that wants
    "the model always calls a tool" scripts one tool response rather than
    guessing how many rounds the loop will take.
    """

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[RecordedRequest] = []
        self._index = 0
        self.messages = _Endpoint(self, used_beta=False)
        self.beta = _Beta(self)

    def _next(self) -> FakeMessage:
        if not self.script:
            raise AssertionError("FakeAnthropic called with an empty script")
        entry = self.script[min(self._index, len(self.script) - 1)]
        self._index += 1
        if isinstance(entry, Exception):
            raise entry
        return entry

    # -- assertions shared by several tests --------------------------------

    def assert_roles_alternate(self) -> None:
        """The Messages API rejects two consecutive turns of the same role."""
        for request in self.requests:
            roles = request.roles
            for earlier, later in zip(roles, roles[1:]):
                assert earlier != later, (
                    f"consecutive {earlier!r} messages sent to the API: {roles}")

    def assert_tool_results_pair_with_tool_uses(self) -> None:
        """Every tool_use id must be answered by a tool_result in the next turn."""
        for request in self.requests:
            pending: set[str] = set()
            for message in request.messages:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                if message["role"] == "assistant":
                    pending = {
                        b.id for b in content
                        if getattr(b, "type", None) == "tool_use"
                    }
                elif pending:
                    answered = {
                        b.get("tool_use_id") for b in content
                        if isinstance(b, dict) and b.get("type") == "tool_result"
                    }
                    assert pending <= answered, (
                        f"unanswered tool_use ids: {pending - answered}")
                    pending = set()
