"""Process-wide configuration, read once from the environment."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SECTORLENS_", extra="ignore")

    llm_provider: str = "anthropic"
    model: str = "claude-opus-5"
    effort: str = "medium"
    max_tokens: int = 16000
    max_tool_rounds: int = 6

    db_path: Path = Path("data/sector_intel.db")
    sec_user_agent: str = "Sector Lens contact@example.com"
    api_base_url: str = "http://127.0.0.1:8000"

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    @property
    def resolved_db_path(self) -> Path:
        """Absolute DB path, so the MCP subprocess resolves it identically."""
        p = self.db_path
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def effective_provider(self) -> str:
        """Fall back to the deterministic provider when no key is available.

        This keeps every entry point runnable without credentials instead of
        failing at the first request; the response labels which provider ran.
        """
        if self.llm_provider == "anthropic" and not self.has_anthropic_key:
            return "deterministic"
        return self.llm_provider


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
