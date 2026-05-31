"""Unit tests for the auth + rate-limit dependency seams."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest

from core.config import Settings
from core.exceptions import AuthError, RateLimitError
from core.security import RateLimiter, require_api_key


def _req(headers=None, host="1.2.3.4"):
    return SimpleNamespace(headers=headers or {}, client=SimpleNamespace(host=host))


def _settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# --- require_api_key -------------------------------------------------------

class TestRequireApiKey:
    def test_disabled_allows_anything(self):
        require_api_key(_req(), _settings(API_AUTH_TOKEN=""))  # no raise

    def test_enabled_missing_key_raises(self):
        with pytest.raises(AuthError):
            require_api_key(_req(), _settings(API_AUTH_TOKEN="secret"))

    def test_enabled_correct_x_api_key(self):
        require_api_key(_req({"X-API-Key": "secret"}), _settings(API_AUTH_TOKEN="secret"))

    def test_enabled_correct_bearer(self):
        require_api_key(
            _req({"Authorization": "Bearer secret"}), _settings(API_AUTH_TOKEN="secret")
        )

    def test_enabled_wrong_key_raises(self):
        with pytest.raises(AuthError):
            require_api_key(_req({"X-API-Key": "nope"}), _settings(API_AUTH_TOKEN="secret"))


# --- RateLimiter -----------------------------------------------------------

class TestRateLimiter:
    def test_disabled_never_limits(self):
        rl = RateLimiter()
        s = _settings(RATE_LIMIT_ENABLED=False, RATE_LIMIT_PER_MINUTE=1)
        for _ in range(10):
            rl(_req(), s)  # no raise

    def test_allows_up_to_limit_then_blocks(self):
        rl = RateLimiter()
        s = _settings(RATE_LIMIT_ENABLED=True, RATE_LIMIT_PER_MINUTE=3)
        req = _req(host="9.9.9.9")
        rl(req, s)
        rl(req, s)
        rl(req, s)
        with pytest.raises(RateLimitError):
            rl(req, s)

    def test_separate_clients_have_separate_budgets(self):
        rl = RateLimiter()
        s = _settings(RATE_LIMIT_ENABLED=True, RATE_LIMIT_PER_MINUTE=1)
        rl(_req(host="a"), s)
        rl(_req(host="b"), s)  # different host → own budget, no raise
        with pytest.raises(RateLimitError):
            rl(_req(host="a"), s)

    def test_reset_clears_state(self):
        rl = RateLimiter()
        s = _settings(RATE_LIMIT_ENABLED=True, RATE_LIMIT_PER_MINUTE=1)
        req = _req(host="x")
        rl(req, s)
        rl.reset()
        rl(req, s)  # budget restored, no raise

    def test_unknown_client_host(self):
        rl = RateLimiter()
        s = _settings(RATE_LIMIT_ENABLED=True, RATE_LIMIT_PER_MINUTE=1)
        req = SimpleNamespace(headers={}, client=None)
        rl(req, s)
        with pytest.raises(RateLimitError):
            rl(req, s)
