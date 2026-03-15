"""Check that .env is loaded and WANDB_API_KEY is set. Skips if .env is missing or key not set."""

import os

import pytest


def test_wandb_api_key_loaded():
    """After importing load_env, WANDB_API_KEY must be in os.environ and non-empty."""
    import load_env  # noqa: F401 — loads .env into os.environ

    key = os.environ.get("WANDB_API_KEY")
    if not key or not key.strip():
        pytest.skip("WANDB_API_KEY not set (add WANDB_API_KEY=... to .env)")
    assert key.strip(), "WANDB_API_KEY must be non-empty"
