"""Tests for environment-file loading."""

import os
from pathlib import Path

import pytest

from trading_script_anatomy.env import load_dotenv


def test_load_dotenv_sets_missing_and_preserves_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fill in missing variables without overriding the real environment."""
    monkeypatch.setenv("TSA_TEST_NEW", "placeholder")
    monkeypatch.delenv("TSA_TEST_NEW")
    monkeypatch.setenv("TSA_TEST_EXISTING", "keep")
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nTSA_TEST_NEW='abc'\nTSA_TEST_EXISTING=override\nBROKEN LINE\n"
    )

    load_dotenv(env_file)

    assert os.environ["TSA_TEST_NEW"] == "abc"
    assert os.environ["TSA_TEST_EXISTING"] == "keep"


def test_load_dotenv_ignores_missing_files(tmp_path: Path) -> None:
    """Treat a missing dotenv file as a no-op."""
    load_dotenv(tmp_path / "does-not-exist.env")
