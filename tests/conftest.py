import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ekb.pipeline import build_chunks  # noqa: E402
from ekb.rbac import load_entitlements  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def chunks(root):
    return build_chunks(root / "data" / "raw", root / "data" / "metadata" / "documents.csv")


@pytest.fixture(scope="session")
def users(root):
    return load_entitlements(root / "data" / "rbac" / "user_entitlements.csv")
