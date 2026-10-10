import os
import tempfile
from pathlib import Path

# Must be set BEFORE maritime.ui is imported anywhere (it reads Settings at import).
_TMP = Path(tempfile.mkdtemp(prefix="maritime-test-"))
os.environ.update({
    "AIS_DATA_MODE": "demo", "AUTH_MODE": "demo", "ARTIFACTS_DIR": str(_TMP / "artifacts"),
    "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}", "SIMULATED_FLEET_SIZE": "60",
})
os.environ.pop("AUTH_USERS_JSON", None)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from maritime.models.registry import train_and_save  # noqa: E402

NOW = pd.Timestamp("2026-10-10 06:00", tz="UTC")


@pytest.fixture(scope="session")
def bundle(tmp_path_factory):
    """A small but real trained model set (fewer voyages than production for speed)."""
    return train_and_save(tmp_path_factory.mktemp("artifacts"), n_voyages=1500, seed=0)


@pytest.fixture(scope="session")
def now():
    return NOW
