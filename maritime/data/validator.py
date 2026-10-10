"""Row-level validation and a transparent data-quality score."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from maritime.data.schema import POSITION_COLUMNS, REQUIRED_COLUMNS

MAX_SOG_KN = 40.0


@dataclass
class QualityReport:
    total_rows: int
    valid_rows: int
    dropped: dict[str, int] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Share of input rows that passed every rule (0-100)."""
        return 100.0 * self.valid_rows / self.total_rows if self.total_rows else 0.0

    def as_dict(self) -> dict:
        return {"total_rows": self.total_rows, "valid_rows": self.valid_rows,
                "quality_score": round(self.score, 2), "dropped": dict(self.dropped)}


def validate_positions(df: pd.DataFrame) -> tuple[pd.DataFrame, QualityReport]:
    """Return (clean_df, report). Never mutates the input."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Position data is missing required columns: {missing}")

    out = df.copy()
    for col in POSITION_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    for col in ["lat", "lon", "sog", "cog", "heading", "draught"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    total = len(out)
    dropped: dict[str, int] = {}

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal out
        n = int(mask.sum())
        if n:
            dropped[reason] = n
            out = out[~mask]

    drop(out["mmsi"].isna() | out["timestamp"].isna() | out["lat"].isna() | out["lon"].isna(), "missing_required_value")
    drop(~out["lat"].between(-90, 90) | ~out["lon"].between(-180, 180), "coordinates_out_of_range")
    drop((out["lat"] == 0) & (out["lon"] == 0), "null_island")
    drop(out["sog"].notna() & ~out["sog"].between(0, MAX_SOG_KN), "implausible_speed")
    drop(out.duplicated(subset=["mmsi", "timestamp"], keep="last"), "duplicate_report")

    out["cog"] = out["cog"].where(out["cog"].between(0, 360))
    out["mmsi"] = out["mmsi"].astype(str)
    out = out.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)
    return out[POSITION_COLUMNS], QualityReport(total, len(out), dropped)
