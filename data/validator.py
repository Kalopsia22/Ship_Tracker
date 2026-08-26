"""
Validation & cleaning layer for incoming AIS data (demo or live).

Applies sanity checks before data is persisted or shown on the dashboard,
and flags data-quality issues rather than silently dropping records where
possible, so the UI can display appropriate warnings.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

MAX_PLAUSIBLE_SPEED_KNOTS = 30.0
STALE_THRESHOLD_MINUTES = 30
VALID_MMSI_LENGTH = 9


class ValidationReport:
    """Aggregated data-quality findings for a batch of AIS records."""

    def __init__(self):
        self.total_records = 0
        self.invalid_coordinates = 0
        self.invalid_mmsi = 0
        self.stale_records = 0
        self.excessive_speed = 0
        self.duplicate_records = 0
        self.missing_eta = 0

    def as_dict(self) -> dict:
        return self.__dict__

    @property
    def data_quality_score(self) -> float:
        """A simple 0-100 composite score; 100 = no issues detected."""
        if self.total_records == 0:
            return 100.0
        issues = (
            self.invalid_coordinates
            + self.invalid_mmsi
            + self.stale_records
            + self.excessive_speed
            + self.duplicate_records
        )
        return max(0.0, 100.0 * (1 - issues / max(1, self.total_records)))


def validate_positions(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate a batch of AIS position rows. Returns the (possibly annotated)
    dataframe plus a ValidationReport summarizing issues found.

    Rather than dropping bad rows outright, invalid rows are flagged with a
    `data_quality_flag` column so the UI/analytics layer can decide how to
    handle them (e.g. exclude from the map but still show in an audit list).
    """
    report = ValidationReport()
    if df.empty:
        return df, report

    df = df.copy()
    report.total_records = len(df)

    df["data_quality_flag"] = ""

    invalid_coords = (
        df["latitude"].isna() | df["longitude"].isna()
        | ~df["latitude"].between(-90, 90) | ~df["longitude"].between(-180, 180)
        | ((df["latitude"] == 0) & (df["longitude"] == 0))
    )
    report.invalid_coordinates = int(invalid_coords.sum())
    df.loc[invalid_coords, "data_quality_flag"] += "invalid_coordinates;"

    invalid_mmsi = df["mmsi"].astype(str).str.len() != VALID_MMSI_LENGTH
    report.invalid_mmsi = int(invalid_mmsi.sum())
    df.loc[invalid_mmsi, "data_quality_flag"] += "invalid_mmsi;"

    if "last_ais_update" in df.columns:
        now = pd.Timestamp.utcnow().tz_localize(None)
        ts = pd.to_datetime(df["last_ais_update"], errors="coerce")
        age_minutes = (now - ts).dt.total_seconds() / 60
        stale = age_minutes > STALE_THRESHOLD_MINUTES
        report.stale_records = int(stale.fillna(False).sum())
        df.loc[stale.fillna(False), "data_quality_flag"] += "stale_data;"
        df["ais_age_minutes"] = age_minutes.round(1)

    excessive_speed = df["speed_knots"] > MAX_PLAUSIBLE_SPEED_KNOTS
    report.excessive_speed = int(excessive_speed.sum())
    df.loc[excessive_speed, "data_quality_flag"] += "excessive_speed;"

    duplicates = df.duplicated(subset=["mmsi"], keep="last")
    report.duplicate_records = int(duplicates.sum())
    df = df[~duplicates]

    if "eta_scheduled" in df.columns:
        report.missing_eta = int(df["eta_scheduled"].isna().sum())

    logger.info(
        "Validated %d records: %d invalid coords, %d stale, %d excessive speed",
        report.total_records, report.invalid_coordinates, report.stale_records,
        report.excessive_speed,
    )
    return df, report
