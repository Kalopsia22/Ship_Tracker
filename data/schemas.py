"""
Pydantic data models used throughout the application for validation.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class VesselClass(str, Enum):
    PANAMAX = "Panamax"
    AFRAMAX = "Aframax"
    SUEZMAX = "Suezmax"
    VLCC = "VLCC"
    ULCC = "ULCC"


class NavigationStatus(str, Enum):
    UNDER_WAY = "Under way using engine"
    AT_ANCHOR = "At anchor"
    MOORED = "Moored"
    IN_PORT = "In port"
    RESTRICTED_MANOEUVRABILITY = "Restricted manoeuvrability"
    NOT_UNDER_COMMAND = "Not under command"
    AGROUND = "Aground"


class DataSource(str, Enum):
    DEMO = "DEMO DATA"
    LIVE = "LIVE AIS"


class AISPosition(BaseModel):
    """A single AIS position report for a vessel."""

    mmsi: str = Field(..., min_length=9, max_length=9)
    imo: Optional[str] = None
    vessel_name: str
    vessel_class: VesselClass
    flag: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    speed_knots: float = Field(..., ge=0, le=40)
    course_deg: float = Field(..., ge=0, lt=360)
    heading_deg: Optional[float] = Field(None, ge=0, lt=360)
    nav_status: NavigationStatus
    destination: str
    eta_scheduled: Optional[datetime] = None
    draught_m: float = Field(..., ge=0, le=25)
    length_m: float = Field(..., gt=0)
    width_m: float = Field(..., gt=0)
    cargo_category: str
    ais_timestamp: datetime
    data_source: DataSource = DataSource.DEMO
    source_timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("mmsi")
    @classmethod
    def mmsi_numeric(cls, v: str) -> str:
        if not v.isdigit():
            raise ValueError("MMSI must be numeric")
        return v

    @property
    def is_stale(self) -> bool:
        age_minutes = (datetime.utcnow() - self.ais_timestamp).total_seconds() / 60
        return age_minutes > 30

    @property
    def has_valid_coordinates(self) -> bool:
        return not (self.latitude == 0 and self.longitude == 0)


class RiskFactorContribution(BaseModel):
    """One contributing factor to an explainable AI output."""

    label: str
    value: str
    weight: float = Field(..., ge=0, le=1)


class ETAPrediction(BaseModel):
    mmsi: str
    predicted_eta: datetime
    scheduled_eta: Optional[datetime]
    eta_deviation_hours: Optional[float]
    confidence: float = Field(..., ge=0, le=1)
    model_version: str
    predicted_at: datetime = Field(default_factory=datetime.utcnow)


class AnomalyResult(BaseModel):
    mmsi: str
    anomaly_score: float = Field(..., ge=0, le=100)
    is_anomalous: bool
    contributing_factors: list[RiskFactorContribution] = []
    model_version: str
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class RiskAssessment(BaseModel):
    mmsi: str
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: str  # LOW / MEDIUM / HIGH
    contributing_factors: list[RiskFactorContribution] = []
    model_version: str
    assessed_at: datetime = Field(default_factory=datetime.utcnow)
    disclaimer: str = (
        "This is an Operational Risk Indicator derived from observable AIS "
        "behavior. It is a decision-support signal, not a security or "
        "criminal classification, and does not represent vessel intent."
    )


class Alert(BaseModel):
    alert_id: str
    mmsi: str
    vessel_name: str
    alert_type: str
    severity: str  # INFO / WARNING / CRITICAL
    message: str
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    acknowledged: bool = False
