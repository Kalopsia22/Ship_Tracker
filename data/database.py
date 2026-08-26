"""
Database abstraction layer.

Uses SQLAlchemy so the prototype can run on local SQLite and later be
pointed at PostgreSQL/PostGIS in production purely by changing
`DATABASE_URL` — no application code changes required.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from utils.configuration import config
from utils.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models — mirrors the data model described in the project spec
# (vessels, ais_positions, voyages, ports, port_calls, predictions,
#  anomalies, alerts, model_versions, users, audit_logs)
# ---------------------------------------------------------------------------


class Vessel(Base):
    __tablename__ = "vessels"
    mmsi = Column(String(9), primary_key=True)
    imo = Column(String(20))
    vessel_name = Column(String(100))
    vessel_class = Column(String(20), index=True)
    flag = Column(String(50))
    length_m = Column(Float)
    width_m = Column(Float)


class AISPositionRecord(Base):
    __tablename__ = "ais_positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mmsi = Column(String(9), ForeignKey("vessels.mmsi"), index=True)
    timestamp = Column(DateTime, index=True)
    latitude = Column(Float)
    longitude = Column(Float)
    speed_knots = Column(Float)
    course_deg = Column(Float)
    heading_deg = Column(Float)
    nav_status = Column(String(50))
    draught_m = Column(Float)
    data_source = Column(String(20))
    data_quality_flag = Column(String(100), default="")


class Voyage(Base):
    __tablename__ = "voyages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mmsi = Column(String(9), ForeignKey("vessels.mmsi"), index=True)
    origin_port = Column(String(100))
    destination_port = Column(String(100))
    departure_time = Column(DateTime)
    eta_scheduled = Column(DateTime)
    eta_actual = Column(DateTime, nullable=True)
    cargo_category = Column(String(50))


class Port(Base):
    __tablename__ = "ports"
    name = Column(String(100), primary_key=True)
    country = Column(String(100))
    latitude = Column(Float)
    longitude = Column(Float)


class PortCall(Base):
    __tablename__ = "port_calls"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mmsi = Column(String(9), index=True)
    port_name = Column(String(100))
    arrival_time = Column(DateTime)
    departure_time = Column(DateTime, nullable=True)
    waiting_time_hours = Column(Float, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mmsi = Column(String(9), index=True)
    prediction_type = Column(String(30))  # eta / congestion / fuel
    predicted_value = Column(Text)
    confidence = Column(Float)
    model_version = Column(String(30))
    predicted_at = Column(DateTime, default=datetime.utcnow)


class AnomalyRecord(Base):
    __tablename__ = "anomalies"
    id = Column(Integer, primary_key=True, autoincrement=True)
    mmsi = Column(String(9), index=True)
    anomaly_score = Column(Float)
    is_anomalous = Column(Boolean)
    contributing_factors = Column(Text)
    model_version = Column(String(30))
    detected_at = Column(DateTime, default=datetime.utcnow)


class AlertRecord(Base):
    __tablename__ = "alerts"
    alert_id = Column(String(40), primary_key=True)
    mmsi = Column(String(9), index=True)
    vessel_name = Column(String(100))
    alert_type = Column(String(50))
    severity = Column(String(20))
    message = Column(Text)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50))
    version = Column(String(20))
    trained_at = Column(DateTime)
    training_data_description = Column(Text)
    notes = Column(Text)


class User(Base):
    __tablename__ = "users"
    username = Column(String(50), primary_key=True)
    role = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50))
    action = Column(String(100))
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Engine / session management
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        connect_args = {"check_same_thread": False} if config.database_url.startswith("sqlite") else {}
        _engine = create_engine(config.database_url, connect_args=connect_args)
        logger.info("Database engine created for %s", config.database_url.split("://")[0])
    return _engine


def init_db():
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(get_engine())
    logger.info("Database schema initialized")


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def log_audit_event(username: str, action: str, details: str = ""):
    """Record an entry in the audit log — used to trace AI-assisted decisions."""
    session = get_session()
    try:
        session.add(AuditLog(username=username, action=action, details=details))
        session.commit()
    finally:
        session.close()
