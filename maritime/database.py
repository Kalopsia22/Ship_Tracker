"""Persistence for alerts (with acknowledgment workflow) and the audit log."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class AlertRow(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(160), index=True)
    alert_type: Mapped[str] = mapped_column(String(40))
    severity: Mapped[str] = mapped_column(String(16))
    subject: Mapped[str] = mapped_column(String(120))
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)  # open|acknowledged|resolved
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    ack_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    ack_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditRow(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    username: Mapped[str] = mapped_column(String(80))
    action: Mapped[str] = mapped_column(String(60))
    target: Mapped[str] = mapped_column(String(160), default="")
    detail: Mapped[str] = mapped_column(Text, default="")


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        from pathlib import Path
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, future=True)
    Base.metadata.create_all(engine)
    return engine


class Repository:
    def __init__(self, engine: Engine):
        self.engine = engine

    # ---- audit -----------------------------------------------------------
    def audit(self, username: str, action: str, target: str = "", detail: str = "") -> None:
        with Session(self.engine) as s, s.begin():
            s.add(AuditRow(username=username, action=action, target=target, detail=detail))

    def audit_log(self, limit: int = 200) -> list[dict]:
        with Session(self.engine) as s:
            rows = s.scalars(select(AuditRow).order_by(AuditRow.id.desc()).limit(limit)).all()
            return [{"ts": r.ts, "user": r.username, "action": r.action, "target": r.target, "detail": r.detail} for r in rows]

    # ---- alerts ----------------------------------------------------------
    def sync_alerts(self, candidates) -> dict:
        """Upsert current candidates; auto-resolve alerts whose condition cleared."""
        created = updated = resolved = 0
        live = {c.fingerprint for c in candidates}
        with Session(self.engine) as s, s.begin():
            active = {a.fingerprint: a for a in s.scalars(select(AlertRow).where(AlertRow.status.in_(["open", "acknowledged"])))}
            for c in candidates:
                row = active.get(c.fingerprint)
                if row:
                    row.message, row.severity, row.details, row.updated_at = c.message, c.severity, json.dumps(c.details), _now()
                    updated += 1
                else:
                    s.add(AlertRow(fingerprint=c.fingerprint, alert_type=c.alert_type, severity=c.severity,
                                   subject=c.subject, message=c.message, details=json.dumps(c.details)))
                    created += 1
            for fp, row in active.items():
                if fp not in live:
                    row.status, row.updated_at = "resolved", _now()
                    resolved += 1
        return {"created": created, "updated": updated, "resolved": resolved}

    def list_alerts(self, statuses=("open", "acknowledged"), limit: int = 500) -> list[dict]:
        with Session(self.engine) as s:
            q = select(AlertRow).where(AlertRow.status.in_(list(statuses))).order_by(AlertRow.updated_at.desc()).limit(limit)
            return [{"id": a.id, "type": a.alert_type, "severity": a.severity, "subject": a.subject, "message": a.message,
                     "status": a.status, "created_at": a.created_at, "updated_at": a.updated_at, "ack_by": a.ack_by}
                    for a in s.scalars(q)]

    def acknowledge(self, alert_id: int, username: str) -> bool:
        with Session(self.engine) as s, s.begin():
            row = s.get(AlertRow, alert_id)
            if not row or row.status != "open":
                return False
            row.status, row.ack_by, row.ack_at, row.updated_at = "acknowledged", username, _now(), _now()
            s.add(AuditRow(username=username, action="alert.acknowledge", target=f"alert:{alert_id}", detail=row.fingerprint))
            return True
