"""Rule-based alert generation with configurable thresholds."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class AlertRules:
    risk_score_min: float = 50.0
    ais_gap_hours: float = 6.0
    route_deviation_nm: float = 60.0
    port_congestion_idx: float = 1.2


@dataclass(frozen=True)
class AlertCandidate:
    alert_type: str
    severity: str  # Info | Warning | Critical
    subject: str
    message: str
    details: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return f"{self.alert_type}:{self.subject}"


def generate_alerts(vessels: pd.DataFrame, ports: pd.DataFrame, rules: AlertRules) -> list[AlertCandidate]:
    out: list[AlertCandidate] = []
    for mmsi, v in vessels.iterrows():
        label = f"{v['vessel_name']} ({mmsi})"
        if v["risk_score"] >= rules.risk_score_min:
            out.append(AlertCandidate("HIGH_RISK", "Critical", str(mmsi),
                f"{label}: risk {v['risk_score']:.0f}/100, driven by {v['top_factor'].lower()}.",
                {"risk_score": round(float(v["risk_score"]), 1), "top_factor": v["top_factor"]}))
        if v["max_gap_h"] >= rules.ais_gap_hours:
            out.append(AlertCandidate("AIS_GAP", "Warning", str(mmsi),
                f"{label}: no AIS reports for up to {v['max_gap_h']:.1f} h in the last 24 h.", {"max_gap_h": round(float(v["max_gap_h"]), 1)}))
        if v["route_known"] and v["xtrack_last_nm"] >= rules.route_deviation_nm:
            out.append(AlertCandidate("ROUTE_DEVIATION", "Warning", str(mmsi),
                f"{label}: {v['xtrack_last_nm']:.0f} nm off the planned great-circle route.", {"xtrack_nm": round(float(v["xtrack_last_nm"]), 1)}))
    for _, p in ports.iterrows():
        if p["congestion_idx"] >= rules.port_congestion_idx:
            out.append(AlertCandidate("PORT_CONGESTION", "Warning", p["port"],
                f"{p['port']}: {p['inbound_48h']} vessels due within 48 h vs capacity {p['capacity']} (≈{p['expected_wait_h']:.0f} h wait).",
                {"congestion_idx": round(float(p["congestion_idx"]), 2)}))
    return out
