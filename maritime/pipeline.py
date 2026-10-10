"""ingest -> validate -> features -> ML -> risk -> ports -> alerts."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from maritime.alerts import AlertCandidate, AlertRules, generate_alerts
from maritime.analytics.port_analysis import port_congestion
from maritime.data.providers import AISProvider
from maritime.data.schema import max_draught
from maritime.data.validator import QualityReport, validate_positions
from maritime.models.features import add_route_geometry, compute_vessel_features, latest_positions
from maritime.models.registry import ModelBundle
from maritime.ports import PORTS


@dataclass
class PipelineResult:
    positions: pd.DataFrame
    vessels: pd.DataFrame  # one row per vessel, indexed by mmsi
    ports: pd.DataFrame
    quality: QualityReport
    alerts: list[AlertCandidate]
    generated_at: pd.Timestamp
    provider: str
    is_synthetic: bool
    notes: list[str] = field(default_factory=list)


def run_pipeline(provider: AISProvider, models: ModelBundle, rules: AlertRules | None = None, hours: int = 72) -> PipelineResult:
    raw = provider.positions(hours=hours)
    positions, quality = validate_positions(raw)
    if positions.empty:
        raise ValueError("No valid position reports after validation")
    notes: list[str] = []
    now = positions["timestamp"].max()

    feats = compute_vessel_features(positions, now=now)
    latest = add_route_geometry(latest_positions(positions)).set_index("mmsi").loc[feats.index]
    vessels = latest.join(feats, how="inner")
    vessels["anomaly_score"], vessels["is_anomaly"] = [models.anomaly.score(vessels)[c] for c in ("anomaly_score", "is_anomaly")]
    vessels = vessels.join(models.risk.score(vessels))

    ports = port_congestion(vessels)
    cong = ports.set_index("port")["congestion_idx"]

    # ETA only where origin & destination are known ports
    known = vessels["destination"].isin(PORTS) & vessels["origin"].isin(PORTS)
    if known.any():
        v = vessels[known]
        six_h = positions[positions["timestamp"] > now - pd.Timedelta(hours=6)].groupby("mmsi")["sog"].mean()
        X = pd.DataFrame({
            "vessel_class": v["vessel_class"].fillna("Other"),
            "dist_remaining_nm": v["dist_to_dest_nm"],
            "dist_total_nm": v["dist_to_dest_nm"] + v["dist_to_origin_nm"],
            "sog_now": v["sog"].clip(lower=0.5),
            "sog_mean_6h": six_h.reindex(v.index).fillna(v["sog"]).clip(lower=0.5),
            "draught_ratio": (v["draught"] / v["vessel_class"].map(max_draught)).clip(0.3, 1.0).fillna(0.8),
            "dest_congestion_idx": v["destination"].map(cong).fillna(0.0),
            "hour_utc": now.hour,
        }, index=v.index)
        eta = models.eta.predict(X)
        vessels = vessels.join(eta)
        vessels["eta_utc"] = now + pd.to_timedelta(vessels["eta_hours"], unit="h")
        # ETA is only trustworthy for vessels behaving normally and making way
        vessels["eta_reliable"] = (~vessels["is_anomaly"]) & (vessels["recent_sog_ratio"] > 0.5)
    else:
        for c in ("eta_hours", "eta_low_h", "eta_high_h"):
            vessels[c] = np.nan
        vessels["eta_utc"], vessels["eta_reliable"] = pd.NaT, False
        notes.append("No vessels with known origin/destination ports: ETA and route features unavailable for this data source.")

    vessels["progress_pct"] = np.where(
        vessels["route_known"], 100 * vessels["dist_to_origin_nm"] / (vessels["dist_to_origin_nm"] + vessels["dist_to_dest_nm"]), np.nan)
    alerts = generate_alerts(vessels, ports, rules or AlertRules())
    return PipelineResult(positions, vessels, ports, quality, alerts, now, provider.describe(), provider.is_synthetic, notes)
