#!/usr/bin/env python3
"""Train and persist model artifacts offline (so the app never fits models per session).

  python scripts/train.py                          # synthetic demo models
  python scripts/train.py --source csv --csv ais.csv   # anomaly model on REAL tracks
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maritime.config import ROOT  # noqa: E402
from maritime.data.providers import CsvProvider  # noqa: E402
from maritime.data.validator import validate_positions  # noqa: E402
from maritime.models.features import compute_vessel_features  # noqa: E402
from maritime.models.registry import train_and_save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["demo", "csv"], default="demo")
    ap.add_argument("--csv", help="AIS CSV (NOAA MarineCadastre column names) when --source csv")
    ap.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    ap.add_argument("--voyages", type=int, default=3000, help="synthetic voyages for the ETA model")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    feats, source = None, "synthetic simulator"
    if a.source == "csv":
        if not a.csv:
            ap.error("--csv is required with --source csv")
        positions, rep = validate_positions(CsvProvider(a.csv).positions(hours=72))
        feats = compute_vessel_features(positions)
        if len(feats) < 50:
            ap.error(f"only {len(feats)} vessels in the data; need >= 50 for a meaningful anomaly model")
        source = f"real AIS: {Path(a.csv).name} ({len(feats)} vessels, quality {rep.score:.1f}%)"
    bundle = train_and_save(Path(a.artifacts), anomaly_features=feats, anomaly_source=source, n_voyages=a.voyages, seed=a.seed)
    print(json.dumps({"version": bundle.metadata["version"], "eta": bundle.metadata["eta"]["metrics"],
                      "anomaly_source": source}, indent=2, default=float))
    if a.source == "csv":
        print("\nNote: ETA stays trained on synthetic voyages; anomaly metrics are on a synthetic holdout. "
              "Real-data anomaly quality needs human review.", file=sys.stderr)


if __name__ == "__main__":
    main()
