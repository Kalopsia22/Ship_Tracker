"""Vectorised great-circle helpers (nautical miles / degrees)."""
from __future__ import annotations

import numpy as np

R_NM = 3440.065  # mean Earth radius in nautical miles


def haversine_nm(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def destination_point(lat, lon, bearing, dist_nm):
    """Point reached travelling `dist_nm` from (lat, lon) on `bearing` degrees."""
    lat, lon, brg = (np.radians(np.asarray(v, dtype=float)) for v in (lat, lon, bearing))
    d = np.asarray(dist_nm, dtype=float) / R_NM
    lat2 = np.arcsin(np.sin(lat) * np.cos(d) + np.cos(lat) * np.sin(d) * np.cos(brg))
    lon2 = lon + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(lat), np.cos(d) - np.sin(lat) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lon2) + 540) % 360 - 180


def _vec(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def interpolate_gc(lat1, lon1, lat2, lon2, fraction):
    """Points at `fraction` (0..1) along the great circle between two points."""
    f = np.asarray(fraction, dtype=float)
    a, b = _vec(lat1, lon1), _vec(lat2, lon2)
    omega = float(np.arccos(np.clip(np.dot(a, b), -1, 1)))
    if omega < 1e-9:
        return np.full_like(f, lat1), np.full_like(f, lon1)
    s = np.sin(omega)
    v = (np.sin((1 - f) * omega) / s)[..., None] * a + (np.sin(f * omega) / s)[..., None] * b
    lat = np.degrees(np.arctan2(v[..., 2], np.hypot(v[..., 0], v[..., 1])))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0]))
    return lat, lon


def cross_track_nm(lat, lon, lat1, lon1, lat2, lon2):
    """Absolute distance (nm) of point(s) from the great circle through (1)->(2)."""
    d13 = haversine_nm(lat1, lon1, lat, lon) / R_NM
    b13 = np.radians(bearing_deg(lat1, lon1, lat, lon))
    b12 = np.radians(bearing_deg(lat1, lon1, lat2, lon2))
    return np.abs(np.arcsin(np.clip(np.sin(d13) * np.sin(b13 - b12), -1, 1))) * R_NM
