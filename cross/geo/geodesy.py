"""WGS84 geodesy: geodetic (lat, lon, alt) <-> ECEF <-> local East-North-Up (ENU).

All functions are vectorised over leading dimensions; angles in degrees, distances in metres.  A map is anchored to a
local ENU frame whose origin (a geodetic point) is stored with the map (`LocalFrame.state`), so every keyframe position
in the map frame has a latitude / longitude once the map frame is aligned to that ENU frame (cross.geo.anchor)."""
from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_EP2 = (WGS84_A ** 2 - WGS84_B ** 2) / WGS84_B ** 2


def lla_to_ecef(lat, lon, alt) -> np.ndarray:
    """(..., ) degrees, degrees, metres -> (..., 3) ECEF metres."""
    lat, lon, alt = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float)), np.asarray(alt, float)
    s, c = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * s * s)
    return np.stack([(n + alt) * c * np.cos(lon), (n + alt) * c * np.sin(lon), (n * (1.0 - WGS84_E2) + alt) * s], -1)


def ecef_to_lla(xyz) -> np.ndarray:
    """(..., 3) ECEF -> (..., 3) [lat deg, lon deg, alt m] (Bowring's method with two refinements: < 1 mm)."""
    xyz = np.asarray(xyz, float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    th = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * np.sin(th) ** 3, p - WGS84_E2 * WGS84_A * np.cos(th) ** 3)
    for _ in range(2):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        alt = p / np.maximum(np.cos(lat), 1e-12) - n
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    alt = np.where(np.abs(np.cos(lat)) > 1e-9, p / np.maximum(np.cos(lat), 1e-12) - n, np.abs(z) - WGS84_B)
    return np.stack([np.degrees(lat), np.degrees(lon), alt], -1)


def enu_rotation(lat0: float, lon0: float) -> np.ndarray:
    """3x3 rotation whose rows are the East, North, Up unit vectors at (lat0, lon0) in ECEF: enu = R @ (ecef - o)."""
    la, lo = np.radians(lat0), np.radians(lon0)
    sla, cla, slo, clo = np.sin(la), np.cos(la), np.sin(lo), np.cos(lo)
    return np.array([[-slo, clo, 0.0],
                     [-sla * clo, -sla * slo, cla],
                     [cla * clo, cla * slo, sla]])


class LocalFrame:
    """A local ENU frame at a geodetic origin."""

    def __init__(self, lat0: float, lon0: float, alt0: float = 0.0):
        self.lat0, self.lon0, self.alt0 = float(lat0), float(lon0), float(alt0)
        self.R = enu_rotation(self.lat0, self.lon0)
        self.o = lla_to_ecef(self.lat0, self.lon0, self.alt0)

    def to_enu(self, lat, lon, alt=None) -> np.ndarray:
        alt = self.alt0 if alt is None else alt
        lat, lon, alt = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float), np.asarray(alt, float))
        return (lla_to_ecef(lat, lon, alt) - self.o) @ self.R.T

    def to_lla(self, enu) -> np.ndarray:
        return ecef_to_lla(np.asarray(enu, float) @ self.R + self.o)

    def state(self) -> dict:
        return {"lat0": self.lat0, "lon0": self.lon0, "alt0": self.alt0}

    @classmethod
    def from_state(cls, s: dict) -> "LocalFrame":
        return cls(s["lat0"], s["lon0"], s.get("alt0", 0.0))


def geojson_linestring(lla: np.ndarray, properties: dict | None = None) -> dict:
    """GeoJSON Feature (LineString) of an (N, 3) lat/lon/alt track (GeoJSON order: lon, lat)."""
    coords = [[round(float(lo), 8), round(float(la), 8)] for la, lo, _ in np.asarray(lla)]
    return {"type": "Feature", "properties": properties or {}, "geometry": {"type": "LineString", "coordinates": coords}}


def geojson_points(lla: np.ndarray, properties: list | None = None) -> list:
    """GeoJSON Point features of an (N, 3) lat/lon/alt array."""
    out = []
    for i, (la, lo, al) in enumerate(np.asarray(lla)):
        out.append({"type": "Feature", "properties": (properties[i] if properties else {}),
                    "geometry": {"type": "Point", "coordinates": [round(float(lo), 8), round(float(la), 8)]}})
    return out


def kml_linestring(lla: np.ndarray, name: str = "track") -> str:
    """Minimal KML document with one LineString (clamped to ground)."""
    coords = " ".join(f"{lo:.8f},{la:.8f},{al:.2f}" for la, lo, al in np.asarray(lla))
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<kml xmlns=\"http://www.opengis.net/kml/2.2\"><Document>"
            f"<Placemark><name>{name}</name><LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates>"
            "</LineString></Placemark></Document></kml>\n")
