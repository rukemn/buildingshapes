#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

import requests
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely import wkt as shapely_wkt

import json
import urllib.parse
from shapely.geometry import mapping




# -----------------------------
# Configs
# -----------------------------

@dataclass
class NominatimConfig:
    base_url: str = "https://nominatim.openstreetmap.org"
    min_interval_s: float = 1.0
    user_agent: str = "osm-to-wkt-polygons/1.0 (contact: ruke@uni-bonn.de)"
    email: Optional[str] = None
    timeout_s: float = 2.0


@dataclass
class OverpassConfig:
    base_url: str = "https://overpass-api.de/api/interpreter"
    min_interval_s: float = 1.0
    user_agent: str = "osm-to-wkt-polygons/1.0 (contact: ruke@uni-bonn.de)"
    timeout_s: float = 60.0


# -----------------------------
# Nominatim client (rate-limited)
# -----------------------------

class NominatimClient:
    def __init__(self, cfg: NominatimConfig):
        self.cfg = cfg
        self._last_call_ts = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": cfg.user_agent,
            "Accept": "application/json",
        })

    def _sleep_if_needed(self):
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.cfg.min_interval_s:
            time.sleep(self.cfg.min_interval_s - elapsed)

    def _get(self, path: str, params: dict) -> Dict[str, Any]:
        self._sleep_if_needed()

        if self.cfg.email:
            params = dict(params)
            params["email"] = self.cfg.email

        url = self.cfg.base_url.rstrip("/") + path
        resp = self._session.get(url, params=params, timeout=self.cfg.timeout_s)
        self._last_call_ts = time.time()

        if resp.status_code == 403:
            raise requests.HTTPError(
                f"403 Forbidden from Nominatim. Check User-Agent + rate limit. "
                f"UA={self._session.headers.get('User-Agent')!r} URL={resp.url}",
                response=resp,
            )
        resp.raise_for_status()
        return resp.json()

    def reverse_polygon_geojson(self, lat: float, lon: float) -> Dict[str, Any]:
        data = self._get(
            "/reverse",
            {
                "format": "jsonv2",
                "lat": lat,
                "lon": lon,
                "polygon_geojson": 1,
                "addressdetails": 0,
            },
        )
        geo = data.get("geojson")
        if not isinstance(geo, dict):
            raise ValueError("Nominatim reverse returned no geojson polygon for that point")
        return geo


# -----------------------------
# Overpass client (rate-limited)
# -----------------------------

class OverpassClient:
    def __init__(self, cfg: OverpassConfig):
        self.cfg = cfg
        self._last_call_ts = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": cfg.user_agent,
            "Accept": "application/json",
        })

    def _sleep_if_needed(self):
        elapsed = time.time() - self._last_call_ts
        if elapsed < self.cfg.min_interval_s:
            time.sleep(self.cfg.min_interval_s - elapsed)

    def query(self, ql: str) -> Dict[str, Any]:
        self._sleep_if_needed()
        resp = self._session.post(
            self.cfg.base_url,
            data={"data": ql},
            timeout=self.cfg.timeout_s,
        )
        self._last_call_ts = time.time()
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or "elements" not in data:
            raise ValueError("Overpass returned unexpected response")
        return data


# -----------------------------
# Parsing helpers
# -----------------------------

# WKT point: POINT(lon lat)
POINT_RE = re.compile(
    r"^\s*POINT\s*\(\s*([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*\)\s*$",
    re.IGNORECASE,
)

OSM_URL_RE = re.compile(
    r"https?://(?:www\.)?openstreetmap\.org/(node|way|relation)/(\d+)",
    re.IGNORECASE,
)

def parse_wkt_point(wkt_point: str) -> Tuple[float, float]:
    """Return (lat, lon) for Nominatim. Input is POINT(lon lat)."""
    m = POINT_RE.match(wkt_point or "")
    if not m:
        raise ValueError(f"Invalid WKT POINT: {wkt_point!r}")
    lon = float(m.group(1))
    lat = float(m.group(2))
    return lat, lon

def parse_osm_url(url: str) -> Tuple[str, int]:
    m = OSM_URL_RE.search(url or "")
    if not m:
        raise ValueError(f"Invalid OSM URL (expected /node|way|relation/<id>): {url!r}")
    return m.group(1).lower(), int(m.group(2))


# -----------------------------
# Geometry helpers (simple polygons only)
# -----------------------------


def ensure_simple_polygon(g: BaseGeometry, context: str) -> Polygon:
    if g.is_empty:
        raise ValueError(f"{context}: geometry is empty")

    if g.geom_type != "Polygon":
        raise ValueError(f"{context}: geometry is not a simple Polygon (got {g.geom_type})")

    poly = cast(Polygon, g)

    if len(poly.interiors) != 0:
        raise ValueError(f"{context}: polygon has holes; simple polygons only")

    if not poly.is_valid:
        raise ValueError(f"{context}: polygon is invalid")

    return poly

def polygon_to_wkt(poly: Polygon) -> str:
    return poly.wkt

def geojsonio_url_from_polygon(poly: Polygon) -> str:
    """
    Create a shareable geojson.io URL showing the polygon.
    """
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": mapping(poly),
    }
    fc = {
        "type": "FeatureCollection",
        "features": [feature],
    }

    payload = json.dumps(fc, separators=(",", ":"))
    encoded = urllib.parse.quote(payload)
    return f"https://geojson.io/#data=data:application/json,{encoded}"


def ring_from_overpass_geom(geom: Any) -> List[Tuple[float, float]]:
    """
    Convert Overpass 'geometry' (list of {'lat','lon'}) to ring coords as (lon, lat).
    """
    if not isinstance(geom, list) or not geom:
        raise ValueError("Overpass element has no geometry")
    ring: List[Tuple[float, float]] = []
    for pt in geom:
        if not isinstance(pt, dict) or "lat" not in pt or "lon" not in pt:
            raise ValueError("Overpass geometry point missing lat/lon")
        ring.append((float(pt["lon"]), float(pt["lat"])))
    return ring

def ensure_closed_ring(ring: List[Tuple[float, float]], context: str) -> List[Tuple[float, float]]:
    if len(ring) < 4:
        raise ValueError(f"{context}: ring too short to form polygon")
    if ring[0] != ring[-1]:
        # If Overpass returns an unclosed ring (some ways), close it.
        ring = ring + [ring[0]]
    return ring


# -----------------------------
# Overpass: fetch polygon by way/relation id
# -----------------------------

def fetch_way_polygon(overpass: OverpassClient, way_id: int) -> Polygon:
    ql = f"""
    [out:json];
    way({way_id});
    out geom;
    """
    data = overpass.query(ql)
    elems = data.get("elements", [])
    way = next((e for e in elems if isinstance(e, dict) and e.get("type") == "way" and e.get("id") == way_id), None)
    if way is None:
        raise ValueError(f"Overpass: way {way_id} not found")
    ring = ensure_closed_ring(ring_from_overpass_geom(way.get("geometry")), f"way {way_id}")
    poly = Polygon(ring)
    poly = orient(poly, sign=1.0)  # consistent winding
    return ensure_simple_polygon(poly, f"way {way_id}")

def fetch_relation_polygon(overpass: OverpassClient, rel_id: int) -> Polygon:
    ql = f"""
    [out:json];
    relation({rel_id});
    out geom;
    """
    data = overpass.query(ql)
    elems = data.get("elements", [])
    rel = next((e for e in elems if isinstance(e, dict) and e.get("type") == "relation" and e.get("id") == rel_id), None)
    if rel is None:
        raise ValueError(f"Overpass: relation {rel_id} not found")

    members = rel.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"relation {rel_id}: no members")

    outers: List[List[Tuple[float, float]]] = []
    inners: List[List[Tuple[float, float]]] = []

    for m in members:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip().lower()
        mtype = m.get("type")
        geom = m.get("geometry")
        if mtype != "way" or not geom:
            continue  # ignore non-way members for polygon outline

        ring = ensure_closed_ring(ring_from_overpass_geom(geom), f"relation {rel_id} member")
        if role == "outer":
            outers.append(ring)
        elif role == "inner":
            inners.append(ring)

    if inners:
        raise ValueError(f"relation {rel_id}: has inner rings (holes); simple polygons only")
    if len(outers) != 1:
        raise ValueError(f"relation {rel_id}: expected exactly 1 outer ring, got {len(outers)} (not a simple polygon)")

    poly = Polygon(outers[0])
    poly = orient(poly, sign=1.0)
    return ensure_simple_polygon(poly, f"relation {rel_id}")

def fetch_osm_url_polygon(overpass: OverpassClient, osm_url: str) -> Polygon:
    osm_type, osm_id = parse_osm_url(osm_url)
    if osm_type == "way":
        return fetch_way_polygon(overpass, osm_id)
    if osm_type == "relation":
        return fetch_relation_polygon(overpass, osm_id)
    raise ValueError(f"OSM URL type '{osm_type}' not supported for polygons (use way or relation)")


# -----------------------------
# Expression evaluation (normal infix)
# Supports: names, |, -, parentheses
# -----------------------------

TOK_EXPR = re.compile(r"\s*(?:(?P<NAME>[A-Za-z_][A-Za-z0-9_.-]*)|(?P<OP>[\|\-\(\)]))")
Token = Tuple[str, str]
PRECEDENCE = {"|": 10, "-": 10}

def tokenize_infix(expr: str) -> List[Token]:
    pos = 0
    out: List[Token] = []
    expr = expr or ""
    while pos < len(expr):
        m = TOK_EXPR.match(expr, pos)
        if not m:
            raise ValueError(f"Invalid token near: {expr[pos:pos+40]!r}")
        if m.lastgroup == "NAME":
            out.append(("NAME", m.group("NAME")))
        else:
            out.append(("OP", m.group("OP")))
        pos = m.end()
    if not out:
        raise ValueError("Empty expression")
    return out

@dataclass
class Parser:
    tokens: List[Token]
    i: int = 0
    def peek(self) -> Optional[Token]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None
    def pop(self) -> Token:
        if self.i >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        t = self.tokens[self.i]
        self.i += 1
        return t

def parse_primary(p: Parser, env: Dict[str, Polygon]) -> Polygon:
    t = p.peek()
    if t is None:
        raise ValueError("Expected a polygon name or '('")
    typ, val = t
    if typ == "OP" and val == "(":
        p.pop()
        inside = parse_expr(p, env, min_prec=0)
        t2 = p.pop()
        if t2 != ("OP", ")"):
            raise ValueError("Expected ')'")
        return inside
    if typ == "NAME":
        p.pop()
        if val not in env:
            raise ValueError(f"Unknown polygon name in expression: {val!r}")
        return env[val]
    raise ValueError(f"Unexpected token: {t}")

def parse_expr(p: Parser, env: Dict[str, Polygon], min_prec: int = 0) -> Polygon:
    left = parse_primary(p, env)
    while True:
        t = p.peek()
        if t is None:
            break
        typ, op = t
        if typ != "OP" or op not in PRECEDENCE:
            break
        prec = PRECEDENCE[op]
        if prec < min_prec:
            break
        p.pop()
        right = parse_expr(p, env, min_prec=prec + 1)
        if op == "|":
            left = ensure_simple_polygon(left.union(right), "union")
        elif op == "-":
            left = ensure_simple_polygon(left.difference(right), "difference")
        else:
            raise ValueError(f"Unsupported operator: {op}")
        if left.geom_type != "Polygon":
            raise ValueError(f"{op}: result is not a simple Polygon (got {left.geom_type})")
    if left.geom_type != "Polygon":
        raise ValueError(f"expression: result is not a Polygon (got {left.geom_type})")
    return left  # type: ignore[return-value]

def eval_infix_expression(expr: str, env: Dict[str, Polygon]) -> Polygon:
    tokens = tokenize_infix(expr)
    p = Parser(tokens)
    result = parse_expr(p, env, min_prec=0)
    if p.peek() is not None:
        raise ValueError(f"Unexpected trailing tokens near: {p.tokens[p.i:]}")
    return result


# -----------------------------
# Main CSV pipeline
# -----------------------------

def build_polygon_from_row(
    row: dict,
    nom: NominatimClient,
    ov: OverpassClient,
    cache: Dict[str, Polygon],
    env: Dict[str, Polygon],
) -> Polygon:
    kind = (row.get("kind") or "").strip().lower()
    name = (row.get("name") or "").strip()
    if not name:
        raise ValueError("Row is missing 'name'")

    if kind == "point":
        wkt_point = row.get("wkt_point") or ""
        lat, lon = parse_wkt_point(wkt_point)
        geo = nom.reverse_polygon_geojson(lat=lat, lon=lon)

        # Nominatim gives GeoJSON; accept only Polygon w/out holes
        gtype = geo.get("type")
        coords = geo.get("coordinates")
        if gtype != "Polygon" or not isinstance(coords, list) or not coords:
            raise ValueError("Nominatim reverse did not return a simple Polygon geojson")

        # coords: [outer_ring, hole1, ...] where ring is [[lon,lat],...]
        if len(coords) != 1:
            raise ValueError("Nominatim polygon has holes; simple polygons only")
        outer = coords[0]
        if not isinstance(outer, list) or len(outer) < 4:
            raise ValueError("Nominatim polygon outer ring too short")
        ring = [(float(x), float(y)) for x, y in outer]
        poly = ensure_simple_polygon(orient(Polygon(ring), sign=1.0), "nominatim")

    elif kind == "url":
        url = row.get("osm_url") or ""
        osm_type, osm_id = parse_osm_url(url)
        cache_key = f"{osm_type}:{osm_id}"
        if cache_key in cache:
            poly = cache[cache_key]
        else:
            poly = fetch_osm_url_polygon(ov, url)
            cache[cache_key] = poly

    elif kind == "expr":
        expr = row.get("expr") or ""
        poly = eval_infix_expression(expr, env)

    elif kind == "wkt":
        w = row.get("wkt") or ""
        geom = shapely_wkt.loads(w)
        if geom.geom_type != "Polygon":
            raise ValueError(f"wkt: geometry is not a Polygon (got {geom.geom_type})")
        poly = ensure_simple_polygon(geom, "wkt")  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unknown kind {kind!r} for row {name!r}. Use: point, url, expr (or wkt).")

    return poly


def run(input_csv: str, output_csv: str, nom_cfg: NominatimConfig, ov_cfg: OverpassConfig) -> None:
    nom = NominatimClient(nom_cfg)
    ov = OverpassClient(ov_cfg)

    cache: Dict[str, Polygon] = {}
    env: Dict[str, Polygon] = {}
    rows_out: List[dict] = []

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"name", "kind"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            row_name = (row.get("name") or "").strip()
            display_name = row_name or f"(row {line_no})"
            try:
                poly = build_polygon_from_row(row, nom, ov, cache, env)
                if not row_name:
                    raise ValueError("Row is missing 'name'")
                env[row_name] = poly
                rows_out.append({"name": row_name, "wkt": polygon_to_wkt(poly), "status": "ok", "error": ""})
                map_url = geojsonio_url_from_polygon(poly)
                print(f"[OK] {row_name}")
                print(f"     Map URL: {map_url}")

            except Exception as e:
                rows_out.append({"name": display_name, "wkt": "", "status": "error", "error": f"{type(e).__name__}: {e}"})

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "wkt", "status", "error"])
        writer.writeheader()
        writer.writerows(rows_out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create simple WKT polygons from CSV: Nominatim reverse for points, Overpass for OSM URLs, infix expr ops.")
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--output", required=True, help="Output CSV path")

    ap.add_argument("--user-agent", default=NominatimConfig.user_agent, help="User-Agent header (identify your app + contact)")
    ap.add_argument("--min-interval", type=float, default=1.0, help="Min seconds between requests (applies to both services)")

    ap.add_argument("--nominatim-base-url", default=NominatimConfig.base_url, help="Nominatim base URL")
    ap.add_argument("--overpass-base-url", default=OverpassConfig.base_url, help="Overpass interpreter URL")
    ap.add_argument("--email", default=None, help="Optional email query parameter for Nominatim")

    ap.add_argument("--nominatim-timeout", type=float, default=30.0, help="Nominatim timeout seconds")
    ap.add_argument("--overpass-timeout", type=float, default=60.0, help="Overpass timeout seconds")

    args = ap.parse_args()

    nom_cfg = NominatimConfig(
        base_url=args.nominatim_base_url,
        min_interval_s=args.min_interval,
        user_agent=args.user_agent,
        email=args.email,
        timeout_s=args.nominatim_timeout,
    )
    ov_cfg = OverpassConfig(
        base_url=args.overpass_base_url,
        min_interval_s=args.min_interval,
        user_agent=args.user_agent,
        timeout_s=args.overpass_timeout,
    )

    try:
        run(args.input, args.output, nom_cfg, ov_cfg)
        return 0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
