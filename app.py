import io
import json
import os
import time
from collections import OrderedDict
from datetime import datetime
import numpy as np
import requests
from PIL import Image
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from google import genai
from google.genai import types

app = Flask(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6JUeazRiSan95etivoS4DXr0UIqvl7c90TeM4qvI7T4Tw")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]

class BoundedCache(OrderedDict):
    def __init__(self, maxsize=20):
        super().__init__()
        self.maxsize = maxsize

    def get_item(self, key):
        if key not in self:
            return None
        self.move_to_end(key)
        return self[key]

    def set_item(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)

IMAGE_CACHE = BoundedCache(maxsize=16)
CMR_CACHE = BoundedCache(maxsize=30)

PRODUCT_COLLECTIONS = {
    "GCOV": ["NISAR_L2_GCOV_PROVISIONAL_V1", "NISAR_L2_GCOV_BETA_V1"],
    "GUNW": ["NISAR_L2_GUNW_PROVISIONAL_V1", "NISAR_L2_GUNW_BETA_V1"],
    "GSLC": ["NISAR_L2_GSLC_PROVISIONAL_V1", "NISAR_L2_GSLC_BETA_V1"],
    "GOFF": ["NISAR_L2_GOFF_PROVISIONAL_V1", "NISAR_L2_GOFF_BETA_V1"],
    "RSLC": ["NISAR_L1_RSLC_PROVISIONAL_V1", "NISAR_L1_RSLC_BETA_V1"],
    "ALL": [
        "NISAR_L2_GCOV_PROVISIONAL_V1", "NISAR_L2_GCOV_BETA_V1",
        "NISAR_L2_GUNW_PROVISIONAL_V1", "NISAR_L2_GUNW_BETA_V1",
        "NISAR_L2_GSLC_PROVISIONAL_V1", "NISAR_L2_GSLC_BETA_V1",
        "NISAR_L2_GOFF_PROVISIONAL_V1", "NISAR_L2_GOFF_BETA_V1",
        "NISAR_L1_RSLC_PROVISIONAL_V1", "NISAR_L1_RSLC_BETA_V1",
    ],
}

POLARIZATION_LABELS = {
    "DHDH": "Dual-Pol HH + HV (Horizontal Tx)",
    "DVDV": "Dual-Pol VV + VH (Vertical Tx)",
    "SHSH": "Single-Pol HH",
    "SVSV": "Single-Pol VV",
    "QPDH": "Quad-Pol (HH + HV + VH + VV)",
    "CLDH": "Compact-Pol (Circular Tx, H/V Rx)",
}

def send_event(step, percent, message, payload=None):
    return f"data: {json.dumps({'step': step, 'percent': percent, 'message': message, 'payload': payload})}\n\n"

def normalize_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0

def fetch_cached_image(url):
    cached = IMAGE_CACHE.get_item(url)
    if cached is not None:
        return cached
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    IMAGE_CACHE.set_item(url, img)
    return img

def pick_default_browse(links):
    """Picks the original standard browse resolution (skips tiny _thumb previews)."""
    fallback = None
    for link in links:
        href = link.get("href", "")
        rel = link.get("rel", "")
        if "browse#" in rel and href.lower().endswith((".png", ".jpg", ".jpeg")):
            if "thumb" not in href.lower():
                return href
            if not fallback:
                fallback = href
    return fallback

def parse_granule_metadata(title):
    parts = title.split("_")
    orbit_dir = "Ascending" if "_A_" in title else ("Descending" if "_D_" in title else "Unknown")
    pol_code = next((p for p in parts if p in POLARIZATION_LABELS), "DHDH")
    return {
        "orbit_dir": orbit_dir,
        "pol_code": pol_code,
        "pol_desc": POLARIZATION_LABELS.get(pol_code, pol_code),
    }

def point_in_polygon(lat, lon, poly):
    inside = False
    n = len(poly)
    if n < 3:
        return False
    p1_lat, p1_lon = poly[0]
    for i in range(1, n + 1):
        p2_lat, p2_lon = poly[i % n]
        if lon > min(p1_lon, p2_lon):
            if lon <= max(p1_lon, p2_lon):
                if lat <= max(p1_lat, p2_lat):
                    if p1_lon != p2_lon:
                        xinters = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                    if p1_lat == p2_lat or lat <= xinters:
                        inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon
    return inside

def fast_swath_overlap_pct(bbox, footprint):
    """Instant 0-byte geometric overlap check (zero image downloads during search)."""
    if not footprint or len(footprint) < 3:
        return 0

    c_lat = sum(p[0] for p in footprint) / len(footprint)
    c_lon = sum(p[1] for p in footprint) / len(footprint)

    inner_poly = [
        [c_lat + (p[0] - c_lat) * 0.88, c_lon + (p[1] - c_lon) * 0.88]
        for p in footprint
    ]

    lon_min, lat_min, lon_max, lat_max = bbox
    steps = 4
    hits = 0
    for i in range(steps):
        lat = lat_min + (lat_max - lat_min) * (i / (steps - 1))
        for j in range(steps):
            lon = lon_min + (lon_max - lon_min) * (j / (steps - 1))
            if point_in_polygon(lat, lon, inner_poly):
                hits += 1
    return int(round((hits / (steps * steps)) * 100))

def parse_iso_date(date_str):
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None

def build_timeline_nodes(granules, min_days_apart=12, max_nodes=16):
    if not granules:
        return []
    full_cover = [g for g in granules if g["overlap_pct"] >= 90]
    partial_cover = [g for g in granules if 50 <= g["overlap_pct"] < 90]
    pool = full_cover if len(full_cover) >= 2 else (
        full_cover + partial_cover if len(full_cover + partial_cover) >= 2 else granules
    )

    valid = [(parse_iso_date(g["time"]), g) for g in pool if parse_iso_date(g["time"])]
    if not valid:
        return []
    valid.sort(key=lambda item: item[0])

    first_dt = valid[0][0]
    nodes = []
    last_kept_dt = None

    for idx, (dt, g) in enumerate(valid):
        is_last = (idx == len(valid) - 1)
        if last_kept_dt is None or (dt - last_kept_dt).days >= min_days_apart or is_last:
            g["days_since_start"] = (dt - first_dt).days
            g["days_since_prev"] = (dt - last_kept_dt).days if last_kept_dt else 0
            g["timestamp_ms"] = int(dt.timestamp() * 1000)
            nodes.append(g)
            last_kept_dt = dt
            if len(nodes) >= max_nodes and not is_last:
                pass

    return nodes

def crop_aligned_region(browse_url, footprint, user_bbox, offset):
    img = fetch_cached_image(browse_url)
    w, h = img.size

    lats = [p[0] for p in footprint]
    lons = [p[1] for p in footprint]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    d_lat = float(offset.get("dLat", 0))
    d_lon = float(offset.get("dLon", 0))
    scale = float(offset.get("scale", 1.0))

    c_lat = (min_lat + max_lat) / 2.0 + d_lat
    c_lon = (min_lon + max_lon) / 2.0 + d_lon
    half_h = ((max_lat - min_lat) / 2.0) * scale
    half_w = ((max_lon - min_lon) / 2.0) * scale

    adj_min_lon, adj_max_lon = c_lon - half_w, c_lon + half_w
    adj_min_lat, adj_max_lat = c_lat - half_h, c_lat + half_h

    u_lon_min, u_lat_min, u_lon_max, u_lat_max = user_bbox

    x0 = max(0, min(w - 1, int(round(((u_lon_min - adj_min_lon) / max(1e-9, adj_max_lon - adj_min_lon)) * w))))
    x1 = max(x0 + 1, min(w, int(round(((u_lon_max - adj_min_lon) / max(1e-9, adj_max_lon - adj_min_lon)) * w))))
    y0 = max(0, min(h - 1, int(round(((adj_max_lat - u_lat_max) / max(1e-9, adj_max_lat - adj_min_lat)) * h))))
    y1 = max(y0 + 1, min(h, int(round(((adj_max_lat - u_lat_min) / max(1e-9, adj_max_lat - adj_min_lat)) * h))))

    cropped = img.crop((x0, y0, x1, y1)).convert("RGB")
    if cropped.width < 384 or cropped.height < 384:
        ratio = 384 / max(cropped.width, cropped.height)
        cropped = cropped.resize((int(cropped.width * ratio), int(cropped.height * ratio)), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def compute_hh_hv_stats(img_a_bytes, img_b_bytes):
    arr_a = np.asarray(Image.open(io.BytesIO(img_a_bytes)).convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(Image.open(io.BytesIO(img_b_bytes)).convert("RGB"), dtype=np.float32)

    def channel_metrics(arr):
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        co_pol = ((r + b) * 0.5) - g
        cross_pol = g - ((r + b) * 0.5)
        brightness = (r + g + b) / 3.0

        total_px = max(1, arr.shape[0] * arr.shape[1])
        hh_dom_pct = round(float(np.sum(co_pol > 20) / total_px * 100), 1)
        hv_dom_pct = round(float(np.sum(cross_pol > 12) / total_px * 100), 1)
        water_pct = round(float(np.sum(brightness < 38) / total_px * 100), 1)
        return hh_dom_pct, hv_dom_pct, water_pct

    hh_a, hv_a, w_a = channel_metrics(arr_a)
    hh_b, hv_b, w_b = channel_metrics(arr_b)

    return {
        "hh_before": hh_a, "hh_after": hh_b, "hh_delta": round(hh_b - hh_a, 1),
        "hv_before": hv_a, "hv_after": hv_b, "hv_delta": round(hv_b - hv_a, 1),
        "water_before": w_a, "water_after": w_b, "water_delta": round(w_b - w_a, 1),
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-change', methods=['POST'])
def analyze_change():
    data = request.json
    node_a = data.get('nodeA')
    node_b = data.get('nodeB')
    offset_a = data.get('offsetA', {})
    offset_b = data.get('offsetB', {})
    user_bbox = data.get('bbox')

    if not node_a or not node_b or not user_bbox:
        return jsonify({'error': 'Select two timeline nodes (A and B) first.'}), 400

    try:
        img_a_bytes = crop_aligned_region(node_a['browse_image'], node_a['footprint'], user_bbox, offset_a)
        img_b_bytes = crop_aligned_region(node_b['browse_image'], node_b['footprint'], user_bbox, offset_b)
        stats = compute_hh_hv_stats(img_a_bytes, img_b_bytes)

        days_diff = abs(node_b.get('days_since_start', 0) - node_a.get('days_since_start', 0))
        pol_a = node_a.get('pol_desc', 'Dual-Pol HH + HV')
        pol_b = node_b.get('pol_desc', 'Dual-Pol HH + HV')
        orbit_a = node_a.get('orbit_dir', 'Unknown')
        orbit_b = node_b.get('orbit_dir', 'Unknown')
        same_geom = (orbit_a == orbit_b) and (node_a.get('pol_code') == node_b.get('pol_code'))

        stats_summary = (
            f"Quantitative Polarization Shift ({node_a.get('date_label')} → {node_b.get('date_label')}, {days_diff}d):\n"
            f"• Co-Pol HH/VV (Magenta — soil moisture, flooded vegetation, double-bounce): {stats['hh_before']}% → {stats['hh_after']}% ({stats['hh_delta']:+.1f}%)\n"
            f"• Cross-Pol HV/VH (Green — dense canopy & crop volume scattering): {stats['hv_before']}% → {stats['hv_after']}% ({stats['hv_delta']:+.1f}%)\n"
            f"• Low-Backscatter Water (Black — smooth open water/inundation): {stats['water_before']}% → {stats['water_after']}% ({stats['water_delta']:+.1f}%)\n"
            f"• Sensor Geometry: {orbit_a} ({pol_a}) vs {orbit_b} ({pol_b})"
        )

        prompt = f"""You are an L-band Synthetic Aperture Radar (SAR) Earth observation analyst comparing two NISAR {node_a.get('product_type', 'GCOV')} images over bounding box {user_bbox}:

{stats_summary}

Polarimetric Physics Guide:
- **HH / VV Co-Pol (Magenta):** Surface & double-bounce scattering. Spikes when soil moisture rises after rain, when water floods beneath vegetation trunks/stems, or when crops are harvested exposing bare soil.
- **HV / VH Cross-Pol (Green):** Volume scattering from randomly oriented leaves and branches. High in dense forests and mature leafy crops; drops when vegetation is cleared or flooded.
- **Low HH + Low HV (Dark/Black):** Specular reflection off calm open water.
- **Geometry Check:** {"Both passes share the exact same orbit direction and polarization mode, confirming color shifts represent real environmental changes." if same_geom else "Orbit direction or polarization mode differs between passes — note how viewing geometry contributes to the color shift."}

Based on the two images and the measured HH/HV/VH metrics above, provide a concise environmental change breakdown:
1. **HH / VV Co-Pol (Soil Moisture & Flooded Vegetation):** Explain what the {stats['hh_delta']:+.1f}% magenta shift indicates on the ground.
2. **HV / VH Cross-Pol (Canopy & Biomass):** Explain what the {stats['hv_delta']:+.1f}% green volume-scattering shift indicates.
3. **Open Water & Hydrology:** Note any changes in dark river channels or flooded basins ({stats['water_delta']:+.1f}%).
4. **Takeaway:** 2-sentence summary of the environmental change."""

        contents = [
            types.Part.from_bytes(data=img_a_bytes, mime_type="image/jpeg"),
            types.Part.from_bytes(data=img_b_bytes, mime_type="image/jpeg"),
            prompt
        ]

        last_error = None
        for model_name in GEMINI_MODELS:
            try:
                response = gemini_client.models.generate_content(model=model_name, contents=contents)
                if not response.text:
                    raise ValueError("Empty response")
                return jsonify({
                    "analysis": f"{stats_summary}\n\n---\n\n{response.text}",
                    "stats": stats,
                    "dateA": node_a.get('date_label'),
                    "dateB": node_b.get('date_label'),
                    "daysDiff": days_diff
                })
            except Exception as model_err:
                last_error = str(model_err)
                continue

        return jsonify({
            "analysis": f"{stats_summary}\n\n(AI narrative unavailable: {last_error})",
            "stats": stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search-stream')
def search_stream():
    bbox_raw = request.args.get('bbox', '')
    product = request.args.get('product', 'GCOV').upper()
    collections = PRODUCT_COLLECTIONS.get(product, PRODUCT_COLLECTIONS["GCOV"])

    @stream_with_context
    def generate():
        try:
            yield send_event("validate", 15, f"Validating box for {product}...")
            parts = [float(x) for x in bbox_raw.split(',')]
            if len(parts) != 4:
                yield send_event("error", 100, "Invalid bounding box.")
                return

            lon_min, lat_min, lon_max, lat_max = parts
            lon_min = round(normalize_lon(lon_min), 4)
            lon_max = round(normalize_lon(lon_max), 4)
            lat_min = round(max(-85.0, min(85.0, lat_min)), 4)
            lat_max = round(max(-85.0, min(85.0, lat_max)), 4)

            if lon_min > lon_max:
                lon_min, lon_max = lon_max, lon_min
            if lat_min > lat_max:
                lat_min, lat_max = lat_max, lat_min

            parts = [lon_min, lat_min, lon_max, lat_max]
            bbox_str = ",".join(map(str, parts))
            cache_key = f"{product}:{bbox_str}"

            t0 = time.time()
            entries = CMR_CACHE.get_item(cache_key)
            if entries is None:
                yield send_event("query", 45, f"Querying NASA CMR for {product}...")
                resp = requests.get(
                    "https://cmr.earthdata.nasa.gov/search/granules.json",
                    params={
                        "provider": "ASF",
                        "short_name[]": collections,
                        "bounding_box": bbox_str,
                        "page_size": 100,
                        "sort_key": "start_date"
                    },
                    timeout=10
                )
                if resp.status_code != 200:
                    yield send_event("error", 100, f"NASA CMR Error ({resp.status_code}): {resp.text}")
                    return
                entries = resp.json().get("feed", {}).get("entry", [])
                CMR_CACHE.set_item(cache_key, entries)

            yield send_event("filtering", 80, f"Scoring swath geometry across {len(entries)} passes...")

            candidates = []
            for entry in entries:
                browse_url = pick_default_browse(entry.get("links", []))
                if not browse_url:
                    continue
                poly_str = entry.get("polygons", [[""]])[0][0]
                coords = [float(x) for x in poly_str.split()]
                footprint = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]

                overlap_pct = fast_swath_overlap_pct(parts, footprint)
                if overlap_pct == 0:
                    continue

                title = entry.get("title", "")
                detected_type = product if product != "ALL" else next(
                    (p for p in ["GCOV", "GUNW", "GSLC", "GOFF", "RSLC"] if f"_{p}_" in title), "NISAR"
                )
                meta_info = parse_granule_metadata(title)
                candidates.append({
                    "id": title,
                    "product_type": detected_type,
                    "orbit_dir": meta_info["orbit_dir"],
                    "pol_code": meta_info["pol_code"],
                    "pol_desc": meta_info["pol_desc"],
                    "overlap_pct": overlap_pct,
                    "time": entry.get("time_start"),
                    "date_label": (entry.get("time_start") or "")[:10],
                    "browse_image": browse_url,
                    "footprint": footprint
                })

            nodes = build_timeline_nodes(candidates, min_days_apart=12, max_nodes=16)
            elapsed = round(time.time() - t0, 2)

            yield send_event(
                "done", 100,
                f"Loaded {len(nodes)} timeline nodes ({nodes[0]['date_label']} → {nodes[-1]['date_label']}) in {elapsed}s." if nodes else "No overlapping radar passes found.",
                {"count": len(nodes), "product": product, "granules": nodes}
            )
        except Exception as e:
            yield send_event("error", 100, f"Error: {str(e)}")

    return Response(generate(), mimetype="text/event-stream")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
