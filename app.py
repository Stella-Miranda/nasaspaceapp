import io
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from google import genai
from google.genai import types
import numpy as np

app = Flask(__name__)

# Hardcoded Gemini API Key (keep private)
GEMINI_API_KEY = "AQ.Ab8RN6JUeazRiSan95etivoS4DXr0UIqvl7c90TeM4qvI7T4Tw"
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

IMAGE_CACHE = {}
POLARIZATION_LABELS = {
    "DHDH": "Dual-Pol HH + HV (Horizontal Tx)",
    "DVDV": "Dual-Pol VV + VH (Vertical Tx)",
    "SHSH": "Single-Pol HH",
    "SVSV": "Single-Pol VV",
    "QPDH": "Quad-Pol (HH + HV + VH + VV)",
    "CLDH": "Compact-Pol (Circular Tx, H/V Rx)",
}

def parse_granule_metadata(title):
    """Extracts Orbit Direction (Asc/Desc), Track, and Polarization Mode (HH/HV/VH) from NISAR filename."""
    parts = title.split("_")
    # Standard NISAR L2 naming: NISAR_L2_PR_GCOV_025_039_A_010_4005_DHDH_A_...
    orbit_dir = "Ascending" if "_A_" in title else ("Descending" if "_D_" in title else "Unknown")
    pol_code = next((p for p in parts if p in POLARIZATION_LABELS), "DHDH")
    pol_desc = POLARIZATION_LABELS.get(pol_code, pol_code)
    return {
        "orbit_dir": orbit_dir,
        "pol_code": pol_code,
        "pol_desc": pol_desc
    }

def compute_hh_hv_stats(img_a_bytes, img_b_bytes):
    """
    Measures pixel-level RGB shifts between Before (A) and After (B) crops:
    - Magenta intensity ((R + B)/2 - G) tracks Co-Pol (HH or VV: soil moisture, double-bounce, flooded vegetation)
    - Green intensity (G - (R + B)/2) tracks Cross-Pol (HV or VH: 3D vegetation canopy volume scattering)
    - Dark intensity (low R, G, B) tracks smooth open water
    """
    arr_a = np.asarray(Image.open(io.BytesIO(img_a_bytes)).convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(Image.open(io.BytesIO(img_b_bytes)).convert("RGB"), dtype=np.float32)

    def channel_metrics(arr):
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        co_pol = ((r + b) * 0.5) - g       # HH / VV dominance (Magenta)
        cross_pol = g - ((r + b) * 0.5)    # HV / VH dominance (Green)
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

def pick_lowest_res_browse(links):
    """Picks the smallest, fastest-loading low-resolution browse thumbnail."""
    candidates = []
    for link in links:
        href = link.get("href", "")
        rel = link.get("rel", "")
        if "browse#" in rel and href.lower().endswith((".jpg", ".jpeg", ".png")):
            score = 0
            lower = href.lower()
            if "thumb" in lower or "small" in lower or "low" in lower:
                score += 20
            if lower.endswith((".jpg", ".jpeg")):
                score += 10  # JPEGs are much smaller than full-res PNGs
            if "high" in lower or "full" in lower or "large" in lower:
                score -= 20
            candidates.append((score, href))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def normalize_lon(lon):
    """Wraps any longitude (e.g., 1000 or -540) cleanly into [-180, 180]."""
    return ((lon + 180.0) % 360.0) - 180.0


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

def send_event(step, percent, message, payload=None):
    print(f"[{percent}%] {step}: {message}", flush=True)
    return f"data: {json.dumps({'step': step, 'percent': percent, 'message': message, 'payload': payload})}\n\n"

def fetch_cached_image(url):
    if url in IMAGE_CACHE:
        return IMAGE_CACHE[url]
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    IMAGE_CACHE[url] = img
    return img

def pick_highest_res_browse(links):
    candidates = []
    for link in links:
        href = link.get("href", "")
        rel = link.get("rel", "")
        if "browse#" in rel and href.lower().endswith((".png", ".jpg", ".jpeg")):
            score = 10 if href.lower().endswith(".png") else 0
            lower = href.lower()
            if any(k in lower for k in ("thumb", "small", "low")):
                score -= 20
            if any(k in lower for k in ("high", "full", "large")):
                score += 15
            candidates.append((score, href))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def get_footprint_bounds(footprint):
    lats = [p[0] for p in footprint]
    lons = [p[1] for p in footprint]
    return min(lons), min(lats), max(lons), max(lats)

def measure_true_radar_pixel_coverage(bbox, footprint, browse_url):
    if not footprint or len(footprint) < 3:
        return 0
    u_lon_min, u_lat_min, u_lon_max, u_lat_max = bbox
    f_lon_min, f_lat_min, f_lon_max, f_lat_max = get_footprint_bounds(footprint)

    try:
        img = fetch_cached_image(browse_url)
        w, h = img.size
        x0 = int(round(((u_lon_min - f_lon_min) / max(1e-9, f_lon_max - f_lon_min)) * w))
        x1 = int(round(((u_lon_max - f_lon_min) / max(1e-9, f_lon_max - f_lon_min)) * w))
        y0 = int(round(((f_lat_max - u_lat_max) / max(1e-9, f_lat_max - f_lat_min)) * h))
        y1 = int(round(((f_lat_max - u_lat_min) / max(1e-9, f_lat_max - f_lat_min)) * h))

        cx0, cx1 = max(0, x0), min(w, max(x0 + 1, x1))
        cy0, cy1 = max(0, y0), min(h, max(y0 + 1, y1))
        if cx1 <= cx0 or cy1 <= cy0:
            return 0

        total_box_px = max(1, (x1 - x0) * (y1 - y0))
        cropped_alpha = img.crop((cx0, cy0, cx1, cy1)).getchannel("A")
        valid_pixels = sum(1 for a in cropped_alpha.getdata() if a > 10)
        return min(100, max(0, int(round((valid_pixels / total_box_px) * 100))))
    except Exception:
        return 0

def parse_iso_date(date_str):
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return None

def build_timeline_nodes(granules, min_days_apart=12, max_nodes=18):
    """
    Returns chronological nodes from first occurrence to last occurrence,
    prioritizing 100% overlap and filtering out near-duplicate passes closer than `min_days_apart`.
    """
    if not granules:
        return []
    full_cover = [g for g in granules if g["overlap_pct"] >= 98]
    partial_cover = [g for g in granules if 50 <= g["overlap_pct"] < 98]
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
        # Always include the very first and very last occurrence
        is_last = (idx == len(valid) - 1)
        if last_kept_dt is None or (dt - last_kept_dt).days >= min_days_apart or is_last:
            g["days_since_start"] = (dt - first_dt).days
            g["days_since_prev"] = (dt - last_kept_dt).days if last_kept_dt else 0
            g["timestamp_ms"] = int(dt.timestamp() * 1000)
            nodes.append(g)
            last_kept_dt = dt
            if len(nodes) >= max_nodes and not is_last:
                # Replace penultimate if needed so final occurrence stays included
                pass

    return nodes

def crop_aligned_region(browse_url, footprint, user_bbox, offset):
    """Crops the user's bounding box from the full image using their manual alignment offset."""
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
    # Upscale small crops slightly so Gemini can inspect fine coastal/urban details clearly
    if cropped.width < 512 or cropped.height < 512:
        ratio = 512 / max(cropped.width, cropped.height)
        cropped = cropped.resize((int(cropped.width * ratio), int(cropped.height * ratio)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()

@app.route('/')
def index():
    return render_template('index.html')


# Ordered fallback list of Gemini models (tries 1 by 1 until one succeeds)
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.0-flash",
    "gemini-2.5-pro",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

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
- **HH / VV Co-Pol (Magenta):** Surface & double-bounce scattering. Spikes when soil moisture rises after rain, when water floods beneath vegetation trunks/stems (L-band penetrates canopy to bounce off water + stalks), or when crops are harvested exposing bare soil.
- **HV / VH Cross-Pol (Green):** Volume scattering from randomly oriented leaves and branches. High in dense forests and mature leafy crops; drops when vegetation is cleared or flooded.
- **Low HH + Low HV (Dark/Black):** Specular reflection off calm open water (rivers, lakes, flooded bare fields).
- **Geometry Check:** {"Both passes share the exact same orbit direction and polarization mode, confirming color shifts represent real environmental changes." if same_geom else "Orbit direction or polarization mode differs between passes — explicitly note how viewing geometry contributes to the color shift."}

Based on the two images and the measured HH/HV/VH metrics above, provide a concise environmental change breakdown:
1. **HH / VV Co-Pol (Soil Moisture & Flooded Vegetation):** Explain what the {stats['hh_delta']:+.1f}% magenta shift indicates on the ground.
2. **HV / VH Cross-Pol (Canopy & Biomass):** Explain what the {stats['hv_delta']:+.1f}% green volume-scattering shift indicates.
3. **Open Water & Hydrology:** Note any changes in dark river channels or flooded basins ({stats['water_delta']:+.1f}%).
4. **Takeaway:** 2-sentence summary of the environmental change."""

        contents = [
            types.Part.from_bytes(data=img_a_bytes, mime_type="image/png"),
            types.Part.from_bytes(data=img_b_bytes, mime_type="image/png"),
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

        # Fallback: even if Gemini API fails, still return the exact quantitative HH/HV/VH pixel breakdown
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
            yield send_event("validate", 10, f"Validating box for {product}...")
            parts = [float(x) for x in bbox_raw.split(',')]
            lon_min, lat_min, lon_max, lat_max = parts
            lon_min = round(normalize_lon(lon_min), 5)
            lon_max = round(normalize_lon(lon_max), 5)
            lat_min = round(max(-85.0, min(85.0, lat_min)), 5)
            lat_max = round(max(-85.0, min(85.0, lat_max)), 5)

            # Handle boxes that cross the antimeridian or were drawn backwards
            if lon_min > lon_max:
                lon_min, lon_max = lon_max, lon_min
            if lat_min > lat_max:
                lat_min, lat_max = lat_max, lat_min

            parts = [lon_min, lat_min, lon_max, lat_max]
            if len(parts) != 4:
                yield send_event("error", 100, "Invalid bounding box.")
                return

            lon_min, lat_min, lon_max, lat_max = parts
            if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180 and -90 <= lat_min <= 90 and -90 <= lat_max <= 90):
                yield send_event("error", 100, "Coordinates out of valid Earth range.")
                return

            bbox_str = ",".join(map(str, parts))
            yield send_event("query", 30, f"Querying NASA CMR for all {product} occurrences...")
            t0 = time.time()
            resp = requests.get(
                "https://cmr.earthdata.nasa.gov/search/granules.json",
                params={
                    "provider": "ASF",
                    "short_name[]": collections,
                    "bounding_box": bbox_str,
                    "page_size": 100,
                    "sort_key": "start_date"
                },
                timeout=12
            )
            if resp.status_code != 200:
                yield send_event("error", 100, f"NASA CMR Error ({resp.status_code}): {resp.text}")
                return

            entries = resp.json().get("feed", {}).get("entry", [])
            yield send_event("inspecting", 50, f"Checking radar pixel coverage across {len(entries)} passes...")

            raw_items = []
            for entry in entries:
                low_res_url = pick_lowest_res_browse(entry.get("links", []))
                if not low_res_url:
                    continue
                poly_str = entry.get("polygons", [[""]])[0][0]
                coords = [float(x) for x in poly_str.split()]
                footprint = [[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)]

                title = entry.get("title", "")
                detected_type = product if product != "ALL" else next(
                    (p for p in ["GCOV", "GUNW", "GSLC", "GOFF", "RSLC"] if f"_{p}_" in title), "NISAR"
                )
                meta_info = parse_granule_metadata(title)
                raw_items.append({
                    "id": title,
                    "product_type": detected_type,
                    "orbit_dir": meta_info["orbit_dir"],
                    "pol_code": meta_info["pol_code"],
                    "pol_desc": meta_info["pol_desc"],
                    "time": entry.get("time_start"),
                    "date_label": (entry.get("time_start") or "")[:10],
                    "browse_image": hi_res_url,
                    "footprint": footprint
                })

            candidates = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                future_map = {
                    pool.submit(measure_true_radar_pixel_coverage, parts, item["footprint"], item["browse_image"]): item
                    for item in raw_items
                }
                completed, total = 0, max(1, len(future_map))
                for future in as_completed(future_map):
                    item = future_map[future]
                    pct = future.result()
                    completed += 1
                    if pct > 0:
                        item["overlap_pct"] = pct
                        candidates.append(item)
                    if completed % 5 == 0 or completed == total:
                        yield send_event("inspecting", 50 + int((completed / total) * 40), f"Checked {completed}/{total} images...")

            nodes = build_timeline_nodes(candidates, min_days_apart=12, max_nodes=18)
            elapsed = round(time.time() - t0, 2)

            yield send_event(
                "done", 100,
                f"Loaded {len(nodes)} timeline nodes ({nodes[0]['date_label']} → {nodes[-1]['date_label']}) in {elapsed}s." if nodes else "No overlapping radar passes found.",
                {"count": len(nodes), "product": product, "granules": nodes}
            )
        except Exception as e:
            yield send_event("error", 100, f"Error: {str(e)}")

    return Response(generate(), mimetype="text/event-stream")

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True)
