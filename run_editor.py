#!/usr/bin/env python3
"""
WiFi Planner Editor

Launches an interactive browser-based WiFi planning editor.
Optionally imports an existing .netspp file for editing.

Usage:
    python run_editor.py                           # Start fresh project
    python run_editor.py "Aqua-Com.netspp"         # Import existing .netspp
    python run_editor.py --output ./my_project     # Custom output dir
    python run_editor.py --port 8766               # Custom port
"""

import argparse
import http.server
import json
import os
import shutil
import sqlite3
import sys
import threading
import webbrowser
import zipfile
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


# ── Extraction (reused from run_planning.py) ─────────────────────────────────

def extract_netspp(netspp_path, extract_dir):
    """Extract .netspp (zip) file. Returns db_path."""
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(netspp_path, 'r') as z:
        z.extractall(extract_dir)
    db_path = os.path.join(extract_dir, 'database.db3')
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database.db3 not found in {netspp_path}")
    return db_path


def get_project_info(db_path):
    """Read project metadata from PredictiveData table."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT Name, TimeCreated, TimeModified FROM PredictiveData LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row:
        return {'name': row[0], 'created': row[1], 'modified': row[2]}
    return {'name': 'Unknown Project', 'created': '', 'modified': ''}


def get_zones(db_path, extract_dir):
    """Read zone info from DB. Returns list of zone dicts with map paths."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT z._id as zone_id, z.Name, z.Scale, m.PngName
        FROM Zones z JOIN Maps m ON m._id = z.MapId
        ORDER BY z._id
    """)
    zones = []
    for row in cur.fetchall():
        map_path = os.path.join(extract_dir, 'Maps', row['PngName'])
        if not os.path.exists(map_path):
            print(f"  Warning: map file not found: {map_path}", file=sys.stderr)
            continue
        zones.append({
            'id': row['zone_id'],
            'name': row['Name'],
            'scale': row['Scale'],
            'map_file': row['PngName'],
            'map_path': map_path,
        })
    conn.close()
    return zones


def get_routers(db_path, snapshot_id, coord_scale):
    """Read routers for a snapshot, group by Alias to pair 2.4G + 5G radios."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT Alias, x, y, Channel, TransmitPower, ChannelWidth, Model, Height, WiFiBand
        FROM Routers
        WHERE SnapId = ?
        ORDER BY Alias, WiFiBand
    """, (snapshot_id,))

    grouped = defaultdict(list)
    for row in cur.fetchall():
        grouped[row['Alias']].append(dict(row))
    conn.close()

    aps = []
    for alias, radios in grouped.items():
        first = radios[0]
        px = first['x'] * coord_scale
        py = first['y'] * coord_scale

        radio_data = {}
        for r in radios:
            band_key = '2g' if r['WiFiBand'] == 0 else '5g'
            radio_data[band_key] = {
                'channel': r['Channel'],
                'power': r['TransmitPower'],
                'width': r['ChannelWidth'],
            }

        aps.append({
            'alias': alias,
            'px': round(px, 1),
            'py': round(py, 1),
            'model': first['Model'],
            'height': first['Height'],
            'radios': radio_data,
        })

    return aps


def get_walls(db_path, zone_id, coord_scale):
    """Read walls and their points for a zone, convert to pixel coordinates."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT w._id as wall_id, w.Absorption, w.Material, w.Color, w.Reflection,
               wp1.X as x1, wp1.Y as y1, wp2.X as x2, wp2.Y as y2
        FROM Walls w
        JOIN WallsPoints wp1 ON wp1._id = w.StartPointId
        JOIN WallsPoints wp2 ON wp2._id = w.EndPointId
        WHERE w.ZoneId = ?
        ORDER BY w._id
    """, (zone_id,))

    walls = []
    for row in cur.fetchall():
        px1 = round(row['x1'] * coord_scale, 1)
        py1 = round(row['y1'] * coord_scale, 1)
        px2 = round(row['x2'] * coord_scale, 1)
        py2 = round(row['y2'] * coord_scale, 1)

        walls.append({
            'start': [px1, py1],
            'end': [px2, py2],
            'absorption': row['Absorption'],
            'material': row['Material'],
            'color': row['Color'] or '#FF5C5C5C',
        })

    conn.close()
    return walls


def get_snapshots(db_path):
    """Read snapshots grouped by zone."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT _id, ZoneId, Name FROM Snapshots ORDER BY _id")
    snapshots = []
    for row in cur.fetchall():
        snapshots.append({
            'id': row['_id'],
            'zone_id': row['ZoneId'],
            'name': row['Name'] or f'Scenario {row["_id"]}',
        })
    conn.close()
    return snapshots


def get_image_size(path):
    """Get image dimensions. Uses PIL if available, else basic PNG/JPEG header parsing."""
    if Image:
        with Image.open(path) as img:
            return img.size

    # Fallback: read PNG/JPEG headers
    with open(path, 'rb') as f:
        header = f.read(32)
        # PNG
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            w = int.from_bytes(header[16:20], 'big')
            h = int.from_bytes(header[20:24], 'big')
            return (w, h)
        # JPEG
        f.seek(0)
        data = f.read()
        idx = 2
        while idx < len(data) - 1:
            if data[idx] != 0xFF:
                break
            marker = data[idx + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h = int.from_bytes(data[idx+5:idx+7], 'big')
                w = int.from_bytes(data[idx+7:idx+9], 'big')
                return (w, h)
            length = int.from_bytes(data[idx+2:idx+4], 'big')
            idx += 2 + length
    # Fallback
    return (1000, 1000)


# ── Preload.js generation ────────────────────────────────────────────────────

def generate_preload_js(db_path, zones, extract_dir, output_path):
    """Generate preload.js with project data for the editor."""
    project = get_project_info(db_path)
    snapshots = get_snapshots(db_path)

    zone_data = []
    total_aps = 0
    total_walls = 0

    for z in zones:
        coord_scale = z['scale'] * 100
        img_w, img_h = get_image_size(z['map_path'])

        # Get first snapshot for this zone (editor loads one scenario)
        zone_snaps = [s for s in snapshots if s['zone_id'] == z['id']]
        aps = []
        if zone_snaps:
            aps = get_routers(db_path, zone_snaps[0]['id'], coord_scale)
        total_aps += len(aps)

        walls = get_walls(db_path, z['id'], coord_scale)
        total_walls += len(walls)

        print(f"  Zone \"{z['name']}\": {img_w}x{img_h}, scale={coord_scale:.2f} px/m, "
              f"{len(aps)} APs, {len(walls)} walls")

        zone_data.append({
            'name': z['name'],
            'map_file': z['map_file'],
            'scale': coord_scale,
            'image': {'width': img_w, 'height': img_h},
            'walls': walls,
            'aps': aps,
        })

    data = {
        'project': project,
        'zones': zone_data,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('const PRELOAD = ')
        json.dump(data, f, indent=1)
        f.write(';\n')

    print(f"  Total: {total_aps} APs, {total_walls} walls across {len(zones)} zone(s)")
    return data


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Launch the WiFi Planner Editor. Optionally import an existing .netspp file.')
    parser.add_argument('netspp', nargs='?', default=None,
                        help='Path to .netspp planning file to import (optional)')
    parser.add_argument('--output', '-o', default='./output_editor',
                        help='Output directory (default: ./output_editor)')
    parser.add_argument('--port', '-p', type=int, default=8766,
                        help='HTTP server port (default: 8766)')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== WiFi Planner Editor ===")

    # Step 1: Setup editor files
    print(f"\n[1/2] Setting up editor...")
    os.makedirs(output_dir, exist_ok=True)

    # Copy editor HTML
    src_editor = os.path.join(script_dir, 'website_editor', 'index.html')
    shutil.copy2(src_editor, os.path.join(output_dir, 'index.html'))

    # Copy leaflet.js + leaflet.css from website/
    leaflet_dir = os.path.join(script_dir, 'website')
    for fname in ['leaflet.js', 'leaflet.css']:
        src = os.path.join(leaflet_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
    print(f"  Copied editor files to {output_dir}")

    # Step 2: Import .netspp if provided
    if args.netspp:
        netspp_path = os.path.abspath(args.netspp)
        if not os.path.exists(netspp_path):
            print(f"Error: file not found: {netspp_path}", file=sys.stderr)
            sys.exit(1)

        project_name = Path(netspp_path).stem
        print(f"\n[2/2] Importing planning data from {project_name}...")

        extract_dir = os.path.join(output_dir, '_extracted')
        try:
            db_path = extract_netspp(netspp_path, extract_dir)
            zones = get_zones(db_path, extract_dir)
            print(f"  Found {len(zones)} zone(s): {', '.join(z['name'] for z in zones)}")

            # Copy map images to output dir
            for z in zones:
                shutil.copy2(z['map_path'], os.path.join(output_dir, z['map_file']))

            # Generate preload.js
            preload_path = os.path.join(output_dir, 'preload.js')
            generate_preload_js(db_path, zones, extract_dir, preload_path)
        finally:
            # Cleanup extracted files
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
    else:
        print(f"\n[2/2] No .netspp file provided, starting fresh project")

    # Start server and open browser
    url = f"http://localhost:{args.port}"
    print(f"\n=== Editor ready! ===")
    print(f"  Starting server: {url}")
    print(f"  Press Ctrl+C to stop\n")

    # Open browser after a short delay
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    # Serve files
    os.chdir(output_dir)
    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.update({
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.html': 'text/html',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
    })

    try:
        with http.server.HTTPServer(('', args.port), handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == '__main__':
    main()
