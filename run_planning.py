#!/usr/bin/env python3
"""
WiFi Planning Viewer (.netspp)

Extracts a .netspp planning file and generates an interactive website
with draggable AP markers and predictive signal heatmap.

Usage:
    python run_planning.py "Aqua-Com.netspp"
    python run_planning.py "Aqua-Com.netspp" --output ./output_planning
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_netspp(netspp_path, extract_dir):
    """Extract .netspp (zip) file. Returns db_path."""
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(netspp_path, 'r') as z:
        z.extractall(extract_dir)
    db_path = os.path.join(extract_dir, 'database.db3')
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database.db3 not found in {netspp_path}")
    return db_path


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


# ── Planning Data Extraction ─────────────────────────────────────────────────

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


def get_routers(db_path, snapshot_id, coord_scale, img_height):
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
            'x': round(first['x'], 2),
            'y': round(first['y'], 2),
            'px': round(px, 1),
            'py': round(py, 1),
            'model': first['Model'],
            'height': first['Height'],
            'radios': radio_data,
        })

    return aps


def get_walls(db_path, zone_id, coord_scale, img_height):
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
            'id': row['wall_id'],
            'start': [px1, py1],
            'end': [px2, py2],
            'absorption': row['Absorption'],
            'material': row['Material'],
            'color': row['Color'] or '#FF5C5C5C',
        })

    conn.close()
    return walls


# ── Data JS Generation ───────────────────────────────────────────────────────

def generate_data_js(db_path, zones, extract_dir, output_path):
    """Generate data.js for the planning viewer."""
    project = get_project_info(db_path)
    snapshots = get_snapshots(db_path)

    zone_data = []
    total_aps = 0
    total_walls = 0

    for z in zones:
        coord_scale = z['scale'] * 100
        with Image.open(z['map_path']) as img:
            img_w, img_h = img.size

        # Get snapshots for this zone
        zone_snaps = [s for s in snapshots if s['zone_id'] == z['id']]

        snap_data = []
        for snap in zone_snaps:
            aps = get_routers(db_path, snap['id'], coord_scale, img_h)
            total_aps += len(aps)
            snap_data.append({
                'id': snap['id'],
                'name': snap['name'],
                'aps': aps,
            })

        walls = get_walls(db_path, z['id'], coord_scale, img_h)
        total_walls += len(walls)

        print(f"  Zone \"{z['name']}\": {img_w}x{img_h}, scale={coord_scale:.4f}, "
              f"{len(zone_snaps)} snapshot(s), "
              f"{sum(len(s['aps']) for s in snap_data)} APs, {len(walls)} walls")

        zone_data.append({
            'id': z['id'],
            'name': z['name'],
            'map_file': z['map_file'],
            'image': {'width': img_w, 'height': img_h},
            'coord_scale': coord_scale,
            'snapshots': snap_data,
            'walls': walls,
        })

    data = {
        'project': project,
        'zones': zone_data,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('const DATA = ')
        json.dump(data, f, indent=1)
        f.write(';\n')

    print(f"  Exported {total_aps} APs, {total_walls} walls across {len(zones)} zone(s)")
    return data


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive planning viewer from a .netspp file.')
    parser.add_argument('netspp', help='Path to .netspp planning file')
    parser.add_argument('--output', '-o', default='./output_planning',
                        help='Output directory (default: ./output_planning)')
    args = parser.parse_args()

    netspp_path = os.path.abspath(args.netspp)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(netspp_path):
        print(f"Error: file not found: {netspp_path}", file=sys.stderr)
        sys.exit(1)

    project_basename = Path(netspp_path).stem
    print(f"=== WiFi Planning: {project_basename} ===")

    # Step 1: Extract .netspp
    extract_dir = os.path.join(output_dir, '_extracted')
    print(f"\n[1/3] Extracting {Path(netspp_path).name}...")
    db_path = extract_netspp(netspp_path, extract_dir)
    zones = get_zones(db_path, extract_dir)
    print(f"  Found {len(zones)} zone(s): {', '.join(z['name'] for z in zones)}")

    # Step 2: Read planning data and generate data.js
    website_dir = os.path.join(output_dir, 'website')
    os.makedirs(website_dir, exist_ok=True)
    data_js_path = os.path.join(website_dir, 'data.js')
    print(f"\n[2/3] Reading planning data...")
    generate_data_js(db_path, zones, extract_dir, data_js_path)

    # Step 3: Copy website assets
    print(f"\n[3/3] Generating website...")
    for z in zones:
        shutil.copy2(z['map_path'], os.path.join(website_dir, z['map_file']))

    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'website_planning')
    shutil.copy2(os.path.join(src_dir, 'index.html'), os.path.join(website_dir, 'index.html'))

    # Copy leaflet from existing website/
    leaflet_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'website')
    for fname in ['leaflet.js', 'leaflet.css']:
        shutil.copy2(os.path.join(leaflet_dir, fname), os.path.join(website_dir, fname))

    print(f"  Website ready: {website_dir}")

    # Cleanup
    shutil.rmtree(extract_dir)

    print(f"\n=== Done! ===")
    print(f"  Website: {website_dir}")
    print(f"\n  To view: python -m http.server 8765 -d \"{website_dir}\"")


if __name__ == '__main__':
    main()
