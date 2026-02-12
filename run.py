#!/usr/bin/env python3
"""
Unified WiFi Survey Script

Extracts a .netspu file and generates:
  - Interactive website (website/ folder with index.html, data.js, map images)
  - PDF heatmap report

Supports single-zone and multi-zone surveys.

Usage:
    python run.py "survey.netspu"
    python run.py "survey.netspu" --output ./my_output
    python run.py "survey.netspu" --ssid-prefix "Corp-"
"""

import argparse
import json
import math
import os
import shutil
import sqlite3
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from PIL import Image


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_netspu(netspu_path, extract_dir):
    """Extract .netspu (zip) file. Returns db_path."""
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(netspu_path, 'r') as z:
        z.extractall(extract_dir)
    db_path = os.path.join(extract_dir, 'database.db3')
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database.db3 not found in {netspu_path}")
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


# ── SSID Detection ───────────────────────────────────────────────────────────

def detect_ssid_prefix(db_path):
    """Auto-detect the most common SSID prefix (e.g. 'PG-') from the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT SSID, COUNT(*) as cnt
        FROM WlanNetworks WHERE SSID != ''
        GROUP BY SSID ORDER BY cnt DESC
    """)
    rows = cur.fetchall()
    conn.close()

    prefix_counts = defaultdict(int)
    for ssid, cnt in rows:
        if '-' in ssid:
            prefix = ssid.split('-')[0] + '-'
            prefix_counts[prefix] += cnt

    if prefix_counts:
        return max(prefix_counts, key=prefix_counts.get)
    return None


def detect_main_ssids(db_path, ssid_prefix):
    """Detect main SSIDs matching the prefix, ordered by observation count."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT SSID, COUNT(*) as cnt
        FROM WlanNetworks WHERE SSID LIKE ? AND SSID != ''
        GROUP BY SSID ORDER BY cnt DESC
    """, (ssid_prefix + '%',))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return []
    max_count = rows[0][1]
    threshold = max_count * 0.1
    return [ssid for ssid, cnt in rows if cnt >= threshold]


# ── Interference Calculation ─────────────────────────────────────────────────

def dbm_to_mw(dbm):
    return 10 ** (dbm / 10)


def mw_to_dbm(mw):
    if mw <= 0:
        return -100
    return 10 * math.log10(mw)


def freq_range_5g(center_ch, width_str):
    center_freq = 5000 + center_ch * 5
    half = {"MHZ_20": 10, "MHZ_40": 20, "MHZ_80": 40, "MHZ_160": 80}.get(width_str, 10)
    return (center_freq - half, center_freq + half)


def freq_range_2g(channel):
    center_freq = 2407 + channel * 5
    return (center_freq - 11, center_freq + 11)


def ranges_overlap(r1, r2):
    return r1[0] < r2[1] and r2[0] < r1[1]


def calc_interference(networks, ssid_prefix):
    """Calculate co-channel interference for networks matching the prefix."""
    prefix_nets = [n for n in networks if n["ssid"].startswith(ssid_prefix)]

    phys_aps_2g = defaultdict(list)
    phys_aps_5g = defaultdict(list)
    for n in prefix_nets:
        base_mac = n["bssid"][3:]
        if n["band"] == "2.4 GHz":
            phys_aps_2g[base_mac].append(n)
        else:
            phys_aps_5g[base_mac].append(n)

    def build_ap_info(phys_aps, band):
        ap_list = []
        for mac, nets in phys_aps.items():
            best_net = max(nets, key=lambda n: n["rssi"])
            if band == "2.4 GHz":
                fr = freq_range_2g(best_net["channel"])
            else:
                fr = freq_range_5g(best_net["center"], best_net["width"])
            ap_list.append({
                "mac": mac, "rssi": round(best_net["rssi"], 1),
                "ch": best_net["channel"],
                "center": best_net.get("center", best_net["channel"]),
                "width": best_net["width"], "freq": fr,
            })
        return ap_list

    aps_2g = build_ap_info(phys_aps_2g, "2.4 GHz")
    aps_5g = build_ap_info(phys_aps_5g, "5 GHz")

    def calc_band_interference(ap_list):
        if not ap_list:
            return {"max_cci": 0, "worst_sir": None, "worst_ch": None, "overlap_groups": []}
        overlap_groups = []
        for i, ap in enumerate(ap_list):
            overlapping = []
            for j, other in enumerate(ap_list):
                if i != j and ranges_overlap(ap["freq"], other["freq"]):
                    overlapping.append(other)
            if overlapping:
                ap_mw = dbm_to_mw(ap["rssi"])
                intf_mw = sum(dbm_to_mw(o["rssi"]) for o in overlapping)
                sir = round(mw_to_dbm(ap_mw) - mw_to_dbm(intf_mw), 1)
                overlap_groups.append({
                    "ap": {"mac": ap["mac"], "rssi": ap["rssi"],
                           "ch": ap["ch"], "width": ap["width"],
                           "freq_lo": ap["freq"][0], "freq_hi": ap["freq"][1]},
                    "interferers": [{"mac": o["mac"], "rssi": o["rssi"],
                                     "ch": o["ch"], "width": o["width"],
                                     "freq_lo": o["freq"][0], "freq_hi": o["freq"][1]}
                                    for o in overlapping],
                    "competing": len(overlapping), "sir": sir,
                })
        max_cci = max((g["competing"] for g in overlap_groups), default=0)
        sirs = [g["sir"] for g in overlap_groups]
        worst_sir = min(sirs) if sirs else None
        worst_group = max(overlap_groups, key=lambda g: g["competing"]) if overlap_groups else None
        return {
            "max_cci": max_cci, "worst_sir": worst_sir,
            "worst_ch": worst_group["ap"]["ch"] if worst_group else None,
            "overlap_groups": overlap_groups,
        }

    summary_2g = calc_band_interference(aps_2g)
    summary_5g = calc_band_interference(aps_5g)
    overall_max_cci = max(summary_2g["max_cci"], summary_5g["max_cci"])
    sirs = [s for s in [summary_2g["worst_sir"], summary_5g["worst_sir"]] if s is not None]

    return {
        "max_cci": overall_max_cci,
        "worst_sir": min(sirs) if sirs else None,
        "band_2g": summary_2g, "band_5g": summary_5g,
    }


# ── Per-zone scan point extraction ──────────────────────────────────────────

def build_zone_scan_points(cur, zone_id, coord_scale, main_ssids, ssid_prefix):
    """Extract scan points for a single zone, with signal and interference data."""
    cur.execute("""
        SELECT sp._id, sp.x, sp.y, sp.Time
        FROM ScanPoints sp
        JOIN Snapshots s ON s._id = sp.SnapId
        WHERE s.ZoneId = ?
        ORDER BY sp._id
    """, (zone_id,))
    scan_points_raw = cur.fetchall()

    sp_ids = [r[0] for r in scan_points_raw]
    if not sp_ids:
        return []

    placeholders = ','.join('?' * len(sp_ids))
    cur.execute(f"""
        SELECT ScanPointId, BSSID, SSID, RSSI, Noise,
               ChannelPrimary, ChannelCenter, ChannelWidth, WiFiBand,
               SecurityMode, PHYModes
        FROM WlanNetworks WHERE SSID != '' AND ScanPointId IN ({placeholders})
        ORDER BY ScanPointId, SSID, RSSI DESC
    """, sp_ids)
    all_networks = cur.fetchall()

    point_networks = {}
    for net in all_networks:
        pid = net[0]
        if pid not in point_networks:
            point_networks[pid] = []
        point_networks[pid].append({
            "bssid": net[1], "ssid": net[2], "rssi": net[3],
            "noise": net[4], "channel": net[5],
            "center": net[6], "width": net[7],
            "band": "5 GHz" if net[8] == 1 else "2.4 GHz",
            "security": net[9], "phy": net[10],
        })

    scan_points = []
    for sp in scan_points_raw:
        pid, x, y, time = sp
        px = x * coord_scale
        py = y * coord_scale
        networks = point_networks.get(pid, [])

        best_per_ssid = {}
        best_per_ssid_2g = {}
        best_per_ssid_5g = {}
        for net in networks:
            ssid = net["ssid"]
            rssi = net["rssi"]
            if ssid not in best_per_ssid or rssi > best_per_ssid[ssid]:
                best_per_ssid[ssid] = rssi
            if net["band"] == "2.4 GHz":
                if ssid not in best_per_ssid_2g or rssi > best_per_ssid_2g[ssid]:
                    best_per_ssid_2g[ssid] = rssi
            else:
                if ssid not in best_per_ssid_5g or rssi > best_per_ssid_5g[ssid]:
                    best_per_ssid_5g[ssid] = rssi

        main_signals = [best_per_ssid[s] for s in main_ssids if s in best_per_ssid]
        best_signal = max(main_signals) if main_signals else None
        cci = calc_interference(networks, ssid_prefix)

        scan_points.append({
            "id": pid, "x": round(px, 1), "y": round(py, 1), "time": time,
            "best_signal": best_signal,
            "signals_any": best_per_ssid, "signals_2g": best_per_ssid_2g,
            "signals_5g": best_per_ssid_5g,
            "networks": networks, "interference": cci,
        })

    return scan_points


# ── Data JS Generation ───────────────────────────────────────────────────────

def generate_data_js(db_path, zones, output_path, main_ssids, ssid_prefix):
    """Generate data.js for the interactive website (multi-zone)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT Name, TimeCreated, TimeModified FROM SurveyData LIMIT 1")
    survey_name, time_created, time_modified = cur.fetchone()

    cur.execute("SELECT DISTINCT SSID FROM WlanNetworks WHERE SSID != '' ORDER BY SSID")
    all_ssids = [r[0] for r in cur.fetchall()]

    zone_data = []
    total_points = 0
    for z in zones:
        coord_scale = z['scale'] * 100
        with Image.open(z['map_path']) as img:
            img_w, img_h = img.size

        scan_points = build_zone_scan_points(cur, z['id'], coord_scale, main_ssids, ssid_prefix)
        total_points += len(scan_points)

        print(f"  Zone \"{z['name']}\": {img_w}x{img_h}, scale={coord_scale:.4f}, {len(scan_points)} points")

        zone_data.append({
            "id": z['id'],
            "name": z['name'],
            "map_file": z['map_file'],
            "image": {"width": img_w, "height": img_h},
            "coord_scale": coord_scale,
            "scan_points": scan_points,
        })

    conn.close()

    data = {
        "survey": {"name": survey_name, "created": time_created, "modified": time_modified},
        "ssid_prefix": ssid_prefix,
        "main_ssids": main_ssids,
        "all_ssids": all_ssids,
        "zones": zone_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("const DATA = ")
        json.dump(data, f, indent=1)
        f.write(";\n")

    print(f"  Exported {total_points} scan points across {len(zones)} zone(s)")
    print(f"  Main SSIDs: {main_ssids}")


# ── PDF Report Generation ───────────────────────────────────────────────────

def generate_pdf_report(db_path, zones, output_path, main_ssids, ssid_prefix):
    """Generate a multi-page PDF heatmap report (per-zone)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT Name, AppVersion, TimeCreated FROM SurveyData LIMIT 1")
    survey_name, app_version, time_created = cur.fetchone()

    like_pattern = ssid_prefix + '%'

    # Global stats
    cur.execute("SELECT COUNT(*) FROM WlanNetworks")
    total_observations = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ScanPoints")
    total_points = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT BSSID) FROM WlanNetworks WHERE SSID LIKE ?", (like_pattern,))
    total_prefix_aps = cur.fetchone()[0]

    placeholders = ','.join('?' * len(main_ssids))
    cur.execute(
        f"SELECT SSID, WiFiBand, COUNT(DISTINCT BSSID) FROM WlanNetworks "
        f"WHERE SSID IN ({placeholders}) GROUP BY SSID, WiFiBand ORDER BY SSID, WiFiBand",
        main_ssids)
    ssid_ap_counts = cur.fetchall()

    cur.execute("SELECT ChannelPrimary, WiFiBand, COUNT(DISTINCT BSSID) FROM WlanNetworks "
                "WHERE SSID LIKE ? AND WiFiBand=0 GROUP BY ChannelPrimary ORDER BY ChannelPrimary",
                (like_pattern,))
    channels_24 = cur.fetchall()

    cur.execute("SELECT MIN(Time), MAX(Time) FROM ScanPoints")
    time_range = cur.fetchone()

    # Gather all best RSSI values across all zones for coverage summary
    cur.execute("SELECT _id FROM ScanPoints ORDER BY _id")
    all_sp_ids = [r[0] for r in cur.fetchall()]
    all_best_rssi = []
    for sp_id in all_sp_ids:
        cur.execute("SELECT MAX(RSSI) FROM WlanNetworks WHERE ScanPointId=? AND SSID LIKE ?",
                     (sp_id, like_pattern))
        val = cur.fetchone()[0]
        all_best_rssi.append(val if val else -100)
    all_best_rssi = np.array(all_best_rssi)

    n_excellent = int(np.sum(all_best_rssi >= -65))
    n_good = int(np.sum((all_best_rssi >= -75) & (all_best_rssi < -65)))
    n_fair = int(np.sum((all_best_rssi >= -80) & (all_best_rssi < -75)))
    n_weak = int(np.sum(all_best_rssi < -80))

    colors_hm = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
    cmap_wifi = LinearSegmentedColormap.from_list('wifi', colors_hm, N=256)

    print(f"  Writing PDF: {output_path}")
    with PdfPages(output_path) as pdf:

        # ── PAGE 1: Title / Summary ──────────────────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')
        fig.text(0.5, 0.88, 'WiFi Site Survey Report', ha='center', fontsize=26,
                 fontweight='bold', color='#1a1a2e')
        fig.text(0.5, 0.82, survey_name, ha='center', fontsize=18, color='#16213e')
        fig.text(0.5, 0.77, f'Survey Date: {time_created[:10]}  |  NetSpot v{app_version}',
                 ha='center', fontsize=11, color='#666')
        fig.text(0.5, 0.73, f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                 ha='center', fontsize=10, color='#999')

        sy = 0.62
        fig.text(0.08, sy, 'Survey Summary', fontsize=14, fontweight='bold', color='#1a1a2e')

        ssid_data = {}
        for ssid, band, count in ssid_ap_counts:
            if ssid not in ssid_data:
                ssid_data[ssid] = [0, 0]
            ssid_data[ssid][band] = count

        st = time_range[0][11:16] if time_range[0] else '?'
        et = time_range[1][11:16] if time_range[1] else '?'

        lines = [
            f'Measurement Points:  {total_points}',
            f'Zones:  {len(zones)} ({", ".join(z["name"] for z in zones)})',
            f'Total Network Observations:  {total_observations:,}',
            f'Survey Duration:  {st} - {et}',
            f'Unique BSSIDs ({ssid_prefix}*):  {total_prefix_aps}',
            '',
            'Corporate SSIDs:',
        ]
        for ssid in main_ssids:
            if ssid in ssid_data:
                d = ssid_data[ssid]
                lines.append(f'  {ssid:<20} {d[0]+d[1]:>3} APs ({d[0]} x 2.4GHz + {d[1]} x 5GHz)')
        lines += [
            '',
            'Coverage Quality (all zones):',
            f'  Excellent (>= -65 dBm):  {n_excellent} / {total_points}  ({n_excellent*100//max(total_points,1)}%)',
            f'  Good (-65 to -75 dBm):   {n_good} / {total_points}  ({n_good*100//max(total_points,1)}%)',
            f'  Fair (-75 to -80 dBm):   {n_fair} / {total_points}  ({n_fair*100//max(total_points,1)}%)',
            f'  Weak (< -80 dBm):        {n_weak} / {total_points}  ({n_weak*100//max(total_points,1)}%)',
        ]
        for i, line in enumerate(lines):
            fig.text(0.10, sy - 0.035 - i * 0.028, line, fontsize=9.5, fontfamily='monospace', color='#333')

        fig.text(0.08, 0.08, 'Signal Strength Reference:', fontsize=10, fontweight='bold', color='#1a1a2e')
        legend_items = [
            ('Excellent', '>= -65 dBm', '#1a9850'), ('Good', '-65 to -75', '#a6d96a'),
            ('Fair', '-75 to -80', '#fee08b'), ('Weak', '-80 to -85', '#f46d43'),
            ('Very Weak', '< -85 dBm', '#d73027')]
        for i, (lb, desc, col) in enumerate(legend_items):
            xp = 0.10 + i * 0.17
            fig.patches.append(plt.Rectangle((xp, 0.035), 0.02, 0.025,
                               transform=fig.transFigure, facecolor=col, edgecolor='#333', linewidth=0.5))
            fig.text(xp + 0.025, 0.043, f'{lb} ({desc})', fontsize=8, color='#333')
        pdf.savefig(fig, dpi=150); plt.close(fig)

        # ── Per-zone pages ───────────────────────────────────────────────
        for z in zones:
            coord_factor = z['scale'] * 100
            floor_img = Image.open(z['map_path'])
            img_w, img_h = floor_img.size

            # Get scan points for this zone
            cur.execute("""
                SELECT sp._id, sp.x, sp.y, sp.Time
                FROM ScanPoints sp JOIN Snapshots s ON s._id = sp.SnapId
                WHERE s.ZoneId = ? ORDER BY sp._id
            """, (z['id'],))
            zone_sps = cur.fetchall()
            zone_sp_ids = [r[0] for r in zone_sps]
            n_zone_pts = len(zone_sp_ids)

            if n_zone_pts < 3:
                floor_img.close()
                continue

            sp_x = np.array([r[1] * coord_factor for r in zone_sps])
            sp_y = np.array([r[2] * coord_factor for r in zone_sps])

            best_rssi = []
            for sp_id in zone_sp_ids:
                cur.execute("SELECT MAX(RSSI) FROM WlanNetworks WHERE ScanPointId=? AND SSID LIKE ?",
                             (sp_id, like_pattern))
                val = cur.fetchone()[0]
                best_rssi.append(val if val else -100)
            best_rssi = np.array(best_rssi)

            def get_best_rssi_zone(ssid, band=None):
                vals = []
                for sp_id in zone_sp_ids:
                    if band is not None:
                        cur.execute("SELECT MAX(RSSI) FROM WlanNetworks WHERE ScanPointId=? AND SSID=? AND WiFiBand=?",
                                     (sp_id, ssid, band))
                    else:
                        cur.execute("SELECT MAX(RSSI) FROM WlanNetworks WHERE ScanPointId=? AND SSID=?",
                                     (sp_id, ssid))
                    val = cur.fetchone()[0]
                    vals.append(val if val else -100)
                return np.array(vals)

            grid_res = 400
            xi = np.linspace(0, img_w, grid_res)
            yi = np.linspace(0, img_h, grid_res)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            def make_heatmap(ax, rssi_values, title, vmin=-95, vmax=-45):
                ax.imshow(floor_img, extent=[0, img_w, img_h, 0], aspect='equal', zorder=0)
                zi = griddata((sp_x, sp_y), rssi_values, (xi_grid, yi_grid), method='cubic')
                im = ax.imshow(zi, extent=[0, img_w, img_h, 0], origin='upper',
                               cmap=cmap_wifi, alpha=0.55, vmin=vmin, vmax=vmax, zorder=1)
                ax.scatter(sp_x, sp_y, c='white', s=10, edgecolors='black', linewidths=0.3, zorder=3)
                ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
                ax.set_xticks([]); ax.set_yticks([])
                return im

            zone_label = z['name']

            # Overall heatmap for this zone
            fig, ax = plt.subplots(1, 1, figsize=(11.69, 8.27))
            fig.patch.set_facecolor('white')
            im = make_heatmap(ax, best_rssi, f'{zone_label} - Overall Best Signal')
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label('Signal Strength (dBm)', fontsize=10)
            fig.suptitle(f'WiFi Signal Heatmap - {zone_label} ({n_zone_pts} points)',
                         fontsize=14, fontweight='bold', y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, dpi=150); plt.close(fig)

            # Per-SSID heatmaps for this zone
            display_ssids = main_ssids[:4]
            if display_ssids:
                fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
                fig.patch.set_facecolor('white')
                fig.suptitle(f'{zone_label} - Per SSID (Best Signal Any Band)',
                             fontsize=14, fontweight='bold', y=0.99)
                im = None
                for idx, ssid in enumerate(display_ssids):
                    im = make_heatmap(axes[idx // 2][idx % 2], get_best_rssi_zone(ssid), ssid)
                for idx in range(len(display_ssids), 4):
                    axes[idx // 2][idx % 2].axis('off')
                fig.tight_layout(rect=[0, 0.02, 0.92, 0.95])
                cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
                fig.colorbar(im, cax=cbar_ax).set_label('Signal Strength (dBm)', fontsize=9)
                pdf.savefig(fig, dpi=150); plt.close(fig)

            # 2.4 vs 5 GHz for this zone
            primary_ssid = main_ssids[0] if main_ssids else None
            if primary_ssid:
                fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
                fig.patch.set_facecolor('white')
                fig.suptitle(f'{zone_label} - 2.4 GHz vs 5 GHz ({primary_ssid})',
                             fontsize=14, fontweight='bold', y=0.98)
                make_heatmap(axes[0], get_best_rssi_zone(primary_ssid, band=0),
                             f'{primary_ssid} - 2.4 GHz')
                im2 = make_heatmap(axes[1], get_best_rssi_zone(primary_ssid, band=1),
                                   f'{primary_ssid} - 5 GHz')
                fig.tight_layout(rect=[0, 0.02, 0.92, 0.94])
                cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
                fig.colorbar(im2, cax=cbar_ax).set_label('Signal Strength (dBm)', fontsize=9)
                pdf.savefig(fig, dpi=150); plt.close(fig)

            floor_img.close()

        # ── Channel Utilization (global) ─────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        fig.patch.set_facecolor('white')
        fig.suptitle(f'Channel Utilization - {ssid_prefix}* Networks',
                     fontsize=14, fontweight='bold', y=0.98)

        if channels_24:
            ch_l = [str(r[0]) for r in channels_24]
            ch_c = [r[2] for r in channels_24]
            axes[0].bar(ch_l, ch_c,
                        color=['#4fc3f7' if c != max(ch_c) else '#ef5350' for c in ch_c],
                        edgecolor='#333', linewidth=0.5)
            axes[0].set_title('2.4 GHz', fontsize=11, fontweight='bold')
            axes[0].set_xlabel('Channel'); axes[0].set_ylabel('APs')
            for i, v in enumerate(ch_c):
                axes[0].text(i, v + 0.5, str(v), ha='center', fontsize=10, fontweight='bold')

        cur.execute("SELECT ChannelPrimary, COUNT(DISTINCT BSSID) FROM WlanNetworks "
                    "WHERE SSID LIKE ? AND WiFiBand=1 "
                    "GROUP BY ChannelPrimary ORDER BY ChannelPrimary", (like_pattern,))
        ch5 = cur.fetchall()
        if ch5:
            axes[1].bar([str(r[0]) for r in ch5], [r[1] for r in ch5],
                        color='#81c784', edgecolor='#333', linewidth=0.5)
            axes[1].set_title('5 GHz', fontsize=11, fontweight='bold')
            axes[1].set_xlabel('Channel'); axes[1].set_ylabel('APs')
            axes[1].tick_params(axis='x', rotation=45)
            for i, v in enumerate([r[1] for r in ch5]):
                axes[1].text(i, v + 0.3, str(v), ha='center', fontsize=7, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig, dpi=150); plt.close(fig)

        # ── Signal tables (global) ───────────────────────────────────────
        table_ssids = main_ssids[:4]
        hdr = ['Pt', 'Zone', 'Time'] + [s.replace(ssid_prefix, '') for s in table_ssids] + ['Best', 'Rating']

        # Build zone name lookup
        cur.execute("""
            SELECT sp._id, z.Name FROM ScanPoints sp
            JOIN Snapshots s ON s._id = sp.SnapId
            JOIN Zones z ON z._id = s.ZoneId
        """)
        sp_zone_name = dict(cur.fetchall())

        cur.execute("SELECT _id, Time FROM ScanPoints ORDER BY _id")
        all_sps = cur.fetchall()
        tbl = []
        for sp_id, tm in all_sps:
            zone_name = sp_zone_name.get(sp_id, '?')
            row = [str(sp_id), zone_name[:8], tm[11:19]]
            bv = -100
            for ssid in table_ssids:
                cur.execute("SELECT MAX(RSSI) FROM WlanNetworks WHERE ScanPointId=? AND SSID=?",
                             (sp_id, ssid))
                v = cur.fetchone()[0]
                if v:
                    row.append(f'{v:.1f}'); bv = max(bv, v)
                else:
                    row.append('-')
            row.append(f'{bv:.1f}')
            if bv >= -65: row.append('Excellent')
            elif bv >= -75: row.append('Good')
            elif bv >= -80: row.append('Fair')
            else: row.append('Weak')
            tbl.append(row)

        def mk_tbl(data, label):
            f = plt.figure(figsize=(11.69, 8.27)); f.patch.set_facecolor('white')
            f.suptitle(f'Signal Per Point ({label})', fontsize=14, fontweight='bold', y=0.98)
            a = f.add_subplot(111); a.axis('off')
            cc = []
            for r in data:
                rc = ['white'] * len(r)
                if r[-1] == 'Excellent': rc[-1] = rc[-2] = '#c8e6c9'
                elif r[-1] == 'Good': rc[-1] = rc[-2] = '#fff9c4'
                elif r[-1] == 'Fair': rc[-1] = rc[-2] = '#ffe0b2'
                else: rc[-1] = rc[-2] = '#ffcdd2'
                cc.append(rc)
            tb = a.table(cellText=data, colLabels=hdr, loc='center', cellLoc='center', cellColours=cc)
            tb.auto_set_font_size(False); tb.set_fontsize(7)
            tb.scale(1, min(0.85 / max(len(data), 1) * 40, 1.0))
            for j in range(len(hdr)):
                tb[0, j].set_facecolor('#1a1a2e')
                tb[0, j].set_text_props(color='white', fontweight='bold')
            f.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
            return f

        n_all = len(tbl)
        mid = (n_all + 1) // 2
        pdf.savefig(mk_tbl(tbl[:mid], f'Pts 1-{mid}'), dpi=150); plt.close()
        if tbl[mid:]:
            pdf.savefig(mk_tbl(tbl[mid:], f'Pts {mid+1}-{n_all}'), dpi=150); plt.close()

    conn.close()
    print(f"  PDF report: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate WiFi survey report and interactive website from a .netspu file.')
    parser.add_argument('netspu', help='Path to .netspu survey file')
    parser.add_argument('--output', '-o', default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--ssid-prefix',
                        help='SSID prefix to filter on (auto-detected if omitted)')
    args = parser.parse_args()

    netspu_path = os.path.abspath(args.netspu)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(netspu_path):
        print(f"Error: file not found: {netspu_path}", file=sys.stderr)
        sys.exit(1)

    survey_basename = Path(netspu_path).stem
    safe_name = survey_basename.replace(' ', '_')

    print(f"=== WiFi Survey: {survey_basename} ===")

    # Step 1: Extract .netspu
    extract_dir = os.path.join(output_dir, '_extracted')
    print(f"\n[1/5] Extracting {Path(netspu_path).name}...")
    db_path = extract_netspu(netspu_path, extract_dir)
    zones = get_zones(db_path, extract_dir)
    print(f"  Found {len(zones)} zone(s): {', '.join(z['name'] for z in zones)}")

    # Step 2: Detect SSID prefix
    if args.ssid_prefix:
        ssid_prefix = args.ssid_prefix
        print(f"\n[2/5] Using specified SSID prefix: {ssid_prefix}")
    else:
        print("\n[2/5] Auto-detecting SSID prefix...")
        ssid_prefix = detect_ssid_prefix(db_path)
        if not ssid_prefix:
            print("  Warning: could not detect SSID prefix, using all SSIDs.", file=sys.stderr)
            ssid_prefix = ""
        else:
            print(f"  Detected prefix: {ssid_prefix}")

    main_ssids = detect_main_ssids(db_path, ssid_prefix)
    print(f"  Main SSIDs: {main_ssids}")

    # Step 3: Generate website data
    website_dir = os.path.join(output_dir, 'website')
    os.makedirs(website_dir, exist_ok=True)
    print(f"\n[3/5] Generating data.js...")
    generate_data_js(db_path, zones, os.path.join(website_dir, 'data.js'),
                     main_ssids, ssid_prefix)

    # Step 4: Copy website assets
    print(f"\n[4/5] Copying website assets...")
    for z in zones:
        shutil.copy2(z['map_path'], os.path.join(website_dir, z['map_file']))
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'website')
    for fname in ['index.html', 'leaflet.js', 'leaflet.css']:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(website_dir, fname))
    print(f"  Website ready: {website_dir}")

    # Step 5: Generate PDF report
    pdf_path = os.path.join(output_dir, f'WiFi_Report_{safe_name}.pdf')
    print(f"\n[5/5] Generating PDF report...")
    generate_pdf_report(db_path, zones, pdf_path, main_ssids, ssid_prefix)

    # Cleanup
    shutil.rmtree(extract_dir)

    print(f"\n=== Done! ===")
    print(f"  Website: {website_dir}")
    print(f"  PDF:     {pdf_path}")
    print(f"\n  To view: open {os.path.join(website_dir, 'index.html')}")


if __name__ == '__main__':
    main()
