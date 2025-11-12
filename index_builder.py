#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate S3-hosted indices for client-side search:
  - Indices/clusters.json.gz   (ALL MGCLS rows + AWS availability flags)
  - Indices/galaxies/<cluster>.json.gz  (one per staged cluster)

Run:
  python index_builder.py \
    --aws-profile default \
    --region af-south-1 \
    --bucket ratt-public-data \
    --root MGCLS_HI/Datasets/koketso-HI-MGCLS-data/HI-MGCLS/ \
    --mgcls-hi-local "./MGCLS HI.txt"  # or omit to read from S3 key "MGCLS HI.txt"

Requires: boto3, pandas, numpy
"""

import argparse, io, json, gzip, sys, re, math, time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import boto3
except ImportError:
    print("Please `pip install boto3 pandas numpy` in your environment.", file=sys.stderr)
    raise

# ---------- Config dataclass ----------

@dataclass
class S3Layout:
    bucket: str
    root: str  # must end with '/'
    # Subpaths under root:
    key_mgcls_hi: str = "MGCLS HI.txt"
    pfx_indices: str = "Indices/"
    pfx_indices_galaxies: str = "Indices/galaxies/"
    pfx_catalogues: str = "Catalogues/"
    pfx_cluster_cubes: str = "Cluster-Cubes/"
    pfx_cubelets: str = "Galaxy-Cubelets/"
    pfx_cluster_figs: str = "Cluster-Figures/"
    pfx_gal_figs: str = "Galaxy-Figures/"

    def abs_key(self, rel: str) -> str:
        return self.root + rel

# ---------- Helpers: RA/Dec parsing + sexagesimal ----------

def hms_to_deg(h: float, m: float, s: float) -> float:
    return 15.0 * (abs(h) + m/60.0 + s/3600.0) * (1.0 if h >= 0 else -1.0)

def dms_to_deg(d: float, m: float, s: float) -> float:
    sign = -1.0 if str(d).strip().startswith('-') else 1.0
    return sign * (abs(d) + m/60.0 + s/3600.0)

def sexa_to_deg_ra(text: str) -> Optional[float]:
    t = text.strip()
    if not t:
        return None
    # Accept "hh:mm:ss.s" or "hh mm ss.s" or "hhhmhss.s" forms
    parts = re.split(r'[:\s]+', t.replace('h', ':').replace('m', ':').replace('s', '').strip())
    try:
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return hms_to_deg(h, m, s)
    except Exception:
        try:
            # Try degrees already
            return float(t)
        except Exception:
            return None

def sexa_to_deg_dec(text: str) -> Optional[float]:
    t = text.strip()
    if not t:
        return None
    parts = re.split(r'[:\s]+', t.replace('d', ':').replace('m', ':').replace('s', '').strip())
    try:
        d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return dms_to_deg(d, m, s)
    except Exception:
        try:
            return float(t)
        except Exception:
            return None

def deg_to_hms(ra_deg: float) -> str:
    ra = ra_deg / 15.0
    h = int(ra)
    m = int((ra - h) * 60)
    s = (ra - h - m / 60.0) * 3600.0
    return f"{h:02d}:{m:02d}:{s:04.1f}"

def deg_to_dms(dec_deg: float) -> str:
    sign = '-' if dec_deg < 0 else '+'
    dabs = abs(dec_deg)
    d = int(dabs)
    m = int((dabs - d) * 60)
    s = (dabs - d - m / 60.0) * 3600.0
    return f"{sign}{d:02d}:{m:02d}:{s:04.1f}"

# ---------- S3 I/O ----------

def get_s3_clients(profile: Optional[str], region: str):
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region)
    else:
        session = boto3.Session(region_name=region)
    return session.client("s3"), session.resource("s3")

def s3_key_exists(s3c, bucket: str, key: str) -> bool:
    try:
        s3c.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def s3_prefix_exists(s3c, bucket: str, prefix: str) -> bool:
    try:
        resp = s3c.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return "Contents" in resp
    except Exception:
        return False

def s3_get_text(s3c, bucket: str, key: str) -> str:
    obj = s3c.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8", errors="ignore")

def s3_put_json_gz(s3c, bucket: str, key: str, data: Any, cache_seconds: int = 86400*30):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    gz = gzip.compress(payload)
    s3c.put_object(
        Bucket=bucket,
        Key=key,
        Body=gz,
        ContentType="application/json",
        ContentEncoding="gzip",
        CacheControl=f"public, max-age={cache_seconds}",
    )

# ---------- Parsers ----------

def read_mgcls_hi_table(path_local: Optional[str], s3c, layout: S3Layout) -> pd.DataFrame:
    """
    Reads the MGCLS HI master TXT. Supports:
      - local path via --mgcls-hi-local
      - S3 key "MGCLS HI.txt" under root
    """
    if path_local:
        txt = open(path_local, "r", encoding="utf-8", errors="ignore").read()
    else:
        txt = s3_get_text(s3c, layout.bucket, layout.abs_key(layout.key_mgcls_hi))

    # Try robust pandas read: tab or multiple spaces, quoted strings, comments '#'
    buf = io.StringIO(txt)
    try:
        df = pd.read_csv(buf, sep=r"\t+", engine="python", comment="#", dtype=str)
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, delim_whitespace=True, engine="python", comment="#", dtype=str)

    # Normalize column names (strip/upper for matching, but keep original too)
    df.columns = [c.strip() for c in df.columns]
    return df

def choose(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    # fuzzy fallback
    for c in df.columns:
        lc = c.lower()
        for n in names:
            if n.lower() in lc:
                return c
    return None

def mgcls_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Column picks (tolerant to small naming differences)
    col_cluster = choose(df, ["ID", "Name", "Cluster", "CLUSTER"])
    col_mgcls = choose(df, ["MGCLS_Name", "MGCLS", "MGCLSName"])
    col_ra = choose(df, ["RA"])
    col_dec = choose(df, ["DEC", "Dec"])
    col_mz = choose(df, ["M_Z", "MZ", "Z"])
    col_date = choose(df, ["DATE"])
    col_sbid = choose(df, ["SBID"])
    col_capture = choose(df, ["CAPTURE_ID", "CAPTURE ID", "CAPTURE"])
    col_sigma = choose(df, ["SIGMA_V", "SIGMA-V", "Sigma_V"])
    col_sofia = choose(df, ["SOFIA_DETS", "SOFIA", "SOFIA_DET"])
    col_rms = choose(df, ["RMS"])
    col_vmin = choose(df, ["V_min", "Vmin"])
    col_vmax = choose(df, ["V_max", "Vmax"])
    col_bmin = choose(df, ["BMIN"])
    col_bmaj = choose(df, ["BMAJ"])
    col_bpa = choose(df, ["BPA"])
    col_counter = choose(df, ["COUNTERPART"])

    rows = []
    for _, r in df.iterrows():
        def g(c): return None if c is None else (None if pd.isna(r[c]) else str(r[c]).strip())
        cluster = g(col_cluster)
        if not cluster:
            continue
        ra = g(col_ra) or ""
        dec = g(col_dec) or ""
        ra_deg = sexa_to_deg_ra(ra)
        dec_deg = sexa_to_deg_dec(dec)
        rows.append({
            "cluster": cluster.strip('"'),
            "MGCLS_Name": (g(col_mgcls) or "").strip('"'),
            "RA": ra, "Dec": dec,
            "RA_deg": ra_deg, "Dec_deg": dec_deg,
            "M_Z": g(col_mz),
            "DATE": g(col_date),
            "SBID": g(col_sbid),
            "CAPTURE_ID": g(col_capture),
            "SIGMA_V": g(col_sigma),
            "SOFIA_DETS": g(col_sofia),
            "RMS": g(col_rms),
            "V_min": g(col_vmin),
            "V_max": g(col_vmax),
            "BMIN": g(col_bmin),
            "BMAJ": g(col_bmaj),
            "BPA": g(col_bpa),
            "COUNTERPART": g(col_counter),
        })
    return rows

def read_sofia_catalogue(s3c, layout: S3Layout, cluster: str) -> Optional[pd.DataFrame]:
    key = layout.abs_key(layout.pfx_catalogues + f"{cluster}_cat.txt")
    if not s3_key_exists(s3c, layout.bucket, key):
        return None
    txt = s3_get_text(s3c, layout.bucket, key)
    buf = io.StringIO(txt)
    # Try TSV or whitespace with comments
    try:
        df = pd.read_csv(buf, sep=r"\t+|,;", engine="python", comment="#", dtype=str)
    except Exception:
        buf.seek(0)
        df = pd.read_csv(buf, delim_whitespace=True, engine="python", comment="#", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def galaxies_index_from_cat(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Column resolver
    c_id = choose(df, ["id", "ID"])
    c_ra = choose(df, ["ra", "RA"])
    c_dec = choose(df, ["dec", "DEC"])
    c_v = choose(df, ["v_rad", "cz", "v", "VEL", "VELO"])
    c_w20 = choose(df, ["w20", "W20"])
    c_w50 = choose(df, ["w50", "W50"])
    c_fsum = choose(df, ["f_sum", "F_SUM", "S_int", "SINT", "fint"])
    c_rms = choose(df, ["rms", "RMS"])
    c_emaj = choose(df, ["ell_maj", "ellmaj", "maj"])
    c_emin = choose(df, ["ell_min", "ellmin", "min"])
    c_epa = choose(df, ["ell_pa", "ellpa", "pa"])

    rows = []
    for _, r in df.iterrows():
        def g(c): return None if c is None else (None if pd.isna(r[c]) else str(r[c]).strip())
        ra_s = g(c_ra) or ""
        dec_s = g(c_dec) or ""
        # Convert to degrees for math; keep sexagesimal for display if the input looks sexa-ish
        ra_deg = sexa_to_deg_ra(ra_s)
        dec_deg = sexa_to_deg_dec(dec_s)
        # If degrees given, render sexa for display
        ra_show = ra_s if (":" in ra_s or "h" in ra_s) and ra_deg is not None else (deg_to_hms(ra_deg) if ra_deg is not None else "")
        dec_show = dec_s if (":" in dec_s or "d" in dec_s) and dec_deg is not None else (deg_to_dms(dec_deg) if dec_deg is not None else "")

        # v_rad: prefer explicit velocity; deriving from z/freq is out of scope for index (we expect v)
        try:
            v = float(g(c_v)) if c_v else None
        except Exception:
            v = None

        # Cast numeric-friendly fields where possible; indices remain simple JSON
        def fnum(x):
            try:
                return float(x)
            except Exception:
                return None

        rows.append({
            "id": g(c_id),
            "RA": ra_show, "Dec": dec_show,
            "ra_deg": ra_deg, "dec_deg": dec_deg,
            "v_rad": v,
            "w20": fnum(g(c_w20)),
            "w50": fnum(g(c_w50)),
            "f_sum": fnum(g(c_fsum)),
            "rms": g(c_rms),
            "ell_maj": fnum(g(c_emaj)),
            "ell_min": fnum(g(c_emin)),
            "ell_pa": fnum(g(c_epa)),
        })
    return rows

# ---------- Availability probing ----------

def availability_for_cluster(s3c, layout: S3Layout, cluster: str) -> Dict[str, Any]:
    # cluster-level files
    kk = lambda rel: layout.abs_key(rel)
    exists = lambda rel: s3_key_exists(s3c, layout.bucket, kk(rel))

    cluster_cubes = {
        "main_cube": exists(layout.pfx_cluster_cubes + f"{cluster}.fits"),
        "chan":      exists(layout.pfx_cluster_cubes + f"{cluster}_chan.fits"),
        "mask":      exists(layout.pfx_cluster_cubes + f"{cluster}_mask.fits"),
        "mom0":      exists(layout.pfx_cluster_cubes + f"{cluster}_mom0.fits"),
        "mom1":      exists(layout.pfx_cluster_cubes + f"{cluster}_mom1.fits"),
        "mom2":      exists(layout.pfx_cluster_cubes + f"{cluster}_mom2.fits"),
        "noise":     exists(layout.pfx_cluster_cubes + f"{cluster}_noise.fits"),
        "rel":       exists(layout.pfx_cluster_cubes + f"{cluster}_rel.eps"),
        "skellam":   exists(layout.pfx_cluster_cubes + f"{cluster}_skellam.eps"),
    }
    catalogue = exists(layout.pfx_catalogues + f"{cluster}_cat.txt")

    # figures
    cluster_figs = {
        "completeness": exists(layout.pfx_cluster_figs + f"{cluster}_completeness.png"),
        "mom0":         exists(layout.pfx_cluster_figs + f"{cluster}_mom0.png"),
    }

    # cubelets folder presence (any object under it)
    cubelets_folder = s3_prefix_exists(s3c, layout.bucket, layout.abs_key(layout.pfx_cubelets + f"{cluster}/"))

    return {
        "catalogue": catalogue,
        "cluster_figs": cluster_figs,
        "cluster_cubes": cluster_cubes,
        "cubelets_folder": cubelets_folder,
    }

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aws-profile", default=None)
    ap.add_argument("--region", default="af-south-1")
    ap.add_argument("--bucket", default="ratt-public-data")
    ap.add_argument("--root", required=True, help="Prefix under bucket ending with '/'")
    ap.add_argument("--mgcls-hi-local", default=None, help="Optional local path to 'MGCLS HI.txt'. If omitted, read from S3.")
    args = ap.parse_args()

    s3c, _ = get_s3_clients(args.aws_profile, args.region)
    layout = S3Layout(bucket=args.bucket, root=args.root if args.root.endswith("/") else args.root + "/")

    print("Reading MGCLS HI table...")
    df = read_mgcls_hi_table(args.mgcls_hi_local, s3c, layout)
    rows = mgcls_rows(df)
    print(f"MGCLS rows parsed: {len(rows)}")

    print("Probing S3 availability per cluster (this may take a bit)...")
    for r in rows:
        r["aws"] = availability_for_cluster(s3c, layout, r["cluster"])

    # Write clusters.json.gz
    clusters_key = layout.abs_key(layout.pfx_indices + "clusters.json.gz")
    print(f"Uploading clusters index to s3://{layout.bucket}/{clusters_key}")
    s3_put_json_gz(s3c, layout.bucket, clusters_key, rows)

    # Per-cluster galaxies indices for staged clusters (catalogue present)
    staged = [r for r in rows if r["aws"].get("catalogue")]
    print(f"Staged clusters with catalogues: {len(staged)}")
    for r in staged:
        cluster = r["cluster"]
        cat_df = read_sofia_catalogue(s3c, layout, cluster)
        if cat_df is None or cat_df.empty:
            print(f" - {cluster}: catalogue missing or empty; skipping.")
            continue
        grows = galaxies_index_from_cat(cat_df)
        gkey = layout.abs_key(layout.pfx_indices_galaxies + f"{cluster}.json.gz")
        print(f"Uploading galaxies index for {cluster} → s3://{layout.bucket}/{gkey} ({len(grows)} rows)")
        s3_put_json_gz(s3c, layout.bucket, gkey, grows)

    print("Done.")

if __name__ == "__main__":
    main()