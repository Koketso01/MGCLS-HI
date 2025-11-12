
from __future__ import annotations

import io
import re
import csv
import zipfile
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

# --- Optional boto3 (not required for these endpoints) ---
try:
    import boto3
    from botocore.config import Config
    from botocore import UNSIGNED
except Exception:
    boto3 = None
    Config = None  # type: ignore
    UNSIGNED = None  # type: ignore

# Use stdlib HTTP so we don't depend on "requests"
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ------------------------ S3 CONFIG ------------------------
S3_BUCKET  = "ratt-public-data"
S3_REGION  = "af-south-1"
S3_PREFIX  = "MGCLS_HI/Datasets/koketso-HI-MGCLS-data/HI-MGCLS/"

S3_CLUSTER_CATALOGUE_KEY = S3_PREFIX + "Cluster_catalogue/MGCLS HI.txt"
S3_CATALOGUES_PREFIX     = S3_PREFIX + "Catalogues/"
S3_CLUSTER_FIGURES_PREFIX= S3_PREFIX + "Cluster-Figures/"
S3_GALAXY_FIGURES_PREFIX = S3_PREFIX + "Galaxy-Figures/"
S3_CLUSTER_CUBES_PREFIX  = S3_PREFIX + "Cluster-Cubes/"
S3_CLUSTER_MASKS_PREFIX  = S3_PREFIX + "Cluster-Masks/"
S3_CLUSTER_MOMS_PREFIX   = S3_PREFIX + "Cluster-Moms/"
S3_GALAXY_CUBELETS_PREFIX= S3_PREFIX + "Galaxy-Cubelets/"

C_LIGHT = 299792.458  # km/s

# ------------------------ SMALL HELPERS ------------------------
def _strip_quotes(s: str | None) -> str:
    if s is None:
        return ""
    return str(s).strip().strip('"').strip("'")

def _have_boto() -> bool:
    return boto3 is not None and Config is not None

def s3_http_url(bucket: str, region: str, key: str) -> str:
    key_q = quote(key, safe="/+-_.")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key_q}"

def _http_get_text(url: str, timeout: int = 20) -> str:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data.decode("utf-8", errors="replace")

def _http_get_bytes(url: str, timeout: int = 60) -> bytes:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()

def _http_head_ok(url: str, timeout: int = 10) -> bool:
    # HEAD exists since Py3.3; falls back to GET if server rejects HEAD
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except HTTPError as e:
        if e.code == 405:
            try:
                req = Request(url, method="GET")
                with urlopen(req, timeout=timeout) as resp:
                    return 200 <= resp.status < 300
            except Exception:
                return False
        return False
    except Exception:
        return False

def _s3_get_text(bucket: str, region: str, key: str) -> str:
    url = s3_http_url(bucket, region, key)
    return _http_get_text(url)

def _s3_get_bytes(bucket: str, region: str, key: str) -> Optional[bytes]:
    url = s3_http_url(bucket, region, key)
    try:
        return _http_get_bytes(url)
    except Exception:
        return None

def _s3_exists(bucket: str, region: str, key: str) -> bool:
    url = s3_http_url(bucket, region, key)
    return _http_head_ok(url)

def _first_existing(keys: List[str]) -> Optional[str]:
    for k in keys:
        if _s3_exists(S3_BUCKET, S3_REGION, k):
            return k
    return None

# ------------------------ MASTER CATALOGUE ------------------------
# --- utils.py: robust parsing for master cluster catalogue ---


import io
import pandas as pd

# --- Master catalogue robust parsing (drop this into utils.py) ---
import io
import pandas as pd

# utils.py — robust master-catalogue parsing
import io, csv
import pandas as pd

def _clean_catalogue_text(txt: str) -> str:
    #Remove comments and blank lines; normalise newlines.
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    kept = []
    for ln in txt.split("\n"):
        if not ln.strip():
            continue
        if ln.lstrip().startswith("#"):
            # ignore comment lines like the SoFiA header block you printed
            continue
        kept.append(ln)
    return "\n".join(kept)

def _strip_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
    drop_cols = [c for c in df.columns if not c or str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def _parse_master_catalogue_text(txt: str) -> pd.DataFrame:
    
    #Robustly parse the MGCLS-HI master cluster catalogue.
    #Supports '|' tables (optionally with a second units row) or plain whitespace tables.
    
    clean = _clean_catalogue_text(txt)
    if not clean.strip():
        raise ValueError("Master catalogue file is empty after removing comments/blank lines")

    lines = clean.split("\n")
    sample = lines[:10]
    # Heuristic: consider 'pipe' if multiple early lines have >=2 pipes and not just separators
    pipe_like = [ln for ln in sample if ln.count("|") >= 2 and set(ln) - set("|- ")]

    has_pipe = len(pipe_like) >= max(2, len(sample) // 3)

    if has_pipe:
        # Single-char sep → quoting works
        df = pd.read_csv(
            io.StringIO(clean),
            sep="|",
            engine="python",
            dtype=str,
            skip_blank_lines=True,
            na_filter=False,
            skipinitialspace=True,
            quoting=csv.QUOTE_MINIMAL,
        )
        df = _strip_df(df)
    else:
        # Whitespace table, but **not** regex with multi-char sep; use r'\s+' safely
        df = pd.read_csv(
            io.StringIO(clean),
            sep=r"\s+",
            engine="python",
            dtype=str,
            skip_blank_lines=True,
            na_filter=False,
            quoting=csv.QUOTE_MINIMAL,
        )
        df = _strip_df(df)

    # Drop a possible units row
    if len(df) >= 1:
        unit_tokens = {"km/s", "jy/beam", "arcsec", "deg", "—", "hh:mm:ss", "dd:mm:ss", "mhz"}
        first_vals = [str(v).strip().lower() for v in df.iloc[0].tolist()]
        if any(tok in first_vals for tok in unit_tokens):
            df = df.iloc[1:].reset_index(drop=True)

    # Canonicalise columns used elsewhere
    rename_map = {
        "name": "Name", "NAME": "Name", "Cluster": "Name", "ID": "Name",
        "ra": "RA", "RA": "RA", "Right Ascension (RA, J2000)": "RA",
        "dec": "DEC", "Dec": "DEC", "DEC": "DEC", "Declination (DEC, J2000)": "DEC",
        "M_Z": "M_Z", "z": "M_Z", "redshift": "M_Z",
        "SBID": "SBID",
        "Capture ID": "CAPTURE_ID", "capture_id": "CAPTURE_ID", "CAPTURE_ID": "CAPTURE_ID",
        "RMS": "RMS",
        "V_min": "V_min", "Vmin": "V_min",
        "V_max": "V_max", "Vmax": "V_max",
        "BMIN": "BMIN", "BMAJ": "BMAJ", "BPA": "BPA",
        "MGCLS_Name": "MGCLS_Name", "MGCLS": "MGCLS_Name",
    }
    present = set(df.columns)
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in present})
    return df

def _read_master_catalogue_text(txt: str) -> pd.DataFrame:
    df = _parse_master_catalogue_text(txt)

    needed = ["Name", "RA", "DEC", "M_Z", "SBID", "CAPTURE_ID",
              "RMS", "V_min", "V_max", "BMIN", "BMAJ", "BPA", "MGCLS_Name"]
    for c in needed:
        if c not in df.columns:
            df[c] = "—"

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace({"": "—"}).dropna(how="all").reset_index(drop=True)

    if "ID" not in df.columns:
        if "Name" in df.columns:
            df["ID"] = df["Name"]
        elif "MGCLS_Name" in df.columns:
            df["ID"] = df["MGCLS_Name"]
        else:
            df["ID"] = [f"Cluster-{i+1}" for i in range(len(df))]
    return df


def _read_text_from_path(path_or_s3: str) -> str:
    if path_or_s3.startswith("s3://"):
        m = re.match(r"^s3://([^/]+)/(.+)$", path_or_s3)
        if not m:
            raise ValueError(f"Bad S3 URL: {path_or_s3}")
        bkt, key = m.group(1), m.group(2)
        return _s3_get_text(bkt, S3_REGION, key)
    else:
        with open(path_or_s3, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

@lru_cache(maxsize=2)
def load_master_catalogue() -> pd.DataFrame:
    
    #Load the master MGCLS HI cluster catalogue from S3 and tidy field names.
    
    txt = _s3_get_text(S3_BUCKET, S3_REGION, S3_CLUSTER_CATALOGUE_KEY)
    df = _read_master_catalogue_text(txt)
    # Harmonise a couple of expected columns (ID, RA, DEC, v_sys if present)
    # We keep everything as string here; numeric casting is per-use.
    if "ID" not in df.columns and "Name" in df.columns:
        df["ID"] = df["Name"]
    return df

# ------------------------ LANDING TABLE ------------------------
def build_landing_table_spec(df: pd.DataFrame, page: int = 1, page_size: int = 25, quick_filter: str = "") -> Dict[str, Any]:
    if df is None or df.empty:
        return {"header_labels": [], "header_units": [], "rows": [], "pagination": {"page": 1, "page_size": 0, "total": 0, "num_pages": 1}}

    sub = df.copy()
    q = (quick_filter or "").strip().lower()
    if q:
        # A simple filter: contains in ID or MGCLS_Name if present.
        mask = pd.Series(False, index=sub.index)
        for col in [c for c in sub.columns if c.lower() in ["id", "mgcls_name"]]:
            mask |= sub[col].astype(str).str.lower().str.contains(q, na=False)
        sub = sub[mask]

    total = len(sub)
    num_pages = max(1, int(np.ceil(total / page_size)))
    page = min(max(1, page), num_pages)
    sub = sub.iloc[(page - 1) * page_size : page * page_size]

    # Choose a set of default columns (keep your existing ones if present)
    cols = [c for c in ["ID", "M_Z", "RA", "DEC", "SBID", "CAPTURE_ID", "MGCLS_Name", "R200", "SIGMA_V"] if c in df.columns]
    rows = sub[cols].fillna("").astype(str).values.tolist()
    units = ["", "", "deg", "deg", "", "", "", "Mpc", "km/s"][: len(cols)]

    return {
        "header_labels": cols,
        "header_units": units,
        "rows": rows,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "num_pages": num_pages,
        }
    }

# ------------------------ COMMON ASTRO HELPERS ------------------------
def _norm_angle_to_deg(x: str) -> float:
    x = (x or "").strip()
    if not x:
        raise ValueError("blank angle")
    if ":" in x:
        # sexagesimal
        fields = [float(f) for f in x.split(":")]
        if len(fields) != 3:
            raise ValueError("need hh:mm:ss or dd:mm:ss")
        sign = -1.0 if x.strip().startswith("-") else 1.0
        fields = [abs(f) for f in fields]
        if " " in x:
            # be forgiving
            pass
        # Heuristic: treat as RA if first field > 0..24
        deg = sign * (abs(fields[0]) + fields[1] / 60.0 + fields[2] / 3600.0)
        if deg <= 24.0:  # likely RA hours
            return deg * 15.0
        return deg
    # decimal
    return float(x)

def parse_radius_text_to_deg(rad_text: str) -> float:
    txt = (rad_text or "").strip()
    if not txt:
        raise ValueError("blank radius")
    parts = txt.split()
    if len(parts) == 1:
        # default arcmin if unit omitted
        val = float(parts[0])
        return val / 60.0
    val = float(parts[0])
    unit = parts[1].lower()
    if unit.startswith("deg"):
        return val
    if unit.startswith("arcmin"):
        return val / 60.0
    if unit.startswith("arcsec"):
        return val / 3600.0
    # fallback: assume deg
    return val

def _cluster_name_variants(name: str) -> List[str]:
    
    #Generate reasonable filename variants for a cluster:
    #Abell-194 → ["Abell-194", "abell-194", "Abell_194", "abell_194", "Abell194", "abell194"]
    
    base = _strip_quotes(name)
    variants = set()
    v = base
    variants.add(v)
    variants.add(v.replace(" ", "-"))
    variants.add(v.replace(" ", "_"))
    variants.add(v.replace("-", "_"))
    variants.add(v.replace("_", "-"))
    variants.add(v.replace("-", "").replace("_", "").replace(" ", ""))
    low = v.lower()
    variants.add(low)
    variants.add(low.replace(" ", "-"))
    variants.add(low.replace(" ", "_"))
    variants.add(low.replace("-", "_"))
    variants.add(low.replace("_", "-"))
    variants.add(low.replace("-", "").replace("_", "").replace(" ", ""))
    return list(variants)

# ------------------------ CLUSTER SEARCH ------------------------
def _cluster_preview_urls(cluster_name: str) -> Dict[str, Optional[str]]:
    
    #Build preview figure URLs for a cluster: <cluster>_Intensity_SNR.png
    
    variants = _cluster_name_variants(cluster_name)
    for cand in variants:
        key = f"{S3_CLUSTER_FIGURES_PREFIX}{cand}_Intensity_SNR.png"
        if _s3_exists(S3_BUCKET, S3_REGION, key):
            return {"preview": s3_http_url(S3_BUCKET, S3_REGION, key)}
    return {"preview": None}

def search_clusters(
    master_df: pd.DataFrame,
    name_query: str = "",
    mgcls_query: str = "",
    sbid: str = "",
    capture_id: str = "",
    ra_txt: str = "",
    dec_txt: str = "",
    radius_text: str = "",
    vel_center: str = "",
    vel_tol: str = "",
    vel_min: str = "",
    vel_max: str = "",
    page: int = 1,
    page_size: int = 25,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], List[str]]:
    errors: List[str] = []
    warns:  List[str] = []

    df = master_df.copy()
    if df is None or df.empty:
        return {"header_labels": [], "header_units": [], "rows": [], "pagination": {}}, [], ["Empty master catalogue"], []

    # Simple filters
    if name_query:
        df = df[df["ID"].astype(str).str.contains(name_query, case=False, na=False)]
    if mgcls_query and "MGCLS_Name" in df.columns:
        df = df[df["MGCLS_Name"].astype(str).str.contains(mgcls_query, case=False, na=False)]
    if sbid and "SBID" in df.columns:
        df = df[df["SBID"].astype(str).str.contains(sbid, case=False, na=False)]
    if capture_id and "CAPTURE_ID" in df.columns:
        df = df[df["CAPTURE_ID"].astype(str).str.contains(capture_id, case=False, na=False)]

    # Spatial cone filter if all present
    if ra_txt and dec_txt and radius_text:
        try:
            ra_deg = _norm_angle_to_deg(ra_txt)
            dec_deg = _norm_angle_to_deg(dec_txt)
            rad_deg = parse_radius_text_to_deg(radius_text)

            ra_col  = pd.to_numeric(df.get("RA"),  errors="coerce")
            dec_col = pd.to_numeric(df.get("DEC"), errors="coerce")
            dra  = (ra_col - ra_deg) * np.cos(np.deg2rad(dec_deg))
            ddec = (dec_col - dec_deg)
            sep = np.sqrt(dra*dra + ddec*ddec)
            df = df[sep <= rad_deg]
        except Exception:
            warns.append("Could not parse spatial filter; ignoring.")

    # Velocity window: we try best-effort if M_Z (z) or SIGMA_V present, but keep simple
    # (left as-is for now to preserve your existing behaviour)

    total = len(df)
    num_pages = max(1, int(np.ceil(total / page_size)))
    page = min(max(1, page), num_pages)
    df = df.iloc[(page - 1) * page_size : page * page_size]

    cols = [c for c in ["ID", "M_Z", "RA", "DEC", "SBID", "CAPTURE_ID", "MGCLS_Name", "R200", "SIGMA_V"] if c in master_df.columns]
    units = ["", "", "deg", "deg", "", "", "", "Mpc", "km/s"][: len(cols)]
    rows = df[cols].fillna("").astype(str).values.tolist()

    # Previews
    previews = []
    for _, r in df.iterrows():
        p = _cluster_preview_urls(r["ID"])
        if p.get("preview"):
            previews.append({"cluster": r["ID"], "url": p["preview"]})

    spec = {
        "header_labels": cols + [],  # keep as-is
        "header_units": units + [],
        "rows": rows,
        "pagination": {
            "page": page, "page_size": page_size, "total": total, "num_pages": num_pages
        },
    }
    if df.empty:
        warns.append("No clusters matched your filters.")

    return spec, previews, [], warns  # (errors empty by design)
# :contentReference[oaicite:4]{index=4}

# ------------------------ SoFiA ASCII (galaxies) ------------------------
def _parse_sofia_ascii_text(text: str) -> pd.DataFrame:
    
    #Parse SoFiA 'plain text' catalogue export with the 3-line header
    #(numbers / names / units). Split on 2+ spaces to keep quoted "name".
    
    lines = text.splitlines()
    idx_num = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*1\s+2\s+3\s+4\s+', ln):
            idx_num = i
            break
    if idx_num is None:
        raise ValueError("SoFiA ASCII parse: could not find header number row")

    names_line = lines[idx_num + 1]
    data_lines = lines[idx_num + 4:]

    raw_names = re.split(r'\s{2,}', names_line.strip())
    col_names = []
    seen: Dict[str, int] = {}
    for n in raw_names:
        n = 'ellipsis' if n.strip() == '...' else n.strip()
        c = seen.get(n, 0) + 1
        seen[n] = c
        col_names.append(n if c == 1 else f"{n}_{c}")

    block = "\n".join([ln for ln in data_lines if ln.strip() != ""])
    df = pd.read_csv(
        io.StringIO(block),
        sep=r"\s{2,}",
        engine="python",
        names=col_names,
        quoting=csv.QUOTE_NONE
    )
    return df
# :contentReference[oaicite:5]{index=5}

def load_sofia_catalogue(cluster_name: str, base: Optional[str] = None) -> pd.DataFrame:
    
    #Return tidy DF with: name (str), id (int), ra, dec (deg), v_rad (m/s),
    #f_sum, rms, w50, dv_mps (m/s) when discoverable.
    #Tries common <cluster> variants: spaces/underscore/hyphen/no-space/lower.
    
    cluster_clean = _strip_quotes(cluster_name)

    candidates: List[str] = []
    if base:
        candidates.append(base)
    else:
        for cand in _cluster_name_variants(cluster_clean):
            candidates.append(f"s3://{S3_BUCKET}/{S3_CATALOGUES_PREFIX}{cand}_cat.txt")

    last_err = None
    df = None
    for path in candidates:
        try:
            txt = _read_text_from_path(path)
            df  = _parse_sofia_ascii_text(txt)
            break
        except Exception as e:
            last_err = e
            continue
    if df is None:
        raise RuntimeError(f'Could not load SoFiA catalogue for "{cluster_clean}": {last_err}')

    # Normalise essential columns if present
    # Expected columns include at least: id, name, ra, dec, v_rad, f_sum, rms, dv_mps (optional)
    for col_try, canonical in [
        ("id", "id"),
        ("name", "name"),
        ("ra", "ra"),
        ("dec", "dec"),
        ("v_rad", "v_rad"),
        ("f_sum", "f_sum"),
        ("rms", "rms"),
        ("dv_mps", "dv_mps"),
    ]:
        if col_try in df.columns and canonical not in df.columns:
            df[canonical] = df[col_try]

    # Cast some numeric helpers
    with pd.option_context("mode.chained_assignment", None):
        if "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce")
        if "ra" in df.columns:
            df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
        if "dec" in df.columns:
            df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
        if "v_rad" in df.columns:
            df["v_rad"] = pd.to_numeric(df["v_rad"], errors="coerce")
        if "f_sum" in df.columns:
            df["f_sum"] = pd.to_numeric(df["f_sum"], errors="coerce")

    return df

# ------------------------ GALAXY SEARCH ------------------------
def _galaxy_preview_urls(cluster: str, gid: int | str, name: str | None) -> Dict[str, Optional[str]]:
    
    #Resolve the three preview images for a detection, with fallbacks:
    #mom0: prefer *_mom0_sdss.png, else *_mom0_panstarrs.png, else *_mom0_wisew1.png
    #mom1: *_mom1.png
    #mom2: *_mom2.png
    
    cluster_v = _cluster_name_variants(cluster)
    gid_s = str(int(gid)) if str(gid).strip().isdigit() else str(gid)
    # build candidate keys
    mom0_candidates = []
    for c in cluster_v:
        base = f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_{gid_s}_mom0"
        mom0_candidates += [base + "_sdss.png", base + "_panstarrs.png", base + "_wisew1.png"]
    mom1_candidates = [f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_{gid_s}_mom1.png" for c in cluster_v]
    mom2_candidates = [f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_{gid_s}_mom2.png" for c in cluster_v]

    mom0_key = _first_existing(mom0_candidates)
    mom1_key = _first_existing(mom1_candidates)
    mom2_key = _first_existing(mom2_candidates)

    return {
        "mom0": s3_http_url(S3_BUCKET, S3_REGION, mom0_key) if mom0_key else None,
        "mom1": s3_http_url(S3_BUCKET, S3_REGION, mom1_key) if mom1_key else None,
        "mom2": s3_http_url(S3_BUCKET, S3_REGION, mom2_key) if mom2_key else None,
    }

def search_galaxies(
    master_df: pd.DataFrame,
    name_query: str = "",
    ra_txt: str = "",
    dec_txt: str = "",
    radius_text: str = "",
    vel_center_kms: str = "",
    vel_tol_kms: str = "",
    cluster_scope: Optional[List[str]] = None,
    page: int = 1,
    page_size: int = 25,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], List[str]]:
    errors: List[str] = []
    warns:  List[str] = []

    # Clean up cluster scope (strip quotes/spaces)
    all_clusters = sorted(master_df["ID"].dropna().astype(str).map(_strip_quotes).unique().tolist()) if master_df is not None and "ID" in master_df.columns else []
    clusters = [ _strip_quotes(c) for c in cluster_scope ] if (cluster_scope and len(cluster_scope) > 0) else all_clusters

    ra_deg  = _norm_angle_to_deg(ra_txt) if ra_txt else None
    dec_deg = _norm_angle_to_deg(dec_txt) if dec_txt else None
    rad_deg = parse_radius_text_to_deg(radius_text) if radius_text else None

    v0 = dv = None
    if vel_center_kms:
        try:
            v0 = float(vel_center_kms)
            dv = float(vel_tol_kms or "0")
        except Exception:
            warns.append("Galaxy velocity center/tolerance not numeric; ignoring velocity filter.")
            v0 = dv = None

    name_sub = (name_query or "").strip().lower()

    rows: List[List[str]] = []
    row_meta: List[Dict[str, str]] = []
    previews: List[Dict[str, Any]] = []

    for clus in clusters:
        try:
            cat = load_sofia_catalogue(clus)
        except Exception as e:
            warns.append(f'"{clus}": failed to load catalogue ({e})')
            continue

        if cat is None or cat.empty:
            continue

        cat["v_kms"] = pd.to_numeric(cat.get("v_rad"), errors="coerce") / 1000.0

        mask = pd.Series(True, index=cat.index)

        if name_sub:
            mask &= cat["name"].astype(str).str.lower().str.contains(name_sub, na=False)

        if ra_deg is not None and dec_deg is not None and rad_deg is not None:
            ra_col  = pd.to_numeric(cat.get("ra"),  errors="coerce")
            dec_col = pd.to_numeric(cat.get("dec"), errors="coerce")
            dra  = (ra_col - ra_deg) * np.cos(np.deg2rad(dec_deg))
            ddec = (dec_col - dec_deg)
            sep = np.sqrt(dra*dra + ddec*ddec)
            mask &= (sep <= rad_deg)

        if v0 is not None and dv is not None:
            mask &= (cat["v_kms"] >= (v0 - dv)) & (cat["v_kms"] <= (v0 + dv))

        sub = cat[mask].copy()
        if sub.empty:
            continue

        # Build table rows
        # Choose a concise set of columns for the table
        for _, r in sub.iterrows():
            gid = r.get("id")
            gname = r.get("name")
            ra   = r.get("ra")
            dec  = r.get("dec")
            vkms = r.get("v_kms")
            fsum = r.get("f_sum")

            rows.append([clus, str(gname) if pd.notna(gname) else "", f"{gid:.0f}" if pd.notna(gid) else "", f"{ra:.6f}" if pd.notna(ra) else "", f"{dec:.6f}" if pd.notna(dec) else "", f"{vkms:.1f}" if pd.notna(vkms) else "", f"{fsum:.6g}" if pd.notna(fsum) else ""])
            row_meta.append({"cluster": str(clus), "id": f"{int(gid)}" if pd.notna(gid) else "", "name": str(gname) if pd.notna(gname) else ""})

            # previews
            if pd.notna(gid):
                prev = _galaxy_preview_urls(clus, gid, gname)
                if any([prev.get("mom0"), prev.get("mom1"), prev.get("mom2")]):
                    previews.append({"cluster": clus, "id": int(gid), "name": gname, "mom0": prev.get("mom0"), "mom1": prev.get("mom1"), "mom2": prev.get("mom2")})

    # Paginate AFTER collecting across clusters
    total = len(rows)
    num_pages = max(1, int(np.ceil(total / page_size)))
    page = min(max(1, page), num_pages)
    start = (page - 1) * page_size
    end   = page * page_size

    rows_page = rows[start:end]
    meta_page = row_meta[start:end]

    spec = {
        "header_labels": ["Cluster", "Galaxy Name", "ID", "RA", "Dec", "v (km/s)", "f_sum"],
        "header_units":  ["", "", "", "deg", "deg", "km/s", ""],
        "rows": rows_page,
        "row_meta": meta_page,
        "pagination": {"page": page, "page_size": page_size, "total": total, "num_pages": num_pages},
    }

    if total == 0:
        warns.append("No galaxies matched your filters.")

    return spec, previews, errors, warns
# :contentReference[oaicite:6]{index=6}

# ------------------------ DOWNLOAD MANIFESTS ------------------------
def _cluster_package_pairs(cluster: str) -> List[Tuple[str, str]]:
    
    #Build a list of (arcname, s3key) for cluster-level products:
    #  - Catalogues/<cluster>_cat.txt
    #  - Cluster-Cubes/<cluster>.fits
    #  - Cluster-Masks/<cluster>_mask.fits
    #  - Cluster-Moms/<cluster>_mom0/1/2.fits
    #NO figures, NO noise, NO *_chan for clusters (per spec).
    
    pairs: List[Tuple[str, str]] = []
    top = f"{_strip_quotes(cluster)}/cluster/"

    variants = _cluster_name_variants(cluster)
    # catalogue
    cat_keys = [f"{S3_CATALOGUES_PREFIX}{v}_cat.txt" for v in variants]
    k = _first_existing(cat_keys)
    if k: pairs.append((top + k.split("/")[-1], k))

    # cube
    cube_keys = [f"{S3_CLUSTER_CUBES_PREFIX}{v}.fits" for v in variants]
    k = _first_existing(cube_keys)
    if k: pairs.append((top + k.split("/")[-1], k))

    # mask
    mask_keys = [f"{S3_CLUSTER_MASKS_PREFIX}{v}_mask.fits" for v in variants]
    k = _first_existing(mask_keys)
    if k: pairs.append((top + k.split("/")[-1], k))

    # moms
    for mi in (0, 1, 2):
        keys = [f"{S3_CLUSTER_MOMS_PREFIX}{v}_mom{mi}.fits" for v in variants]
        k = _first_existing(keys)
        if k: pairs.append((top + k.split("/")[-1], k))

    return pairs

def _galaxy_package_pairs(cluster: str, gid: str | int) -> List[Tuple[str, str]]:
    
    #Build a list of (arcname, s3key) for the galaxy's cubelets/products:
    #Include *_chan.fits here (galaxies only), plus cube/mask/mom0/1/2, pv, snr, spec.txt (if they exist).
    #All live under Galaxy-Cubelets/<cluster>/ with filename prefix <cluster>_<gid>_*
    
    pairs: List[Tuple[str, str]] = []
    top = f"{_strip_quotes(cluster)}/cubelets/"

    gid_s = str(int(gid)) if str(gid).strip().isdigit() else str(gid)
    variants = _cluster_name_variants(cluster)

    # All candidate product suffixes (no figures)
    suffixes = [
        "_chan.fits",  # allowed for galaxies
        "_cube.fits",
        "_mask.fits",
        "_mom0.fits",
        "_mom1.fits",
        "_mom2.fits",
        "_pv.fits",
        "_snr.fits",
        "_spec.txt",
    ]

    for v in variants:
        base_dir = f"{S3_GALAXY_CUBELETS_PREFIX}{v}/"
        for suf in suffixes:
            key = f"{base_dir}{v}_{gid_s}{suf}"
            if _s3_exists(S3_BUCKET, S3_REGION, key):
                pairs.append((top + key.split("/")[-1], key))

    # Deduplicate (same file could match across variants)
    seen = set()
    unique_pairs = []
    for arc, key in pairs:
        if key not in seen:
            seen.add(key)
            unique_pairs.append((arc, key))
    return unique_pairs

def build_zip_from_pairs(pairs: List[Tuple[str, str]], title: str) -> io.BytesIO:
    
    #Fetch each S3 key (if reachable) and write to a ZIP in-memory.
    #This is suitable for the typical per-galaxy/per-cluster payload sizes.
    
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, key in pairs:
            data = _s3_get_bytes(S3_BUCKET, S3_REGION, key)
            if data is not None:
                zf.writestr(arcname, data)
    buf.seek(0)
    return buf

# ------------------------ PREVIEW HELPERS (used by search.html) ------------------------
def cluster_preview_for_template(cluster_name: str) -> Optional[str]:
    return _cluster_preview_urls(cluster_name).get("preview")

def galaxy_previews_for_template(cluster: str, gid: str | int, name: str | None) -> Dict[str, Optional[str]]:
    return _galaxy_preview_urls(cluster, gid, name)



"""

from __future__ import annotations

import io
import os
import re
import csv
import zipfile
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd

# --- Optional boto3 (we will use unsigned HTTP; boto3 is not required) ---
try:
    import boto3
    from botocore.config import Config
    from botocore import UNSIGNED
except Exception:
    boto3 = None
    Config = None  # type: ignore
    UNSIGNED = None  # type: ignore

# Use stdlib HTTP so we don't depend on "requests"
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# ------------------------ S3 CONFIG ------------------------
S3_BUCKET  = "ratt-public-data"
S3_REGION  = "af-south-1"
S3_PREFIX  = "MGCLS_HI/Datasets/koketso-HI-MGCLS-data/HI-MGCLS/"

S3_CLUSTER_CATALOGUE_KEY = S3_PREFIX + "Cluster_catalogue/MGCLS HI.txt"
S3_CATALOGUES_PREFIX     = S3_PREFIX + "Catalogues/"
S3_CLUSTER_FIGURES_PREFIX= S3_PREFIX + "Cluster-Figures/"
S3_GALAXY_FIGURES_PREFIX = S3_PREFIX + "Galaxy-Figures/"
S3_CLUSTER_CUBES_PREFIX  = S3_PREFIX + "Cluster-Cubes/"
S3_CLUSTER_MASKS_PREFIX  = S3_PREFIX + "Cluster-Masks/"
S3_CLUSTER_MOMS_PREFIX   = S3_PREFIX + "Cluster-Moms/"
S3_GALAXY_CUBELETS_PREFIX= S3_PREFIX + "Galaxy-Cubelets/"

C_LIGHT = 299792.458  # km/s

# ------------------------ SMALL HELPERS ------------------------
def _strip_quotes(s: str | None) -> str:
    if s is None:
        return ""
    return str(s).strip().strip('"').strip("'")

def s3_http_url(bucket: str, region: str, key: str) -> str:
    key_q = quote(key, safe="/+-_.")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key_q}"

def _http_get_text(url: str, timeout: int = 20) -> str:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data.decode("utf-8", errors="replace")

def _http_get_bytes(url: str, timeout: int = 60) -> Optional[bytes]:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()

def _http_head_ok(url: str, timeout: int = 10) -> bool:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except HTTPError as e:
        if e.code == 405:
            # Some S3 endpoints disallow HEAD; fallback to GET
            try:
                req = Request(url, method="GET")
                with urlopen(req, timeout=timeout) as resp:
                    return 200 <= resp.status < 300
            except Exception:
                return False
        return False
    except Exception:
        return False

def _s3_get_text(bucket: str, region: str, key: str) -> str:
    return _http_get_text(s3_http_url(bucket, region, key))

def _s3_get_bytes(bucket: str, region: str, key: str) -> Optional[bytes]:
    try:
        return _http_get_bytes(s3_http_url(bucket, region, key))
    except Exception:
        return None

def _s3_exists(bucket: str, region: str, key: str) -> bool:
    return _http_head_ok(s3_http_url(bucket, region, key))

def _first_existing(keys: List[str]) -> Optional[str]:
    for k in keys:
        if _s3_exists(S3_BUCKET, S3_REGION, k):
            return k
    return None

# ------------------------ MASTER CATALOGUE ------------------------
def _parse_master_catalogue_text(txt: str) -> pd.DataFrame:
    
    #Robustly parse the master MGCLS HI cluster catalogue text.

    #Handles both:
    #  • pipe-separated tables (with an optional second 'units' row),
    #  • whitespace-separated tables.

    #Also:
    #  • strips comment lines starting with '#',
    #  • avoids regex separators that can break quotes,
    #  • trims blank/unnamed columns,
    #  • drops a first data row that looks like "units".
    
    # Normalise newlines and strip empty/comment lines
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = txt.split("\n")
    lines = []
    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):  # drop comment blocks (your file has many)
            continue
        lines.append(ln)

    if not lines:
        raise ValueError("Master catalogue file is empty or only comments.")

    has_pipe = any("|" in ln for ln in lines[:10])

    def _strip_df(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
        # drop blank/unnamed columns
        drop_cols = [c for c in df.columns if c == "" or c.lower().startswith("unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df

    # Try preferred parse
    try:
        if has_pipe:
            # Single-character sep keeps proper quote handling
            df = pd.read_csv(
                io.StringIO("\n".join(lines)),
                sep="\\|",
                engine="python",
                header=0,
                skip_blank_lines=True,
                skipinitialspace=True,
                dtype=str,
                na_filter=False,
            )
            df = _strip_df(df)
        else:
            df = pd.read_csv(
                io.StringIO("\n".join(lines)),
                delim_whitespace=True,     # OK here because comments already removed
                engine="python",
                header=0,
                skip_blank_lines=True,
                dtype=str,
                na_filter=False,
            )
            df = _strip_df(df)
    except Exception:
        # Fallback: manual split for pipe tables
        if has_pipe:
            rows = []
            for ln in lines:
                rows.append([p.strip() for p in ln.split("|")])
            maxcols = max(len(r) for r in rows) if rows else 0
            rows = [r + [""] * (maxcols - len(r)) for r in rows]
            header = [h.strip() or f"col{i}" for i, h in enumerate(rows[0])]
            df = pd.DataFrame(rows[1:], columns=header)
            df = _strip_df(df)
        else:
            raise

    # Detect and drop a 'units' row if the first data row looks like units
    if len(df) >= 1:
        unit_tokens = {"km/s", "jy/beam", "arcsec", "deg", "—", "hh:mm:ss", "dd:mm:ss", "mhz"}
        first_vals = [str(v).strip().lower() for v in df.iloc[0].tolist()]
        if any(tok in first_vals for tok in unit_tokens):
            df = df.iloc[1:].reset_index(drop=True)

    # Normalize to canonical names used by the app
    rename_map = {
        "name": "Name", "NAME": "Name", "Cluster": "Name", "ID": "Name",
        "ra": "RA", "RA": "RA", "Right Ascension (RA, J2000)": "RA",
        "dec": "DEC", "Dec": "DEC", "DEC": "DEC", "Declination (DEC, J2000)": "DEC",
        "M_Z": "M_Z", "z": "M_Z", "redshift": "M_Z",
        "SBID": "SBID",
        "Capture ID": "CAPTURE_ID", "capture_id": "CAPTURE_ID", "CAPTURE_ID": "CAPTURE_ID",
        "RMS": "RMS", "rms": "RMS",
        "V_min": "V_min", "Vmin": "V_min",
        "V_max": "V_max", "Vmax": "V_max",
        "BMIN": "BMIN", "BMAJ": "BMAJ", "BPA": "BPA",
        "MGCLS_Name": "MGCLS_Name", "MGCLS": "MGCLS_Name",
    }
    cols_before = set(df.columns)
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in cols_before})
    return df

def _read_master_catalogue_text(txt: str) -> pd.DataFrame:
    df = _parse_master_catalogue_text(txt)

    # Ensure canonical columns exist (fill missing with "—")
    needed = [
        "Name", "RA", "DEC", "M_Z", "SBID", "CAPTURE_ID",
        "RMS", "V_min", "V_max", "BMIN", "BMAJ", "BPA", "MGCLS_Name",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = "—"

    # Add an ID column used widely in your UI (mirror Name)
    if "ID" not in df.columns and "Name" in df.columns:
        df["ID"] = df["Name"]

    # trim + cleanup
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace({"": "—"})
    df = df.dropna(how="all").reset_index(drop=True)
    return df

@lru_cache(maxsize=2)
def load_master_catalogue() -> pd.DataFrame:
    txt = _s3_get_text(S3_BUCKET, S3_REGION, S3_CLUSTER_CATALOGUE_KEY)
    return _read_master_catalogue_text(txt)

# ------------------------ Landing Table ------------------------
def build_landing_table_spec(df: pd.DataFrame, page: int = 1, page_size: int = 25, quick_filter: str = "") -> Dict[str, Any]:
    if df is None or df.empty:
        return {"header_labels": [], "header_units": [], "rows": [], "pagination": {"page": 1, "page_size": 0, "total": 0, "num_pages": 1}}

    sub = df.copy()
    q = (quick_filter or "").strip().lower()
    if q:
        mask = pd.Series(False, index=sub.index)
        for col in [c for c in sub.columns if c.lower() in ["id", "mgcls_name"]]:
            mask |= sub[col].astype(str).str.lower().str.contains(q, na=False)
        sub = sub[mask]

    total = len(sub)
    num_pages = max(1, int(np.ceil(total / page_size)))
    page = min(max(1, page), num_pages)
    sub = sub.iloc[(page - 1) * page_size : page * page_size]

    cols = [c for c in ["ID", "M_Z", "RA", "DEC", "SBID", "CAPTURE_ID", "MGCLS_Name", "R200", "SIGMA_V"] if c in df.columns]
    rows = sub[cols].fillna("").astype(str).values.tolist()
    units = ["", "", "deg", "deg", "", "", "", "Mpc", "km/s"][: len(cols)]

    return {
        "header_labels": cols,
        "header_units": units,
        "rows": rows,
        "pagination": {"page": page, "page_size": page_size, "total": total, "num_pages": num_pages},
    }

# ------------------------ Angle/Radius Helpers ------------------------
def _norm_angle_to_deg(x: str) -> float:
    x = (x or "").strip()
    if not x:
        raise ValueError("blank angle")
    if ":" in x:
        # sexagesimal
        fields = [float(f) for f in x.split(":")]
        if len(fields) != 3:
            raise ValueError("need hh:mm:ss or dd:mm:ss")
        sign = -1.0 if x.strip().startswith("-") else 1.0
        fields = [abs(f) for f in fields]
        deg = sign * (abs(fields[0]) + fields[1] / 60.0 + fields[2] / 3600.0)
        # heuristic: RA hours if <= 24
        if deg <= 24.0:
            return deg * 15.0
        return deg
    return float(x)

def parse_radius_text_to_deg(rad_text: str) -> float:
    txt = (rad_text or "").strip()
    if not txt:
        raise ValueError("blank radius")
    parts = txt.split()
    if len(parts) == 1:
        return float(parts[0]) / 60.0  # default arcmin
    val = float(parts[0])
    unit = parts[1].lower()
    if unit.startswith("deg"):
        return val
    if unit.startswith("arcmin"):
        return val / 60.0
    if unit.startswith("arcsec"):
        return val / 3600.0
    return val

def _cluster_name_variants(name: str) -> List[str]:
    
    #Generate filename variants for a cluster:
    #Abell-194 → ["Abell-194", "abell-194", "Abell_194", "abell_194", "Abell194", "abell194"]
    
    base = _strip_quotes(name)
    variants = set()
    v = base
    variants.add(v)
    variants.add(v.replace(" ", "-"))
    variants.add(v.replace(" ", "_"))
    variants.add(v.replace("-", "_"))
    variants.add(v.replace("_", "-"))
    variants.add(v.replace("-", "").replace("_", "").replace(" ", ""))
    low = v.lower()
    variants.add(low)
    variants.add(low.replace(" ", "-"))
    variants.add(low.replace(" ", "_"))
    variants.add(low.replace("-", "_"))
    variants.add(low.replace("_", "-"))
    variants.add(low.replace("-", "").replace("_", "").replace(" ", ""))
    return list(variants)

# ------------------------ Cluster search + previews ------------------------
def _cluster_preview_urls(cluster_name: str) -> Dict[str, Optional[str]]:
    for cand in _cluster_name_variants(cluster_name):
        key = f"{S3_CLUSTER_FIGURES_PREFIX}{cand}_Intensity_SNR.png"
        if _s3_exists(S3_BUCKET, S3_REGION, key):
            return {"preview": s3_http_url(S3_BUCKET, S3_REGION, key)}
    return {"preview": None}

def search_clusters(
    master_df: pd.DataFrame,
    name_query: str = "",
    mgcls_query: str = "",
    sbid: str = "",
    capture_id: str = "",
    ra_txt: str = "",
    dec_txt: str = "",
    radius_text: str = "",
    vel_center: str = "",
    vel_tol: str = "",
    vel_min: str = "",
    vel_max: str = "",
    page: int = 1,
    page_size: int = 25,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], List[str]]:
    warns: List[str] = []
    df = master_df.copy()
    if df is None or df.empty:
        return {"header_labels": [], "header_units": [], "rows": [], "pagination": {}}, [], ["Empty master catalogue"], []

    if name_query:
        df = df[df["ID"].astype(str).str.contains(name_query, case=False, na=False)]
    if mgcls_query and "MGCLS_Name" in df.columns:
        df = df[df["MGCLS_Name"].astype(str).str.contains(mgcls_query, case=False, na=False)]
    if sbid and "SBID" in df.columns:
        df = df[df["SBID"].astype(str).str.contains(sbid, case=False, na=False)]
    if capture_id and "CAPTURE_ID" in df.columns:
        df = df[df["CAPTURE_ID"].astype(str).str.contains(capture_id, case=False, na=False)]

    if ra_txt and dec_txt and radius_text:
        try:
            ra_deg = _norm_angle_to_deg(ra_txt)
            dec_deg = _norm_angle_to_deg(dec_txt)
            rad_deg = parse_radius_text_to_deg(radius_text)
            ra_col  = pd.to_numeric(df.get("RA"),  errors="coerce")
            dec_col = pd.to_numeric(df.get("DEC"), errors="coerce")
            dra  = (ra_col - ra_deg) * np.cos(np.deg2rad(dec_deg))
            ddec = (dec_col - dec_deg)
            sep = np.sqrt(dra*dra + ddec*ddec)
            df = df[sep <= rad_deg]
        except Exception:
            warns.append("Could not parse spatial filter; ignoring.")

    total = len(df)
    num_pages = max(1, int(np.ceil(total / page_size)))
    page = min(max(1, page), num_pages)
    df = df.iloc[(page - 1) * page_size : page * page_size]

    cols = [c for c in ["ID", "M_Z", "RA", "DEC", "SBID", "CAPTURE_ID", "MGCLS_Name", "R200", "SIGMA_V"] if c in master_df.columns]
    units = ["", "", "deg", "deg", "", "", "", "Mpc", "km/s"][: len(cols)]
    rows = df[cols].fillna("").astype(str).values.tolist()

    previews = []
    for _, r in df.iterrows():
        p = _cluster_preview_urls(r["ID"])
        if p.get("preview"):
            previews.append({"cluster": r["ID"], "url": p["preview"]})

    spec = {
        "header_labels": cols,
        "header_units": units,
        "rows": rows,
        "pagination": {"page": page, "page_size": page_size, "total": total, "num_pages": num_pages},
    }
    if df.empty:
        warns.append("No clusters matched your filters.")
    return spec, previews, [], warns

# ------------------------ SoFiA ASCII (galaxies) ------------------------
def _parse_sofia_ascii_text(text: str) -> pd.DataFrame:
    
    #Parse SoFiA 'plain text' export with the 3-line header (numbers / names / units).
    #We split on 2+ spaces to keep quoted 'name' intact.
    
    lines = text.splitlines()
    idx_num = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*1\s+2\s+3\s+4\s+', ln):
            idx_num = i
            break
    if idx_num is None:
        raise ValueError("SoFiA ASCII parse: could not find header number row")

    names_line = lines[idx_num + 1]
    data_lines = lines[idx_num + 4:]

    raw_names = re.split(r'\s{2,}', names_line.strip())
    col_names = []
    seen: Dict[str, int] = {}
    for n in raw_names:
        n = 'ellipsis' if n.strip() == '...' else n.strip()
        c = seen.get(n, 0) + 1
        seen[n] = c
        col_names.append(n if c == 1 else f"{n}_{c}")

    block = "\n".join([ln for ln in data_lines if ln.strip() != ""])
    df = pd.read_csv(
        io.StringIO(block),
        sep=r"\s{2,}",
        engine="python",
        names=col_names,
        quoting=csv.QUOTE_NONE,
    )
    return df

def load_sofia_catalogue(cluster_name: str, base: Optional[str] = None) -> pd.DataFrame:
    
    #Return a tidy DF with: name (str), id (int), ra, dec (deg), v_rad (m/s),
    #f_sum, rms, w50, dv_mps (m/s) when discoverable.
    #Tries common <cluster> variants (case/dash/underscore).
    
    cluster_clean = _strip_quotes(cluster_name)
    candidates: List[str] = []
    if base:
        candidates.append(base)
    else:
        for cand in _cluster_name_variants(cluster_clean):
            candidates.append(f"s3://{S3_BUCKET}/{S3_CATALOGUES_PREFIX}{cand}_cat.txt")

    last_err = None
    df = None
    for path in candidates:
        try:
            txt = _s3_get_text(S3_BUCKET, S3_REGION, path.split(f"s3://{S3_BUCKET}/", 1)[1])
            df  = _parse_sofia_ascii_text(txt)
            break
        except Exception as e:
            last_err = e
            continue
    if df is None:
        raise RuntimeError(f'Could not load SoFiA catalogue for "{cluster_clean}": {last_err}')

    # Normalize essential columns if present
    for col_try, canonical in [
        ("id", "id"),
        ("name", "name"),
        ("ra", "ra"),
        ("dec", "dec"),
        ("v_rad", "v_rad"),
        ("f_sum", "f_sum"),
        ("rms", "rms"),
        ("dv_mps", "dv_mps"),
    ]:
        if col_try in df.columns and canonical not in df.columns:
            df[canonical] = df[col_try]

    # Numeric casts
    with pd.option_context("mode.chained_assignment", None):
        if "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce")
        if "ra" in df.columns:
            df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
        if "dec" in df.columns:
            df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
        if "v_rad" in df.columns:
            df["v_rad"] = pd.to_numeric(df["v_rad"], errors="coerce")
        if "f_sum" in df.columns:
            df["f_sum"] = pd.to_numeric(df["f_sum"], errors="coerce")

    return df

# ------------------------ Galaxy search + previews ------------------------
def _galaxy_preview_urls(cluster: str, gid: int | str, name: str | None) -> Dict[str, Optional[str]]:
    cluster_v = _cluster_name_variants(cluster)
    gid_s = str(int(gid)) if str(gid).strip().isdigit() else str(gid)
    # mom0: prefer *_sdss.png, then *_panstarrs.png, then *_wisew1.png
    mom0_candidates = []
    for c in cluster_v:
        base = f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_{gid_s}_mom0"
        mom0_candidates += [base + "_sdss.png", base + "_panstarrs.png", base + "_wisew1.png"]
    mom1_candidates = [f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_{gid_s}_mom1.png" for c in cluster_v]
    mom2_candidates = [f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_figures/{c}_{gid_s}_mom2.png".replace(f"{c}_figures/{c}_figures", f"{c}_figures") for c in cluster_v]

    mom0_key = _first_existing(mom0_candidates)
    mom1_key = _first_existing(mom1_candidates)
    mom2_key = _first_existing(mom2_candidates)

    return {
        "mom0": s3_http_url(S3_BUCKET, S3_REGION, mom0_key) if mom0_key else None,
        "mom1": s3_http_url(S3_BUCKET, S3_REGION, mom1_key) if mom1_key else None,
        "mom2": s3_http_url(S3_BUCKET, S3_REGION, mom2_key) if mom2_key else None,
    }

def search_galaxies(
    master_df: pd.DataFrame,
    name_query: str = "",
    ra_txt: str = "",
    dec_txt: str = "",
    radius_text: str = "",
    vel_center_kms: str = "",
    vel_tol_kms: str = "",
    cluster_scope: Optional[List[str]] = None,
    page: int = 1,
    page_size: int = 25,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], List[str]]:
    warns: List[str] = []

    all_clusters = sorted(master_df["ID"].dropna().astype(str).map(_strip_quotes).unique().tolist()) if master_df is not None and "ID" in master_df.columns else []
    clusters = [ _strip_quotes(c) for c in cluster_scope ] if (cluster_scope and len(cluster_scope) > 0) else all_clusters

    ra_deg  = _norm_angle_to_deg(ra_txt) if ra_txt else None
    dec_deg = _norm_angle_to_deg(dec_txt) if dec_txt else None
    rad_deg = parse_radius_text_to_deg(radius_text) if radius_text else None

    v0 = dv = None
    if vel_center_kms:
        try:
            v0 = float(vel_center_kms); dv = float(vel_tol_kms or "0")
        except Exception:
            warns.append("Galaxy velocity center/tolerance not numeric; ignoring velocity filter.")
            v0 = dv = None

    name_sub = (name_query or "").strip().lower()

    rows: List[List[str]] = []
    row_meta: List[Dict[str, str]] = []
    previews: List[Dict[str, Any]] = []

    for clus in clusters:
        try:
            cat = load_sofia_catalogue(clus)
        except Exception as e:
            warns.append(f'"{clus}": failed to load catalogue ({e})')
            continue
        if cat is None or cat.empty:
            continue

        cat["v_kms"] = pd.to_numeric(cat.get("v_rad"), errors="coerce") / 1000.0

        mask = pd.Series(True, index=cat.index)
        if name_sub:
            mask &= cat["name"].astype(str).str.lower().str.contains(name_sub, na=False)

        if ra_deg is not None and dec_deg is not None and rad_deg is not None:
            ra_col  = pd.to_numeric(cat.get("ra"),  errors="coerce")
            dec_col = pd.to_numeric(cat.get("dec"), errors="coerce")
            dra  = (ra_col - ra_deg) * np.cos(np.deg2rad(dec_deg))
            ddec = (dec_col - dec_deg)
            sep = np.sqrt(dra*dra + ddec*ddec)
            mask &= (sep <= rad_deg)

        if v0 is not None and dv is not None:
            mask &= (cat["v_kms"] >= (v0 - dv)) & (cat["v_kms"] <= (v0 + dv))

        sub = cat[mask].copy()
        if sub.empty:
            continue

        for _, r in sub.iterrows():
            gid = r.get("id")
            gname = r.get("name")
            ra   = r.get("ra")
            dec  = r.get("dec")
            vkms = r.get("v_kms")
            fsum = r.get("f_sum")

            rows.append([
                clus,
                str(gname) if pd.notna(gname) else "",
                f"{gid:.0f}" if pd.notna(gid) else "",
                f"{ra:.6f}" if pd.notna(ra) else "",
                f"{dec:.6f}" if pd.notna(dec) else "",
                f"{vkms:.1f}" if pd.notna(vkms) else "",
                f"{fsum:.6g}" if pd.notna(fsum) else "",
            ])
            row_meta.append({
                "cluster": str(clus),
                "id": f"{int(gid)}" if pd.notna(gid) else "",
                "name": str(gname) if pd.notna(gname) else "",
            })

            # figure previews
            if pd.notna(gid):
                prev = _galaxy_preview_urls(clus, gid, gname)
                if any([prev.get("mom0"), prev.get("mom1"), prev.get("mom2")]):
                    previews.append({
                        "cluster": clus,
                        "id": int(gid),
                        "name": gname,
                        "mom0": prev.get("mom0"),
                        "mom1": prev.get("mom1"),
                        "mom2": prev.get("mom2"),
                    })

    # Pagination AFTER collecting across clusters
    total = len(rows)
    num_pages = max(1, int(np.ceil(total / page_size)))
    page = min(max(1, page), num_pages)
    start = (page - 1) * page_size
    end   = page * page_size

    rows_page = rows[start:end]
    meta_page = row_meta[start:end]

    spec = {
        "header_labels": ["Cluster", "Galaxy Name", "ID", "RA", "Dec", "v (km/s)", "f_sum"],
        "header_units":  ["", "", "", "deg", "deg", "km/s", ""],
        "rows": rows_page,
        "row_meta": meta_page,
        "pagination": {"page": page, "page_size": page_size, "total": total, "num_pages": num_pages},
    }
    if total == 0:
        warns.append("No galaxies matched your filters.")
    return spec, previews, [], warns

# ------------------------ Download manifests + ZIP builder ------------------------
def _cluster_package_pairs(cluster: str) -> List[Tuple[str, str]]:
    
    #Cluster-level products (no figures, no *_chan):
    #  - Catalogues/<cluster>_cat.txt
    #  - Cluster-Cubes/<cluster>.fits
    #  - Cluster-Masks/<cluster>_mask.fits
    #  - Cluster-Moms/<cluster>_mom0/1/2.fits
    
    pairs: List[Tuple[str, str]] = []
    top = f"{_strip_quotes(cluster)}/cluster/"

    variants = _cluster_name_variants(cluster)

    # catalogue
    k = _first_existing([f"{S3_CATALOGUES_PREFIX}{v}_cat.txt" for v in variants])
    if k: pairs.append((top + os.path.basename(k), k))

    # cube
    k = _first_existing([f"{S3_CLUSTER_CUBES_PREFIX}{v}.fits" for v in variants])
    if k: pairs.append((top + os.path.basename(k), k))

    # mask
    k = _first_existing([f"{S3_CLUSTER_MASKS_PREFIX}{v}_mask.fits" for v in variants])
    if k: pairs.append((top + os.path.basename(k), k))

    # mom0/1/2
    for mi in (0, 1, 2):
        k = _first_existing([f"{S3_CLUSTER_MOMS_PREFIX}{v}_mom{mi}.fits" for v in variants])
        if k: pairs.append((top + os.path.basename(k), k))

    return pairs

def _galaxy_package_pairs(cluster: str, gid: str | int) -> List[Tuple[str, str]]:
    
    #Galaxy-level products (may include *_chan.fits):
    #  Galaxy-Cubelets/<cluster>/<cluster>_<gid>_<type>.<ext>
    #Types: chan, cube, mask, mom0, mom1, mom2, pv, snr, spec.txt
    
    pairs: List[Tuple[str, str]] = []
    top = f"{_strip_quotes(cluster)}/cubelets/"

    gid_s = str(int(gid)) if str(gid).strip().isdigit() else str(gid)
    variants = _cluster_name_variants(cluster)

    suffixes = [
        "_chan.fits",
        "_cube.fits",
        "_mask.fits",
        "_mom0.fits",
        "_mom1.fits",
        "_mom2.fits",
        "_pv.fits",
        "_snr.fits",
        "_spec.txt",
    ]

    for v in variants:
        base_dir = f"{S3_GALAXY_CUBELETS_PREFIX}{v}/"
        for suf in suffixes:
            key = f"{base_dir}{v}_{gid_s}{suf}"
            if _s3_exists(S3_BUCKET, S3_REGION, key):
                pairs.append((top + os.path.basename(key), key))

    # Deduplicate across variant matches
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for arc, key in pairs:
        if key not in seen:
            seen.add(key)
            uniq.append((arc, key))
    return uniq

def build_zip_from_pairs(pairs: List[Tuple[str, str]], title: str) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, key in pairs:
            data = _s3_get_bytes(S3_BUCKET, S3_REGION, key)
            if data is not None:
                zf.writestr(arcname, data)
    buf.seek(0)
    return buf

# ------------------------ Preview helpers (template) ------------------------
def cluster_preview_for_template(cluster_name: str) -> Optional[str]:
    return _cluster_preview_urls(cluster_name).get("preview")

def galaxy_previews_for_template(cluster: str, gid: str | int, name: str | None) -> Dict[str, Optional[str]]:
    return _galaxy_preview_urls(cluster, gid, name)




"""