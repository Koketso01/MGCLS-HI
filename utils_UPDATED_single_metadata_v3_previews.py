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
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from pathlib import Path
# ==== Publications loader =====================================================
def load_publications(csv_path: str | None = None):
    """
    Load a publications CSV and normalize to a list of dicts with keys:
      year (int), title (str), authors (str), abstract (str),
      link (str), sample (str)
    The CSV can use flexible headings:
      Year/Published/Date, Title, Authors, Abstract, Link/URL/DOI, Sample/Objects/Cluster
    Looks for: data/publications.csv, then publications.csv, unless csv_path is given.
    """
    import pandas as pd
    # Find a CSV path
    here = Path(__file__).resolve().parent
    candidates = []
    if csv_path:
        candidates.append(Path(csv_path))
    candidates += [
        here / "data" / "publications.csv",
        here.parent / "data" / "publications.csv",
        here / "publications.csv",
        Path("data/publications.csv"),
        Path("publications.csv"),
    ]
    csv_file = next((p for p in candidates if p.exists()), None)
    if not csv_file:
        return [], ["No publications.csv found. Place one at data/publications.csv"], {"labels": [], "counts": []}
    df = pd.read_csv(csv_file)
    # Normalize column names
    lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None
    col_year     = pick("year", "published", "date")
    col_title    = pick("title")
    col_authors  = pick("authors", "author")
    col_abstract = pick("abstract", "summary")
    col_link     = pick("link", "url", "doi")
    col_sample   = pick("sample", "objects", "cluster", "galaxy", "targets")
    # Build clean records
    pubs = []
    for _, row in df.iterrows():
        year_raw = str(row[col_year]).strip() if col_year else ""
        # try to coerce year
        try:
            year = int(float(year_raw)) if year_raw not in ("", "nan", "NaN", "None") else None
        except Exception:
            year = None
        title    = (str(row[col_title]).strip() if col_title else "").replace("\n", " ")
        authors  = (str(row[col_authors]).strip() if col_authors else "")
        abstract = (str(row[col_abstract]).strip() if col_abstract else "")
        link     = (str(row[col_link]).strip() if col_link else "")
        sample   = (str(row[col_sample]).strip() if col_sample else "")
        # Normalize DOI-only links
        if link and link.lower().startswith("10."):
            link = f"https://doi.org/{link}"
        pubs.append({
            "year": year,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "link": link,
            "sample": sample,
        })
    # Bar chart data (papers per year)
    counts = {}
    for p in pubs:
        if p["year"]:
            counts[p["year"]] = counts.get(p["year"], 0) + 1
    labels = sorted(counts.keys())
    data = [counts[y] for y in labels]
    chart = {"labels": labels, "counts": data}
    return pubs, [], chart
# ------------------------ S3 CONFIG ------------------------
S3_BUCKET  = "ratt-public-data"
S3_REGION  = "af-south-1"
S3_PREFIX  = "MGCLS_HI/Datasets/koketso-HI-MGCLS-data/HI-MGCLS/"
S3_HTTP_BASE = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{S3_PREFIX.rstrip('/')}"
S3_CLUSTER_CATALOGUE_KEY_OLD  = f"{S3_PREFIX}Cluster_catalogue/MGCLS_HI_15clusters.txt"
S3_CLUSTER_CATALOGUE_KEY      = f"{S3_PREFIX}Cluster_catalogue/MGCLS_HI_15clusters.txt"
S3_CATALOGUES_PREFIX      = f"{S3_PREFIX}Catalogues/"
S3_CLUSTER_FIGURES_PREFIX = f"{S3_PREFIX}Cluster-Figures/"
S3_COMPLETENESS_CURVES_PREFIX = f"{S3_PREFIX}Completeness-Curves/"
S3_GALAXY_FIGURES_PREFIX  = f"{S3_PREFIX}Galaxy-Figures/"
S3_CLUSTER_CUBES_PREFIX   = f"{S3_PREFIX}Cluster-Cubes/"
S3_CLUSTER_MASKS_PREFIX   = f"{S3_PREFIX}Cluster-Masks/"
S3_CLUSTER_MOMS_PREFIX    = f"{S3_PREFIX}Cluster-Moms/"
S3_GALAXY_CUBELETS_PREFIX = f"{S3_PREFIX}Galaxy-Cubelets/"
C_LIGHT = 299792.458  # km/s
# ---- Combined metadata file helpers -----------------------------------------
# MGCLS_HI_Final.txt contains:
#   (1) a cluster-level master catalogue at the top; and
#   (2) per-cluster SoFiA catalogues appended below, each preceded by '#<ClusterID>'.
#@lru_cache(maxsize=1)
#def _load_combined_metadata_text() -> str:
#    """Load the single local metadata text file used by the app.
#
#    Search only for MGCLS_HI_15clusters.txt locally. No S3 fallback is used.
#    Resolution order:
#      1. MGCLS_HI_METADATA_PATH environment variable, if set
#      2. MGCLS_HI_15clusters.txt next to this utils file
#      3. MGCLS_HI_15clusters.txt in the current working directory
#    """
#    candidates = []
#
#    local_path = os.environ.get("MGCLS_HI_METADATA_PATH")
#    if local_path:
#        candidates.append(Path(local_path))
#
#    here = Path(__file__).resolve().parent
#    candidates.append(here / "MGCLS_HI_15clusters.txt")
#    candidates.append(Path.cwd() / "MGCLS_HI_15clusters.txt")
#
#    for p in candidates:
#        try:
#            if p.exists() and p.is_file():
#                return p.read_text(encoding="utf-8", errors="replace")
#        except Exception:
#            continue
#
#    searched = "\n".join(f"- {str(p)}" for p in candidates)
#    raise FileNotFoundError(
#        "Could not find MGCLS_HI_15clusters.txt locally. Checked:\n" + searched
#    )
@lru_cache(maxsize=1)
def _load_combined_metadata_text() -> str:
    """Load the single combined metadata text file from S3."""
    return _s3_get_text(S3_BUCKET, S3_REGION, S3_CLUSTER_CATALOGUE_KEY)
@lru_cache(maxsize=1)
def _split_combined_metadata() -> Tuple[str, Dict[str, str]]:
    """
    Split MGCLS_HI_Final.txt into:
      - cluster_master_text: the top master cluster table
      - sofia_blocks: dict mapping cluster id -> that cluster's SoFiA ASCII catalogue text
    """
    txt = _load_combined_metadata_text().replace("\r\n", "\n").replace("\r", "\n")
    lines = txt.split("\n")
    # Find first marker line of the form '#Abell-133' etc.
    first_marker_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("#") and len(s) > 1 and not s.startswith("# "):
            first_marker_idx = i
            break
    if first_marker_idx is None:
        # No appended SoFiA blocks found; treat whole file as master catalogue.
        return txt, {}
    master_lines = lines[:first_marker_idx]
    blocks_lines = lines[first_marker_idx:]
    sofia_blocks: Dict[str, List[str]] = {}
    current_key: Optional[str] = None
    for ln in blocks_lines:
        s = ln.strip()
        if s.startswith("#") and len(s) > 1 and not s.startswith("# "):
            current_key = s[1:].strip()
            sofia_blocks[current_key] = [ln]
            continue
        if current_key is None:
            # skip any padding/separator lines before the first marker
            continue
        sofia_blocks[current_key].append(ln)
    sofia_blocks_text = {k: "\n".join(v).strip() for k, v in sofia_blocks.items()}
    master_text = "\n".join(master_lines).strip()
    return master_text, sofia_blocks_text
# ---- Landing table (Home) display order, labels, and units ----
# Keep "ID" first so the first cell still links to the cluster detail page.
DISPLAY_ORDER = [
    "ID",           # Cluster Name (ID)
    "RA",           # RA (J2000)
    "DEC",          # DEC (J2000)
    "M_Z",          # z
    "MGCLS_Name",   # MGCLS Field Name    
    "SBID",         # SBID
    "CAPTURE_ID",   # Capture ID
    "R200",         # R200
    "M200",         # M200
    "SIGMA_V",      # σ_v
    "SOFIA_DETS",   # HI Detections
    "RMS",          # rms
    "V_min",        # v_min
    "V_max",        # v_max
    "BMIN",         # BMIN
    "BMAJ",         # BMAJ
    "BPA",          # BPA
]
# Labels you asked for. (Using Unicode for σ and subscripts so we don’t need MathJax.)
DISPLAY_LABELS = {
    "ID": "Cluster Name (ID)",
    "RA": "RA",
    "DEC": "DEC",
    "M_Z": "z",
    "MGCLS_Name": "MGCLS Field Name",    
    "SBID": "SBID",
    "CAPTURE_ID": "Capture ID",
    "R200": "R₂₀₀",
    "M200": "M₂₀₀",
    "SIGMA_V": "σᵥ",
    "SOFIA_DETS": "HI Detections",
    "RMS": "RMS",
    "V_min": "V (min)",
    "V_max": "V (max)",
    "BMIN": "BMIN",
    "BMAJ": "BMAJ",
    "BPA": "BPA",
}
# Optional units row. Keep only what you're certain about to avoid misinformation.
DISPLAY_UNITS = {
    "ID": "",
    "RA": "[HH:MM:SS.ss]",
    "DEC": "[±DD:MM:SS.ss]",
    "M_Z": "",
    "MGCLS_Name": "jHHMMSS±DDMMSS",
    "SBID": "YYYYMMDD-NNNN",
    "CAPTURE_ID": "",
    "R200": "[Mpc]",
    "M200": "[M☉]",
    "SIGMA_V": "[km/s]",
    "SOFIA_DETS": "",
    "RMS": "[mJy/beam]",
    "V_min": "[km/s]",
    "V_max": "[km/s]",
    "BMIN": "[arcsec]",
    "BMAJ": "[arcsec]",
    "BPA": "[deg]",
    # If you want: "R200": "Mpc", "M200": "M☉", "RMS": "Jy/beam" (or mJy/beam) — add once confirmed
}
# ------------------------ BASIC HTTP HELPERS ------------------------
def _strip_quotes(s: str | None) -> str:
    if s is None:
        return ""
    return str(s).strip().strip('"').strip("'")
def s3_http_url(bucket: str, region: str, key: str) -> str:
    # key should be relative to bucket (no leading slash)
    key_rel = key.lstrip("/")
    key_q = quote(key_rel, safe="/+-_.")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key_q}"
def _http_get_text(url: str, timeout: int = 25) -> str:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return data.decode("utf-8", errors="replace")
def _http_get_bytes(url: str, timeout: int = 60) -> Optional[bytes]:
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()
def _http_head_ok(url: str, timeout: int = 10) -> bool:
    """
    Some S3 endpoints may disallow HEAD; if 405 occurs, fall back to GET.
    """
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
    except URLError:
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
# ------------------------ NAME VARIANTS ------------------------
def _cluster_name_variants(name: str) -> List[str]:
    """
    Generate filename variants for a cluster:
    Abell 194 → ["Abell 194", "Abell-194", "Abell_194", "Abell194", lower-case versions...]
    """
    base = _strip_quotes(name)
    if not base:
        return []
    vs = set()
    v = base
    z = v.replace("-", "").replace("_", "").replace(" ", "")
    # As-is
    vs.add(v)
    # Space/hyphen/underscore variants
    vs.add(v.replace(" ", "-"))
    vs.add(v.replace(" ", "_"))
    vs.add(v.replace("-", "_"))
    vs.add(v.replace("_", "-"))
    vs.add(z)  # no separators
    # Lower-case
    low = v.lower()
    zlow = z.lower()
    vs.add(low)
    vs.add(low.replace(" ", "-"))
    vs.add(low.replace(" ", "_"))
    vs.add(low.replace("-", "_"))
    vs.add(low.replace("_", "-"))
    vs.add(zlow)
    return list(vs)


def _unique_preserve(items: List[str]) -> List[str]:
    """Return unique non-empty strings while preserving order."""
    out: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _search_key(value: Any) -> str:
    """Normalize text for forgiving search matching."""
    if value is None:
        return ""
    s = _strip_quotes(str(value)).casefold()
    return re.sub(r"[^0-9a-z]+", "", s)


def _split_search_terms(query: str) -> List[str]:
    """Split a query into comma-separated terms, preserving single-query behaviour."""
    q_raw = _strip_quotes(query)
    if not q_raw:
        return []
    terms = [_strip_quotes(part) for part in str(q_raw).split(",")]
    clean = [term for term in terms if term]
    return clean or [q_raw]


def _series_search_match(series: pd.Series, query: str) -> pd.Series:
    """Separator-insensitive substring match for pandas Series.

    Supports comma-separated OR queries such as
    ``Abell-548, Abell-168`` or ``20181108-0033, 20181027-0003``.
    """
    terms = _split_search_terms(query)
    if not terms:
        return pd.Series(True, index=series.index)

    s_raw = series.astype(str).fillna("")
    s_key = s_raw.map(_search_key)
    mask = pd.Series(False, index=series.index)

    for term in terms:
        term_mask = s_raw.str.contains(term, case=False, na=False, regex=False)
        term_key = _search_key(term)
        if term_key:
            term_mask = term_mask | s_key.str.contains(term_key, na=False, regex=False)
        mask = mask | term_mask.fillna(False)

    return mask.fillna(False)


def _text_search_match(text: str, query: str) -> bool:
    """Separator-insensitive substring match for plain strings.

    Supports comma-separated OR queries.
    """
    terms = _split_search_terms(query)
    if not terms:
        return True

    text_raw = "" if text is None else str(text)
    text_casefold = text_raw.casefold()
    text_key = _search_key(text_raw)

    for term in terms:
        if term.casefold() in text_casefold:
            return True
        term_key = _search_key(term)
        if term_key and term_key in text_key:
            return True
    return False

# ------------------------ MASTER CATALOGUE ------------------------
def _parse_master_catalogue_text(txt: str) -> pd.DataFrame:
    """
    Robust parse of the master MGCLS_HI cluster catalogue.
    Handles:
      • pipe-separated tables (with/without a units row),
      • whitespace-separated tables.
    Removes lines starting with '#'.
    """
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = txt.split("\n")
    lines = []
    for ln in raw_lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(ln)
    if not lines:
        raise ValueError("Master catalogue file is empty (or all comments).")
    has_pipe = any("|" in ln for ln in lines[:10])
    def _strip_df(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
        # drop blank/unnamed columns
        drop_cols = [c for c in df.columns if not c or c.lower().startswith("unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df
    try:
        if has_pipe:
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
            #df = pd.read_csv(
            #    io.StringIO("\n".join(lines)),
            #    delim_whitespace=True,
            #    engine="python",
            #    header=0,
            #    skip_blank_lines=True,
            #    dtype=str,
            #    na_filter=False,
            #)
            df = pd.read_csv(
                io.StringIO("\n".join(lines)),
                sep=r"\s+",
                engine="python",
                header=0,
                skip_blank_lines=True,
                dtype=str,
                na_filter=False,
            )            
            df = _strip_df(df)
    except Exception:
        # Fallback for pipe tables: manual split
        if has_pipe:
            rows = [[p.strip() for p in ln.split("|")] for ln in lines]
            maxcols = max((len(r) for r in rows), default=0)
            rows = [r + [""] * (maxcols - len(r)) for r in rows]
            header = [h.strip() or f"col{i}" for i, h in enumerate(rows[0])]
            df = pd.DataFrame(rows[1:], columns=header)
            df = _strip_df(df)
        else:
            raise
    # Normalize columns you display
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
        "R200": "R200", "SIGMA_V": "SIGMA_V",
    }
    present = set(df.columns)
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in present})
    # Ensure canonical columns exist
    needed = [
        "Name", "RA", "DEC", "M_Z", "SBID", "CAPTURE_ID",
        "RMS", "V_min", "V_max", "BMIN", "BMAJ", "BPA", "MGCLS_Name",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = "—"
    # Add ID (what your UI uses)
    if "ID" not in df.columns and "Name" in df.columns:
        df["ID"] = df["Name"]
    #df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace({"": "—"})
    df = df.dropna(how="all").reset_index(drop=True)
    return df
@lru_cache(maxsize=2)
def load_master_catalogue() -> pd.DataFrame:
    # Read the combined metadata file and parse only the top (cluster-level) section.
    master_txt, _ = _split_combined_metadata()
    return _parse_master_catalogue_text(master_txt)
@lru_cache(maxsize=1)
def compute_global_hi_stats() -> Dict[str, str]:
    """
    Compute global MGCLS–HI DR1-style summary stats from the master catalogue.
    Uses:
      - SOFIA_DETS      → total HI detections
      - M_Z             → redshift coverage
      - V_min, V_max    → velocity coverage [km/s]
      - FREQ_min/max    → frequency coverage [MHz]
      - RMS             → image RMS [Jy/beam]
      - NHI_min/max     → column density sensitivity [cm^-2]
      - BMIN/BMAJ       → beam FWHM [arcsec]
      - R200            → cluster radius [Mpc]
      - Mass            → cluster dynamical mass [M_sun]
    """
    df = load_master_catalogue().copy()
    def _num(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce")
    # Restrict to clusters that actually have some HI detections, if possible
    sof = _num("SOFIA_DETS")
    mask_hi = sof > 0
    if mask_hi.any():
        df_hi = df[mask_hi].copy()
        sof_hi = sof[mask_hi]
    else:
        df_hi = df
        sof_hi = sof
    # Basic counts
    num_clusters = int(df_hi["ID"].nunique()) if "ID" in df_hi.columns else int(len(df_hi))
    num_galaxies = int(sof_hi.fillna(0).sum()) if not sof_hi.empty else 0
    # Redshift
    z = _num("M_Z")
    z_min = float(z.min()) if len(z.dropna()) else float("nan")
    z_max = float(z.max()) if len(z.dropna()) else float("nan")
    # Velocity coverage [km/s]
    vmin = _num("V_min")
    vmax = _num("V_max")
    v_min = float(vmin.min()) if len(vmin.dropna()) else float("nan")
    v_max = float(vmax.max()) if len(vmax.dropna()) else float("nan")
    # Frequency coverage [MHz]
    fmin = _num("FREQ_min")
    fmax = _num("FREQ_max")
    f_min = float(fmin.min()) if len(fmin.dropna()) else float("nan")
    f_max = float(fmax.max()) if len(fmax.dropna()) else float("nan")
    # RMS [Jy/beam] → mJy/beam
    rms = _num("RMS")
    rms_mjy = rms * 1000.0
    rms_min = float(rms_mjy.min()) if len(rms_mjy.dropna()) else float("nan")
    rms_max = float(rms_mjy.max()) if len(rms_mjy.dropna()) else float("nan")
    # Column density [cm^-2]
    nhi_min = _num("NHI_min")
    nhi_max = _num("NHI_max")
    n_min = float(nhi_min.min()) if len(nhi_min.dropna()) else float("nan")
    n_max = float(nhi_max.max()) if len(nhi_max.dropna()) else float("nan")
    # Beam [arcsec]
    bmin = _num("BMIN")
    bmaj = _num("BMAJ")
    if len(bmin.dropna()) or len(bmaj.dropna()):
        beam_min = float(min(bmin.min(), bmaj.min()))
        beam_max = float(max(bmin.max(), bmaj.max()))
    else:
        beam_min = beam_max = float("nan")
    # R200 [Mpc]
    r200 = _num("R200")
    r200_min = float(r200.min()) if len(r200.dropna()) else float("nan")
    r200_max = float(r200.max()) if len(r200.dropna()) else float("nan")
    # Cluster halo mass [M_sun] (dynamical masses from MGCLS_HI.txt "Mass" column)
    mass = _num("Mass")
    mass_min = float(mass.min()) if len(mass.dropna()) else float("nan")
    mass_max = float(mass.max()) if len(mass.dropna()) else float("nan")
    def _fmt(val: float, fmt: str) -> str:
        return "—" if (val is None or np.isnan(val)) else fmt.format(val)
    def _fmt_sci_plain(val: float) -> str:
        """Return e.g. 3.4×10^19 as a plain string."""
        if val is None or np.isnan(val):
            return "—"
        mant, exp = f"{val:.1e}".split("e")
        exp_i = int(exp)
        return f"{mant}×10^{exp_i}"
    def _fmt_sci_html(val: float) -> str:
        """Return e.g. 3.4×10<sup>19</sup> for HTML."""
        if val is None or np.isnan(val):
            return "—"
        mant, exp = f"{val:.1e}".split("e")
        exp_i = int(exp)
        return f"{mant}×10<sup>{exp_i}</sup>"
    stats = {
        "num_clusters": num_clusters,
        "num_galaxies": num_galaxies,
        "z_min": _fmt(z_min, "{:.3f}"),
        "z_max": _fmt(z_max, "{:.3f}"),
        "v_min": _fmt(v_min, "{:.0f}"),
        "v_max": _fmt(v_max, "{:.0f}"),
        "f_min": _fmt(f_min, "{:.0f}"),
        "f_max": _fmt(f_max, "{:.0f}"),
        "rms_min_mjy": _fmt(rms_min, "{:.2f}"),
        "rms_max_mjy": _fmt(rms_max, "{:.2f}"),
        "nhi_min": _fmt(n_min, "{:.1e}"),
        "nhi_max": _fmt(n_max, "{:.1e}"),
        # Scientific notation (plain + HTML with <sup>)
        "nhi_min_sci": _fmt_sci_plain(n_min),
        "nhi_max_sci": _fmt_sci_plain(n_max),
        "nhi_min_html": _fmt_sci_html(n_min),
        "nhi_max_html": _fmt_sci_html(n_max),
        "beam_min": _fmt(beam_min, "{:.1f}"),
        "beam_max": _fmt(beam_max, "{:.1f}"),
        "r200_min": _fmt(r200_min, "{:.2f}"),
        "r200_max": _fmt(r200_max, "{:.2f}"),
        # Old 10^14-scaling (kept for backwards-compatibility)
        "mass_min_14": _fmt(mass_min / 1e14 if not np.isnan(mass_min) else float("nan"), "{:.1f}"),
        "mass_max_14": _fmt(mass_max / 1e14 if not np.isnan(mass_max) else float("nan"), "{:.1f}"),
        # Full M_sun scientific notation, plain and HTML
        "mass_min_sci": _fmt_sci_plain(mass_min),
        "mass_max_sci": _fmt_sci_plain(mass_max),
        "mass_min_html": _fmt_sci_html(mass_min),
        "mass_max_html": _fmt_sci_html(mass_max),
    }
    return stats
def build_landing_table_spec(
    df: pd.DataFrame, page: int = 1, page_size: int = 25, quick_filter: str = ""
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "header_labels": [],
            "header_units": [],
            "rows": [],
            "pagination": {"page": 1, "page_size": 0, "total": 0, "num_pages": 1},
        }
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
    sub = sub.iloc[(page - 1) * page_size: page * page_size]
    # Column selection in the exact order you requested, but only keep those that exist
    #cols = [c for c in DISPLAY_ORDER if c in df.columns]
    ## Pretty labels + units for thead rows
    #labels = [DISPLAY_LABELS.get(c, c) for c in cols]
    #units  = [DISPLAY_UNITS.get(c, "") for c in cols]
    ## The Home table expects rows as a list-of-lists, with the FIRST column being the cluster ID
    #sub = sub.copy()
    #rows = sub[cols].fillna("").astype(str).values.tolist()
    # Column selection in the exact order you requested, but only keep those that exist
    cols = [c for c in DISPLAY_ORDER if c in df.columns]
    # Pretty labels + units
    labels = [DISPLAY_LABELS.get(c, c) for c in cols]
    units  = [DISPLAY_UNITS.get(c, "") for c in cols]
    # Make a working copy for formatting
    sub = sub.copy()
    rows = _format_cluster_rows(sub, cols)
    return {
        "header_labels": labels,
        "header_units": units,
        "rows": rows,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "num_pages": num_pages,
        },
    }
def _format_decimal(value: Any, ndigits: int = 2, scientific: bool = False) -> str:
    try:
        if pd.isna(value):
            return ""
        f = float(value)
    except Exception:
        return "" if value is None else str(value)
    if scientific:
        return f"{f:.{ndigits}e}"
    return f"{f:.{ndigits}f}"


def _format_cluster_value(column: str, value: Any) -> str:
    if value is None or (isinstance(value, str) and value == ""):
        return ""

    if column == "M_Z":
        return "" if pd.isna(value) else str(value)

    if column == "RA":
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return _deg_to_hms_str(num) if pd.notna(num) else str(value)

    if column == "DEC":
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return _deg_to_dms_str(num) if pd.notna(num) else str(value)

    if column in {"R200", "SIGMA_V", "RMS", "V_min", "V_max", "BMIN", "BMAJ", "BPA"}:
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return _format_decimal(num, 2) if pd.notna(num) else str(value)

    if column == "M200":
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return _format_decimal(num, 2, scientific=True) if pd.notna(num) else str(value)

    if column == "SOFIA_DETS":
        num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return str(int(num)) if pd.notna(num) else str(value)

    return "" if pd.isna(value) else str(value)


def _format_cluster_rows(df: pd.DataFrame, cols: List[str]) -> List[List[str]]:
    rows: List[List[str]] = []
    if df is None or df.empty:
        return rows
    for _, row in df.iterrows():
        rows.append([_format_cluster_value(col, row.get(col, "")) for col in cols])
    return rows


def _drop_placeholder_galaxy_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    work = df.copy()
    if "name" not in work.columns:
        return work

    name = work["name"].astype(str).fillna("").str.strip()
    lower = name.str.lower()
    header_like_name = (
        name.eq("")
        | name.eq("#")
        | name.str.startswith("#")
        | lower.isin({"name", "id", "source name", "mktcs-hi name", "nan"})
    )

    numeric_cols = [c for c in ["id", "ra", "dec", "v_rad", "f_sum", "w20", "w50", "vrad", "sint"] if c in work.columns]
    if numeric_cols:
        numeric_missing = pd.Series(True, index=work.index)
        for c in numeric_cols:
            numeric_missing &= pd.to_numeric(work[c], errors="coerce").isna()
    else:
        numeric_missing = pd.Series(True, index=work.index)

    return work[~(header_like_name & numeric_missing)].copy()


# ------------------------ ANGLE / RADIUS HELPERS ------------------------
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
        if deg <= 24.0:  # likely hours
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
# --- Sexagesimal formatters ---
def _deg_to_hms_str(ra_deg) -> str:
    """RA degrees -> 'HH:MM:SS.ss' (2 dec places)"""
    try:
        if pd.isna(ra_deg):
            return ""
        hours = (float(ra_deg) / 15.0) % 24.0
        h = int(hours)
        m_float = (hours - h) * 60.0
        m = int(m_float)
        s = (m_float - m) * 60.0
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    except Exception:
        return ""
def _deg_to_dms_str(dec_deg) -> str:
    """Dec degrees -> '+DD:MM:SS.ss' (2 dec places; includes +/−)"""
    try:
        if pd.isna(dec_deg):
            return ""
        val = float(dec_deg)
        sign = "-" if val < 0 else "+"
        d_abs = abs(val)
        d = int(d_abs)
        m_float = (d_abs - d) * 60.0
        m = int(m_float)
        s = (m_float - m) * 60.0
        return f"{sign}{d:02d}:{m:02d}:{s:05.2f}"
    except Exception:
        return ""
def _to_kms_like(series: pd.Series) -> pd.Series:
    """
    Convert a velocity/width series to km/s if values look like m/s.
    Heuristic: values > 2000 are treated as m/s and divided by 1000.
    Leaves NaNs and already-km/s values as-is.
    """
    s = pd.to_numeric(series, errors="coerce")
    out = s.copy()
    mask = s > 2000
    out[mask] = s[mask] / 1000.0
    return out
# ------------------------ CLUSTER FIGURE URL ------------------------
def cluster_figure_url(cluster_id: str) -> str:
    """Backward-compatible helper: return the deterministic intensity preview URL."""
    c = _strip_quotes(cluster_id)
    key = f"{S3_CLUSTER_FIGURES_PREFIX}{c}_Intensity.png"
    return s3_http_url(S3_BUCKET, S3_REGION, key)
# ------------------------ SoFiA CATALOGUE PARSERS ------------------------
def _strip_comment_lines(txt: str) -> str:
    lines = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        lines.append(ln)
    return "\n".join(lines)
def _try_parse_sofia_ascii(text: str) -> Optional[pd.DataFrame]:
    """
    Try SoFiA 'plain text' export with the 3-line header (numbers/names/units).
    Return DataFrame or None if this layout isn't detected.
    """
    lines = text.splitlines()
    idx_num = None
    for i, ln in enumerate(lines):
        if re.match(r'^\s*1\s+2\s+3\s+4\s+', ln):
            idx_num = i
            break
    if idx_num is None:
        return None
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
    block = "\n".join([ln for ln in data_lines if ln.strip()])
    df = pd.read_csv(
        io.StringIO(block),
        sep=r"\s{2,}",
        engine="python",
        names=col_names,
        quoting=csv.QUOTE_NONE,
        dtype=str
    )
    return df
def _try_parse_sofia_names_units(txt: str) -> Optional[pd.DataFrame]:
    """
    Parse a SoFiA-style ASCII catalogue where:
      - first non-marker line is a column-name header (fixed-width, 2+ spaces between cols)
      - second line is a units row
      - data rows follow, and the first column may be a quoted string (e.g. source name)
    This is tolerant of quoted names and does not rely on commas/tabs.
    """
    lines = [ln for ln in txt.replace("\r\n","\n").replace("\r","\n").split("\n") if ln.strip()]
    # Drop marker lines like '#Abell-168' (but keep other content)
    lines = [ln for ln in lines if not (ln.strip().startswith("#") and not ln.strip().startswith("# "))]
    if len(lines) < 3:
        return None
    header = lines[0].rstrip("\n")
    # Heuristic: header has many columns separated by 2+ spaces
    cols = [c.strip() for c in re.split(r"\s{2,}", header.strip()) if c.strip()]
    if len(cols) < 8:
        return None
    # De-duplicate column names
    seen: Dict[str, int] = {}
    col_names: List[str] = []
    for c in cols:
        n = seen.get(c, 0) + 1
        seen[c] = n
        col_names.append(c if n == 1 else f"{c}_{n}")
    data_lines = lines[2:]
    rows: List[List[str]] = []
    for ln in data_lines:
        s = ln.rstrip()
        if not s:
            continue
        tokens: List[str] = []
        if s.lstrip().startswith('"'):
            m = re.match(r'^\s*"([^"]*)"\s*(.*)$', s)
            if not m:
                continue
            tokens.append(m.group(1))
            rest = m.group(2).strip()
            if rest:
                tokens.extend(re.split(r"\s+", rest))
        else:
            tokens = re.split(r"\s+", s.strip())
        if len(tokens) < len(col_names):
            tokens = tokens + [""] * (len(col_names) - len(tokens))
        elif len(tokens) > len(col_names):
            tokens = tokens[:len(col_names)]
        rows.append(tokens)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=[c.strip().lower() for c in col_names])
def _parse_sofia_flexible(txt: str) -> pd.DataFrame:
    """
    Generic parser: skip '#'-comments and try comma, tab, whitespace.
    """
    body = _strip_comment_lines(txt)
    attempts = [
        dict(sep=",", engine="python"),
        dict(sep="\t", engine="python"),
        dict(sep=r"\s+", engine="python"),
        dict(sep=r"[,\t; ]+", engine="python"),
    ]
    last_err: Optional[Exception] = None
    for opts in attempts:
        try:
            df = pd.read_csv(io.StringIO(body), dtype=str, na_filter=False, **opts)
            if df.shape[1] >= 3:
                return df
        except Exception as e:
            last_err = e
    # Last resort: let pandas auto-detect
    if last_err:
        try:
            return pd.read_csv(io.StringIO(body), dtype=str, na_filter=False, engine="python")
        except Exception:
            raise last_err
    raise ValueError("Empty or unparseable catalogue text.")
@lru_cache(maxsize=128)
def _load_sofia_catalogue_cached(cluster_name: str) -> pd.DataFrame:
    """
    Load and parse a SoFiA catalogue for one cluster.
    The parsed DataFrame is cached because galaxy search can touch many
    catalogues repeatedly across requests.
    """
    cluster_clean = _strip_quotes(cluster_name)
    if not cluster_clean:
        return pd.DataFrame()
    # Try to load embedded SoFiA catalogue blocks from the combined metadata file first.
    last_err: Optional[Exception] = None
    txt: Optional[str] = None
    which_key: Optional[str] = None
    try:
        _, sofia_blocks = _split_combined_metadata()
    except Exception as e:
        sofia_blocks = {}
        last_err = e
    for v in _cluster_name_variants(cluster_clean):
        if v in sofia_blocks:
            txt = sofia_blocks[v]
            which_key = f"embedded:{v}"
            break
    # Fallback: older layout with per-cluster SoFiA catalogues living under Catalogues/
    if txt is None:
        for v in _cluster_name_variants(cluster_clean):
            key = f"{S3_CATALOGUES_PREFIX}{v}_cat.txt"
            url = s3_http_url(S3_BUCKET, S3_REGION, key)
            try:
                txt = _http_get_text(url)
                which_key = key
                break
            except Exception as e:
                last_err = e
                continue
    if txt is None:
        raise RuntimeError(f'Could not load SoFiA catalogue for "{cluster_clean}": {last_err}')
    # Parse: try known SoFiA ASCII variants first, then fall back to flexible parsing.
    df_ascii = _try_parse_sofia_ascii(txt)
    if df_ascii is not None:
        df = df_ascii
    else:
        df_compact = _try_parse_sofia_names_units(txt)
        if df_compact is not None:
            df = df_compact
        else:
            df = _parse_sofia_flexible(txt)
    # Normalize column names (lower/strip)
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", c.strip().lower()) for c in df.columns]
    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None
    col_id   = pick("id")
    col_name = pick("name", "mktcs name", "mktcs-hi name", "source name", "object name", "src_name")
    col_ra   = pick("ra", "ra deg", "ra (j2000)", "ra[deg]", "ra deg (j2000)", "ra hms", "ra(hms)", "ra_peak")
    col_dec  = pick("dec", "dec deg", "dec (j2000)", "dec[deg]", "decl", "dec hms", "dec_peak")
    col_v    = pick("v_rad", "vrad", "v", "velocity", "v[km/s]", "radial velocity", "v_cen", "v_rad_peak", "v_peak", "vhel")
    col_f    = pick("f_sum", "sint", "s int", "integrated flux", "flux", "fint", "s_int")
    col_w20  = pick("w20", "w 20", "w20[km/s]", "w_20", "w_20[km/s]")
    col_w50  = pick("w50", "w 50", "w50[km/s]", "w_50", "w_50[km/s]")
    out = pd.DataFrame()
    if col_id:   out["id"]    = df[col_id]
    if col_name: out["name"]  = df[col_name]
    if col_ra:   out["ra"]    = df[col_ra]
    if col_dec:  out["dec"]   = df[col_dec]
    if col_v:    out["v_rad"] = df[col_v]
    if col_f:    out["f_sum"] = df[col_f]
    if col_w20:  out["w20"]   = df[col_w20]
    if col_w50:  out["w50"]   = df[col_w50]
    # Ensure columns exist
    for c in ["id", "name", "ra", "dec", "v_rad", "f_sum", "w20", "w50"]:
        if c not in out.columns:
            out[c] = ""
    # Numeric conversions where possible
    with pd.option_context("mode.chained_assignment", None):
        for c in ["id", "ra", "dec", "v_rad", "f_sum", "w20", "w50"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        # If v_rad seems like m/s (very big), convert to km/s later in formatter
    return out
def load_sofia_catalogue(cluster_name: str) -> pd.DataFrame:
    """
    Public wrapper around the cached loader.
    Returns a copy so callers can safely mutate columns without polluting the cache.
    """
    return _load_sofia_catalogue_cached(_strip_quotes(cluster_name)).copy()
# ------------------------ CLUSTER DETAIL: GALAXY ROWS ------------------------
def cluster_galaxy_rows(cluster_id: str) -> list[dict]:
    """
    Records for the detail table, including the SoFiA id so we can build
    per-galaxy download links.
    Output rows have keys:
      id, name, ra, dec, vrad, sint, w20, w50
    """
    try:
        df = load_sofia_catalogue(cluster_id)  # id,name,ra,dec,v_rad,f_sum,w20,w50 (flexible)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    df = df.copy()
    # Ensure columns exist
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)
    # Normalize velocity & flux field names
    if "vrad" not in df.columns and "v_rad" in df.columns:
        df["vrad"] = df["v_rad"]
    if "sint" not in df.columns and "f_sum" in df.columns:
        df["sint"] = df["f_sum"]
    if "name" not in df.columns:
        df["name"] = df["id"].astype(str)

    df = _drop_placeholder_galaxy_rows(df)
    if df.empty:
        return []

    # Numeric casts for formatting
    with pd.option_context("mode.chained_assignment", None):
        df["id"]   = pd.to_numeric(df.get("id"),   errors="coerce")
        df["ra"]   = pd.to_numeric(df.get("ra"),   errors="coerce")
        df["dec"]  = pd.to_numeric(df.get("dec"),  errors="coerce")
        df["vrad"] = pd.to_numeric(df.get("vrad"), errors="coerce")
        df["sint"] = pd.to_numeric(df.get("sint"), errors="coerce")
        if "w20" in df: df["w20"] = pd.to_numeric(df.get("w20"), errors="coerce")
        if "w50" in df: df["w50"] = pd.to_numeric(df.get("w50"), errors="coerce")

    # Convert vrad to km/s if it looks like m/s, and widths likewise.
    use_kms_v = (df["vrad"].abs() > 2e4).fillna(False)
    df["vrad_display"] = np.where(use_kms_v, df["vrad"] / 1000.0, df["vrad"])

    if "w20" in df:
        use_kms_w20 = (df["w20"].abs() > 2e3).fillna(False)
        df["w20_display"] = np.where(use_kms_w20, df["w20"] / 1000.0, df["w20"])
    else:
        df["w20_display"] = np.nan

    if "w50" in df:
        use_kms_w50 = (df["w50"].abs() > 2e3).fillna(False)
        df["w50_display"] = np.where(use_kms_w50, df["w50"] / 1000.0, df["w50"])
    else:
        df["w50_display"] = np.nan

    # RA/DEC -> sexagesimal strings
    ra_hms  = df["ra"].map(_deg_to_hms_str)  if "ra"  in df else pd.Series(dtype=str)
    dec_dms = df["dec"].map(_deg_to_dms_str) if "dec" in df else pd.Series(dtype=str)

    out = pd.DataFrame({
        "id":   df["id"].map(lambda v: "" if pd.isna(v) else f"{int(v)}"),
        "name": df["name"].astype(str).fillna(""),
        "ra":   ra_hms,
        "dec":  dec_dms,
        "vrad": df["vrad_display"].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}"),
        "sint": df["sint"].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}"),
        "w20":  df["w20_display"].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}"),
        "w50":  df["w50_display"].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}"),
    })
    out = out.sort_values(by=["name"], kind="stable")
    return out.to_dict(orient="records")
# ------------------------ CLUSTER SEARCH + PREVIEWS ------------------------
def _cluster_preview_urls(cluster_name: str) -> Dict[str, Optional[str]]:
    """Return deterministic cluster preview image URLs.
    Naming rules:
      - Cluster-Figures/<cluster-name>_Intensity.png
      - Cluster-Figures/<cluster-name>_Noise_Footprints.png
      - Completeness-Curves/<clustername_no_hyphen>_completeness_matched_v8.png
    The metadata cluster name matches the S3 naming exactly, so we build the URLs
    directly instead of probing S3.
    """
    cluster_name = _strip_quotes(cluster_name)
    compact = str(cluster_name).replace('-', '')
    return {
        "intensity": s3_http_url(S3_BUCKET, S3_REGION, f"{S3_CLUSTER_FIGURES_PREFIX}{cluster_name}_Intensity.png"),
        "noise": s3_http_url(S3_BUCKET, S3_REGION, f"{S3_CLUSTER_FIGURES_PREFIX}{cluster_name}_Noise_Footprints.png"),
        "completeness": s3_http_url(S3_BUCKET, S3_REGION, f"{S3_COMPLETENESS_CURVES_PREFIX}{compact}_completeness_matched_v8.png"),
    }
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
        df = df[_series_search_match(df["ID"].astype(str), name_query)]
    if mgcls_query and "MGCLS_Name" in df.columns:
        df = df[_series_search_match(df["MGCLS_Name"].astype(str), mgcls_query)]
    if sbid and "SBID" in df.columns:
        df = df[_series_search_match(df["SBID"].astype(str), sbid)]
    if capture_id and "CAPTURE_ID" in df.columns:
        df = df[_series_search_match(df["CAPTURE_ID"].astype(str), capture_id)]
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
    df = df.iloc[(page - 1) * page_size: page * page_size].copy()

    cols = [c for c in DISPLAY_ORDER if c in master_df.columns]
    labels = [DISPLAY_LABELS.get(c, c) for c in cols]
    units = [DISPLAY_UNITS.get(c, "") for c in cols]
    rows = _format_cluster_rows(df, cols)

    previews = []
    for _, r in df.iterrows():
        p = _cluster_preview_urls(r["ID"])
        if any(v for v in p.values()):
            #previews.append({"cluster": r["ID"], "url": p["preview"]})
            previews.append({"cluster": str(r["ID"]), **p})
    spec = {
        "header_labels": labels,          # <-- use pretty labels
        "header_units": units,            # unchanged
        "rows": rows,                     # unchanged
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "num_pages": num_pages,
        },
    }
    if df.empty:
        warns.append("No clusters matched your filters.")
    return spec, previews, [], warns
# ------------------------ GALAXY SEARCH + PREVIEWS ------------------------
def _galaxy_preview_urls(cluster: str, gid: int | str, name: str | None) -> Dict[str, Any]:
    """Return deterministic galaxy preview URLs without server-side S3 probes.

    The previous implementation performed many per-result HEAD requests against S3
    to discover which preview figure variant existed. That made search results slow
    and led to galaxy previews being disabled entirely on the search page.

    Here we build ordered candidate URL lists instead:
      - moment-0 tries SDSS first, then Pan-STARRS, then WISE W1
      - moment-1 / moment-2 / spectrum use the direct deterministic filenames
      - a small set of cluster-name separator variants is included for safety

    The template/JS can then try these candidates client-side until one loads,
    without blocking the Flask response.
    """
    cluster_clean = _strip_quotes(cluster)
    gid_raw = str(gid).strip()
    if not cluster_clean or not gid_raw:
        return {
            "mom0": None, "mom1": None, "mom2": None, "spec": None,
            "mom0_candidates": [], "mom1_candidates": [],
            "mom2_candidates": [], "spec_candidates": [],
        }

    gid_s = str(int(gid_raw)) if gid_raw.isdigit() else gid_raw
    cluster_variants = _unique_preserve([cluster_clean] + _cluster_name_variants(cluster_clean))

    def _urls(keys: List[str]) -> List[str]:
        return [s3_http_url(S3_BUCKET, S3_REGION, key) for key in _unique_preserve(keys)]

    mom0_keys: List[str] = []
    for c in cluster_variants:
        base = f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_{gid_s}_mom0"
        mom0_keys.extend([
            base + "_sdss.png",
            base + "_panstarrs.png",
            base + "_wisew1.png",
        ])

    mom1_keys = [f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_{gid_s}_mom1.png" for c in cluster_variants]
    mom2_keys = [f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_{gid_s}_mom2.png" for c in cluster_variants]
    spec_keys = [f"{S3_GALAXY_FIGURES_PREFIX}{c}_figures/{c}_{gid_s}_spec.png" for c in cluster_variants]

    mom0_candidates = _urls(mom0_keys)
    mom1_candidates = _urls(mom1_keys)
    mom2_candidates = _urls(mom2_keys)
    spec_candidates = _urls(spec_keys)

    return {
        "mom0": mom0_candidates[0] if mom0_candidates else None,
        "mom1": mom1_candidates[0] if mom1_candidates else None,
        "mom2": mom2_candidates[0] if mom2_candidates else None,
        "spec": spec_candidates[0] if spec_candidates else None,
        "mom0_candidates": mom0_candidates,
        "mom1_candidates": mom1_candidates,
        "mom2_candidates": mom2_candidates,
        "spec_candidates": spec_candidates,
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
    all_clusters = (
        sorted(master_df["ID"].dropna().astype(str).map(_strip_quotes).unique().tolist())
        if master_df is not None and "ID" in master_df.columns else []
    )
    clusters = [_strip_quotes(c) for c in (cluster_scope or [])] or all_clusters
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
    if name_sub:
        try:
            _, sofia_blocks = _split_combined_metadata()
            narrowed: List[str] = []
            unresolved: List[str] = []
            for clus in clusters:
                block_hits = []
                for v in _cluster_name_variants(clus):
                    blk = sofia_blocks.get(v)
                    if blk:
                        block_hits.append(blk)
                if block_hits:
                    haystack = "\n".join(block_hits)
                    if _text_search_match(haystack, name_query):
                        narrowed.append(clus)
                else:
                    unresolved.append(clus)
            clusters = narrowed if narrowed else unresolved
        except Exception:
            pass

    rows: List[List[str]] = []
    row_meta: List[Dict[str, str]] = []
    for clus in clusters:
        try:
            cat = load_sofia_catalogue(clus)
        except Exception as e:
            warns.append(f'"{clus}": failed to load catalogue ({e})')
            continue
        if cat is None or cat.empty:
            continue

        cat = _drop_placeholder_galaxy_rows(cat)
        if cat.empty:
            continue

        vraw = pd.to_numeric(cat.get("v_rad"), errors="coerce")
        vkms = vraw.copy()
        vkms[vraw.abs() > 2.0e4] = vraw[vraw.abs() > 2.0e4] / 1000.0
        cat["v_kms"] = vkms

        w20_raw = pd.to_numeric(cat.get("w20"), errors="coerce")
        w20_kms = w20_raw.copy()
        w20_kms[w20_raw.abs() > 2.0e3] = w20_raw[w20_raw.abs() > 2.0e3] / 1000.0
        cat["w20_kms"] = w20_kms

        w50_raw = pd.to_numeric(cat.get("w50"), errors="coerce")
        w50_kms = w50_raw.copy()
        w50_kms[w50_raw.abs() > 2.0e3] = w50_raw[w50_raw.abs() > 2.0e3] / 1000.0
        cat["w50_kms"] = w50_kms

        mask = pd.Series(True, index=cat.index)
        if name_sub:
            mask &= _series_search_match(cat["name"].astype(str), name_query)
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
            w20  = r.get("w20_kms")
            w50  = r.get("w50_kms")
            rows.append([
                str(clus),
                str(gname) if pd.notna(gname) else "",
                _deg_to_hms_str(ra),
                _deg_to_dms_str(dec),
                f"{float(vkms):.2f}" if pd.notna(vkms) else "",
                f"{float(fsum):.2f}" if pd.notna(fsum) else "",
                f"{float(w20):.2f}" if pd.notna(w20) else "",
                f"{float(w50):.2f}" if pd.notna(w50) else "",
            ])
            row_meta.append({
                "cluster": str(clus),
                "id": f"{int(gid)}" if pd.notna(gid) else "",
                "name": str(gname) if pd.notna(gname) else "",
            })
    total = len(rows)
    num_pages = max(1, int(np.ceil(total / page_size)))
    page = min(max(1, page), num_pages)
    start = (page - 1) * page_size
    end   = page * page_size
    rows_page = rows[start:end]
    meta_page = row_meta[start:end]
    previews: List[Dict[str, Any]] = []
    for meta in meta_page:
        cluster_name = str(meta.get("cluster", "") or "").strip()
        gid = str(meta.get("id", "") or "").strip()
        gname = str(meta.get("name", "") or "").strip()
        if not cluster_name or not gid:
            continue
        p = _galaxy_preview_urls(cluster_name, gid, gname)
        previews.append({
            "cluster": cluster_name,
            "id": gid,
            "name": gname,
            **p,
        })
    spec = {
        "header_labels": ["Cluster", "Galaxy Name", "RA", "DEC", "V<sub>rad</sub>", "S<sub>int</sub>", "W<sub>20</sub>", "W<sub>50</sub>"],
        "header_units":  ["", "MKTCS-HI JHHMMSS.ss-DDMMSS.s", "[HH:MM:SS.ss]", "[±DD:MM:SS.ss]", "[km/s]", "[Jy km/s]", "[km/s]", "[km/s]"],
        "rows": rows_page,
        "row_meta": meta_page,
        "pagination": {"page": page, "page_size": page_size, "total": total, "num_pages": num_pages},
    }
    if total == 0:
        warns.append("No galaxies matched your filters.")
    return spec, previews, [], warns
# ------------------------ DOWNLOAD MANIFESTS + ZIP ------------------------
def _cluster_package_pairs(cluster: str) -> List[Tuple[str, str]]:
    """
    Cluster-level products (no figures, no *_chan):
      - Catalogues/<cluster>_cat.txt
      - Cluster-Cubes/<cluster>.fits
      - Cluster-Masks/<cluster>_mask.fits
      - Cluster-Moms/<cluster>_mom0/1/2.fits
    """
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
    """
    Deterministic galaxy-level product paths.
    The previous implementation probed S3 for every candidate product/variant
    before returning anything. That was reliable locally but too slow for the
    deployed API, especially when building cluster-level galaxy download scripts.
    The public bucket uses predictable keys of the form:
      Galaxy-Cubelets/<cluster>/<cluster>_<gid>_<type>.<ext>
    So we build those keys directly and let the generated shell script fetch
    them from S3.
    """
    cluster_clean = _strip_quotes(cluster)
    gid_s = str(int(gid)) if str(gid).strip().isdigit() else str(gid)
    top = f"{cluster_clean}/cubelets/"
    base_dir = f"{S3_GALAXY_CUBELETS_PREFIX}{cluster_clean}/"
    suffixes = [
        "_chan.fits", "_cube.fits", "_mask.fits",
        "_mom0.fits", "_mom1.fits", "_mom2.fits",
        "_pv.fits", "_snr.fits", "_spec.txt",
    ]
    pairs: List[Tuple[str, str]] = []
    for suf in suffixes:
        key = f"{base_dir}{cluster_clean}_{gid_s}{suf}"
        pairs.append((top + os.path.basename(key), key))
    return pairs
def build_zip_from_pairs(pairs: List[Tuple[str, str]], title: str) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, key in pairs:
            data = _s3_get_bytes(S3_BUCKET, S3_REGION, key)
            if data is not None:
                zf.writestr(arcname, data)
    buf.seek(0)
    return buf
# ------------------------ TEMPLATE HELPERS ------------------------
def cluster_previews_for_template(cluster_name: str) -> Dict[str, Optional[str]]:
    """Preferred helper: returns {'intensity','noise','completeness'} URLs."""
    return _cluster_preview_urls(cluster_name)
def cluster_preview_for_template(cluster_name: str) -> Optional[str]:
    """Backward-compatible: return the intensity preview if available."""
    return _cluster_preview_urls(cluster_name).get("intensity")
def galaxy_previews_for_template(cluster: str, gid: str | int, name: str | None) -> Dict[str, Optional[str]]:
    return _galaxy_preview_urls(cluster, gid, name)
