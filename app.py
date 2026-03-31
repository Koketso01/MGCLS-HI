from __future__ import annotations

import os
import shlex
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict

from flask import (
    Flask,
    Response,
    render_template,
    request,
    send_file,
    abort,
    url_for,
)

import pandas as pd

import logging
import utils_UPDATED_single_metadata_v3_previews as utils  # our helpers (parsing, S3 fetches, searching, zipping)

# -----------------------------------------------------------------------------
# Flask app + logging
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-not-secret")

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s.%(msecs)03dZ [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("mgcls-hi")


def _public_base_url() -> str:
    """Best-effort canonical site base URL."""
    configured = (os.environ.get("SITE_URL") or "").strip().rstrip("/")
    if configured:
        return configured
    try:
        return request.url_root.rstrip("/")
    except Exception:
        return ""

#S3 URLs for the navbar images
@app.context_processor
def inject_navbar_assets():
    base = f"{utils.S3_PREFIX.rstrip('/')}/navbar_figures/"
    site_url = _public_base_url()
    endpoint = (request.endpoint or "").strip()
    descriptions = {
        "index": "MGCLS–H I DR1: browse MGCLS galaxy clusters, H I detections, preview figures, catalogues, cubes and cubelets.",
        "search": "Search MGCLS–H I clusters and galaxies by name, position, velocity and observing metadata.",
        "about": "Learn about the MGCLS–H I database, survey scope, data products and scientific motivation.",
        "help": "MGCLS–H I help and field definitions for cluster and galaxy catalogue columns and downloads.",
        "publications": "Publications and references related to the MGCLS–H I database and the parent MGCLS survey.",
        "people": "Contributors and collaborators behind the MGCLS–H I database.",
        "contact": "Contact the MGCLS–H I team and RATT admin for data, website and support queries.",
    }
    return {
        "NAVBAR_IMG": utils.s3_http_url(utils.S3_BUCKET, utils.S3_REGION, base + "navbar_image.png"),
        "LOGO_IMG": utils.s3_http_url(utils.S3_BUCKET, utils.S3_REGION, base + "logo.png"),
        "FAVICON_IMG": utils.s3_http_url(utils.S3_BUCKET, utils.S3_REGION, base + "hi_mgcls_icon.png"),
        "HERO_BG_IMG": utils.s3_http_url(utils.S3_BUCKET, utils.S3_REGION, f"{utils.S3_PREFIX.rstrip('/')}/OtherFigures/noise_background.png"),
        "FOOTER_BASE": utils.s3_http_url(utils.S3_BUCKET, utils.S3_REGION, f"{utils.S3_PREFIX.rstrip('/')}/FooterFigures").rstrip("/"),
        "SITE_URL": site_url,
        "SEO_DESCRIPTION": descriptions.get(endpoint, descriptions["index"]),
        "CURRENT_YEAR": datetime.now(timezone.utc).year,
    }

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

@app.before_request
def _trace_inbound() -> None:
    try:
        if request.method == "GET":
            logger.info("[%s] INBOUND GET %s args=%s", _ts(), request.path, dict(request.args))
        elif request.method == "POST":
            form_preview = {k: request.form.get(k) for k in request.form.keys()}
            logger.info("[%s] INBOUND POST %s form=%s", _ts(), request.path, form_preview)
    except Exception:
        logger.exception("[%s] inbound-trace failed", _ts())


# -----------------------------------------------------------------------------
# Landing page
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    logger.info("[%s] index(): loading master catalogue…", _ts())
    try:
        master = utils.load_master_catalogue()
        spec = utils.build_landing_table_spec(
            master,
            page=int(request.args.get("page", "1") or "1"),
            page_size=int(request.args.get("page_size", "25") or "25"),
            quick_filter=(request.args.get("q") or ""),
        )
        hi_stats = utils.compute_global_hi_stats()

        logger.info(
            "[%s] index(): rows=%d (page %d/%d, total=%d)",
            _ts(), len(spec["rows"]), spec["pagination"]["page"],
            spec["pagination"]["num_pages"], spec["pagination"]["total"]
        )
        return render_template(
            "index.html",
            header_labels=spec["header_labels"],
            header_units=spec["header_units"],
            rows=spec["rows"],
            pagination=spec["pagination"],
            hi_stats=hi_stats,
        )
    except Exception as e:
        logger.exception("[%s] index(): failed to render", _ts())
        abort(500, f"Landing page failed: {e}")


# -----------------------------------------------------------------------------
# Cluster detail page
# -----------------------------------------------------------------------------
@app.route("/cluster/<cluster_id>")
def cluster_detail(cluster_id: str):
    """
    Render a cluster detail page, given a cluster ID.

    :param cluster_id: The ID of the cluster to render.
    :return: A rendered template for the cluster detail page.
    """
    previews = utils.cluster_previews_for_template(cluster_id)
    rows = utils.cluster_galaxy_rows(cluster_id)
    logger.info("[%s] cluster_detail: %s · galaxies=%d", _ts(), cluster_id, len(rows))
    return render_template(
        "cluster_detail.html",
        cluster_id=cluster_id,
        previews=previews,
        galaxy_rows=rows,
        all_galaxies_script_url=url_for("download_all_galaxies", cluster=cluster_id, _external=True) + "?format=script",
    )


# -----------------------------------------------------------------------------
# Search page: clusters & galaxies
# -----------------------------------------------------------------------------
@app.route("/search", methods=["GET", "POST"])
def search():
    did_submit = request.method == "POST"
    target_choice = (
        request.form.get("target_choice") if did_submit
        else (request.args.get("target_choice") or "clusters")
    )

    logger.info("[%s] search(): target_choice=%s did_submit=%s", _ts(), target_choice, did_submit)
    master = utils.load_master_catalogue()

    tpl_ctx: Dict[str, Any] = {
        "did_submit": did_submit,
        "target_choice": target_choice,
        "cluster_choices": sorted(master["ID"].dropna().astype(str).unique().tolist()) if "ID" in master.columns else [],
        "form": {},
    }

    # ----------------- clusters -----------------
    # Always expose form fields (so the page renders), but only run a search after submit
    form = {
        "name_query": request.form.get("name_query", "") if did_submit else request.args.get("name_query", ""),
        "mgcls_query": request.form.get("mgcls_query", "") if did_submit else request.args.get("mgcls_query", ""),
        "sbid": request.form.get("sbid", "") if did_submit else request.args.get("sbid", ""),
        "capture_id": request.form.get("capture_id", "") if did_submit else request.args.get("capture_id", ""),
        "ra": request.form.get("ra", "") if did_submit else request.args.get("ra", ""),
        "dec": request.form.get("dec", "") if did_submit else request.args.get("dec", ""),
        "radius": request.form.get("radius", "") if did_submit else request.args.get("radius", ""),
        "radius_unit": request.form.get("radius_unit", "arcmin") if did_submit else request.args.get("radius_unit", "arcmin"),
        "vel_center": request.form.get("vel_center", "") if did_submit else request.args.get("vel_center", ""),
        "vel_tol": request.form.get("vel_tol", "") if did_submit else request.args.get("vel_tol", ""),
        "vel_min": request.form.get("vel_min", "") if did_submit else request.args.get("vel_min", ""),
        "vel_max": request.form.get("vel_max", "") if did_submit else request.args.get("vel_max", ""),
    }
    tpl_ctx["form"].update(form)

    # default: no results until a search is submitted
    tpl_ctx.update({
        "cluster_header_labels": [],
        "cluster_header_units": [],
        "cluster_rows": [],
        "cluster_previews": [],
        "cluster_results": [],
        "cluster_errors": [],
        "cluster_warnings": [],
        "galaxy_header_labels": [],
        "galaxy_header_units": [],
        "galaxy_rows": [],
        "galaxy_row_meta": [],
        "galaxy_previews": [],
        "galaxy_results": [],
        "galaxy_errors": [],
        "galaxy_warnings": [],
    })

    # Only search after submit, and only when the “Clusters” tab is chosen
    if did_submit and target_choice != "galaxies":
        radius_text = f'{form["radius"]} {form.get("radius_unit","arcmin")}' if form["radius"] else ""
        logger.info("[%s] search(): cluster form parsed -> %s", _ts(), form)

        spec, previews, errors, warns = utils.search_clusters(
            master_df=master,
            name_query=form["name_query"],
            mgcls_query=form["mgcls_query"],
            sbid=form["sbid"],
            capture_id=form["capture_id"],
            ra_txt=form["ra"],
            dec_txt=form["dec"],
            radius_text=radius_text,
            vel_center=form["vel_center"],
            vel_tol=form["vel_tol"],
            vel_min=form["vel_min"],
            vel_max=form["vel_max"],
            page=int(request.args.get("page", "1") or "1"),
            page_size=25,
        )

        logger.info("[%s] search(): cluster results rows=%d previews=%d warns=%s",
                    _ts(), len(spec["rows"]), len(previews), warns)

        cluster_preview_map = {
            str(item.get("cluster", "")): item
            for item in previews
            if item.get("cluster")
        }
        cluster_results = [
            {
                "row": row,
                "preview": cluster_preview_map.get(str(row[0]), {}),
            }
            for row in spec["rows"]
        ]

        tpl_ctx.update({
            "cluster_header_labels": spec["header_labels"],
            "cluster_header_units": spec["header_units"],
            "cluster_rows": spec["rows"],
            "cluster_previews": previews,
            "cluster_results": cluster_results,
            "cluster_errors": errors,
            "cluster_warnings": warns,
        })


    # ---------- galaxies ----------
    if did_submit and target_choice == "galaxies":
        scope = request.form.getlist("cluster_scope") or []
        scope_clean = [s.strip() for s in scope if s and s.strip()]
        scope_join = ",".join(scope_clean)

        form = {
            "galaxy_name": request.form.get("galaxy_name", ""),
            "cluster_scope": scope_join,
            "g_ra": request.form.get("g_ra", ""),
            "g_dec": request.form.get("g_dec", ""),
            "g_radius": request.form.get("g_radius", ""),
            "g_radius_unit": request.form.get("g_radius_unit", "arcmin"),
            "g_vel_center": request.form.get("g_vel_center", ""),
            "g_vel_tol": request.form.get("g_vel_tol", ""),
        }
        tpl_ctx["form"].update(form)

        radius_text = f'{form["g_radius"]} {form.get("g_radius_unit","arcmin")}' if form["g_radius"] else ""

        spec, previews, errors, warns = utils.search_galaxies(
            master_df=master,
            name_query=form["galaxy_name"],
            ra_txt=form["g_ra"],
            dec_txt=form["g_dec"],
            radius_text=radius_text,
            vel_center_kms=form["g_vel_center"],
            vel_tol_kms=form["g_vel_tol"],
            cluster_scope=scope_clean,
            page=int(request.args.get("page", "1") or "1"),
            page_size=25,
        )

        galaxy_preview_map = {
            (str(item.get("cluster", "")), str(item.get("id", ""))): item
            for item in previews
            if item.get("cluster") is not None and item.get("id") is not None
        }
        galaxy_results = [
            {
                "row": row,
                "meta": meta,
                "preview": galaxy_preview_map.get(
                    (str(meta.get("cluster", "")), str(meta.get("id", ""))),
                    {},
                ),
            }
            for row, meta in zip(spec["rows"], spec.get("row_meta", []))
        ]

        tpl_ctx.update({
            "galaxy_header_labels": spec["header_labels"],
            "galaxy_header_units": spec["header_units"],
            "galaxy_rows": spec["rows"],
            "galaxy_row_meta": spec.get("row_meta", []),
            "galaxy_previews": previews,
            "galaxy_results": galaxy_results,
            "galaxy_errors": errors,
            "galaxy_warnings": warns,
        })

    return render_template("search.html", **tpl_ctx)




# -----------------------------------------------------------------------------
# Download selection helpers
# -----------------------------------------------------------------------------
CLUSTER_PRODUCT_GROUPS = [
    {
        "title": "Cluster products",
        "help": "Main cluster-level data products.",
        "items": [
            {"key": "cluster_cat", "label": "Cluster catalogue"},
            {"key": "cluster_cube", "label": "Cluster main cube"},
            {"key": "cluster_mask", "label": "Cluster mask"},
            {"key": "cluster_mask_raw", "label": "Cluster raw mask"},
            {"key": "cluster_mom0", "label": "Moment 0"},
            {"key": "cluster_mom1", "label": "Moment 1"},
            {"key": "cluster_mom2", "label": "Moment 2"},
        ],
    },
    {
        "title": "Galaxy cubelet products",
        "help": "Apply selected cubelet product types across all detected galaxies in the cluster.",
        "items": [
            {"key": "gal_chan", "label": "Channel cubelets"},
            {"key": "gal_cube", "label": "Galaxy cubes"},
            {"key": "gal_mask", "label": "Galaxy masks"},
            {"key": "gal_mom0", "label": "Galaxy moment 0"},
            {"key": "gal_mom1", "label": "Galaxy moment 1"},
            {"key": "gal_mom2", "label": "Galaxy moment 2"},
            {"key": "gal_pv", "label": "PV slices"},
            {"key": "gal_snr", "label": "SNR cubes"},
            {"key": "gal_spec", "label": "Spectra"},
        ],
    },
]

GALAXY_PRODUCT_GROUPS = [
    {
        "title": "Galaxy products",
        "help": "Select which products to download for this galaxy.",
        "items": [
            {"key": "gal_chan", "label": "Channel cubelet"},
            {"key": "gal_cube", "label": "Galaxy cube"},
            {"key": "gal_mask", "label": "Galaxy mask"},
            {"key": "gal_mom0", "label": "Moment 0"},
            {"key": "gal_mom1", "label": "Moment 1"},
            {"key": "gal_mom2", "label": "Moment 2"},
            {"key": "gal_pv", "label": "PV slice"},
            {"key": "gal_snr", "label": "SNR cube"},
            {"key": "gal_spec", "label": "Spectrum"},
        ],
    },
]

ALL_CLUSTER_KEYS = [item["key"] for group in CLUSTER_PRODUCT_GROUPS for item in group["items"]]
ALL_GALAXY_KEYS = [item["key"] for group in GALAXY_PRODUCT_GROUPS for item in group["items"]]

GAL_SUFFIX_MAP = {
    "gal_chan": "_chan.fits",
    "gal_cube": "_cube.fits",
    "gal_mask": "_mask.fits",
    "gal_mom0": "_mom0.fits",
    "gal_mom1": "_mom1.fits",
    "gal_mom2": "_mom2.fits",
    "gal_pv": "_pv.fits",
    "gal_snr": "_snr.fits",
    "gal_spec": "_spec.txt",
}

def _render_download_select(*, entity_title: str, subtitle: str, back_href: str, back_label: str, action_url: str, action_url_abs: str, filename: str, groups: list[dict], default_checked: list[str], cli_note: str):
    return render_template(
        "download_select.html",
        entity_title=entity_title,
        subtitle=subtitle,
        back_href=back_href,
        back_label=back_label,
        action_url=action_url,
        action_url_abs=action_url_abs,
        filename=filename,
        groups=groups,
        default_checked=default_checked,
        cli_note=cli_note,
    )


def _pairs_to_shell_script(pairs, script_name: str, banner: str) -> str:
    pairs = _dedupe_pairs(pairs)
    expected = len(pairs)
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        "",
        f'SCRIPT_NAME={shlex.quote(script_name)}',
        f'BANNER={shlex.quote(banner)}',
        f'EXPECTED={expected}',
        'LOG_FILE="download_log.txt"',
        'MISSING_LOG="missing_files.log"',
        'FAILED_LOG="failed_files.log"',
        'DOWNLOADED=0',
        'MISSING=0',
        'FAILED=0',
        'STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date)"',
        ': > "$LOG_FILE"',
        ': > "$MISSING_LOG"',
        ': > "$FAILED_LOG"',
        'exec > >(tee -a "$LOG_FILE") 2>&1',
        '',
        'echo "$BANNER"',
        'echo "Script: $SCRIPT_NAME"',
        'echo "Started: $STARTED_AT"',
        'echo "Expected files: $EXPECTED"',
        'echo "This workflow continues even if an individual file is missing or fails."',
        'echo "Missing or failed files will be listed in local log files at the end."',
        '',
        'if command -v wget >/dev/null 2>&1; then',
        '  DL_TOOL="wget"',
        'elif command -v curl >/dev/null 2>&1; then',
        '  DL_TOOL="curl"',
        'else',
        '  echo "Please install either wget or curl first."',
        '  exit 1',
        'fi',
        'echo "Download tool: $DL_TOOL"',
        '',
        'finish_summary() {',
        '  local status="${1:-completed}"',
        '  echo ""',
        '  echo "Download summary ($status)"',
        '  echo "  Downloaded: $DOWNLOADED / $EXPECTED"',
        '  echo "  Missing:    $MISSING"',
        '  echo "  Failed:     $FAILED"',
        '  echo "  Log file:   $LOG_FILE"',
        '  if [ -s "$MISSING_LOG" ]; then echo "  Missing list: $MISSING_LOG"; else rm -f "$MISSING_LOG"; fi',
        '  if [ -s "$FAILED_LOG" ]; then echo "  Failed list:  $FAILED_LOG"; else rm -f "$FAILED_LOG"; fi',
        '  echo "The MGCLS-HI team thanks you for using our data."',
        '}',
        '',
        'on_interrupt() {',
        '  echo ""',
        '  echo "Download interrupted before completion."',
        '  finish_summary "interrupted"',
        '  exit 130',
        '}',
        'trap on_interrupt INT TERM',
        '',
        'url_exists() {',
        '  local url="$1"',
        '  if command -v curl >/dev/null 2>&1; then',
        '    curl -fsIL "$url" >/dev/null 2>&1',
        '  else',
        '    wget -q --spider "$url" >/dev/null 2>&1',
        '  fi',
        '}',
        '',
        'download_one() {',
        '  local idx="$1"',
        '  local out="$2"',
        '  local url="$3"',
        '  mkdir -p "$(dirname "$out")"',
        '  echo "[$idx/$EXPECTED] $out"',
        '  if [ "$DL_TOOL" = "wget" ]; then',
        '    if wget -c --show-progress -O "$out" "$url"; then',
        '      DOWNLOADED=$((DOWNLOADED + 1))',
        '      echo "  Downloaded"',
        '      return 0',
        '    fi',
        '  else',
        '    if curl -fL -C - "$url" -o "$out"; then',
        '      DOWNLOADED=$((DOWNLOADED + 1))',
        '      echo "  Downloaded"',
        '      return 0',
        '    fi',
        '  fi',
        '  rm -f "$out"',
        '  if url_exists "$url"; then',
        '    FAILED=$((FAILED + 1))',
        '    printf "%s\t%s\n" "$out" "$url" >> "$FAILED_LOG"',
        '    echo "  Failed while fetching (network, permissions, disk space, or interruption)"',
        '  else',
        '    MISSING=$((MISSING + 1))',
        '    printf "%s\t%s\n" "$out" "$url" >> "$MISSING_LOG"',
        '    echo "  Not currently available on the server"',
        '  fi',
        '  return 0',
        '}',
        '',
    ]
    for idx, (arc, key) in enumerate(pairs, start=1):
        rel = arc.lstrip("/")
        url = utils.s3_http_url(utils.S3_BUCKET, utils.S3_REGION, key)
        lines.extend([
            f'download_one {idx} {shlex.quote(rel)} {shlex.quote(url)}',
            "",
        ])
    lines.extend([
        'finish_summary "completed"',
    ])
    return "\n".join(lines).strip() + "\n"


def _requested_product_keys() -> list[str]:
    """
    Robustly read selected ?products=... values from Flask/Zappa requests.

    Handles:
      - repeated query params: ?products=a&products=b
      - comma-joined forms:   ?products=a,b
      - cases where only the raw query string preserves the full list
    """
    raw_vals = request.args.getlist("products")
    if not raw_vals:
        single = request.args.get("products")
        if single:
            raw_vals = [single]

    if not raw_vals and request.query_string:
        try:
            parsed = urllib.parse.parse_qs(
                request.query_string.decode("utf-8", "ignore"),
                keep_blank_values=False,
            )
            raw_vals = parsed.get("products", []) or raw_vals
        except Exception:
            pass

    out = []
    for val in raw_vals:
        for part in str(val).split(","):
            part = part.strip()
            if part:
                out.append(part)

    seen = set()
    uniq = []
    for part in out:
        if part not in seen:
            seen.add(part)
            uniq.append(part)
    return uniq

def _dedupe_pairs(pairs):
    seen=set()
    out=[]
    for arc,key in pairs:
        if key not in seen:
            seen.add(key)
            out.append((arc,key))
    return out

def _collect_cluster_pairs(cluster: str, selected_keys=None):
    cluster = utils._strip_quotes(cluster)
    selected = set(selected_keys or ALL_CLUSTER_KEYS)
    pairs = []
    top = f"{cluster}/cluster/"
    variants = utils._cluster_name_variants(cluster)

    def add_first(arcname, candidates):
        key = utils._first_existing(candidates)
        if key:
            pairs.append((top + arcname, key))

    if "cluster_cat" in selected:
        add_first(f"{cluster}_cat.txt", [f"{utils.S3_CATALOGUES_PREFIX}{v}_cat.txt" for v in variants])
    if "cluster_cube" in selected:
        add_first(f"{cluster}.fits", [f"{utils.S3_CLUSTER_CUBES_PREFIX}{v}.fits" for v in variants])
    if "cluster_mask" in selected:
        add_first(f"{cluster}_mask.fits", [f"{utils.S3_CLUSTER_MASKS_PREFIX}{v}_mask.fits" for v in variants])
    if "cluster_mask_raw" in selected:
        add_first(f"{cluster}_mask-raw.fits", [f"{utils.S3_CLUSTER_MASKS_PREFIX}{v}_mask-raw.fits" for v in variants])
    for mi in (0,1,2):
        keyname=f"cluster_mom{mi}"
        if keyname in selected:
            add_first(f"{cluster}_mom{mi}.fits", [f"{utils.S3_CLUSTER_MOMS_PREFIX}{v}_mom{mi}.fits" for v in variants])

    gal_keys = [k for k in selected if k.startswith("gal_")]
    if gal_keys:
        try:
            cat = utils.load_sofia_catalogue(cluster)
        except Exception:
            cat = pd.DataFrame()
        if cat is not None and not cat.empty and "id" in cat.columns:
            gids = pd.to_numeric(cat["id"], errors="coerce").dropna().astype(int).tolist()
            wanted_suffixes = {GAL_SUFFIX_MAP[k] for k in gal_keys if k in GAL_SUFFIX_MAP}
            for gid in gids:
                for arc, key in utils._galaxy_package_pairs(cluster, gid):
                    if any(key.endswith(suf) for suf in wanted_suffixes):
                        pairs.append((arc, key))

    return _dedupe_pairs(pairs)

def _collect_galaxy_pairs(cluster: str, gid: str, selected_keys=None):
    selected = set(selected_keys or ALL_GALAXY_KEYS)
    wanted_suffixes = {GAL_SUFFIX_MAP[k] for k in selected if k in GAL_SUFFIX_MAP}
    pairs = []
    for arc, key in utils._galaxy_package_pairs(cluster, gid):
        if any(key.endswith(suf) for suf in wanted_suffixes):
            pairs.append((arc, key))
    return _dedupe_pairs(pairs)

# -----------------------------------------------------------------------------
# Download: Cluster bundle
# -----------------------------------------------------------------------------
@app.route("/download/cluster/<cluster>", methods=["GET"])
def download_cluster(cluster: str):
    cluster = utils._strip_quotes(cluster)
    mode = (request.args.get("mode") or "").strip().lower()
    if mode not in {"all", "selected"}:
        return _render_download_select(
            entity_title=f"{cluster} cluster bundle",
            subtitle="Choose which cluster and galaxy-cubelet products to include.",
            back_href=url_for("cluster_detail", cluster_id=cluster),
            back_label="Back to cluster",
            action_url=url_for("download_cluster", cluster=cluster),
            action_url_abs=url_for("download_cluster", cluster=cluster, _external=True),
            filename=f"{cluster}_cluster_products.zip",
            groups=CLUSTER_PRODUCT_GROUPS,
            default_checked=ALL_CLUSTER_KEYS,
            cli_note="Use the generated shell-script commands below to download everything or a selected subset directly to your machine.",
        )

    selected = _requested_product_keys() if mode == "selected" else list(ALL_CLUSTER_KEYS)
    logger.info("[%s] download_cluster(): selected=%s", _ts(), selected)
    if mode == "selected" and not selected:
        abort(400, "No products selected.")
    pairs = _collect_cluster_pairs(cluster, selected)
    if not pairs:
        abort(404, f"No products found for {cluster} with the selected filters")

    if (request.args.get("format") or "").strip().lower() == "script":
        script_name = f"{cluster}_cluster_products_download.sh"
        script = _pairs_to_shell_script(pairs, script_name, f"MGCLS-HI :: {cluster} cluster download")
        return Response(
            script,
            mimetype="text/x-sh; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{script_name}"'},
        )

    zipbuf = utils.build_zip_from_pairs(pairs, title=f"{cluster}-cluster")
    filename = f"{cluster}_cluster_products.zip"
    return send_file(
        zipbuf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename
    )


# -----------------------------------------------------------------------------
# Download: Galaxy bundle for a detection
# -----------------------------------------------------------------------------
def _rename_arcname_to_mktcs(arcname: str, gname: str) -> str:
    """
    Rename archive member like:
      Abell-194/cubelets/Abell-194_2_mom1.fits → MKTCS-HI J..._mom1.fits
    Only when gname is non-empty; else return original arcname.
    """
    gname = (gname or "").strip()
    if not gname:
        return arcname

    base = os.path.basename(arcname)          # Abell-194_2_mom1.fits
    root, ext = os.path.splitext(base)        # Abell-194_2_mom1 , .fits
    typ = root.split("_")[-1].lower()

    if base.lower().endswith("_spec.txt"):
        return f"{gname}_spec.txt"

    allowed = {"chan", "cube", "mask", "mom0", "mom1", "mom2", "pv", "snr"}
    if typ in allowed and ext.lower() == ".fits":
        return f"{gname}_{typ}.fits"

    return arcname


@app.route("/download/galaxy/<cluster>/<gid>", methods=["GET"])
def download_galaxy(cluster: str, gid: str):
    cluster = utils._strip_quotes(cluster)
    gid_txt = f"{int(gid)}" if str(gid).strip().isdigit() else str(gid)

    gname = ""
    try:
        cat = utils.load_sofia_catalogue(cluster)
        if "id" in cat.columns and "name" in cat.columns:
            sub = cat.loc[(cat["id"].astype(str) == gid_txt)]
            if not sub.empty:
                gname = str(sub.iloc[0]["name"])
    except Exception:
        pass

    mode = (request.args.get("mode") or "").strip().lower()
    if mode not in {"all", "selected"}:
        entity_title = gname if gname else f"{cluster} galaxy ID {gid_txt}"
        subtitle = f"Choose which products to include for {cluster} ID {gid_txt}."
        filename = f"{(gname.replace(' ','_') if gname else f'{cluster}_id{gid_txt}')}_galaxy.zip"
        return _render_download_select(
            entity_title=entity_title,
            subtitle=subtitle,
            back_href=url_for("cluster_detail", cluster_id=cluster),
            back_label="Back to cluster",
            action_url=url_for("download_galaxy", cluster=cluster, gid=gid_txt),
            action_url_abs=url_for("download_galaxy", cluster=cluster, gid=gid_txt, _external=True),
            filename=filename,
            groups=GALAXY_PRODUCT_GROUPS,
            default_checked=ALL_GALAXY_KEYS,
            cli_note="Use the generated shell-script commands below to download everything or a selected subset directly to your machine.",
        )

    selected = _requested_product_keys() if mode == "selected" else list(ALL_GALAXY_KEYS)
    logger.info("[%s] download_galaxy(): selected=%s", _ts(), selected)
    if mode == "selected" and not selected:
        abort(400, "No products selected.")
    pairs = _collect_galaxy_pairs(cluster, gid_txt, selected)
    if not pairs:
        abort(404, f"No cubelets/products found for {cluster} ID {gid_txt}")

    if (request.args.get("format") or "").strip().lower() == "script":
        display_name = gname if gname else f"{cluster} ID {gid_txt}"
        script_name = f"{(gname.replace(' ', '_') if gname else f'{cluster}_id{gid_txt}')}_download.sh"
        script = _pairs_to_shell_script(pairs, script_name, f"MGCLS-HI :: {display_name} direct download")
        return Response(
            script,
            mimetype="text/x-sh; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{script_name}"'},
        )

    if gname:
        renamed = [(_rename_arcname_to_mktcs(arc, gname), key) for (arc, key) in pairs]
        title = f"{gname.replace(' ','_')}-galaxy"
    else:
        renamed = pairs
        title = f"{cluster}_id{gid_txt}-galaxy"

    zipbuf = utils.build_zip_from_pairs(renamed, title=title)
    filename = f"{title}.zip"
    return send_file(
        zipbuf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename
    )

@app.route("/download/cluster/<cluster>/galaxies", methods=["GET"])
def download_all_galaxies(cluster: str):
    cluster = utils._strip_quotes(cluster)
    logger.info("[%s] download_all_galaxies: start cluster=%s", _ts(), cluster)

    # Load SoFiA catalogue to enumerate IDs
    try:
        cat = utils.load_sofia_catalogue(cluster)
    except Exception as e:
        logger.warning("[%s] download_all_galaxies: cannot read SoFiA catalogue (%s)", _ts(), e)
        abort(404, f"No SoFiA catalogue found for {cluster}")

    if cat is None or cat.empty or "id" not in cat.columns:
        abort(404, f"No galaxy IDs found for {cluster}")

    ids = pd.to_numeric(cat["id"], errors="coerce").dropna().astype(int).tolist()
    if not ids:
        abort(404, f"No usable galaxy IDs found for {cluster}")

    pairs_all = []
    for gid in ids:
        pairs_all.extend(utils._galaxy_package_pairs(cluster, gid))

    if not pairs_all:
        abort(404, f"No cubelets/products found for any galaxies in {cluster}")

    if (request.args.get("format") or "").strip().lower() == "script":
        script_name = f"{cluster}_all_galaxies_download.sh"
        script = _pairs_to_shell_script(pairs_all, script_name, f"MGCLS-HI :: {cluster} all-galaxies download")
        return Response(
            script,
            mimetype="text/x-sh; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{script_name}"'},
        )

    zipbuf = utils.build_zip_from_pairs(pairs_all, title=f"{cluster}-all-galaxies")
    filename = f"{cluster}_all_galaxies.zip"
    logger.info("[%s] download_all_galaxies: returning ZIP=%s size=%dB", _ts(), filename, len(zipbuf.getbuffer()))
    return send_file(
        zipbuf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename
    )


# -----------------------------------------------------------------------------
# Simple stubs for navbar pages (optional – keep if your templates exist)
# -----------------------------------------------------------------------------
@app.route("/contact")
def contact():
    try:
        return render_template("contact.html")
    except Exception:
        return "Contact page", 200


@app.route("/people")
def people():
    try:
        return render_template("people.html")
    except Exception:
        return "People page", 200


@app.route("/about")
def about():
    try:
        return render_template("about.html")
    except Exception:
        return "About page", 200


@app.route("/publications")
def publications():
    pubs, warns, chart = utils.load_publications()
    return render_template(
        "publications.html",
        pubs=pubs,
        warns=warns,
        chart_labels=chart["labels"],
        chart_counts=chart["counts"],
    )



@app.route("/help")
def help():
    try:
        return render_template("help.html")
    except Exception:
        return "Help page", 200


# -----------------------------------------------------------------------------
# SEO helpers
# -----------------------------------------------------------------------------
@app.route("/robots.txt", methods=["GET"])
def robots_txt():
    base_url = _public_base_url()
    lines = [
        "User-agent: *",
        "Allow: /",
    ]
    if base_url:
        lines.append(f"Sitemap: {base_url}/sitemap.xml")
    lines.append("")
    return Response("\n".join(lines), mimetype="text/plain; charset=utf-8")


@app.route("/sitemap.xml", methods=["GET"])
def sitemap_xml():
    base_url = _public_base_url()
    if not base_url:
        abort(500, "SITE_URL is not configured.")

    static_paths = [
        url_for("index"),
        url_for("search"),
        url_for("about"),
        url_for("publications"),
        url_for("help"),
        url_for("people"),
        url_for("contact"),
    ]

    urls = [f"{base_url}{path}" for path in static_paths]
    try:
        master = utils.load_master_catalogue()
        if "ID" in master.columns:
            for cluster_id in sorted(master["ID"].dropna().astype(str).unique().tolist()):
                urls.append(f"{base_url}{url_for('cluster_detail', cluster_id=cluster_id)}")
    except Exception:
        logger.exception("[%s] sitemap_xml(): failed to expand cluster detail URLs", _ts())

    unique_urls = []
    seen = set()
    for item in urls:
        if item in seen:
            continue
        seen.add(item)
        unique_urls.append(item)

    lastmod = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for item in unique_urls:
        body.extend([
            "  <url>",
            f"    <loc>{item}</loc>",
            f"    <lastmod>{lastmod}</lastmod>",
            "  </url>",
        ])
    body.append("</urlset>")
    return Response("\n".join(body), mimetype="application/xml; charset=utf-8")


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "ts": _ts()}


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = bool(int(os.environ.get("FLASK_DEBUG", "1")))
    logger.info("[%s] starting Flask on %s:%d debug=%s", _ts(), host, port, debug)
    app.run(host=host, port=port, debug=debug)
