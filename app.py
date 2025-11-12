
from __future__ import annotations

from typing import Dict, List
from flask import Flask, render_template, request, send_file, abort

import utils
# in app.py, once at startup, after importing utils
try:
    utils.load_master_catalogue.cache_clear()
except Exception:
    pass

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

def _form_dict(form) -> Dict[str, str]:
    out = {}
    for k in form.keys():
        vals = form.getlist(k)
        out[k] = vals[0] if len(vals) == 1 else ",".join(vals)
    return out

# -------------------------
# Landing page (cluster table)
# -------------------------
@app.route("/", methods=["GET"])
def index():
    q = (request.args.get("q") or "").strip()
    page = int(request.args.get("page", "1") or 1)
    page_size = int(request.args.get("page_size", "25") or 25)

    df = utils.load_master_catalogue()
    spec = utils.build_landing_table_spec(df, page=page, page_size=page_size, quick_filter=q)

    header_labels = spec.get("header_labels", [])
    header_units  = spec.get("header_units", [])
    rows          = spec.get("rows", [])
    pagination    = spec.get("pagination", {
        "page": 1, "page_size": len(rows), "total": len(rows), "num_pages": 1
    })

    ctx = {
        "header_labels": header_labels,
        "header_units":  header_units,
        "rows":          rows,
        "pagination":    pagination,
        "quick_filter":  q,
    }
    return render_template("index.html", **ctx)

# -----------
# Search page
# -----------
@app.route("/search", methods=["GET", "POST"])
def search():
    df = utils.load_master_catalogue()
    form = _form_dict(request.form if request.method == "POST" else request.args)

    did_submit = (request.method == "POST")
    target_choice = (form.get("target_choice") or "clusters").strip()  # clusters | galaxies
    cluster_choices = sorted(df["ID"].astype(str).dropna().unique().tolist()) if "ID" in df.columns else []

    page = int(form.get("page", "1") or "1")
    page_size = int(form.get("page_size", "25") or "25")

    # Initialize outputs
    cluster_rows = cluster_header_labels = cluster_header_units = None
    cluster_previews = None
    cluster_errors, cluster_warnings = [], []

    galaxy_rows = galaxy_header_labels = galaxy_header_units = None
    galaxy_previews = None
    galaxy_row_meta = None
    galaxy_errors, galaxy_warnings = [], []

    if did_submit:
        if target_choice == "galaxies":
            # ---- GALAXY FIELD NAMES (accept both g_* and legacy names) ----
            galaxy_name = form.get("galaxy_name") or form.get("name") or ""

            g_ra   = form.get("g_ra")  or form.get("ra")  or ""
            g_dec  = form.get("g_dec") or form.get("dec") or ""

            # Combine radius numeric + unit into a single parseable string for utils
            g_radius_val  = (form.get("g_radius") or form.get("radius") or "").strip()
            g_radius_unit = (form.get("g_radius_unit") or form.get("radius_unit") or "arcmin").strip()
            g_radius_text = f"{g_radius_val} {g_radius_unit}".strip() if g_radius_val else ""

            g_vc   = form.get("g_vel_center") or form.get("velocity") or ""
            g_vtol = form.get("g_vel_tol")    or form.get("vel_tol")  or ""

            # Optional cluster scope (multi-select → comma-joined in _form_dict)
            scope_raw = form.get("cluster_scope", "")
            cluster_scope = [s.strip() for s in scope_raw.split(",") if s.strip()] if scope_raw else None

            gspec, galaxy_previews, galaxy_errors, galaxy_warnings = utils.search_galaxies(
                df,
                name_query=galaxy_name,
                ra_txt=g_ra,
                dec_txt=g_dec,
                radius_text=g_radius_text,
                vel_center_kms=g_vc,
                vel_tol_kms=g_vtol,
                cluster_scope=cluster_scope,
                page=page,
                page_size=page_size,
            )
            if gspec:
                galaxy_header_labels = gspec.get("header_labels")
                galaxy_header_units  = gspec.get("header_units")
                galaxy_rows          = gspec.get("rows")
                galaxy_row_meta      = gspec.get("row_meta", [])

        else:
            # ---- CLUSTER FIELD NAMES ----
            radius_val  = (form.get("radius") or "").strip()
            radius_unit = (form.get("radius_unit") or "arcmin").strip()
            radius_text = f"{radius_val} {radius_unit}".strip() if radius_val else ""

            cspec, cluster_previews, cluster_errors, cluster_warnings = utils.search_clusters(
                df,
                name_query=form.get("name_query", ""),
                mgcls_query=form.get("mgcls_query", ""),
                sbid=form.get("sbid", ""),
                capture_id=form.get("capture_id", ""),
                ra_txt=form.get("ra", ""),
                dec_txt=form.get("dec", ""),
                radius_text=radius_text,  # unit-aware
                vel_center=form.get("vel_center", ""),
                vel_tol=form.get("vel_tol", ""),
                vel_min=form.get("vel_min", ""),
                vel_max=form.get("vel_max", ""),
                page=page,
                page_size=page_size,
            )
            if cspec:
                cluster_header_labels = cspec.get("header_labels")
                cluster_header_units  = cspec.get("header_units")
                cluster_rows          = cspec.get("rows")

    return render_template(
        "search.html",
        did_submit=did_submit,
        target_choice=target_choice,
        cluster_choices=cluster_choices,
        # clusters
        cluster_rows=cluster_rows,
        cluster_header_labels=cluster_header_labels,
        cluster_header_units=cluster_header_units,
        cluster_previews=cluster_previews,
        cluster_errors=cluster_errors,
        cluster_warnings=cluster_warnings,
        # galaxies
        galaxy_rows=galaxy_rows,
        galaxy_header_labels=galaxy_header_labels,
        galaxy_header_units=galaxy_header_units,
        galaxy_previews=galaxy_previews,
        galaxy_row_meta=galaxy_row_meta or [],
        galaxy_errors=galaxy_errors,
        galaxy_warnings=galaxy_warnings,
        # echo form back to template
        form=form,
    )

# ----------------
# DOWNLOAD ROUTES
# ----------------
@app.route("/download/cluster", methods=["GET"])
def download_cluster():
    cluster = (request.args.get("cluster") or "").strip()
    if not cluster:
        abort(400, "Missing cluster")
    pairs = utils._cluster_package_pairs(cluster)
    if not pairs:
        abort(404, f"No downloadable cluster products found for {cluster}")
    buf = utils.build_zip_from_pairs(pairs, title=f"{cluster}_cluster_package")
    return send_file(buf, as_attachment=True, download_name=f"{cluster}_cluster_package.zip", mimetype="application/zip")

@app.route("/download/galaxy", methods=["GET"])
def download_galaxy():
    cluster = (request.args.get("cluster") or "").strip()
    gid     = (request.args.get("gid") or "").strip()
    if not cluster or not gid:
        abort(400, "cluster and gid are required")
    pairs = utils._galaxy_package_pairs(cluster, gid)
    if not pairs:
        abort(404, f"No cubelets/products found for {cluster} ID {gid}")
    buf = utils.build_zip_from_pairs(pairs, title=f"{cluster}_{gid}_cubelets")
    return send_file(buf, as_attachment=True, download_name=f"{cluster}_{gid}_cubelets.zip", mimetype="application/zip")

# ------------
# Static pages
# ------------
@app.route("/people", methods=["GET"])
def people():
    return render_template("people.html")

@app.route("/help", methods=["GET"])
def help():
    return render_template("help.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@app.route("/publications", methods=["GET"])
def publications():
    return render_template("publications.html")

if __name__ == "__main__":
    import os
    print("\n=== URL MAP ===")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        print(f"{rule.endpoint:20s} {','.join(rule.methods):20s} {rule.rule}")
    print("===============\n")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))

app.config["PREFERRED_URL_SCHEME"] = "https"

"""
#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import sys
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    abort,
)

import logging

import utils  # our helpers (parsing, S3 fetches, searching, zipping)


# -----------------------------------------------------------------------------
# Flask app + logging
# -----------------------------------------------------------------------------
app = Flask(__name__)
# If you use flash() anywhere, set a secret key:
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-not-secret")

# Console logging with timestamps
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s.%(msecs)03dZ [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("mgcls-hi")

def _ts() -> str:
    #UTC ISO8601 timestamp (for consistent logs).
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@app.before_request
def _trace_inbound() -> None:
    #Trace every request entry with method, path, args and tiny body sample.
    try:
        if request.method == "GET":
            logger.info(
                "[%s] INBOUND GET %s args=%s",
                _ts(), request.path, dict(request.args),
            )
        elif request.method == "POST":
            # Safely read form fields for trace (avoid large bodies)
            form_preview = {k: request.form.get(k) for k in request.form.keys()}
            logger.info(
                "[%s] INBOUND POST %s form=%s",
                _ts(), request.path, form_preview,
            )
    except Exception:
        logger.exception("[%s] inbound-trace failed", _ts())


# -----------------------------------------------------------------------------
# Landing page: load master cluster catalogue and show the landing table
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
        logger.info(
            "[%s] index(): landing table rows=%d (page %d/%d, total=%d)",
            _ts(), len(spec["rows"]), spec["pagination"]["page"],
            spec["pagination"]["num_pages"], spec["pagination"]["total"]
        )
        return render_template("index.html", table_spec=spec)
    except Exception as e:
        logger.exception("[%s] index(): failed to render", _ts())
        abort(500, f"Landing page failed: {e}")


# -----------------------------------------------------------------------------
# Search page: clusters & galaxies (POST for form submission; GET to show)
# -----------------------------------------------------------------------------
@app.route("/search", methods=["GET", "POST"])
def search():
    did_submit = request.method == "POST"
    target_choice = (request.form.get("target_choice") if did_submit
                     else (request.args.get("target_choice") or "clusters"))

    logger.info("[%s] search(): target_choice=%s did_submit=%s",
                _ts(), target_choice, did_submit)

    # Always load master once per request (cached inside utils)
    master = utils.load_master_catalogue()

    tpl_ctx: Dict[str, Any] = {
        "did_submit": did_submit,
        "target_choice": target_choice,
        "cluster_choices": sorted(master["ID"].dropna().astype(str).unique().tolist()) if "ID" in master.columns else [],
        "form": {},
    }

    # ----------------- clusters -----------------
    if not did_submit or target_choice != "galaxies":
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

        # Convert radius + unit into a single text (utils parses “<num> <unit>”)
        radius_text = ""
        if form["radius"]:
            radius_text = f'{form["radius"]} {form.get("radius_unit","arcmin")}'

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

        tpl_ctx.update({
            "cluster_header_labels": spec["header_labels"],
            "cluster_header_units": spec["header_units"],
            "cluster_rows": spec["rows"],
            "cluster_previews": previews,
            "cluster_errors": errors,
            "cluster_warnings": warns,
        })

    # ----------------- galaxies -----------------
    if did_submit and target_choice == "galaxies":
        # Parse multi-select cluster scope
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

        radius_text = ""
        if form["g_radius"]:
            radius_text = f'{form["g_radius"]} {form.get("g_radius_unit","arcmin")}'

        logger.info("[%s] search(): galaxy form parsed -> %s", _ts(), form)

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

        logger.info("[%s] search(): galaxy results rows=%d previews=%d warns=%s",
                    _ts(), len(spec["rows"]), len(previews), warns)

        tpl_ctx.update({
            "galaxy_header_labels": spec["header_labels"],
            "galaxy_header_units": spec["header_units"],
            "galaxy_rows": spec["rows"],
            "galaxy_row_meta": spec.get("row_meta", []),  # for per-row download links
            "galaxy_previews": previews,
            "galaxy_errors": errors,
            "galaxy_warnings": warns,
        })

    return render_template("search.html", **tpl_ctx)


# -----------------------------------------------------------------------------
# Download: Cluster bundle (catalogue + cube + mask + mom0/1/2 … no figures)
# -----------------------------------------------------------------------------
@app.route("/download/cluster/<cluster>", methods=["GET"])
def download_cluster(cluster: str):
    cluster = utils._strip_quotes(cluster)
    logger.info("[%s] download_cluster: start cluster=%s", _ts(), cluster)

    pairs = utils._cluster_package_pairs(cluster)
    if not pairs:
        logger.info("[%s] download_cluster: no products for %s", _ts(), cluster)
        abort(404, f"No products found for {cluster}")

    logger.info("[%s] download_cluster: %d files to fetch", _ts(), len(pairs))
    zipbuf = utils.build_zip_from_pairs(pairs, title=f"{cluster}-cluster")
    filename = f"{cluster}_cluster_products.zip"
    logger.info("[%s] download_cluster: returning ZIP=%s size=%dB",
                _ts(), filename, len(zipbuf.getbuffer()))
    return send_file(
        zipbuf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename
    )


# -----------------------------------------------------------------------------
# Download: Galaxy bundle for a single detection (with friendly renaming)
# -----------------------------------------------------------------------------
def _rename_arcname_to_mktcs(arcname: str, gname: str) -> str:
    
    #Convert an archive member name like:
    #    Abell-194/cubelets/Abell-194_2_mom1.fits
    #to:
    #    MKTCS-HI J012345.67-012345.6_mom1.fits
    #(only when gname is non-empty; otherwise return original arcname).
    
    gname = (gname or "").strip()
    if not gname:
        return arcname

    base = os.path.basename(arcname)          # Abell-194_2_mom1.fits
    root, ext = os.path.splitext(base)        # Abell-194_2_mom1 , .fits

    # Determine “type” suffix after the last underscore
    # e.g., mom1, mom2, mask, cube, pv, snr, chan, spec
    typ = root.split("_")[-1].lower()

    # Map spec.txt to "spec.txt" (keeps ext)
    if base.lower().endswith("_spec.txt"):
        return f"{gname}_spec.txt"

    # All the FITS-like types keep their own label + .fits
    allowed = {"chan", "cube", "mask", "mom0", "mom1", "mom2", "pv", "snr"}
    if typ in allowed and ext.lower() == ".fits":
        return f"{gname}_{typ}.fits"

    # Fallback: keep original if we can’t identify it
    return arcname


@app.route("/download/galaxy/<cluster>/<gid>", methods=["GET"])
def download_galaxy(cluster: str, gid: str):
    cluster = utils._strip_quotes(cluster)
    gid_txt = str(int(gid)) if str(gid).strip().isdigit() else str(gid)
    logger.info("[%s] download_galaxy: start cluster=%s gid=%s", _ts(), cluster, gid_txt)

    # Build the raw (arcname, s3key) list from utils
    pairs = utils._galaxy_package_pairs(cluster, gid_txt)
    if not pairs:
        logger.info("[%s] download_galaxy: no products for cluster=%s id=%s",
                    _ts(), cluster, gid_txt)
        abort(404, f"No cubelets/products found for {cluster} ID {gid_txt}")

    # Try to load the SoFiA catalogue for this cluster to fetch the friendly name
    gname = ""
    try:
        cat = utils.load_sofia_catalogue(cluster)
        if "id" in cat.columns and "name" in cat.columns:
            sub = cat.loc[(cat["id"].astype(str) == gid_txt)]
            if not sub.empty:
                gname = str(sub.iloc[0]["name"])
    except Exception as e:
        logger.warning("[%s] download_galaxy: could not read SoFiA for rename (%s)", _ts(), e)

    if gname:
        renamed = [(_rename_arcname_to_mktcs(arc, gname), key) for (arc, key) in pairs]
        title = f"{gname.replace(' ','_')}-galaxy"
    else:
        renamed = pairs
        title = f"{cluster}_id{gid_txt}-galaxy"

    logger.info("[%s] download_galaxy: %d files (rename=%s)",
                _ts(), len(renamed), "yes" if gname else "no")

    zipbuf = utils.build_zip_from_pairs(renamed, title=title)
    filename = f"{title}.zip"
    logger.info("[%s] download_galaxy: returning ZIP=%s size=%dB",
                _ts(), filename, len(zipbuf.getbuffer()))
    return send_file(
        zipbuf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename
    )


# -----------------------------------------------------------------------------
# Health endpoint
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "ts": _ts()}


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow host/port overrides via env
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = bool(int(os.environ.get("FLASK_DEBUG", "1")))
    logger.info("[%s] starting Flask on %s:%d debug=%s", _ts(), host, port, debug)
    app.run(host=host, port=port, debug=debug)


"""