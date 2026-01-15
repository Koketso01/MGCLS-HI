from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    abort,
)

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

#S3 URLs for the navbar images
@app.context_processor
def inject_navbar_assets():
    base = f"{utils.S3_PREFIX.rstrip('/')}/navbar_figures/"
    return {
        "NAVBAR_IMG": utils.s3_http_url(utils.S3_BUCKET, utils.S3_REGION, base + "navbar_image.png"),
        "LOGO_IMG":   utils.s3_http_url(utils.S3_BUCKET, utils.S3_REGION, base + "logo.png"),
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
    fig_url = utils.cluster_figure_url(cluster_id)  # template will show placeholder if 404
    rows = utils.cluster_galaxy_rows(cluster_id)
    logger.info("[%s] cluster_detail: %s · galaxies=%d", _ts(), cluster_id, len(rows))
    return render_template(
        "cluster_detail.html",
        cluster_id=cluster_id,
        fig_url=fig_url,
        galaxy_rows=rows,
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
        "cluster_errors": [],
        "cluster_warnings": [],
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

        tpl_ctx.update({
            "cluster_header_labels": spec["header_labels"],
            "cluster_header_units": spec["header_units"],
            "cluster_rows": spec["rows"],
            "cluster_previews": previews,
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

        tpl_ctx.update({
            "galaxy_header_labels": spec["header_labels"],
            "galaxy_header_units": spec["header_units"],
            "galaxy_rows": spec["rows"],
            "galaxy_row_meta": spec.get("row_meta", []),
            "galaxy_previews": previews,
            "galaxy_errors": errors,
            "galaxy_warnings": warns,
        })

    return render_template("search.html", **tpl_ctx)


# -----------------------------------------------------------------------------
# Download: Cluster bundle
# -----------------------------------------------------------------------------
@app.route("/download/cluster/<cluster>", methods=["GET"])
def download_cluster(cluster: str):
    cluster = utils._strip_quotes(cluster)
    pairs = utils._cluster_package_pairs(cluster)
    if not pairs:
        abort(404, f"No products found for {cluster}")

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
    #gid_txt = str(int(gid)) if str(gid).strip().isdigit() else str(gid)

    gid_txt = f"{int(gid)}" if str(gid).strip().isdigit() else str(gid)


    pairs = utils._galaxy_package_pairs(cluster, gid_txt)
    if not pairs:
        abort(404, f"No cubelets/products found for {cluster} ID {gid_txt}")

    # Try to load SoFiA for a friendly name
    gname = ""
    try:
        cat = utils.load_sofia_catalogue(cluster)
        if "id" in cat.columns and "name" in cat.columns:
            sub = cat.loc[(cat["id"].astype(str) == gid_txt)]
            if not sub.empty:
                gname = str(sub.iloc[0]["name"])
    except Exception:
        pass

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
