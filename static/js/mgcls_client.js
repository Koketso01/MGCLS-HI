<script>
// ====================== Utilities ======================
async function httpText(url) {
  const r = await fetch(url, { mode: "cors" });
  if (!r.ok) throw new Error(`GET ${r.status} ${url}`);
  return await r.text();
}
async function httpHeadExists(url) {
  try {
    const r = await fetch(url, { method: "HEAD", mode: "cors" });
    return r.ok;
  } catch (e) {
    return false;
  }
}
function parseTableText(txt) {
  // Try to detect delimiter from header line
  const lines = txt.split(/\r?\n/).filter(x => x.trim().length);
  if (!lines.length) return { header: [], rows: [] };
  const headerLine = lines[0];
  const delim = headerLine.includes("\t") ? "\t" : /\s{2,}/; // tab or multi-space
  const header = headerLine.split(delim).map(h => h.trim());
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim().startsWith("#")) continue;
    const cols = lines[i].split(delim);
    if (cols.length < 2) continue;
    const obj = {};
    header.forEach((h, idx) => obj[h] = (cols[idx] ?? "").trim());
    rows.push(obj);
  }
  return { header, rows };
}
function getColName(headers, candidates) {
  const lower = headers.map(h => h.toLowerCase());
  for (const c of candidates) {
    const idx = lower.indexOf(c.toLowerCase());
    if (idx >= 0) return headers[idx];
  }
  // fuzzy partial
  for (const h of headers) {
    for (const c of candidates) {
      if (h.toLowerCase().includes(c.toLowerCase())) return h;
    }
  }
  return null;
}
function toNumber(x) {
  if (x === null || x === undefined) return null;
  const n = Number(String(x).replace(/[^\d\.\-+eE]/g, ''));
  return Number.isFinite(n) ? n : null;
}
function parseRA(input) {
  if (!input) return null;
  const t = String(input).trim();
  if (t.includes(":") || /[hms]/i.test(t)) {
    const parts = t.replace(/h/ig,":").replace(/m/ig,":").replace(/s/ig,"").split(":").map(Number);
    if (parts.length < 3 || parts.some(isNaN)) return null;
    const [h,m,s] = parts;
    return 15.0*(Math.abs(h)+m/60+s/3600)*(h>=0?1:-1);
  }
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}
function parseDec(input) {
  if (!input) return null;
  const t = String(input).trim();
  if (t.includes(":") || /[dms]/i.test(t)) {
    const parts = t.replace(/d/ig,":").replace(/m/ig,":").replace(/s/ig,"").split(":").map(Number);
    if (parts.length < 3 || parts.some(isNaN)) return null;
    let [d,m,s] = parts;
    const sign = String(parts[0]).startsWith("-") ? -1 : 1;
    d = Math.abs(d);
    return sign*(d + m/60 + s/3600);
  }
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}
function angsepDeg(ra1, dec1, ra2, dec2) {
  const d2r = Math.PI/180;
  const a1 = ra1*d2r, b1 = dec1*d2r, a2 = ra2*d2r, b2 = dec2*d2r;
  const cos = Math.sin(b1)*Math.sin(b2) + Math.cos(b1)*Math.cos(b2)*Math.cos(Math.abs(a1-a2));
  return Math.acos(Math.min(1,Math.max(-1,cos)))/d2r;
}

// ====================== MGCLS master fetch & shape ======================
async function loadMGCLS() {
  const txt = await httpText(window.MGCLS_CFG.mgclsTxt()); // "MGCLS HI.txt"
  const { header, rows } = parseTableText(txt);
  // pick columns
  const C = (cands) => getColName(header, cands);
  const c_cluster = C(["ID", "Name", "Cluster", "CLUSTER"]);
  const c_mgcls   = C(["MGCLS_Name","MGCLS","MGCLSName"]);
  const c_ra      = C(["RA"]);
  const c_dec     = C(["DEC","Dec"]);
  const c_mz      = C(["M_Z","MZ","Z"]);
  const c_sbid    = C(["SBID"]);
  const c_cap     = C(["CAPTURE_ID","CAPTURE ID","CAPTURE"]);
  const c_sigma   = C(["SIGMA_V","SIGMA-V","Sigma_V"]);
  const c_sofia   = C(["SOFIA_DETS","SOFIA","SOFIA_DET"]);
  const c_rms     = C(["RMS"]);
  const c_vmin    = C(["V_min","Vmin"]);
  const c_vmax    = C(["V_max","Vmax"]);
  const c_bmin    = C(["BMIN"]);
  const c_bmaj    = C(["BMAJ"]);
  const c_bpa     = C(["BPA"]);
  const c_counter = C(["COUNTERPART"]);
  // shape
  const out = rows.map(r => {
    const ra = r[c_ra] ?? "";
    const dec= r[c_dec] ?? "";
    return {
      cluster: (r[c_cluster] ?? "").replace(/^"+|"+$/g,""),
      MGCLS_Name: (r[c_mgcls] ?? "").replace(/^"+|"+$/g,""),
      RA: ra, Dec: dec,
      RA_deg: parseRA(ra), Dec_deg: parseDec(dec),
      M_Z: r[c_mz] ?? "",
      SBID: r[c_sbid] ?? "",
      CAPTURE_ID: r[c_cap] ?? "",
      SIGMA_V: r[c_sigma] ?? "",
      SOFIA_DETS: r[c_sofia] ?? "",
      RMS: r[c_rms] ?? "",
      V_min: r[c_vmin] ?? "",
      V_max: r[c_vmax] ?? "",
      BMIN: r[c_bmin] ?? "",
      BMAJ: r[c_bmaj] ?? "",
      BPA: r[c_bpa] ?? "",
      COUNTERPART: r[c_counter] ?? ""
    };
  });
  return out;
}

// ====================== Availability (client-side HEAD) ======================
async function probeClusterAvailability(cluster) {
  const cc = window.MGCLS_CFG.cc;
  // Required cluster-level artifacts (catalogue count for staging, fits/eps for availability)
  const checks = {
    main_cube: cc(cluster, ".fits"),
    chan:      cc(cluster, "_chan.fits"),
    mask:      cc(cluster, "_mask.fits"),
    mom0:      cc(cluster, "_mom0.fits"),
    mom1:      cc(cluster, "_mom1.fits"),
    mom2:      cc(cluster, "_mom2.fits"),
    noise:     cc(cluster, "_noise.fits"),
    rel:       cc(cluster, "_rel.eps"),
    skellam:   cc(cluster, "_skellam.eps"),
    catalogue: window.MGCLS_CFG.clusterCatalogue(cluster)
  };
  const keys = Object.keys(checks);
  const bools = await Promise.all(keys.map(k => httpHeadExists(checks[k])));
  const res = {};
  keys.forEach((k, i) => res[k] = bools[i]);
  // (Optional) we could try to detect cubelets folder, but listing is usually blocked; treat cubelets as optional.
  return res;
}
function summarizeAvailability(av) {
  if (!av) return { label: "Not available yet", cls: "badge bg-secondary" };
  const required = [av.catalogue, av.main_cube, av.chan, av.mask, av.mom0, av.mom1, av.mom2, av.noise, av.rel, av.skellam];
  const present = required.filter(Boolean).length;
  if (present === 0) return { label: "Not available yet", cls: "badge bg-secondary" };
  if (present === required.length) return { label: "Available", cls: "badge bg-success" };
  return { label: "Partial", cls: "badge bg-warning text-dark" };
}

// ====================== UI: table, filters, figures ======================
function renderClustersTable(rows, targetId="clusterResults") {
  const el = document.getElementById(targetId);
  if (!el) return;
  const labels = ["Cluster","Redshift","RA (J2000)","Dec (J2000)","SBID","Capture ID","σᵥ","SoFiA H I detections","RMS","Velocity range","BMIN","BMAJ","BPA","MGCLS name","AWS","Figures","Download"];
  const units  = ["—","—","hh:mm:ss.s","dd:mm:ss.s","—","—","km/s","—","(as in file)","km/s","arcsec","arcsec","deg","—","—","—","—"];
  let html = `<div class="table-responsive"><table class="table table-striped table-hover table-sm align-middle">
    <thead class="table-light">
      <tr>${labels.map(h=>`<th>${h}</th>`).join("")}</tr>
      <tr>${units.map(u=>`<th class="text-muted small">${u}</th>`).join("")}</tr>
    </thead><tbody>`;
  for (const r of rows) {
    const vmin = r.V_min || "", vmax = r.V_max || "";
    const figs = `
      <div class="d-flex gap-2 align-items-center">
        <img src="${window.MGCLS_CFG.clusterFig(r.cluster, "completeness")}" alt="completeness" style="height:52px" onerror="this.style.display='none'">
        <img src="${window.MGCLS_CFG.clusterFig(r.cluster, "mom0")}"         alt="mom0"         style="height:52px" onerror="this.style.display='none'">
      </div>`;
    const rowId = `aws-${encodeURIComponent(r.cluster)}`;
    html += `<tr>
      <td>${r.cluster}</td>
      <td>${r.M_Z ?? ""}</td>
      <td>${r.RA ?? ""}</td>
      <td>${r.Dec ?? ""}</td>
      <td>${r.SBID ?? ""}</td>
      <td>${r.CAPTURE_ID ?? ""}</td>
      <td>${r.SIGMA_V ?? ""}</td>
      <td>${r.SOFIA_DETS ?? ""}</td>
      <td>${r.RMS ?? ""}</td>
      <td>${(vmin && vmax) ? `${vmin}…${vmax}` : ""}</td>
      <td>${r.BMIN ?? ""}</td>
      <td>${r.BMAJ ?? ""}</td>
      <td>${r.BPA ?? ""}</td>
      <td>${r.MGCLS_Name ?? ""}</td>
      <td><span id="${rowId}" class="badge bg-secondary">Checking…</span></td>
      <td>${figs}</td>
      <td>
        <button class="btn btn-sm btn-outline-primary" onclick="MGCLS.showClusterLinks('${encodeURIComponent(r.cluster)}')">
          Download links…
        </button>
      </td>
    </tr>`;
  }
  html += `</tbody></table></div>`;
  el.innerHTML = html;

  // After render, asynchronously probe availability for *visible* rows
  rows.forEach(async (r) => {
    const badge = document.getElementById(`aws-${encodeURIComponent(r.cluster)}`);
    if (!badge) return;
    try {
      const av = await probeClusterAvailability(r.cluster);
      const s = summarizeAvailability(av);
      badge.className = `badge ${s.cls.split(" ").slice(1).join(" ")}`; // keep 'badge'
      badge.textContent = s.label;
      badge.title = Object.entries(av).map(([k,v])=>`${k}: ${v?'✓':'✗'}`).join("\n");
    } catch (e) {
      badge.className = "badge bg-secondary";
      badge.textContent = "Unknown";
    }
  });
}

function filterClusters(all, form) {
  if (!form) return all;
  const get = (name) => (form.elements[name]?.value || "").trim();
  const name = get("name").toLowerCase();
  const mgcls = get("mgcls_name").toLowerCase();
  const sbid  = get("sbid").toLowerCase();
  const cap   = get("capture_id").toLowerCase();

  // spatial (radius in arcmin)
  const ra = parseRA(get("ra"));
  const dec= parseDec(get("dec"));
  let rad = toNumber(get("radius"));
  rad = Number.isFinite(rad) ? (rad/60.0) : null; // arcmin -> degrees

  // velocity (center ± tol)
  const v = toNumber(get("velocity"));
  const vt= toNumber(get("vel_tol"));
  const hasV = Number.isFinite(v) && Number.isFinite(vt);
  const vmin = hasV ? v - vt : null;
  const vmax = hasV ? v + vt : null;

  return all.filter(r => {
    if (name && !String(r.cluster).toLowerCase().includes(name)) return false;
    if (mgcls && !String(r.MGCLS_Name||"").toLowerCase().includes(mgcls)) return false;
    if (sbid && !String(r.SBID||"").toLowerCase().includes(sbid)) return false;
    if (cap  && !String(r.CAPTURE_ID||"").toLowerCase().includes(cap)) return false;

    if (ra!=null && dec!=null && rad!=null && r.RA_deg!=null && r.Dec_deg!=null) {
      const sep = angsepDeg(ra, dec, r.RA_deg, r.Dec_deg);
      if (sep > rad) return false;
    }
    if (hasV) {
      const rmin = toNumber(r.V_min), rmax = toNumber(r.V_max);
      if (rmin!=null && rmax!=null) {
        const ok = !(vmax < rmin || rmax < vmin);
        if (!ok) return false;
      }
    }
    return true;
  });
}

async function initLandingClusters(options = { formId: "clusterSearchForm", targetId: "clusterResults" }) {
  const all = await loadMGCLS();
  const form = document.getElementById(options.formId);
  const render = () => renderClustersTable(filterClusters(all, form), options.targetId);
  if (form) {
    form.addEventListener("submit", e => { e.preventDefault(); render(); });
    form.addEventListener("input",  e => render());
  }
  render();
}

// ====================== “Download links” (no Lambda yet) ======================
function clusterProductUrls(cluster) {
  // cluster-level
  const cc = window.MGCLS_CFG.cc;
  const urls = [
    cc(cluster,".fits"),
    cc(cluster,"_cat.txt"),
    cc(cluster,"_chan.fits"),
    cc(cluster,"_mask.fits"),
    cc(cluster,"_mom0.fits"),
    cc(cluster,"_mom1.fits"),
    cc(cluster,"_mom2.fits"),
    cc(cluster,"_noise.fits"),
    cc(cluster,"_rel.eps"),
    cc(cluster,"_skellam.eps"),
  ];
  return urls;
}
async function buildClusterLinksHtml(cluster) {
  const urls = clusterProductUrls(cluster);
  // Filter to those that actually exist (HEAD check). If slow, remove this probe and list all.
  const exists = await Promise.all(urls.map(u => httpHeadExists(u)));
  const present = urls.filter((u,i)=>exists[i]);
  const missing = urls.filter((u,i)=>!exists[i]);
  let html = `<h6>Cluster products for <code>${cluster}</code></h6>`;
  if (present.length) {
    html += `<p><strong>Available:</strong></p><ul class="small">` + present.map(u=>`<li><a href="${u}" target="_blank" rel="noopener">${u}</a></li>`).join("") + `</ul>`;
  }
  if (missing.length) {
    html += `<p class="text-muted"><strong>Missing (not staged yet):</strong></p><ul class="small text-muted">` + missing.map(u=>`<li>${u}</li>`).join("") + `</ul>`;
  }
  // Cubelets note for now
  html += `<p class="mt-2"><em>Galaxy cubelets (per-ID) links are available from the Galaxy search page; packaging into a ZIP will come once IAM is granted.</em></p>`;
  return html;
}
window.MGCLS = window.MGCLS || {};
window.MGCLS.showClusterLinks = async function(encodedCluster) {
  const cluster = decodeURIComponent(encodedCluster);
  const box = document.getElementById("downloadLinksBox");
  if (box) {
    box.innerHTML = `<div class="text-muted">Checking S3…</div>`;
    const html = await buildClusterLinksHtml(cluster);
    box.innerHTML = html;
    const modal = new bootstrap.Modal(document.getElementById('downloadLinksModal'));
    modal.show();
  } else {
    alert("Missing #downloadLinksModal in this page. Please add it (see snippet).");
  }
};

</script>
