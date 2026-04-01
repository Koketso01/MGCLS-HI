# MGCLS-HI Data Staging README

## Purpose

This document is a handover guide for the person responsible for staging new or updated MGCLS-HI data products to the public AWS S3 bucket and keeping the website-facing metadata consistent.

This README covers the **post-reduction / post-source-finding** part of the workflow. In other words:

1. data products have already been reduced and prepared,
2. source finding has already been run,
3. the staging person uploads the final products to the correct S3 locations,
4. the staging person updates the combined metadata file so the Flask site can discover and display the data.

---

## Critical principle

The website is tightly coupled to the S3 folder structure, file names, and the combined metadata file.

That means:

- do **not** rename folders casually,
- do **not** rename files casually,
- do **not** move products to a different S3 prefix unless the backend is updated too,
- do **not** change metadata column names unless the backend is updated too,
- do **not** split the metadata into separate files unless the backend is rewritten.

If naming changes, previews, search, cluster pages, and downloads can silently break.

---

## Public S3 location

The site expects data under this prefix:

`ratt-public-data / MGCLS_HI / Datasets / koketso-HI-MGCLS-data / HI-MGCLS /`

Treat `HI-MGCLS/` as the web-facing data root.

---

## Top-level S3 structure

The current web-facing structure is:

- `Catalogues/`
- `Cluster_catalogue/`
- `Cluster-Cubes/`
- `Cluster-Figures/`
- `Cluster-Masks/`
- `Cluster-Moms/`
- `Completeness-Curves/`
- `FooterFigures/`
- `Galaxy-Cubelets/`
- `Galaxy-Figures/`
- `Indices/`
- `navbar_figures/`
- `OtherFigures/`

Only some of these are critical for scientific data staging. The most important ones are described below.

---

## Core staging products required by the site

## 1. Combined metadata file

Location:

`Cluster_catalogue/MGCLS_HI_15clusters.txt`

This is the single most important file for the site.

It contains two logical sections in one text file:

### A. Cluster-level master catalogue at the top

This section stores one row per cluster (or field entry) with the cluster-level metadata used by the landing page, cluster search, and global stats.

### B. Appended per-cluster SoFiA catalogues below

Below the master table, each cluster-specific SoFiA catalogue is appended as a block beginning with a marker line like:

`#Abell-133`

or

`#Abell-194`

Each marker must be followed by that cluster’s source catalogue table.

The Flask backend reads this single text file from S3 and splits it into:

- the top master catalogue,
- the per-cluster SoFiA blocks.

If a cluster is staged but its SoFiA block is missing from this file, the cluster detail galaxy table and galaxy-level discovery logic will be incomplete or broken.

---

## 2. Cluster catalogue text files

Location:

`Catalogues/`

Naming rule:

`<cluster>_cat.txt`

Examples:

- `Abell-133_cat.txt`
- `Abell-168_cat.txt`
- `Abell-548_cat.txt`

These are used for cluster-level downloads.

---

## 3. Cluster cubes

Location:

`Cluster-Cubes/`

Naming rule:

`<cluster>.fits`

Examples:

- `Abell-133.fits`
- `Abell-168.fits`
- `Abell-548.fits`

These are the main cluster cubes exposed by the website.

---

## 4. Cluster masks

Location:

`Cluster-Masks/`

Naming rules:

- `<cluster>_mask.fits`
- `<cluster>_mask-raw.fits`

Examples:

- `Abell-133_mask.fits`
- `Abell-133_mask-raw.fits`
- `Abell-548_mask.fits`
- `Abell-548_mask-raw.fits`

Both versions should be staged when available because the website exposes both.

---

## 5. Cluster moment maps

Location:

`Cluster-Moms/`

Naming rules:

- `<cluster>_mom0.fits`
- `<cluster>_mom1.fits`
- `<cluster>_mom2.fits`

Examples:

- `Abell-133_mom0.fits`
- `Abell-133_mom1.fits`
- `Abell-133_mom2.fits`

---

## 6. Cluster preview figures

Location:

`Cluster-Figures/`

Naming rules:

- `<cluster>_Intensity.png`
- `<cluster>_Noise_Footprints.png`

Examples:

- `Abell-133_Intensity.png`
- `Abell-133_Noise_Footprints.png`
- `Abell-548_Intensity.png`
- `Abell-548_Noise_Footprints.png`

These are used directly by the cluster page and search results.

---

## 7. Completeness curves

Location:

`Completeness-Curves/`

Naming rule:

`<cluster-without-hyphen>_completeness_matched_v8.png`

Important:

- the backend removes hyphens from the cluster name when building this file name,
- this naming is **not** the same as the other cluster products.

Examples:

- `Abell133_completeness_matched_v8.png`
- `Abell168_completeness_matched_v8.png`
- `Abell548_completeness_matched_v8.png`

If this file is missing or named differently, the completeness preview will not appear.

---

## 8. Galaxy figure folders

Location:

`Galaxy-Figures/`

Folder naming rule:

`<cluster>_figures/`

Examples:

- `Abell-133_figures/`
- `Abell-168_figures/`
- `Abell-548_figures/`

Inside each folder, figures are named per galaxy detection ID.

### Common observed figure naming

For a detection ID `1` in `Abell-548`, examples include:

- `Abell-548_1_mom0_dss2blue.png`
- `Abell-548_1_mom0_dss2red.png`
- `Abell-548_1_mom0_panstarrs.png`
- `Abell-548_1_mom0_wisew1.png`
- `Abell-548_1_mom0.png`
- `Abell-548_1_mom1.png`
- `Abell-548_1_mom2.png`
- `Abell-548_1_snr.png`
- `Abell-548_1_spec.png`

Use the same pattern for all detections in that cluster.

### Important note on galaxy figure previews

The current backend builds deterministic preview candidates for:

- `mom0` background variants,
- `mom1`,
- `mom2`,
- `spec`.

So even if you produce more figure types, at minimum ensure the commonly used preview files exist and follow the established naming style.

---

## 9. Galaxy cubelets

Location:

`Galaxy-Cubelets/<cluster>/`

Folder naming rule:

`Galaxy-Cubelets/<cluster>/`

Examples:

- `Galaxy-Cubelets/Abell-133/`
- `Galaxy-Cubelets/Abell-548/`

Inside each cluster folder, the website expects deterministic galaxy products of the form:

`<cluster>_<gid>_<type>.<ext>`

Where `<gid>` is the SoFiA integer detection ID and `<type>` is one of:

- `chan.fits`
- `cube.fits`
- `mask.fits`
- `mom0.fits`
- `mom1.fits`
- `mom2.fits`
- `pv.fits`
- `snr.fits`
- `spec.txt`

Examples:

- `Abell-548_1_chan.fits`
- `Abell-548_1_cube.fits`
- `Abell-548_1_mask.fits`
- `Abell-548_1_mom0.fits`
- `Abell-548_1_mom1.fits`
- `Abell-548_1_mom2.fits`
- `Abell-548_1_pv.fits`
- `Abell-548_1_snr.fits`
- `Abell-548_1_spec.txt`

These are used for galaxy downloads and cluster-wide bundled downloads.

---

## Canonical cluster naming

Use a canonical, hyphenated cluster naming style in S3 and metadata wherever possible.

Examples:

- `Abell-133`
- `Abell-168`
- `Abell-194`
- `Abell-3104`
- `Abell-3266`
- `Abell-3360`
- `Abell-3376`
- `Abell-3562`
- `Abell-3990`
- `Abell-4038`
- `Abell-548`
- `Abell-85`
- `Abell-S405`
- `Abell-S560`
- `Abell-S606`

The backend does contain some name-variant logic for searching and file lookup, but staging should still follow one clean canonical naming scheme.

**Best practice:** stage files using the exact cluster name that appears in the metadata file `ID` column.

---

## Metadata file structure and rules

## Master cluster table

At the top of `MGCLS_HI_15clusters.txt`, maintain a cluster-level table with the standard column names already in use.

Expected columns include at least:

- `ID`
- `SEL`
- `M_Z`
- `Anc_Z`
- `RA`
- `DEC`
- `DATE`
- `SBID`
- `CAPTURE_ID`
- `AWS`
- `R200`
- `SIGMA_V`
- `SOFIA_DETS`
- `RMS`
- `NHI_min`
- `NHI_max`
- `V_min`
- `V_max`
- `NCHAN`
- `FREQ_min`
- `FREQ_max`
- `BMIN`
- `BMAJ`
- `BPA`
- `MGCLS_Name`
- `Alternate_Name`
- `Mass`

### Metadata update rules

When staging a cluster:

1. add or update its row in the master cluster table,
2. set the archive/staging field consistently,
3. ensure numeric fields are populated in the same style as existing rows,
4. keep the cluster `ID` exactly aligned with the S3 naming convention,
5. update `SOFIA_DETS` to match the staged SoFiA catalogue content,
6. append or replace the matching SoFiA block lower down in the same text file.

### Do not do this

- do not rename columns,
- do not reorder columns without a strong reason,
- do not remove the `#<cluster>` block markers,
- do not store the master cluster table and per-cluster SoFiA tables in separate web-facing files unless backend changes are made.

---

## Per-cluster SoFiA block rules

Each cluster-specific SoFiA catalogue must appear in the combined metadata file as a block like:

`#Abell-194`

followed by the SoFiA table.

The parser is fairly tolerant, but the safest approach is:

- keep the current SoFiA-style header layout,
- keep the column names intact,
- keep source names quoted if they already are,
- keep the numeric source `id` column intact,
- do not insert extra prose inside a block.

The website uses these blocks for:

- cluster detail galaxy tables,
- galaxy search,
- galaxy ID lookup,
- per-galaxy downloads.

The galaxy `id` in the SoFiA block must match the `<gid>` used in:

- galaxy figure names,
- galaxy cubelet names.

That mapping must remain one-to-one.

---

## Minimum complete staging set for one new cluster

For one fully staged cluster, the minimum recommended set is:

### Metadata

- update master row in `Cluster_catalogue/MGCLS_HI_15clusters.txt`
- append or refresh `#<cluster>` SoFiA block in the same file
- upload `Catalogues/<cluster>_cat.txt`

### Cluster data products

- `Cluster-Cubes/<cluster>.fits`
- `Cluster-Masks/<cluster>_mask.fits`
- `Cluster-Masks/<cluster>_mask-raw.fits`
- `Cluster-Moms/<cluster>_mom0.fits`
- `Cluster-Moms/<cluster>_mom1.fits`
- `Cluster-Moms/<cluster>_mom2.fits`

### Cluster previews

- `Cluster-Figures/<cluster>_Intensity.png`
- `Cluster-Figures/<cluster>_Noise_Footprints.png`
- `Completeness-Curves/<cluster-no-hyphen>_completeness_matched_v8.png`

### Galaxy products

- `Galaxy-Figures/<cluster>_figures/` with per-galaxy figures
- `Galaxy-Cubelets/<cluster>/` with per-galaxy cubelet products

---

## Recommended staging workflow

## Step 1. Confirm canonical cluster name

Choose the exact cluster name that will be used everywhere.

Use that same canonical string in:

- metadata `ID`
- cluster catalogue filename
- cluster cube filename
- cluster mask filenames
- cluster moment filenames
- cluster figure filenames
- galaxy figure folder name
- galaxy cubelet folder name
- galaxy cubelet filename prefix

## Step 2. Prepare the cluster-level row

Before upload, prepare the cluster row for the top of `MGCLS_HI_15clusters.txt`.

Check carefully:

- RA / DEC
- SBID
- CAPTURE_ID
- SOFIA_DETS
- RMS
- NHI range
- velocity range
- frequency range
- beam parameters
- MGCLS short name
- mass

## Step 3. Prepare the SoFiA block

Export the SoFiA table for that cluster and make sure:

- source names are preserved,
- detection `id` values are stable integers,
- RA / DEC / velocity columns are present,
- `f_sum` / integrated flux is present,
- width columns such as `w20` and `w50` are present when available.

## Step 4. Stage cluster products to S3

Upload the cluster data files into:

- `Catalogues/`
- `Cluster-Cubes/`
- `Cluster-Masks/`
- `Cluster-Moms/`
- `Cluster-Figures/`
- `Completeness-Curves/`

## Step 5. Stage galaxy products to S3

Upload:

- figure PNGs into `Galaxy-Figures/<cluster>_figures/`
- cubelet FITS/TXT products into `Galaxy-Cubelets/<cluster>/`

Make sure every staged galaxy product uses the correct SoFiA integer ID.

## Step 6. Replace or update the combined metadata file

After all products are staged, upload the updated:

`Cluster_catalogue/MGCLS_HI_15clusters.txt`

This should be done carefully because it is the file the site reads directly.

## Step 7. Manual QA in the website

After staging:

1. open landing page and confirm the cluster appears,
2. search by cluster name,
3. open the cluster detail page,
4. check intensity / noise / completeness previews,
5. confirm galaxies appear in the detail table,
6. search by a galaxy name substring if possible,
7. test one galaxy download,
8. test one cluster download,
9. confirm missing products are not silently due to naming mismatch.

---

## Validation checklist

Use this checklist every time you stage a new cluster.

### Metadata

- [ ] cluster row exists in the master table
- [ ] `ID` matches canonical cluster name
- [ ] `SOFIA_DETS` matches the source catalogue
- [ ] `SBID` and `CAPTURE_ID` are filled
- [ ] `AWS` flag is updated
- [ ] appended `#<cluster>` SoFiA block is present

### Cluster products

- [ ] `<cluster>_cat.txt`
- [ ] `<cluster>.fits`
- [ ] `<cluster>_mask.fits`
- [ ] `<cluster>_mask-raw.fits`
- [ ] `<cluster>_mom0.fits`
- [ ] `<cluster>_mom1.fits`
- [ ] `<cluster>_mom2.fits`

### Preview products

- [ ] `<cluster>_Intensity.png`
- [ ] `<cluster>_Noise_Footprints.png`
- [ ] `<cluster-no-hyphen>_completeness_matched_v8.png`

### Galaxy folders

- [ ] `Galaxy-Figures/<cluster>_figures/` exists
- [ ] `Galaxy-Cubelets/<cluster>/` exists

### Galaxy ID consistency

For a sample of detections, confirm that the same `gid` exists consistently in:

- SoFiA block in metadata
- galaxy figures
- galaxy cubelets

---

## What the backend assumes

The backend currently assumes the following:

1. the combined metadata file is read from S3, not from a separate database,
2. the top section of the metadata file is the cluster-level master catalogue,
3. later `#<cluster>` sections are per-cluster SoFiA source catalogues,
4. cluster previews are deterministic and built from file names,
5. completeness file names remove hyphens from the cluster name,
6. galaxy cubelet names are deterministic and built from `<cluster>_<gid>_<suffix>`,
7. cluster and galaxy download bundles are created by composing those exact S3 keys.

Because of this, staging is not just an upload exercise. It is effectively part of the application’s data model.

---

## Common failure modes

## 1. Cluster appears in metadata but not on the site correctly

Likely causes:

- malformed master table row,
- broken delimiter structure in the combined metadata file,
- missing required columns,
- cluster ID mismatch.

## 2. Cluster page opens but galaxy table is empty

Likely causes:

- missing appended `#<cluster>` SoFiA block,
- broken SoFiA block formatting,
- cluster marker name not matching the master `ID`.

## 3. Cluster preview images missing

Likely causes:

- wrong file names in `Cluster-Figures/`,
- wrong completeness naming,
- wrong cluster name string.

## 4. Galaxy previews missing

Likely causes:

- wrong `Galaxy-Figures/<cluster>_figures/` folder name,
- wrong `gid` in filenames,
- unsupported or inconsistent background suffix naming.

## 5. Galaxy download returns missing products

Likely causes:

- missing files in `Galaxy-Cubelets/<cluster>/`,
- wrong `<gid>` in filename,
- missing `_spec.txt`, `_pv.fits`, or other expected suffixes.

---

## Recommended operational practice

- Keep a local staging manifest per cluster before upload.
- Stage all products for a cluster as one coherent batch.
- Update the combined metadata file last, after confirming all S3 objects exist.
- Keep a dated backup copy of the previous metadata file before replacement.
- Do not partially stage a cluster and then mark it as fully staged in metadata.
- If a cluster is incomplete, either keep it off the staged list or document the missing products clearly.

---

## Suggested per-cluster staging manifest template

Use something like this internally for each cluster:

### Cluster

- Canonical name:
- MGCLS short name:
- SBID:
- CAPTURE_ID:
- Number of detections:

### Uploaded cluster products

- [ ] catalogue
- [ ] cube
- [ ] mask
- [ ] raw mask
- [ ] mom0
- [ ] mom1
- [ ] mom2
- [ ] intensity preview
- [ ] noise footprints preview
- [ ] completeness curve

### Uploaded galaxy products

- [ ] galaxy figures folder
- [ ] galaxy cubelets folder
- [ ] all gids cross-checked against SoFiA block

### Metadata

- [ ] master row updated
- [ ] SoFiA block appended or refreshed
- [ ] combined metadata file uploaded

### Site QA

- [ ] landing page
- [ ] cluster page
- [ ] galaxy table
- [ ] cluster download
- [ ] galaxy download

---

## Final handover note

For MGCLS-HI, the S3 bucket is effectively the public data backend, and the combined metadata file is effectively the site index.

If you preserve:

- the exact folder structure,
- the exact naming conventions,
- the one-file metadata model,
- the ID consistency between metadata and S3 products,

then the site should continue to work with minimal backend changes.

If you change any of those, update the Flask backend at the same time.

---

## Items that may still need future documentation

This README documents the web-facing staging contract. A future version can still add:

- exact reduction pipeline commands,
- exact source-finding commands,
- figure-generation commands,
- how completeness plots are produced,
- who is allowed to publish to the bucket,
- versioning / rollback procedure,
- formal release checklist.

