# MGCLS-HI

MGCLS-HI is the Flask-based web application and data access layer for the MGCLS–H I database. It is designed to let users browse MGCLS cluster metadata, search both clusters and H I detections, view preview figures, and download cluster- and galaxy-level products from the public S3-backed data store.

This README is the **main project entry point** for anyone taking over the repository. It focuses on:

- what the project is,
- how the repository is organised,
- how to edit and test locally first,
- how deployment is currently handled,
- where to look for the detailed staging handover guide.

For detailed S3 upload conventions, metadata maintenance, and staging rules, see **`README_data_staging.md`**.

---

## What this project does

The application currently supports:

- a landing page built from the master cluster catalogue,
- cluster detail pages with preview figures and galaxy tables,
- cluster and galaxy search,
- product-selection download pages,
- ZIP downloads and terminal download scripts for selected products,
- supporting pages such as About, Help, Publications, People, and Contact,
- SEO endpoints such as `/robots.txt`, `/sitemap.xml`, and a health endpoint at `/health`.

At a high level, the app is not backed by a relational database. Instead, it reads a combined metadata text file and deterministic S3 object paths.

---

## Repository structure

This repository is intentionally split across branches.

### `main`

The `main` branch contains the backend and project-level files such as:

- `app.py`
- `utils_UPDATED_single_metadata_v3_previews.py`
- `MGCLS_HI_15clusters.txt`
- `README.md`
- deployment/configuration files such as requirements, deployment scripts, and Zappa settings

### `templates`

The `templates` branch contains Flask HTML templates only:

- `templates/index.html`
- `templates/search.html`
- `templates/cluster_detail.html`
- `templates/download_select.html`
- `templates/about.html`
- `templates/help.html`
- `templates/publications.html`
- `templates/people.html`
- `templates/contact.html`
- `templates/layout.html`

### `static`

The `static` branch contains frontend assets only:

- CSS
- JavaScript
- icons
- logos
- PDFs and other bundled static assets used by the site

### Important note about working with this repo

Because the project is branch-separated, you should always be clear about **which branch you are editing**.

A safe pattern is:

1. edit backend logic on `main`,
2. edit HTML templates on `templates`,
3. edit CSS/JS/images on `static`,
4. commit and push each branch independently.

---

## Application overview

### Core backend files

#### `app.py`

This is the Flask application entry point. It wires together:

- the landing page,
- cluster detail pages,
- search,
- download routes,
- supporting informational pages,
- SEO endpoints,
- a `/health` endpoint for deployment checks.

#### `utils_UPDATED_single_metadata_v3_previews.py`

This file contains the data access and helper logic, including:

- S3 path configuration,
- metadata loading and parsing,
- preview URL construction,
- landing table generation,
- search helpers,
- ZIP/download assembly helpers,
- publication loading.

#### `MGCLS_HI_15clusters.txt`

This is the combined metadata file used by the app.

It contains:

1. a cluster-level master catalogue at the top, and
2. appended per-cluster SoFiA catalogues below, separated by cluster marker lines such as `#Abell-133`.

This file is central to the site’s search and discovery logic.

---

## Data model and storage model

The project currently uses a **file-and-S3 driven architecture** rather than a database server.

### Public S3 root

The web-facing data products live under the public S3 prefix:

`ratt-public-data / MGCLS_HI / Datasets / koketso-HI-MGCLS-data / HI-MGCLS /`

### Key prefixes used by the backend

The backend expects deterministic objects under prefixes such as:

- `Catalogues/`
- `Cluster_catalogue/`
- `Cluster-Cubes/`
- `Cluster-Figures/`
- `Cluster-Masks/`
- `Cluster-Moms/`
- `Completeness-Curves/`
- `Galaxy-Cubelets/`
- `Galaxy-Figures/`

Because of this design, folder names and file naming conventions are part of the application contract.

For the full operational rules, refer to **`README_data_staging.md`**.

---

## Pages and routes

The main routes exposed by the app include:

- `/` — landing page
- `/cluster/<cluster_id>` — cluster detail page
- `/search` — cluster and galaxy search
- `/contact`
- `/people`
- `/about`
- `/publications`
- `/help`
- `/robots.txt`
- `/sitemap.xml`
- `/health`

There are also download endpoints for cluster- and galaxy-level products, including ZIP bundles and terminal download scripts.

---

## Local development workflow

This project should be edited **locally first**, tested, and only then deployed.

### 1. Clone the repository

Clone the repo and fetch all branches.

Example:

```bash
git clone https://github.com/Koketso01/MGCLS-HI.git
cd MGCLS-HI
git fetch --all
```

### 2. Choose the branch you want to work on

Examples:

```bash
git switch main
```

```bash
git switch templates
```

```bash
git switch static
```

### 3. Create and activate a virtual environment

The project uses Python 3.11 in deployment-related scripts, so local work should preferably use the same Python version.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

At minimum:

```bash
python -m pip install --upgrade pip wheel 'setuptools<80'
python -m pip install -r requirements.txt
```

### 5. Run the Flask app locally

```bash
python app.py
```

By default, the app runs on:

- host: `0.0.0.0`
- port: `5000`
- debug: enabled when `FLASK_DEBUG=1`

### 6. Suggested local environment variables

Useful variables include:

- `FLASK_SECRET_KEY`
- `SITE_URL`
- `LOGLEVEL`
- `FLASK_HOST`
- `FLASK_PORT`
- `FLASK_DEBUG`

Example:

```bash
export FLASK_SECRET_KEY='change-me'
export SITE_URL='http://127.0.0.1:5000'
export LOGLEVEL='INFO'
export FLASK_DEBUG=1
python app.py
```

### 7. What to test locally before pushing

Before pushing changes, check:

- landing page loads,
- search page loads and submits correctly,
- cluster detail pages load,
- preview images open correctly,
- download selection page renders,
- ZIP and script download routes work,
- navigation/footer assets resolve correctly,
- `/health` returns OK.

---

## Recommended edit workflow

A good working pattern for the next maintainer is:

### Backend changes

1. switch to `main`,
2. update `app.py` or `utils_UPDATED_single_metadata_v3_previews.py`,
3. run locally,
4. test affected pages and routes,
5. commit and push `main`.

### Template changes

1. switch to `templates`,
2. edit files under `templates/`,
3. test locally with the updated template set,
4. commit and push `templates`.

### Static/frontend changes

1. switch to `static`,
2. edit files under `static/`,
3. test styling, scripts, and images locally,
4. commit and push `static`.

### Metadata/staging-related changes

If the change affects S3 staging, metadata structure, or the combined metadata file, read **`README_data_staging.md`** before making changes.

---

## Dependencies

The repository includes:

- `requirements.txt`
- `requirements-deploy.txt`
- `requirements-dev.txt`

These capture the main runtime and deployment dependencies.

Important note: deployment helper scripts may rebuild environments or rewrite packaging-related files. Review them before running them in a production workflow.

---

## Deployment overview

Deployment is currently AWS- and Zappa-based.

### Current deployment pattern

The repository includes deployment helpers such as:

- `deploy_fresh_mgcls_hi.sh`
- `deploy_quick.sh`
- `recover_deploy.sh`
- `rebuild_py311_and_deploy.sh`
- `zappa_settings.json`

These indicate the current deployment model:

- AWS profile: `mgcls`
- AWS region: `us-east-2`
- Zappa stage: `dev`
- runtime: `python3.11`
- app function: `app.app`
- touch path: `/health`
- Lambda layer used for NumPy/Pandas packaging

### Important deployment caution

The deployment scripts are operational helpers, not generic one-size-fits-all scripts.

Some of them:

- rebuild the virtual environment,
- rebuild Lambda layers,
- modify `zappa_settings.json`,
- undeploy and redeploy stacks,
- publish new layer versions,
- update Lambda settings.

Because of that, **always read the script you plan to run before using it**, especially in a handover situation.

### Typical deployment sequence

A safe deployment mindset is:

1. edit locally,
2. test locally,
3. commit and push the relevant branch,
4. confirm deployment settings,
5. run the intended deployment helper,
6. verify `/health`, the landing page, search, and downloads after deployment.

### Quick update vs fresh deploy

In general:

- use the quick update path when only code/templates change and infrastructure does not need rebuilding,
- use the fresh/recovery path when packaging, Lambda layers, or deployment state has broken.

---

## Health, SEO, and operational endpoints

The app exposes:

- `/health` for health checks,
- `/robots.txt` for crawler directives,
- `/sitemap.xml` for sitemap generation.

The sitemap expands cluster detail URLs dynamically from the master catalogue, so metadata integrity affects SEO coverage too.

---

## Known architectural characteristics

A maintainer should know the following up front:

1. the app is tightly coupled to S3 naming conventions,
2. the app is tightly coupled to the combined metadata file format,
3. the combined metadata file acts as both a master cluster index and a per-cluster SoFiA source catalogue store,
4. deterministic file naming is part of how previews and downloads are discovered,
5. changes to naming conventions usually require backend changes as well.

---

## Files you should treat carefully

The following files are especially sensitive:

- `utils_UPDATED_single_metadata_v3_previews.py`
- `MGCLS_HI_15clusters.txt`
- `zappa_settings.json`
- deployment scripts

Changes to these can affect:

- search,
- preview loading,
- downloads,
- metadata parsing,
- deployment behavior.

---

## Handover guidance for the next maintainer

If you are taking over this project, start in this order:

1. read this `README.md`,
2. read `README_data_staging.md`,
3. inspect `app.py`,
4. inspect `utils_UPDATED_single_metadata_v3_previews.py`,
5. inspect `MGCLS_HI_15clusters.txt`,
6. inspect `zappa_settings.json`,
7. review the deployment helper scripts,
8. run the app locally before attempting any deployment.

Do not start by deploying first.

Start by understanding the local application and the S3/metadata contract.

---

## Suggested first local checks for a new maintainer

After cloning and installing dependencies, verify:

- the app starts without import errors,
- the landing page renders,
- the search page loads,
- one cluster page opens,
- one download flow works,
- one publications page load works,
- `/health` returns a JSON OK response.

If those work locally, you are in a much safer position to deploy.

---

## Related documentation

### `README_data_staging.md`

Use this for:

- S3 folder structure,
- file naming conventions,
- metadata update rules,
- staging workflow,
- validation and QA for new cluster uploads.

This main README and the staging README are complementary:

- **`README.md`** = project overview, local editing, repo structure, deployment orientation
- **`README_data_staging.md`** = operational data staging and metadata handover guide

---

## Acknowledgement

This repository supports the MGCLS–H I database and its public-facing access layer, built around MGCLS cluster data products, SoFiA-derived source catalogues, and AWS-hosted delivery.

---

## Minimal quick-start

If you only need the shortest path:

```bash
git clone https://github.com/Koketso01/MGCLS-HI.git
cd MGCLS-HI
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel 'setuptools<80'
python -m pip install -r requirements.txt
export FLASK_SECRET_KEY='change-me'
export SITE_URL='http://127.0.0.1:5000'
python app.py
```

Then open the site locally, test the pages you changed, and only deploy once local checks pass.
