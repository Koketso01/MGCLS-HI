#!/usr/bin/env bash
set -euo pipefail
cd ~/FLASKAPPS
export AWS_PROFILE=mgcls
export AWS_DEFAULT_REGION=us-east-2
./.venv/bin/zappa update dev
./.venv/bin/zappa status dev
