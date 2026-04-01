#!/usr/bin/env bash
set -euo pipefail

cd ~/FLASKAPPS

export AWS_PROFILE=mgcls
export AWS_DEFAULT_REGION=us-east-2

rm -rf .venv python layer.zip build dist *.egg-info
python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel 'setuptools<80'
python -m pip install -r requirements.txt

mkdir -p python
python -m pip install \
  --platform manylinux_2_28_x86_64 \
  --implementation cp \
  --python-version 3.11 \
  --only-binary=:all: \
  --no-compile \
  -t python \
  numpy==2.4.3 \
  pandas==3.0.1

find python -type d -name '__pycache__' -exec rm -rf {} +
find python -type d -name 'tests' -exec rm -rf {} +
find python -type f -name '*.pyc' -delete
rm -rf python/bin
zip -r9 layer.zip python

aws lambda publish-layer-version \
  --profile mgcls \
  --region us-east-2 \
  --layer-name flaskapps-numpy-pandas-py311-x86_64 \
  --description 'NumPy + pandas for Flask app on Python 3.11 x86_64' \
  --zip-file fileb://layer.zip \
  --compatible-runtimes python3.11 \
  --compatible-architectures x86_64

zappa update dev
