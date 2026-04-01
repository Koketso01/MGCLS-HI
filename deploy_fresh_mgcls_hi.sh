#!/usr/bin/env bash
set -euo pipefail

cd ~/FLASKAPPS

export AWS_PROFILE=mgcls
export AWS_DEFAULT_REGION=us-east-2
export PYTHON_BIN=python3.11
export STAGE=dev
export STACK_NAME=mgcls-hi-dev
export LAYER_NAME=flaskapps-numpy-pandas-py311-x86_64

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN is not installed or not on PATH." >&2
  exit 1
fi

rm -rf .venv python layer.zip build dist .zappa *.egg-info

"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel 'setuptools<80'
python -m pip install -r requirements.txt

mkdir -p python
python -m pip install \
  --platform manylinux2014_x86_64 \
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
zip -r9 layer.zip python >/dev/null

LAYER_ARN=$(aws lambda publish-layer-version \
  --profile "$AWS_PROFILE" \
  --region "$AWS_DEFAULT_REGION" \
  --layer-name "$LAYER_NAME" \
  --description 'NumPy + pandas for Flask app on Python 3.11 x86_64' \
  --zip-file fileb://layer.zip \
  --compatible-runtimes python3.11 \
  --compatible-architectures x86_64 \
  --query 'LayerVersionArn' \
  --output text)

echo "Published layer: $LAYER_ARN"

python - <<PY
import json
from pathlib import Path
p = Path('zappa_settings.json')
data = json.loads(p.read_text())
data['dev']['keep_warm'] = False
data['dev']['layers'] = ['$LAYER_ARN']
p.write_text(json.dumps(data, indent=2) + '\n')
print('Updated zappa_settings.json with new layer ARN and keep_warm=false')
PY

zappa unschedule "$STAGE" >/dev/null 2>&1 || true
zappa undeploy "$STAGE" -y >/dev/null 2>&1 || true

if aws cloudformation describe-stacks --stack-name "$STACK_NAME" >/dev/null 2>&1; then
  aws cloudformation delete-stack --stack-name "$STACK_NAME"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME"
fi

zappa deploy "$STAGE"
zappa status "$STAGE"

echo
echo "Fresh deployment complete."
echo "If you later change only code/templates, use: zappa update $STAGE"
echo "If you need logs, use: zappa tail $STAGE --since 10m"
