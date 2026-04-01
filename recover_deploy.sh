#!/usr/bin/env bash
set -euo pipefail

cd ~/FLASKAPPS

export AWS_PROFILE=mgcls
export AWS_DEFAULT_REGION=us-east-2

STAGE=dev
PROJECT_NAME=mgcls-hi
FUNCTION_NAME=mgcls-hi-dev
STACK_NAME=mgcls-hi-dev
BUCKET_NAME=zappa-mgcls-hi-26-03-05-13-24
LAYER_NAME=flaskapps-numpy-pandas-py311-x86_64

echo "== Writing clean requirements files =="

cat > requirements.txt <<'REQ'
Flask==2.2.5
Werkzeug==2.2.3
numpy==1.26.4
pandas==2.2.3
boto3==1.42.71
REQ

cat > requirements-deploy.txt <<'REQ'
-r requirements.txt
zappa==0.62.1
REQ

echo "== Cleaning old build artifacts =="
rm -rf .venv python build dist .zappa *.egg-info layer.zip package.zip
find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f -name '*.pyc' -delete 2>/dev/null || true

echo "== Rebuilding virtualenv =="
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel 'setuptools<80'
python -m pip install -r requirements-deploy.txt

echo "== Building Lambda layer for numpy/pandas =="
mkdir -p python
python -m pip install \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 3.11 \
  --only-binary=:all: \
  --no-compile \
  -t python \
  numpy==1.26.4 \
  pandas==2.2.3

find python -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
find python -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true
find python -type f -name '*.pyc' -delete 2>/dev/null || true
rm -rf python/bin

zip -qr layer.zip python

echo "== Publishing layer =="
LAYER_ARN=$(aws lambda publish-layer-version \
  --profile "$AWS_PROFILE" \
  --region "$AWS_DEFAULT_REGION" \
  --layer-name "$LAYER_NAME" \
  --description "numpy+pandas for MGCLS-HI Flask app on Python 3.11" \
  --zip-file fileb://layer.zip \
  --compatible-runtimes python3.11 \
  --compatible-architectures x86_64 \
  --query 'LayerVersionArn' \
  --output text)

echo "Layer ARN: $LAYER_ARN"

echo "== Writing clean zappa settings =="
cat > zappa_settings.json <<JSON
{
  "dev": {
    "app_function": "app.app",
    "aws_region": "us-east-2",
    "profile_name": "mgcls",
    "project_name": "mgcls-hi",
    "runtime": "python3.11",
    "s3_bucket": "$BUCKET_NAME",
    "binary_support": true,
    "slim_handler": true,
    "memory_size": 1024,
    "timeout_seconds": 60,
    "keep_warm": false,
    "touch_path": "/health",
    "use_precompiled_packages": true,
    "layers": [
      "$LAYER_ARN"
    ],
    "exclude": [
      ".venv",
      "python",
      "layer.zip",
      "build",
      "dist",
      ".zappa",
      "__pycache__",
      "*.pyc",
      "*.pyo",
      "*.egg-info",
      ".git",
      ".pytest_cache",
      ".mypy_cache",
      "node_modules",
      "numpy",
      "pandas",
      "boto3",
      "botocore",
      "s3transfer",
      "dateutil",
      "tests"
    ]
  }
}
JSON

echo "== Showing final config sanity check =="
grep -nE 'Flask|Werkzeug|numpy|pandas|boto3|zappa' requirements.txt requirements-deploy.txt || true
grep -nE 'slim_handler|keep_warm|layers|runtime|project_name' zappa_settings.json || true

echo "== Turning off old keep-warm =="
./.venv/bin/zappa unschedule "$STAGE" >/dev/null 2>&1 || true

echo "== Removing old deployment if present =="
./.venv/bin/zappa undeploy "$STAGE" -y >/dev/null 2>&1 || true

if aws cloudformation describe-stacks --stack-name "$STACK_NAME" >/dev/null 2>&1; then
  aws cloudformation delete-stack --stack-name "$STACK_NAME"
  aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME"
fi

echo "== Fresh deploy =="
./.venv/bin/zappa deploy "$STAGE"

echo "== Increasing Lambda temp storage to 2 GB =="
aws lambda update-function-configuration \
  --profile "$AWS_PROFILE" \
  --region "$AWS_DEFAULT_REGION" \
  --function-name "$FUNCTION_NAME" \
  --ephemeral-storage '{"Size": 2048}'

echo "== Final status =="
./.venv/bin/zappa status "$STAGE"
aws lambda get-function-configuration \
  --profile "$AWS_PROFILE" \
  --region "$AWS_DEFAULT_REGION" \
  --function-name "$FUNCTION_NAME" \
  --query '{FunctionName:FunctionName,Runtime:Runtime,EphemeralStorage:EphemeralStorage,Layers:Layers[*].Arn}' \
  --output table
