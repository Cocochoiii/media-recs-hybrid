#!/usr/bin/env bash
set -euo pipefail
export LOGSTASH_HOST=localhost
export LOGSTASH_PORT=5044
uvicorn api.main:app --host 0.0.0.0 --port 8000
