#!/usr/bin/env bash
set -euo pipefail
python -m cProfile -o ./.logs/profile.out -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
PID=$!
sleep 15
kill $PID || true
python -m pstats ./.logs/profile.out -c "stats 20"
