#!/bin/bash
# Run locally, not on the pod: ships the repo, installs deps, starts data
# prep + training detached in tmux so it survives SSH disconnect.
set -euo pipefail
cd "$(dirname "$0")/.."

INFO=$(cat remote/.pod_info.json)
IP=$(python3 -c "import json,sys; print(json.load(sys.stdin)['ip'])" <<< "$INFO")
PORT=$(python3 -c "import json,sys; print(json.load(sys.stdin)['port'])" <<< "$INFO")

echo "shipping repo to root@$IP:$PORT ..."
rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  --exclude .git --exclude train/data \
  ./ "root@$IP:/workspace/rivaquant/"

ssh -o StrictHostKeyChecking=no -p "$PORT" "root@$IP" bash -s <<'REMOTE'
set -e
cd /workspace/rivaquant
pip install -q -r requirements.txt
mkdir -p /workspace/rivaquant-out
tmux new-session -d -s rivaquant "
  cd /workspace/rivaquant &&
  python3 train/data.py 2>&1 | tee -a /workspace/rivaquant-out/train.log &&
  PYTHONPATH=/workspace/rivaquant python3 train/train.py 2>&1 | tee -a /workspace/rivaquant-out/train.log
"
echo "launched in tmux session 'rivaquant'"
REMOTE
