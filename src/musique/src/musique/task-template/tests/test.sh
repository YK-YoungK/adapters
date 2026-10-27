#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
# Remove stale rewards even if Python cannot import or start the verifier.
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/diagnostics.json /logs/verifier/prediction.json
exec python -E -B /tests/verify.py
