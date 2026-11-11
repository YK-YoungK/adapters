#!/bin/bash
set -euo pipefail

# Always leave a numeric reward even if Python fails before loading the verifier.
mkdir -p /logs/verifier
printf '0\n' > /logs/verifier/reward.txt
exec python /tests/verify.py
