#!/bin/bash
set -euo pipefail

mkdir -p /logs/verifier
printf '0.0\n' > /logs/verifier/reward.txt
printf '{"answer_f1": 0.0, "answer_em": 0.0, "support_f1": 0.0}\n' > /logs/verifier/reward.json

# Missing/malformed predictions are scored by verify.py. A verifier exception
# must fail the trial instead of leaving a valid-looking zero reward behind.
verifier_failed() {
    local status=$?
    rm -f /logs/verifier/reward.txt /logs/verifier/reward.json
    exit "$status"
}
trap verifier_failed ERR

python -I -B /tests/verify.py
