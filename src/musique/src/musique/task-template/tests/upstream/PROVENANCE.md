These files are copied unchanged from StonyBrookNLP/MuSiQue:

- Repository: https://github.com/StonyBrookNLP/MuSiQue
- Commit: `922ac98f19a201998dbdae6d7f2887a5258dbdeb`
- Files: `evaluate_v1.0.py`, `metrics/*.py`, and `LICENSE`
- Authors: Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal
- License: Creative Commons Attribution 4.0 International (see `LICENSE`)

The Harbor-specific wrapper is `../verify.py`; it adapts paths and prediction
validation, calls the original answer/support metric classes, and emits Harbor
reward files. It keeps task scores unrounded because the original evaluator
rounds only after averaging the dataset. No upstream scoring code is modified.
