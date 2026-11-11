FROM python:3.12-slim-bookworm

# CLI agents may need git/curl; the verifier uses only Python's standard library.
RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY screenshot.png /workspace/screenshot.png
