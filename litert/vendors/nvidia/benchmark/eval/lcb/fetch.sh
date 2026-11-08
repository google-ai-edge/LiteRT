#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# Fetches the LiveCodeBench code-generation problems of release v6 (the lite dataset on
# Hugging Face, livecodebench/code_generation_lite: test6.jsonl = the 175 problems added in v6,
# contests of January-April 2025) into data/. select.py then samples and formats them.
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
mkdir -p "$here/data"
curl -L --fail -o "$here/data/test6.jsonl" \
  https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl
wc -l "$here/data/test6.jsonl"
