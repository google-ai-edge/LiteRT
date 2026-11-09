#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# Fetches GPQA's official release (github.com/idavidrein/gpqa, dataset.zip at commit 56686c0,
# password-protected with the password its README states) and extracts gpqa_diamond.csv into
# data/. The questions are not to be posted online: keep data/ out of logs and commits.
set -euo pipefail
here=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
commit=56686c0f3e1d9f8b9b0c0d9f0a5b2a3a3f2a9b7e
mkdir -p "$here/data"
cd "$here/data"
[[ -f dataset.zip ]] || curl -L --fail -o dataset.zip "https://github.com/idavidrein/gpqa/raw/56686c0/dataset.zip"
password=$(curl -sL --fail "https://raw.githubusercontent.com/idavidrein/gpqa/56686c0/README.md" | sed -n 's/.*password-prote[c]*ted with this password: `\([^`]*\)`.*/\1/p' | head -1)
[[ -n $password ]] || password=deserted-untie-orchid  # the README's password at 56686c0
if command -v unzip >/dev/null; then unzip -o -P "$password" dataset.zip 'dataset/gpqa_diamond.csv' >/dev/null
else python3 - "$password" <<'PY'
import sys, zipfile
with zipfile.ZipFile('dataset.zip') as z:
    z.extract('dataset/gpqa_diamond.csv', pwd=sys.argv[1].encode())
PY
fi
mv -f dataset/gpqa_diamond.csv gpqa_diamond.csv && rmdir dataset 2>/dev/null || true
sha256sum gpqa_diamond.csv 2>/dev/null || python3 -c "import hashlib;print(hashlib.sha256(open('gpqa_diamond.csv','rb').read()).hexdigest(), 'gpqa_diamond.csv')"
wc -l gpqa_diamond.csv
