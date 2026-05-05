#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <binary>" >&2
  exit 2
fi

binary="$1"
if nm "$binary" | c++filt | grep -E '(^|[^[:alnum:]_])absl::|Absl|_ZN4absl' >&2; then
  echo "Found Abseil symbols in $binary" >&2
  exit 1
fi
