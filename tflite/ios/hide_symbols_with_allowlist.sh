#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# A script to merge Mach-O object files into a single object file and hide
# their internal symbols. Only allowed symbols will be visible in the
# symbol table after this script.

# To run this script, you must set several variables:
#   INPUT_FRAMEWORK: a zip file containing the iOS static framework.
#   BUNDLE_NAME: the pod/bundle name of the iOS static framework.
#   ALLOWLIST_FILE_PATH: contains the allowed symbols.
#   EXTRACT_SCRIPT_PATH: path to the extract_object_files script.
#   OUTPUT: the output zip file.

#!/usr/bin/env bash
set -euo pipefail

MKTEMP=/usr/bin/mktemp
LD_DEBUGGABLE_FLAGS="-x"

# --- Validate allowlist ---
if grep -q "^__Z" "$ALLOWLIST_FILE_PATH"; then
  echo "ERROR: C++ symbols are not allowed in the allowlist."
  grep "^__Z" "$ALLOWLIST_FILE_PATH"
  exit 1
fi

# --- Workspace ---
framework="$($MKTEMP -d -t framework)"
unzip -q "$INPUT_FRAMEWORK" -d "$framework"

executable_file="$BUNDLE_NAME.framework/$BUNDLE_NAME"
binary="$framework/$executable_file"

# --- Detect architectures ---
archs_str=$(xcrun lipo -info "$binary" | sed -E 's/.*: //')
read -ra archs <<< "$archs_str"

merge_cmd=(xcrun lipo)

for arch in "${archs[@]}"; do
  archdir="$($MKTEMP -d -t "$arch")"
  arch_file="$archdir/$arch"

  # --- Extract architecture ---
  if (( ${#archs[@]} > 1 )); then
    xcrun lipo "$binary" -thin "$arch" -output "$arch_file"
  else
    cp "$binary" "$arch_file"
  fi

  # --- Thread local warning ---
  if [[ "$arch" == "armv7" ]]; then
    thread_locals=$(xcrun nm -m -g "$arch_file" \
      | awk '/__DATA,__thread_vars/ {print $5}' \
      | c++filt || true)

    if [[ -n "$thread_locals" ]]; then
      echo "WARNING: thread local variables detected:"
      echo "$thread_locals"
    fi
  fi

  # --- Extract object files ---
  if [[ -n "${EXTRACT_SCRIPT_PATH:-}" ]]; then
    "$EXTRACT_SCRIPT_PATH" "$arch_file" "$archdir"
  else
    (
      cd "$archdir"
      xcrun ar -x "$arch_file"
    )
  fi

  objects_file_list="$($MKTEMP)"

  find "$archdir" -name "*.o" > "$objects_file_list"

  # --- Bitcode detection (faster) ---
  all_objects_have_bitcode=true
  while read -r obj; do
    if ! otool -arch "$arch" -l "$obj" | grep -q __LLVM; then
      all_objects_have_bitcode=false
      break
    fi
  done < "$objects_file_list"

  output="$arch_file"_processed.o

  if $all_objects_have_bitcode; then
    echo "$arch fully bitcode-enabled"
    xcrun ld -r -bitcode_bundle \
      -exported_symbols_list "$ALLOWLIST_FILE_PATH" \
      $LD_DEBUGGABLE_FLAGS \
      -filelist "$objects_file_list" \
      -o "$output"
  else
    echo "$arch NOT fully bitcode-enabled"
    xcrun ld -r \
      -exported_symbols_list "$ALLOWLIST_FILE_PATH" \
      $LD_DEBUGGABLE_FLAGS \
      -filelist "$objects_file_list" \
      -o "$output"
  fi

  final_obj="$framework/$arch"
  mv "$output" "$final_obj"

  merge_cmd+=(-arch "$arch" "$final_obj")

  rm -rf "$archdir" "$objects_file_list"
done

# --- Merge architectures ---
"${merge_cmd[@]}" -create -output "$BUNDLE_NAME"

chmod +x "$BUNDLE_NAME"

rm "$framework/$executable_file"
mv "$BUNDLE_NAME" "$framework/$executable_file"

# --- Normalize timestamps ---
TZ=UTC find "$framework/$BUNDLE_NAME.framework" -exec touch -h -t 198001010000 {} +

# --- Package ---
(
  cd "$framework"
  zip -qry --compression-method store --symlinks "$OUTPUT" "$BUNDLE_NAME.framework"
)
