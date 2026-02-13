#!/bin/bash

# Copyright 2020 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eu

saw_text_segment=0
saw_vmaddr=0
text_vmaddr=0
saw_lc_main=0
saw_entryoff=0
entryoff=0

while read -r line ; do
  if [[ $saw_text_segment -eq 0 && "$line" =~ 'segname __TEXT' ]] ; then
    # Record that we saw the first __TEXT segment if we haven't seen one yet.
    saw_text_segment=1
  elif [[ $saw_text_segment -eq 1 && $saw_vmaddr -eq 0 && "$line" =~ 'vmaddr '0x([0-9a-fA-F]+) ]] ; then
    # Record the virtual memory address of the first load command that populates
    # the __TEXT segment.
    saw_vmaddr=1
    text_vmaddr="${BASH_REMATCH[1]}"
  elif [[ $saw_vmaddr -eq 1 && "$line" =~ 'cmd LC_MAIN' ]] ; then
    # Record that we saw the LC_MAIN load command (a well-formed binary should
    # have only one).
    saw_lc_main=1
  elif [[ $saw_lc_main -eq 1 && "$line" =~ 'entryoff '([0-9]+) ]] ; then
    # Record the offset of the entry point within the __TEXT segment.
    saw_entryoff=1
    entryoff="${BASH_REMATCH[1]}"
    break
  fi
done < <(xcrun llvm-objdump --macho --private-headers "$BINARY")

# Fail if we didn't get a valid entryoff; something must have gone wrong
# earlier.
[[ $saw_entryoff -eq 1 ]] || \
    fail "Did not see valid llvm-objdump output (no text segment, vmaddr, or LC_MAIN command with entryoff?)"

# Add the entry point offset to the __TEXT segment virtual address to get the
# actual symbol address. Shelling out to a Python one-liner like this is vile,
# but doing 64-bit hex address arithmetic in bash seems worse. (The [2:] strips
# off the leading "0x" that the hex function adds to the string.)
entry_addr=$(/usr/bin/env python3 -c "print(hex(int('${text_vmaddr}', 16) + ${entryoff})[2:])")

# Find the desired entry point symbol in the symbol table with the same address.
#
# The entry point could be a symbol in the binary itself, or it could be an
# indirect symbol stub (e.g., _NSExtensionMain in Foundation.framework). We need
# to dump both tables to make sure we find what we're looking for, and that
# affects the regex slightly; indirect symbol hex addresses are preceded by
# "0x", but regular symbol hex addresses are not. In either case, we allow
# an arbitrary number of leading zeroes so that we don't need to have the Python
# snippet above match the objdump output exactly.
TEMP_OUTPUT="$(mktemp "${TMPDIR:-${TEST_TMPDIR:-/tmp}}/entry_point_output.XXXXXX")"
xcrun llvm-objdump --macho --syms --indirect-symbols "$BINARY" > "$TEMP_OUTPUT"

if ! grep "^\(0x\)\?0*${entry_addr}[[:space:]].*[[:space:]]${ENTRY_POINT}$" "$TEMP_OUTPUT" ; then
  cat "$TEMP_OUTPUT" >>$TEST_log
  fail "Did not find symbol ${ENTRY_POINT} at address 0x${entry_addr}"
fi

rm -rf "$TEMP_OUTPUT"
