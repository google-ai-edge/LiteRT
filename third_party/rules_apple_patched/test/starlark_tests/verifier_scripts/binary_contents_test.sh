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

newline=$'\n'

# This script allows many of the functions in apple_shell_testutils.sh to be
# called through apple_verification_test_runner.sh.template by using environment
# variables.
#
# Supported operations:
#  BINARY_NOT_CONTAINS_ARCHITECTURES: The architectures to verify are not in the
#      assembled binary.
#  BINARY_TEST_FILE: The file to test with.
#  BINARY_TEST_ARCHITECTURE: The architecture to use with
#      `BINARY_CONTAINS_SYMBOLS`.
#  BINARY_CONTAINS_SYMBOLS: Array of symbols that should be present.
#  BINARY_NOT_CONTAINS_SYMBOLS: Array of symbols that should not be present.
#  BINARY_CONTAINS_FILE_INFO: Array of strings that should be present as
#      substrings of the reported `file` information.
#  MACHO_LOAD_COMMANDS_CONTAIN: Array of Mach-O load commands that should
#      be present.
#  MACHO_LOAD_COMMANDS_NOT_CONTAIN: Array of Mach-O load commands that should
#      not be present.
#  PLIST_SECTION_NAME: Name of the plist section to inspect values from. If not
#      supplied, will test the embedded Info.plist slice at __TEXT,__info_plist.
#  PLIST_TEST_VALUES: Array for keys and values in the format "KEY VALUE" where
#      the key is a string without spaces, followed by by a single space,
#      followed by the value to test. * can be used as a wildcard value.

if [[ -n "${BINARY_TEST_FILE-}" ]]; then
  path=$(eval echo "$BINARY_TEST_FILE")
  if [[ ! -e "$path" ]]; then
    fail "Could not find binary at \"$path\""
  fi
  something_tested=false

  if [[ -n "${BINARY_NOT_CONTAINS_ARCHITECTURES-}" ]]; then
    IFS=' ' found_archs=($(lipo -archs "$path"))
    for arch in "${BINARY_NOT_CONTAINS_ARCHITECTURES[@]}"
    do
      for found_arch in "${found_archs[@]}"
      do
        something_tested=true
        arch_found=false
        if [[ "$arch" == "$found_arch" ]]; then
          arch_found=true
          break
        fi
      done
      if [[ "$arch_found" = true ]]; then
        fail "Unexpectedly found architecture \"$arch\". The architectures " \
          "in the binary were:$newline${found_archs[@]}"
      fi
    done
  fi

  if [[ -n "${BINARY_TEST_ARCHITECTURE-}" ]]; then
    arch=$(eval echo "$BINARY_TEST_ARCHITECTURE")
    if [[ ! -n $arch ]]; then
      fail "No architecture specified for binary file at \"$path\""
    fi

    # Filter out undefined symbols from the objdump mach-o symbol output and
    # return the fifth from rightmost values, with the `.hidden` column stripped
    # where applicable.
    IFS=$'\n' actual_symbols=($(objdump --syms --macho --arch="$arch" "$path" | grep -v "*UND*" | awk '{print substr($0,index($0,$5))}' | sed 's/.hidden *//'))
    if [[ -n "${BINARY_CONTAINS_SYMBOLS-}" ]]; then
      for test_symbol in "${BINARY_CONTAINS_SYMBOLS[@]}"
      do
        something_tested=true
        symbol_found=false
        for actual_symbol in "${actual_symbols[@]}"
        do
          if [[ "$actual_symbol" == "$test_symbol" ]]; then
            symbol_found=true
            break
          fi
        done
        if [[ "$symbol_found" = false ]]; then
            fail "Expected symbol \"$test_symbol\" was not found. The " \
              "symbols in the binary were:$newline${actual_symbols[@]}"
        fi
      done
    fi

    if [[ -n "${BINARY_NOT_CONTAINS_SYMBOLS-}" ]]; then
      for test_symbol in "${BINARY_NOT_CONTAINS_SYMBOLS[@]}"
      do
        something_tested=true
        symbol_found=false
        for actual_symbol in "${actual_symbols[@]}"
        do
          if [[ "$actual_symbol" == "$test_symbol" ]]; then
            symbol_found=true
            break
          fi
        done
        if [[ "$symbol_found" = true ]]; then
            fail "Unexpected symbol \"$test_symbol\" was found. The symbols " \
              "in the binary were:$newline${actual_symbols[@]}"
        fi
      done
    fi

  if [[ -n "${MACHO_LOAD_COMMANDS_CONTAIN-}" || -n "${MACHO_LOAD_COMMANDS_NOT_CONTAIN-}" ]]; then
    # The `otool` commands below remove the leftmost white space from the
    # output to make string matching of symbols possible, avoiding the
    # accidental elimination of white space from paths and identifiers.
    IFS=$'\n'
    if [[ -n "${BINARY_TEST_ARCHITECTURE-}" ]]; then
      arch=$(eval echo "$BINARY_TEST_ARCHITECTURE")
      if [[ ! -n $arch ]]; then
        fail "No architecture specified for binary file at \"$path\""
      else
        actual_symbols=($(otool -v -arch "$arch" -l "$path" | awk '{$1=$1}1'))
      fi
    else
      actual_symbols=($(otool -v -l "$path" | awk '{$1=$1}1'))
    fi
    if [[ -n "${MACHO_LOAD_COMMANDS_CONTAIN-}" ]]; then
      for test_symbol in "${MACHO_LOAD_COMMANDS_CONTAIN[@]}"
      do
        something_tested=true
        symbol_found=false
        for actual_symbol in "${actual_symbols[@]}"
        do
          if [[ "$actual_symbol" == "$test_symbol" ]]; then
            symbol_found=true
            break
          fi
        done
        if [[ "$symbol_found" = false ]]; then
            fail "Expected load command \"$test_symbol\" was not found." \
              "The load commands in the binary were:" \
              "$newline${actual_symbols[@]}"
        fi
      done
    fi

    if [[ -n "${MACHO_LOAD_COMMANDS_NOT_CONTAIN-}" ]]; then
      for test_symbol in "${MACHO_LOAD_COMMANDS_NOT_CONTAIN[@]}"
      do
        something_tested=true
        symbol_found=false
        for actual_symbol in "${actual_symbols[@]}"
        do
          if [[ "$actual_symbol" == "$test_symbol" ]]; then
            symbol_found=true
            break
          fi
        done
        if [[ "$symbol_found" = true ]]; then
            fail "Unexpected load command \"$test_symbol\" was found." \
              "The load commands in the binary were:" \
              "$newline${actual_symbols[@]}"
        fi
      done
    fi
  fi

  else
    if [[ -n "${BINARY_CONTAINS_SYMBOLS-}" ]]; then
      fail "Rule Misconfigured: Supposed to look for symbols," \
        "but no arch was set to check: ${BINARY_CONTAINS_SYMBOLS[@]}"
    fi
    if [[ -n "${BINARY_NOT_CONTAINS_SYMBOLS-}" ]]; then
      fail "Rule Misconfigured: Supposed to look for missing symbols," \
        "but no arch was set to check: ${BINARY_NOT_CONTAINS_SYMBOLS[@]}"
    fi
    if [[ -n "${MACHO_LOAD_COMMANDS_CONTAIN-}" ]]; then
      fail "Rule Misconfigured: Supposed to look for macho load commands," \
        "but no arch was set to check: ${MACHO_LOAD_COMMANDS_CONTAIN[@]}"
    fi
    if [[ -n "${MACHO_LOAD_COMMANDS_NOT_CONTAIN-}" ]]; then
      fail "Rule Misconfigured: Supposed to look for missing macho load commands," \
        "but no arch was set to check: ${MACHO_LOAD_COMMANDS_NOT_CONTAIN[@]}"
    fi
  fi

  # Use `file --brief` to verify how the file is recognized by macOS, and other
  # Apple platforms by proxy.
  if [[ -n "${BINARY_CONTAINS_FILE_INFO-}" ]]; then
    IFS=$'\n' file_info_output=($(file --brief "$path" | awk '{$1=$1}1'))
    for file_info_test_substring in "${BINARY_CONTAINS_FILE_INFO[@]}"
    do
      something_tested=true
      output_found=false
      for file_info_line in "${file_info_output[@]}"
      do
        if [[ "$file_info_line" == *"$file_info_test_substring"* ]]; then
          output_found=true
          break
        fi
      done
      if [[ "$output_found" = false ]]; then
          fail "Expected file output \"$file_info_test_substring\" was not " \
            "found. The file output for the binary was:" \
            "$newline${file_info_output[@]}"
      fi
    done
  fi

  # Use `launchctl plist` to test for key/value pairs in an embedded plist file.
  if [[ -n "${PLIST_TEST_VALUES-}" ]]; then
    for test_values in "${PLIST_TEST_VALUES[@]}"
    do
      something_tested=true
      # Keys and expected-values are in the format "KEY VALUE".
      IFS=' ' read -r key expected_value <<< "$test_values"
      if [[ -z "${PLIST_SECTION_NAME-}" ]]; then
        fail "Rule Misconfigured: missing plist section," \
         "but not supposed to check for values: ${PLIST_TEST_VALUES}"
      fi
      plist_section_name="__TEXT,$PLIST_SECTION_NAME"
      # Replace wildcard "*" characters with a sed-friendly ".*" wildcard.
      expected_value=${expected_value/"*"/".*"}
      value="$(launchctl plist $plist_section_name $path | sed -nE "s/.*\"$key\" = \"($expected_value)\";.*/\1/p" || true)"
      if [[ ! -n "$value" ]]; then
        fail "Expected plist key \"$key\" to be \"$expected_value\" in plist " \
            "embedded in \"$path\" at \"$plist_section_name\". Plist " \
            "contents:$newline$(launchctl plist $plist_section_name $path)"
      fi
    done
  else
    # Don't error if PLIST_SECTION_NAME is set because the rule defaults it.
    true
  fi

  if [[ "$something_tested" = false ]]; then
    fail "Rule Misconfigured: Nothing was configured to be validated on the binary \"$path\""
  fi
else
  fail "Rule Misconfigured: No binary was set to be inspected"
fi
