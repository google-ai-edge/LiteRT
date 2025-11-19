# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

reset_color='\033[0m'
host_color='\033[34m'
hightlight_color='\033[36m'
error_color='\033[31m'

# Print a message in the canonical color.
function print() {
  echo -e "${host_color}${1}${reset_color}"
}

# Print a message in the canonical hightlight color.
function print_hightlight() {
  echo -e "${hightlight_color}${1}${reset_color}"
}

# Print a file or host file device file pair.
function print_file() {
  if [[ "$#" -ne 2 ]]; then
    echo -e "    ${1}"
  else
    echo -e "    ${1} => ${2}"
  fi
}

# Print message and exit.
function fatal() {
  echo -e "${error_color}ERROR: ${reset_color}${1}"
  exit 1
}

# Join with delim.
function str_join() {
  local IFS=$1
  shift
  echo "$*"
}
