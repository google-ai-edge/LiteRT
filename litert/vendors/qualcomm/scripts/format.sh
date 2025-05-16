#!/bin/bash

print_usage() {
    echo "Usage: $0 <folder_path> <buildifier_path>"
    echo "  example:"
    echo "    $0 ./litert/vendors/qualcomm/ /usr/local/bin/buildifier"
}

if [ "$#" -eq 1 ]; then
    if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        print_usage
        exit 0
    fi
fi

if [ "$#" -ne 2 ]; then
    print_usage
    exit 1
fi

script_dir=$(dirname "$(readlink -f "$0")")
format_file="$script_dir/.clang-format"
folder_path="$1"
buidifier_path="$2"

# format source code
find "$folder_path" -type f \( -name "*.c" -o -name "*.cc" -o -name "*.h" \) | while read -r file
do
    clang-format -i --style=file:$format_file $file
done

# format bazel files
find "$folder_path" -type f \( -name "BUILD" -o -name "*.bzl" \) | while read -r file
do
    $buidifier_path $file
done
