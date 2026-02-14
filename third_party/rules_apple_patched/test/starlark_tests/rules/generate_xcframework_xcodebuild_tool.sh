#!/bin/bash

set -euo pipefail

libraries=()
headers=()
frameworks=()
debug_symbols=()
output=

while [[ $# -gt 0 ]]; do
    case "$1" in
    --library)
        libraries+=("$2")
        shift
        ;;
    --headers)
        headers+=("$2")
        shift
        ;;
    --framework)
        frameworks+=("$2")
        shift
        ;;
    --debug-symbols)
        debug_symbols+=("$2")
        shift
        ;;
    --output)
        output="$2"
        shift
        ;;
    *)
        echo "Unrecognized option: $1" >&2
        exit 1
        ;;
    esac
    shift
done

if [[ -z "$output" ]]; then
    echo "--output is required for this program, but none was passed"
    exit 1
fi

create_xcframework_args=()

for library in "${libraries[@]+"${libraries[@]}"}"; do
    create_xcframework_args+=("-library" "$library")

    library_dirname="$(dirname "$library")"

    # -headers arguments must be positioned next to their respective -library
    for header in "${headers[@]+"${headers[@]}"}"; do
        if [[ "$header" == $library_dirname/* ]]; then
            create_xcframework_args+=("-headers" "$header")
        fi
    done
done

for framework in "${frameworks[@]+"${frameworks[@]}"}"; do
    create_xcframework_args+=("-framework" "$framework")

    framework_name="$(basename "$framework")"
    framework_target="$(basename "$(dirname "$framework")")"

    # -debug-symbols arguments must be positioned next to their respective -framework,
    # and must be absolute physical paths (no symlinks)
    for dsym in "${debug_symbols[@]+"${debug_symbols[@]}"}"; do
        if [[ "$dsym" == *$framework_target/$framework_name.dSYM ]]; then
            create_xcframework_args+=("-debug-symbols" "$(realpath "$dsym")")
        fi
    done
done

create_xcframework_args+=("-output" "$output")

# We must delete already anything at the output location (such as a previous output) otherwise xcodebuild refuses to produce anything and raises an error.
rm -rf "$output"
xcodebuild -create-xcframework "${create_xcframework_args[@]}"
