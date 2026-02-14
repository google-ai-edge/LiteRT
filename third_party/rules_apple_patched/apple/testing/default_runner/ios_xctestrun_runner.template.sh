#!/bin/bash
# This script replaces the variables in the templated xctestrun file with the
# the specific paths to the test bundle, and the optionally test host

set -euo pipefail

if [[ -n "${TEST_PREMATURE_EXIT_FILE:-}" ]]; then
  touch "$TEST_PREMATURE_EXIT_FILE"
fi

if [[ -n "${DEBUG_XCTESTRUNNER:-}" ]]; then
  set -x
fi

create_xcresult_bundle="%(create_xcresult_bundle)s"
if [[ -n "${CREATE_XCRESULT_BUNDLE:-}" ]]; then
  create_xcresult_bundle=true
fi

custom_xcodebuild_args=(%(xcodebuild_args)s)
simulator_name=""
device_id=""
command_line_args=(%(command_line_args)s)
attachment_lifetime="%(attachment_lifetime)s"
destination_timeout="%(destination_timeout)s"
while [[ $# -gt 0 ]]; do
  arg="$1"
  case $arg in
    --simulator_name=*)
      simulator_name="${arg##*=}"
      ;;
    --xcodebuild_args=*)
      xcodebuild_arg="${arg#--xcodebuild_args=}" # Strip "--xcodebuild_args=" prefix
      custom_xcodebuild_args+=("$xcodebuild_arg")
      ;;
    --destination=platform=iOS,id=*)
      device_id="${arg##*=}"
      ;;
    --command_line_args=*)
      command_line_args+=("${arg##*=}")
      ;;
    --xctestrun_attachment_lifetime=*)
      attachment_lifetime="${arg##*=}"
      ;;
    *)
      echo "error: Unsupported argument '${arg}'" >&2
      exit 1
      ;;
  esac
  shift
done

# Retrieve the basename of a file or folder with an extension.
basename_without_extension() {
  local filename
  filename=$(basename "$1")
  echo "${filename%.*}"
}

test_tmp_dir="$(mktemp -d "${TEST_TMPDIR:-${TMPDIR:-/tmp}}/test_tmp_dir.XXXXXX")"
if [[ -z "${NO_CLEAN:-}" ]]; then
  trap 'rm -rf "${test_tmp_dir}"' EXIT
else
  test_tmp_dir="${TMPDIR:-/tmp}/test_tmp_dir"
  rm -rf "$test_tmp_dir"
  mkdir -p "$test_tmp_dir"
  echo "note: keeping test dir around at: $test_tmp_dir"
fi

test_bundle_path="%(test_bundle_path)s"
test_bundle_name=$(basename_without_extension "$test_bundle_path")
test_bundle_binary="$test_tmp_dir/$test_bundle_name.xctest/$test_bundle_name"

if [[ "$test_bundle_path" == *.xctest ]]; then
  cp -cRL "$test_bundle_path" "$test_tmp_dir"
  # Need to modify permissions as Bazel will set all files to non-writable, and
  # Xcode's test runner requires the files to be writable.
  chmod -R 777 "$test_tmp_dir/$test_bundle_name.xctest"
else
  unzip -qq -d "${test_tmp_dir}" "${test_bundle_path}"
fi

# Delta update won't update the binary if it has the same timestamp
touch "$test_bundle_binary"

build_for_device=false
test_execution_platform="iPhoneSimulator.platform"
if [[ -n "$device_id" ]]; then
  test_execution_platform="iPhoneOS.platform"
  build_for_device=true
fi

# In case there is no test host, test_host_path will be empty
test_host_path="%(test_host_path)s"
if [[ -n "$test_host_path" ]]; then
  test_host_name=$(basename_without_extension "$test_host_path")

  if [[ "$test_host_path" == *.app ]]; then
    cp -cRL "$test_host_path" "$test_tmp_dir"
    # Need to modify permissions as Bazel will set all files to non-writable,
    # and Xcode's test runner requires the files to be writable.
    chmod -R 777 "$test_tmp_dir/$test_host_name.app"
  else
    unzip -qq -d "${test_tmp_dir}" "${test_host_path}"
    mv "$test_tmp_dir"/Payload/*.app "$test_tmp_dir"
    # When extracting an ipa file we don't know the name of the app bundle
    test_tmp_dir_test_host_path=$(find "$test_tmp_dir" -name "*.app" -type d -maxdepth 1 -mindepth 1 -print -quit)
    test_host_name=$(basename_without_extension "$test_tmp_dir_test_host_path")
  fi
fi

# Basic XML character escaping for environment variable substitution.
function escape() {
  local escaped=${1//&/&amp;}
  escaped=${escaped//</&lt;}
  escaped=${escaped//>/&gt;}
  escaped=${escaped//'"'/&quot;}
  echo "$escaped"
}

# Gather command line arguments for `CommandLineArguments` in the xctestrun file
xctestrun_cmd_line_args_section=""
if [[ -n "${command_line_args:-}" ]]; then
  xctestrun_cmd_line_args_section="\n"
  saved_IFS=$IFS
  IFS=","
  for cmd_line_arg in ${command_line_args[@]}; do
    xctestrun_cmd_line_args_section+="      <string>$cmd_line_arg</string>\n"
  done
  IFS=$saved_IFS
  xctestrun_cmd_line_args_section="    <key>CommandLineArguments</key>\n    <array>$xctestrun_cmd_line_args_section    </array>"
fi

# Add the test environment variables into the xctestrun file to propagate them
# to the test runner
default_test_env="TEST_PREMATURE_EXIT_FILE=$TEST_PREMATURE_EXIT_FILE,TEST_SRCDIR=$TEST_SRCDIR,TEST_UNDECLARED_OUTPUTS_DIR=$TEST_UNDECLARED_OUTPUTS_DIR,XML_OUTPUT_FILE=$XML_OUTPUT_FILE"
test_env="%(test_env)s"
env_inherit=%(test_env_inherit)s
for env_var in "${env_inherit[@]:-}"; do
  # If the environment variable is set, add it to the test environment
  if declare -p "$env_var" &>/dev/null; then
    if [[ -n "$test_env" ]]; then
      test_env="$test_env,$env_var=${!env_var}"
    else
      test_env="$env_var=${!env_var}"
    fi
  fi
done

if [[ -n "$device_id" ]]; then
  default_test_env="$default_test_env,BAZEL_DEVICE_UDID=$device_id"
fi

if [[ -n "$test_env" ]]; then
  test_env="$test_env,$default_test_env"
else
  test_env="$default_test_env"
fi

passthrough_env=()
xctestrun_env=""
saved_IFS=$IFS
IFS=","
for test_env_key_value in ${test_env}; do
  IFS="=" read -r key value <<< "$test_env_key_value"
  xctestrun_env+="<key>$(escape "$key")</key><string>$(escape "$value")</string>"
  passthrough_env+=("SIMCTL_CHILD_$key=$value")
done
IFS=$saved_IFS

declare -r sed_delim=$'\001'

xcrun_target_app_path=""
xcrun_test_host_bundle_identifier=""
xcrun_test_bundle_path="__TESTROOT__/$test_bundle_name.xctest"
xcrun_is_xctrunner_hosted_bundle="false"
xcrun_is_ui_test_bundle="false"
test_type="%(test_type)s"
if [[ -n "$test_host_path" ]]; then
  developer_dir=$(xcode-select -p)

  xctestrun_test_host_path="__TESTROOT__/$test_host_name.app"
  xctestrun_test_host_based=true
  # If this is set in the case there is no test host, some tests hang indefinitely
  xctestrun_env+="<key>XCInjectBundleInto</key><string>$(escape "__TESTHOST__/$test_host_name.app/$test_host_name")</string>"

  developer_path="$developer_dir/Platforms/$test_execution_platform/Developer"
  libraries_path="$developer_path/Library"

  # Added in Xcode 16.0
  testing_framework_path="$libraries_path/Frameworks/Testing.framework"
  if [[ -d "$testing_framework_path" ]]; then
    xctestrun_env+="<key>DYLD_FRAMEWORK_PATH</key><string>$libraries_path/Frameworks</string>"
  fi

  if [[ "$test_type" = "XCUITEST" ]]; then
    xcrun_is_xctrunner_hosted_bundle="true"
    xcrun_is_ui_test_bundle="true"
    xcrun_target_app_path="$xctestrun_test_host_path"
    # If ui testing is enabled we need to copy out the XCTRunner app, update its info.plist accordingly and finally
    # copy over the needed frameworks to enable ui testing
    readonly runner_app_name="$test_bundle_name-Runner"
    readonly runner_app="$runner_app_name.app"
    readonly runner_app_destination="$test_tmp_dir/$runner_app"
    cp -R "$libraries_path/Xcode/Agents/XCTRunner.app" "$runner_app_destination"
    chmod -R 777 "$runner_app_destination"
    xctestrun_test_host_path="__TESTROOT__/$runner_app"
    xcrun_test_host_bundle_identifier="com.apple.test.$runner_app_name"
    plugins_path="$test_tmp_dir/$runner_app/PlugIns"
    mkdir -p "$plugins_path"
    mv "$test_tmp_dir/$test_bundle_name.xctest" "$plugins_path"
    test_bundle_binary="$plugins_path/$test_bundle_name.xctest/$test_bundle_name"
    mkdir -p "$plugins_path/$test_bundle_name.xctest/Frameworks"
    # We need this dylib for 14.x OSes. This intentionally doesn't use `test_execution_platform`
    # since this file isn't present in the `iPhoneSimulator.platform`.
    # No longer necessary starting in Xcode 15 - hence the `-f` file existence check
    libswift_concurrency_path="$developer_dir/Platforms/iPhoneOS.platform/Library/Developer/CoreSimulator/Profiles/Runtimes/iOS.simruntime/Contents/Resources/RuntimeRoot/usr/lib/swift/libswift_Concurrency.dylib"
    if [[ -f "$libswift_concurrency_path" ]]; then
      cp "$libswift_concurrency_path" "$plugins_path/$test_bundle_name.xctest/Frameworks/libswift_Concurrency.dylib"
    fi
    xcrun_test_bundle_path="__TESTHOST__/PlugIns/$test_bundle_name.xctest"

    runner_app_infoplist="$runner_app_destination/Info.plist"
    /usr/bin/plutil -convert xml1 "$runner_app_infoplist"
    /usr/bin/sed \
      -e "s${sed_delim}\$(WRAPPEDPRODUCTNAME)${sed_delim}XCTRunner${sed_delim}g"\
      -e "s${sed_delim}WRAPPEDPRODUCTNAME${sed_delim}XCTRunner${sed_delim}g"\
      -e "s${sed_delim}\$(WRAPPEDPRODUCTBUNDLEIDENTIFIER)${sed_delim}$xcrun_test_host_bundle_identifier${sed_delim}g"\
      -e "s${sed_delim}WRAPPEDPRODUCTBUNDLEIDENTIFIER${sed_delim}$xcrun_test_host_bundle_identifier${sed_delim}g"\
      -i "" \
      "$runner_app_infoplist"
    /usr/bin/plutil -convert binary1 "$runner_app_infoplist"

    readonly runner_app_frameworks_destination="$runner_app_destination/Frameworks"
    mkdir -p "$runner_app_frameworks_destination"
    cp -R "$libraries_path/Frameworks/XCTest.framework" "$runner_app_frameworks_destination/XCTest.framework"
    cp -R "$libraries_path/PrivateFrameworks/XCTestCore.framework" "$runner_app_frameworks_destination/XCTestCore.framework"
    cp -R "$libraries_path/PrivateFrameworks/XCTAutomationSupport.framework" "$runner_app_frameworks_destination/XCTAutomationSupport.framework"
    cp -R "$libraries_path/PrivateFrameworks/XCUnit.framework" "$runner_app_frameworks_destination/XCUnit.framework"
    cp "$developer_path/usr/lib/libXCTestSwiftSupport.dylib" "$runner_app_frameworks_destination/libXCTestSwiftSupport.dylib"
    cp "$developer_path/usr/lib/libXCTestBundleInject.dylib" "$runner_app_frameworks_destination/libXCTestBundleInject.dylib"
    # Added in Xcode 14.3
    xctestsupport_framework_path="$libraries_path/PrivateFrameworks/XCTestSupport.framework"
    if [[ -d "$xctestsupport_framework_path" ]]; then
      cp -R "$xctestsupport_framework_path" "$runner_app_frameworks_destination/XCTestSupport.framework"
    fi
    # Added in Xcode 16.0
    if [[ -d "$testing_framework_path" ]]; then
      cp -R "$testing_framework_path" "$runner_app_frameworks_destination/Testing.framework"
    fi

    # On Xcode 16.3 and later, XCUIAutomation is not private anymore
    xcuiautomation_path="$libraries_path/Frameworks/XCUIAutomation.framework"
    if [[ ! -d "$xcuiautomation_path" ]]; then
        xcuiautomation_path="$libraries_path/PrivateFrameworks/XCUIAutomation.framework"
    fi
    cp -R "$xcuiautomation_path" "$runner_app_frameworks_destination/XCUIAutomation.framework"

    if [[ "$build_for_device" == true ]]; then
      # XCTRunner is multi-archs. When launching XCTRunner on arm64e device, it
      # will be launched as arm64e process by default. If the test bundle is arm64
      # bundle, the XCTRunner which hosts the test bundle will fail to be
      # launched. So removing the arm64e arch from XCTRunner can resolve this
      # case.
      /usr/bin/lipo "$test_tmp_dir/$runner_app/XCTRunner" -remove arm64e -output "$test_tmp_dir/$runner_app/XCTRunner"
    fi
    test_host_mobileprovision_path="$test_tmp_dir/$test_host_name.app/embedded.mobileprovision"
    # Only engage signing workflow if the test host is signed
    if [[ -f "$test_host_mobileprovision_path" ]]; then
      cp "$test_host_mobileprovision_path" "$test_tmp_dir/$runner_app/embedded.mobileprovision"
      xctrunner_entitlements="$test_tmp_dir/$runner_app/RunnerEntitlements.plist"
      test_host_binary_path="$test_tmp_dir/$test_host_name.app/$test_host_name"
      codesigning_team_identifier=$(codesign -dvv "$test_host_binary_path"  2>&1 >/dev/null | /usr/bin/sed -n  -E 's/TeamIdentifier=(.*)/\1/p')
      codesigning_authority=$(codesign -dvv "$test_host_binary_path"  2>&1 >/dev/null | /usr/bin/sed -n  -E 's/^Authority=(.*)/\1/p'| head -n 1)
      /usr/bin/sed \
        -e "s${sed_delim}BAZEL_CODESIGNING_TEAM_IDENTIFIER${sed_delim}$codesigning_team_identifier${sed_delim}g" \
        -e "s${sed_delim}BAZEL_TEST_HOST_BUNDLE_IDENTIFIER${sed_delim}$xcrun_test_host_bundle_identifier${sed_delim}g" \
        "%(xctrunner_entitlements_template)s" > "$xctrunner_entitlements"
      codesign -f \
        --entitlements "$xctrunner_entitlements" \
        --timestamp=none -s "$codesigning_authority" \
        "$plugins_path/$test_bundle_name.xctest"
      find "$test_tmp_dir/$runner_app/Frameworks" \
        -type d \
        -name "*.framework" \
        -exec codesign -f --timestamp=none -s "$codesigning_authority" --entitlements "$xctrunner_entitlements" {} \;
      find "$test_tmp_dir/$runner_app/Frameworks" \
        -type f \
        -name "*.dylib" \
        -exec codesign -f --timestamp=none -s "$codesigning_authority" --entitlements "$xctrunner_entitlements" {} \;
      codesign -f \
        --entitlements "$xctrunner_entitlements" \
        --timestamp=none \
        -s "$codesigning_authority" \
        "$test_tmp_dir/$runner_app"
    fi
  fi
else
  xctestrun_test_host_path="__PLATFORMS__/$test_execution_platform/Developer/Library/Xcode/Agents/xctest"
  xctestrun_test_host_based=false
fi

sanitizer_dyld_env=""
readonly sanitizer_root="$test_tmp_dir/$test_bundle_name.xctest/Frameworks"
for sanitizer in "$sanitizer_root"/libclang_rt.*.dylib; do
  [[ -e "$sanitizer" ]] || continue

  if [[ -n "$sanitizer_dyld_env" ]]; then
    sanitizer_dyld_env="$sanitizer_dyld_env:"
  fi
  sanitizer_dyld_env="${sanitizer_dyld_env}${sanitizer}"
done

main_thread_checker_dyld_env=""
readonly main_thread_checker_root="$test_tmp_dir/$test_bundle_name.xctest/Frameworks"
main_thread_checker="$main_thread_checker_root/libMainThreadChecker.dylib"
if [[ -e "$main_thread_checker" ]]; then
    main_thread_checker_dyld_env="$main_thread_checker"
fi

xctestrun_libraries=""
if [[ "$test_type" != "XCUITEST" ]]; then
  xctestrun_libraries="__PLATFORMS__/$test_execution_platform/Developer/usr/lib/libXCTestBundleInject.dylib"
fi

if [[ -n "$sanitizer_dyld_env" ]]; then
  if [[ -n "$xctestrun_libraries" ]]; then
    xctestrun_libraries="${xctestrun_libraries}:${sanitizer_dyld_env}"
  else
    xctestrun_libraries="${sanitizer_dyld_env}"
  fi
fi

if [[ -n "$main_thread_checker_dyld_env" ]]; then
  if [[ -n "$xctestrun_libraries" ]]; then
    xctestrun_libraries="${xctestrun_libraries}:${main_thread_checker_dyld_env}"
  else
    xctestrun_libraries="${main_thread_checker_dyld_env}"
  fi
fi

TEST_FILTER="%(test_filter)s"
xctestrun_skip_test_section=""
xctestrun_only_test_section=""

# Use the 'TESTBRIDGE_TEST_ONLY' environment variable set by Bazel's
# '--test_filter' flag to set the xctestrun's skip/only parameters.
#
# Any test prefixed with '-' will be passed to 'SkipTestIdentifiers'. Otherwise
# the tests is passed to 'OnlyTestIdentifiers',
if [[ -n "${TESTBRIDGE_TEST_ONLY:-}" || -n "${TEST_FILTER:-}" ]]; then
  if [[ -n "${TESTBRIDGE_TEST_ONLY:-}" && -n "${TEST_FILTER:-}" ]]; then
    ALL_TESTS="$TESTBRIDGE_TEST_ONLY,$TEST_FILTER"
  elif [[ -n "${TESTBRIDGE_TEST_ONLY:-}" ]]; then
    ALL_TESTS="$TESTBRIDGE_TEST_ONLY"
  else
    ALL_TESTS="$TEST_FILTER"
  fi

  saved_IFS=$IFS
  IFS=","; for TEST in $ALL_TESTS; do
    if [[ $TEST == -* ]]; then
      if [[ -n "${SKIP_TESTS:-}" ]]; then
        SKIP_TESTS+=",${TEST:1}"
      else
        SKIP_TESTS="${TEST:1}"
      fi
    else
      if [[ -n "${ONLY_TESTS:-}" ]]; then
          ONLY_TESTS+=",$TEST"
      else
          ONLY_TESTS="$TEST"
      fi
    fi
  done

  IFS=$saved_IFS

  if [[ -n "${SKIP_TESTS:-}" ]]; then
    xctestrun_skip_test_section="\n"
    for skip_test in ${SKIP_TESTS//,/ }; do
      xctestrun_skip_test_section+="      <string>$skip_test</string>\n"
    done
    xctestrun_skip_test_section="    <key>SkipTestIdentifiers</key>\n    <array>$xctestrun_skip_test_section    </array>"
  fi

  if [[ -n "${ONLY_TESTS:-}" ]]; then
    xctestrun_only_test_section="\n"
    for only_test in ${ONLY_TESTS//,/ }; do
      xctestrun_only_test_section+="      <string>$only_test</string>\n"
    done
    xctestrun_only_test_section="    <key>OnlyTestIdentifiers</key>\n    <array>$xctestrun_only_test_section    </array>"
  fi
fi

readonly profraw="$test_tmp_dir/coverage.profraw"

simulator_creator_args=(
  "%(os_version)s" \
  "%(device_type)s" \
  --name "$simulator_name"
)

reuse_simulator=%(reuse_simulator)s
if [[ "$reuse_simulator" == true ]]; then
  simulator_creator_args+=(--reuse-simulator)
else
  simulator_creator_args+=(--no-reuse-simulator)
fi

simulator_id="unused"
if [[ "$build_for_device" == false ]]; then
  simulator_id="$("./%(simulator_creator.py)s" \
    "${simulator_creator_args[@]}"
  )"
fi

test_exit_code=0
readonly testlog=$test_tmp_dir/test.log
test_file=$(file "$test_bundle_binary")

intel_simulator_hack=false
architecture="arm64"
if [[ $(arch) == arm64 && "$test_file" != *arm64* ]]; then
  intel_simulator_hack=true
  architecture="x86_64"
fi

should_use_xcodebuild=false
if [[ "$build_for_device" == true  ]]; then
  echo "note: Using 'xcodebuild' because build for device was requested"
  should_use_xcodebuild=true
fi
if [[ -n "$test_host_path" ]]; then
  echo "note: Using 'xcodebuild' because test host was provided"
  should_use_xcodebuild=true
fi
# shellcheck disable=SC2050
if [[ "%(test_order)s" == random ]]; then
  echo "note: Using 'xcodebuild' because random test order was requested"
  should_use_xcodebuild=true
fi
if [[ "$create_xcresult_bundle" == true ]]; then
  echo "note: Using 'xcodebuild' because XCResult bundle was requested"
  should_use_xcodebuild=true
fi
if [[ -n "$xctestrun_cmd_line_args_section" ]]; then
  echo "note: Using 'xcodebuild' because '--command_line_args' was provided"
  should_use_xcodebuild=true
fi
if [[ -n "$xctestrun_skip_test_section" || -n "$xctestrun_only_test_section" ]]; then
  echo "note: Using 'xcodebuild' because test filter was provided"
  should_use_xcodebuild=true
fi
if (( ${#custom_xcodebuild_args[@]} )); then
  echo "note: Using 'xcodebuild' because '--xcodebuild_args' was provided"
  should_use_xcodebuild=true
fi

# Run a pre-action binary, if provided.
pre_action_binary=%(pre_action_binary)s
SIMULATOR_UDID="$simulator_id" \
  "$pre_action_binary"

if [[ "$should_use_xcodebuild" == true ]]; then
  if [[ -z "$test_host_path" && "$intel_simulator_hack" == true ]]; then
    echo "error: running x86_64 tests on arm64 macs using 'xcodebuild' requires a test host" >&2
    exit 1
  fi

  # Set xctest attachment liftime
  xctestrun_attachment_lifetime_section+="    <key>SystemAttachmentLifetime</key>\n"
  xctestrun_attachment_lifetime_section+="    <string>$attachment_lifetime</string>\n"
  xctestrun_attachment_lifetime_section+="    <key>UserAttachmentLifetime</key>\n"
  xctestrun_attachment_lifetime_section+="    <string>$attachment_lifetime</string>"

  readonly xctestrun_file="$test_tmp_dir/tests.xctestrun"
  /usr/bin/sed \
    -e "s${sed_delim}BAZEL_INSERT_LIBRARIES${sed_delim}$xctestrun_libraries${sed_delim}g" \
    -e "s${sed_delim}BAZEL_TEST_BUNDLE_PATH${sed_delim}$xcrun_test_bundle_path${sed_delim}g" \
    -e "s${sed_delim}BAZEL_TEST_ENVIRONMENT${sed_delim}$xctestrun_env${sed_delim}g" \
    -e "s${sed_delim}BAZEL_TEST_HOST_BASED${sed_delim}$xctestrun_test_host_based${sed_delim}g" \
    -e "s${sed_delim}BAZEL_TEST_HOST_PATH${sed_delim}$xctestrun_test_host_path${sed_delim}g" \
    -e "s${sed_delim}BAZEL_TEST_HOST_BUNDLE_IDENTIFIER${sed_delim}$xcrun_test_host_bundle_identifier${sed_delim}g" \
    -e "s${sed_delim}BAZEL_TEST_PRODUCT_MODULE_NAME${sed_delim}${test_bundle_name//-/_}${sed_delim}g" \
    -e "s${sed_delim}BAZEL_IS_XCTRUNNER_HOSTED_BUNDLE${sed_delim}$xcrun_is_xctrunner_hosted_bundle${sed_delim}g" \
    -e "s${sed_delim}BAZEL_IS_UI_TEST_BUNDLE${sed_delim}$xcrun_is_ui_test_bundle${sed_delim}g" \
    -e "s${sed_delim}BAZEL_TARGET_APP_PATH${sed_delim}$xcrun_target_app_path${sed_delim}g" \
    -e "s${sed_delim}BAZEL_TEST_ORDER_STRING${sed_delim}%(test_order)s${sed_delim}g" \
    -e "s${sed_delim}BAZEL_DYLD_LIBRARY_PATH${sed_delim}__PLATFORMS__/$test_execution_platform/Developer/usr/lib${sed_delim}g" \
    -e "s${sed_delim}BAZEL_COVERAGE_OUTPUT_DIR${sed_delim}$test_tmp_dir${sed_delim}g" \
    -e "s${sed_delim}BAZEL_COMMAND_LINE_ARGS_SECTION${sed_delim}$xctestrun_cmd_line_args_section${sed_delim}g" \
    -e "s${sed_delim}BAZEL_ATTACHMENT_LIFETIME_SECTION${sed_delim}$xctestrun_attachment_lifetime_section${sed_delim}g" \
    -e "s${sed_delim}BAZEL_SKIP_TEST_SECTION${sed_delim}$xctestrun_skip_test_section${sed_delim}g" \
    -e "s${sed_delim}BAZEL_ONLY_TEST_SECTION${sed_delim}$xctestrun_only_test_section${sed_delim}g" \
    -e "s${sed_delim}BAZEL_ARCHITECTURE${sed_delim}$architecture${sed_delim}g" \
    -e "s${sed_delim}BAZEL_TEST_BUNDLE_NAME${sed_delim}$test_bundle_name.xctest${sed_delim}g" \
    -e "s${sed_delim}BAZEL_PRODUCT_PATH${sed_delim}$xcrun_test_bundle_path${sed_delim}g" \
    "%(xctestrun_template)s" \
    > "$xctestrun_file"

  if [[ -n "${DEBUG_XCTESTRUNNER:-}" ]]; then
    echo
    echo "xctestrun contents:"
    cat "$xctestrun_file"
    echo
  fi

  args=(
    -xctestrun "$xctestrun_file" \
  )

  if [[ -n "$destination_timeout" ]]; then
    args+=(-destination-timeout "$destination_timeout")
  fi

  if [[ "$build_for_device" == true ]]; then
    args+=(-destination "platform=iOS,id=$device_id")
  else
    args+=(-destination "id=$simulator_id")
  fi

  readonly result_bundle_path="$TEST_UNDECLARED_OUTPUTS_DIR/tests.xcresult"
  # TEST_UNDECLARED_OUTPUTS_DIR isn't cleaned up with multiple retries of flaky tests
  rm -rf "$result_bundle_path"
  if [[ "$create_xcresult_bundle" == true ]]; then
    args+=(-resultBundlePath "$result_bundle_path")
  fi

  if (( ${#custom_xcodebuild_args[@]} )); then
    args+=("${custom_xcodebuild_args[@]}")
  fi

  xcodebuild test-without-building "${args[@]}" \
    2>&1 | tee -i "$testlog" || test_exit_code=$?
else
  developer_dir=$(xcode-select -p)
  platform_developer_dir="$developer_dir/Platforms/$test_execution_platform/Developer"
  xctest_binary="$platform_developer_dir/Library/Xcode/Agents/xctest"
  test_file=$(file "$test_tmp_dir/$test_bundle_name.xctest/$test_bundle_name")
  if [[ "$intel_simulator_hack" == true ]]; then
    sliced_xctest_binary="$test_tmp_dir/xctest_intel_bin"
    lipo -thin x86_64 -output "$sliced_xctest_binary" "$xctest_binary"
    xctest_binary=$sliced_xctest_binary
  fi

  SIMCTL_CHILD_DYLD_LIBRARY_PATH="$platform_developer_dir/usr/lib" \
    SIMCTL_CHILD_DYLD_FALLBACK_FRAMEWORK_PATH="$platform_developer_dir/Library/Frameworks" \
    SIMCTL_CHILD_DYLD_INSERT_LIBRARIES="$sanitizer_dyld_env" \
    SIMCTL_CHILD_LLVM_PROFILE_FILE="$profraw" \
    env "${passthrough_env[@]}" \
    xcrun simctl \
    spawn \
    "$simulator_id" \
    "$xctest_binary" \
    -XCTest All \
    "$test_tmp_dir/$test_bundle_name.xctest" \
    2>&1 | tee -i "$testlog" || test_exit_code=$?
fi

profdata="$test_tmp_dir/$simulator_id/Coverage.profdata"
if [[ "$should_use_xcodebuild" == false ]]; then
  profdata="$test_tmp_dir/coverage.profdata"
fi

if [[ "${COLLECT_PROFDATA:-0}" == "1" && -f "$profdata" ]]; then
  cp -R "$profdata" "$TEST_UNDECLARED_OUTPUTS_DIR"
fi

if [[ "${COVERAGE:-}" -eq 1 && "${APPLE_COVERAGE:-}" -eq 1 ]]; then
  if [[ "$should_use_xcodebuild" == false ]]; then
    xcrun llvm-profdata merge "$profraw" --output "$profdata"
  fi

  lcov_args=(
    -instr-profile "$profdata"
    -ignore-filename-regex='.*external/.+'
    -path-equivalence=".,$PWD"
  )
  has_binary=false
  IFS=";"
  arch=$(uname -m)
  for binary in $TEST_BINARIES_FOR_LLVM_COV; do
    if [[ "$has_binary" == false ]]; then
      lcov_args+=("${binary}")
      has_binary=true
      if ! file "$binary" | grep -q "$arch"; then
        arch=x86_64
      fi
    else
      lcov_args+=(-object "${binary}")
    fi

    lcov_args+=("-arch=$arch")
  done

  llvm_coverage_manifest="$COVERAGE_MANIFEST"
  readonly provided_coverage_manifest="%(test_coverage_manifest)s"
  if [[ -s "${provided_coverage_manifest:-}" ]]; then
    llvm_coverage_manifest="$provided_coverage_manifest"
  fi

  readonly error_file="$test_tmp_dir/llvm-cov-error.txt"
  llvm_cov_status=0
  xcrun llvm-cov \
    export \
    -format lcov \
    "${lcov_args[@]}" \
    @"$llvm_coverage_manifest" \
    > "$COVERAGE_OUTPUT_FILE" \
    2> "$error_file" \
    || llvm_cov_status=$?

  # Error ourselves if lcov outputs warnings, such as if we misconfigure
  # something and the file path of one of the covered files doesn't exist
  if [[ -s "$error_file" || "$llvm_cov_status" -ne 0 ]]; then
    echo "error: while exporting coverage report" >&2
    cat "$error_file" >&2
  fi

  if [[ -n "${COVERAGE_PRODUCE_JSON:-}" ]]; then
    llvm_cov_json_export_status=0
    xcrun llvm-cov \
      export \
      -format text \
      "${lcov_args[@]}" \
      @"$llvm_coverage_manifest" \
      > "$TEST_UNDECLARED_OUTPUTS_DIR/coverage.json" \
      2> "$error_file" \
      || llvm_cov_json_export_status=$?
    if [[ -s "$error_file" || "$llvm_cov_json_export_status" -ne 0 ]]; then
      echo "error: while exporting json coverage report" >&2
      cat "$error_file" >&2
    fi
  fi
fi

# Run a post-action binary, if provided.
post_action_binary=%(post_action_binary)s
post_action_determines_exit_code="%(post_action_determines_exit_code)s"
post_action_exit_code=0
if [[ -n "${result_bundle_path:-}" ]]; then
  TEST_EXIT_CODE=$test_exit_code \
    TEST_LOG_FILE="$testlog" \
    SIMULATOR_UDID="$simulator_id" \
    TEST_XCRESULT_BUNDLE_PATH="$result_bundle_path" \
    LLVM_COV_EXIT_CODE="${llvm_cov_status:-0}" \
    LLVM_COV_JSON_EXPORT_EXIT_CODE="${llvm_cov_json_export_status:-0}" \
    "$post_action_binary" || post_action_exit_code=$?
else
  TEST_EXIT_CODE=$test_exit_code \
    TEST_LOG_FILE="$testlog" \
    SIMULATOR_UDID="$simulator_id" \
    "$post_action_binary" || post_action_exit_code=$?
fi

if [[ "$post_action_determines_exit_code" == true ]]; then
  if [[ "$post_action_exit_code" -ne 0 ]]; then
    echo "error: post_action exited with '$post_action_exit_code'" >&2
    exit "$post_action_exit_code"
  fi
fi

if [[
  "$test_exit_code" -eq 0 &&
  "$create_xcresult_bundle" == true &&
  "${KEEP_XCRESULT_ON_SUCCESS:-1}" != "1"
]]; then
  # Reduce download size by removing the xcresult bundle if the test run was successful
  rm -r "$result_bundle_path"
fi

if [[ "$reuse_simulator" == false ]]; then
  # Delete will shutdown down the simulator if it's still currently running.
  xcrun simctl delete "$simulator_id"
fi

if [[ "$post_action_determines_exit_code" == true ]]; then
  if [[ "$post_action_exit_code" -ne 0 ]]; then
    echo "error: post_action exited with '$post_action_exit_code'" >&2
    exit "$post_action_exit_code"
  fi
else
  if [[ "$test_exit_code" -ne 0 ]]; then
    echo "error: tests exited with '$test_exit_code'" >&2
    exit "$test_exit_code"
  fi
fi

if [[ "${ERROR_ON_NO_TESTS_RAN:-1}" == "1" ]]; then
  parallel_testing_enabled=false
  if grep -q "-parallel-testing-enabled YES" "$testlog"; then
    parallel_testing_enabled=true
  fi

  # Fail when bundle executes nothing
  no_tests_ran=false
  if [[ $parallel_testing_enabled == true ]]; then
    echo "Parallel testing is enabled" >&2
    # When executing tests in parallel, test start markers are absent when no
    # tests are run.
    test_execution_count=$(grep -c -e "Test suite '.*' started.*" "$testlog")
    if [[ "$test_execution_count" == "0" ]]; then
      no_tests_ran=true
    fi
  else
    echo "Testing is serialized" >&2
    # Assume the final 'Executed N tests' or 'Executed 1 test' is the
    # total execution count for the test bundle.
    xctest_target_execution_count=$(grep -e "Executed [[:digit:]]\{1,\} test.*," "$testlog" | tail -n1)
    swift_testing_target_execution_count=$(grep -e "Test run with [[:digit:]]\{1,\} test.*" "$testlog" | tail -n1 || true)
    if echo "$xctest_target_execution_count" | grep -q -e "Executed 0 tests, with 0 failures" && \
      [ -z "$swift_testing_target_execution_count" ] ; then
      echo "No tests ran -> no count lines found" >&2
      no_tests_ran=true
    fi

    if echo "$xctest_target_execution_count" | grep -q -e "Executed 0 tests, with 0 failures" && \
      echo "$swift_testing_target_execution_count" | grep -q -e "Test run with 0 tests" ; then
      echo "No tests ran -> count line was 0" >&2
      no_tests_ran=true
    fi
  fi

  if [[ $no_tests_ran == true ]]; then
    echo "error: no tests were executed, is the test bundle empty?" >&2
    exit 1
  fi
fi

# When tests crash after they have reportedly completed, XCTest marks them as
# a success. These 2 cases are Swift fatalErrors, and C++ exceptions. There
# are likely other cases we can add to this in the future. FB7801959
if grep -q \
  -e "^Fatal error:" \
  -e "^.*:[0-9]\+:\sFatal error:" \
  -e "^libc++abi.dylib: terminating with uncaught exception" \
  "$testlog"
then
  echo "error: log contained test false negative" >&2
  exit 1
fi

if [[ "${llvm_cov_status:-0}" -ne 0 ]]; then
  echo "error: exporting coverage report failed" >&2
  exit "$llvm_cov_status"
fi

if [[ "${llvm_cov_json_export_status:-0}" -ne 0 ]]; then
  echo "error: exporting json coverage report failed" >&2
  exit "$llvm_cov_json_export_status"
fi

if [[ -f "${TEST_PREMATURE_EXIT_FILE:-}" ]]; then
  rm -f "$TEST_PREMATURE_EXIT_FILE"
fi
