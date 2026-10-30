# Copyright 2025 Google LLC
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

# Builds LiteRT for Windows with Bazel and MSVC.
#
# -TargetArch x86_64 (default) builds for x86_64 on an x86_64 host.
# -TargetArch arm64 cross-compiles for ARM64 on an x86_64 host
#   (--config=cross_compile_windows_arm64). Requires the MSVC ARM64 build tools;
#   see ci/setup_windows_arm64_toolchain.ps1.
# -Mode wheel (default) builds the ai-edge-litert wheel into dist/.
# -Mode prebuilts builds the prebuilt binaries (libLiteRt.dll, the GPU
#   accelerator, the Qualcomm plugins and the tools) into
#   dist/prebuilts/windows_arm64/ and dist/litert_prebuilts_windows_arm64.zip.
#   Only supported with -TargetArch arm64; other Windows prebuilts are built by
#   the Linux prebuilt job.
# The target and mode can also be selected with the LITERT_TARGET_ARCH and
# LITERT_BUILD_MODE env variables.
param (
  [ValidateSet('x86_64', 'arm64')]
  [string]$TargetArch = $(if ($env:LITERT_TARGET_ARCH) { $env:LITERT_TARGET_ARCH } else { 'x86_64' }),
  [ValidateSet('wheel', 'prebuilts')]
  [string]$Mode = $(if ($env:LITERT_BUILD_MODE) { $env:LITERT_BUILD_MODE } else { 'wheel' })
)

$ErrorActionPreference = 'Stop'

if ($Mode -eq 'prebuilts' -and $TargetArch -ne 'arm64') {
  throw '-Mode prebuilts is only supported with -TargetArch arm64.'
}

# Windows ARM64 prebuilt binaries: Bazel target -> file name in the archive.
$PrebuiltTargets = [ordered]@{
  '//litert/c:litert_runtime_c_api_dll' = 'libLiteRt.dll'
  '//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_dll' = 'libLiteRtWebGpuAccelerator.dll'
  '//litert/vendors/qualcomm/compiler:LiteRtCompilerPlugin_Qualcomm' = 'LiteRtCompilerPlugin_Qualcomm.dll'
  '//litert/vendors/qualcomm/dispatch:LiteRtDispatch_Qualcomm' = 'LiteRtDispatch_Qualcomm.dll'
  '//litert/tools:benchmark_model' = 'benchmark_model.exe'
  '//litert/tools:run_model' = 'run_model.exe'
  '//litert/tools:gpu_numerics_check' = 'gpu_numerics_check.exe'
  '//litert/tools:npu_numerics_check' = 'npu_numerics_check.exe'
  '//litert/tools:apply_plugin_main' = 'apply_plugin_main.exe'
}
$BuildTargets = if ($Mode -eq 'prebuilts') { @($PrebuiltTargets.Keys) } else { @('//ci/tools/python/wheel:litert_wheel') }

function Convert-WslPathToWindows {
  param (
    [string]$Path
  )
  if ($Path -match '^/mnt/([a-zA-Z])/(.*)$') {
    return "$($matches[1].ToUpper()):\$($matches[2] -replace '/', '\')"
  }
  if ($Path -match '^\\\\wsl\\\$\\[^\\]+\\mnt\\([a-zA-Z])\\(.*)$') {
    return "$($matches[1].ToUpper()):\$($matches[2])"
  }
  return $Path
}

$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$RepoRoot = Split-Path -Parent $ScriptDir
$RepoRoot = Convert-WslPathToWindows $RepoRoot
Set-Location $RepoRoot


# Get the first instance found in PATH.
$Bazel = (Get-Command bazel -ErrorAction SilentlyContinue | Select-Object -First 1).Source

if (-not $Bazel) {
  throw 'Bazel not found in system PATH'
}

if ($Bazel -notmatch '\.exe$') {
  if (Test-Path "$Bazel.exe") {
    $Bazel = "$Bazel.exe"
  }
  elseif (Test-Path $Bazel) {
    # GitHub Actions hosted toolcache might provide 'bazel' without extension on Windows.
    # We copy it to .exe to ensure it's executable by PowerShell.
    Write-Host "Bazel found at $Bazel but lacks extension. Creating .exe wrapper..."
    Copy-Item -Path $Bazel -Destination "$Bazel.exe" -Force
    $Bazel = "$Bazel.exe"
  }
}

if (-not (Test-Path $Bazel)) {
  throw "Bazel not found at $Bazel"
}

Write-Host "Using Bazel located at: $Bazel"

function Get-BazelInfo {
  param (
    [string]$Key
  )
  Write-Host "Getting Bazel info: $Key..."
  $ProcInfo = New-Object System.Diagnostics.ProcessStartInfo
  $ProcInfo.FileName = $Bazel
  if ($env:BAZEL_OUTPUT_USER_ROOT) {
    $ProcInfo.Arguments = "--output_user_root=$($env:BAZEL_OUTPUT_USER_ROOT) info $Key"
  }
  else {
    $ProcInfo.Arguments = "info $Key"
  }
  $ProcInfo.WorkingDirectory = $RepoRoot
  $ProcInfo.RedirectStandardOutput = $true
  $ProcInfo.RedirectStandardError = $true
  $ProcInfo.UseShellExecute = $false
  $ProcInfo.CreateNoWindow = $true

  $Proc = New-Object System.Diagnostics.Process
  $Proc.StartInfo = $ProcInfo
  $Proc.Start() | Out-Null

  # Read Stdout asynchronously to avoid deadlock if Stderr fills up
  $StdoutTask = $Proc.StandardOutput.ReadToEndAsync()
  $Stderr = $Proc.StandardError.ReadToEnd()
  $Proc.WaitForExit()
  $Stdout = $StdoutTask.Result

  if ($Proc.ExitCode -ne 0) {
    throw "Bazel info $Key failed. Exit Code: $($Proc.ExitCode). Stderr: $Stderr"
  }
  if (-not $Stdout) {
    throw "Bazel info $Key returned empty output."
  }
  return $Stdout.Trim()
}

function Replace-InFile {
  param (
    [string]$Path,
    [string]$Old,
    [string]$New
  )
  if (-not (Test-Path $Path)) { return $false }
  $Text = Get-Content $Path -Raw
  if (-not $Text) { return $false }
  if (-not $Text.Contains($Old)) { return $false }
  $Text = $Text.Replace($Old, $New)
  Set-Content -Path $Path -Value $Text
  return $true
}

function Ensure-Include {
  param (
    [string]$Path,
    [string]$Anchor,
    [string]$IncludeLine
  )
  if (-not (Test-Path $Path)) { return $false }
  $Text = Get-Content $Path -Raw
  if ($Text -like "*${IncludeLine}*") { return $false }
  if ($Text -notlike "*${Anchor}*") { return $false }
  $Text = $Text.Replace($Anchor, $Anchor + $IncludeLine)
  Set-Content -Path $Path -Value $Text
  return $true
}

# Remove incompatible flags from .litert_configure.bazelrc if present
$ConfigureBazelrc = Join-Path $RepoRoot '.litert_configure.bazelrc'
if (Test-Path $ConfigureBazelrc) {
  Write-Host "Checking for incompatible flags in $ConfigureBazelrc..."
  Replace-InFile $ConfigureBazelrc '-Wno-sign-compare' '' | Out-Null
  Replace-InFile $ConfigureBazelrc '-Wno-gnu-offsetof-extensions' '' | Out-Null
}

# Verify Bazel version
Write-Host 'Checking Bazel version...'
& $Bazel --version
if ($LASTEXITCODE -ne 0) {
  Write-Warning "Failed to check Bazel version. Exit code: $LASTEXITCODE"
}

if ($env:BAZEL_OUTPUT_USER_ROOT) {
  $env:BAZEL_OUTPUT_USER_ROOT = Convert-WslPathToWindows $env:BAZEL_OUTPUT_USER_ROOT
  if ($env:BAZEL_OUTPUT_USER_ROOT -match '^D:' -and -not (Test-Path 'D:\')) {
    Write-Host 'D: drive not found, redirecting BAZEL_OUTPUT_USER_ROOT to C: drive.'
    $env:BAZEL_OUTPUT_USER_ROOT = $env:BAZEL_OUTPUT_USER_ROOT -replace '^D:', 'C:'
  }
  $OutputUserRootParent = Split-Path -Parent $env:BAZEL_OUTPUT_USER_ROOT
  if (-not (Test-Path $OutputUserRootParent)) {
    New-Item -ItemType Directory -Path $OutputUserRootParent -Force | Out-Null
  }
}

# Fetch dependencies first to ensure the Bazel server is running and external repos are populated.
Write-Host 'Fetching dependencies...'
$BazelStartupOpts = if ($env:BAZEL_OUTPUT_USER_ROOT) {
  @("--output_user_root=$($env:BAZEL_OUTPUT_USER_ROOT)")
} else {
  @()
}
$TargetConfigArgs = if ($TargetArch -eq 'arm64') { @('--config=cross_compile_windows_arm64') } else { @() }
Write-Host "Target architecture: $TargetArch, mode: $Mode"
$FetchArgs = @('fetch', '--config=windows') + $TargetConfigArgs + @('--repo_env=USE_PYWRAP_RULES=True') + $BuildTargets
& $Bazel $BazelStartupOpts $FetchArgs
if ($LASTEXITCODE -ne 0) {
  throw "Bazel fetch failed with exit code $LASTEXITCODE"
}

$OutputBase = Get-BazelInfo 'output_base'
Write-Host "Bazel output_base: $OutputBase"

$ExecRoot = Get-BazelInfo 'execution_root'
Write-Host "Bazel execution_root: $ExecRoot"
if ($ExecRoot) {
  $ExecBazelOut = Join-Path $ExecRoot 'bazel-out'
  $WorkspaceBazelOut = Join-Path $RepoRoot 'bazel-out'
  if (Test-Path $WorkspaceBazelOut) {
    if (Test-Path $ExecBazelOut) {
      & cmd.exe /c "rmdir /s /q `"$ExecBazelOut`"" | Out-Null
    }
    New-Item -ItemType Junction -Path $ExecBazelOut -Target $WorkspaceBazelOut | Out-Null
  }
}

# Deep patch external repositories to remove incompatible flags that might be hardcoded.
Write-Host 'Deep patching external repositories for incompatible flags...'
$ExternalDir = Join-Path $OutputBase 'external'
if (Test-Path $ExternalDir) {
  Get-ChildItem -Path $ExternalDir -Include '*.bazel','*.bzl','*.rc','BUILD*' -Recurse | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
    Replace-InFile $_.FullName '-Wno-sign-compare' '' | Out-Null
  }
}

$LocalXlaTsl = Join-Path $OutputBase 'external\local_xla\xla\tsl\tsl.bzl'
Replace-InFile $LocalXlaTsl 'clean_dep("//xla/tsl:windows"): get_win_copts(is_external, is_msvc = False),' 'clean_dep("//xla/tsl:windows"): get_win_copts(is_external, is_msvc = True),' | Out-Null

$AbslOptions = Join-Path $OutputBase 'external\com_google_absl\absl\base\options.h'
if (-not (Test-Path $AbslOptions)) {
  $AbslOptions = Join-Path $OutputBase 'external\abseil-cpp\absl\base\options.h'
}
if (Test-Path $AbslOptions) {
  Write-Host 'Patching Abseil options.h to disable std::source_location...'
  Replace-InFile $AbslOptions '#define ABSL_OPTION_USE_STD_SOURCE_LOCATION 2' '#define ABSL_OPTION_USE_STD_SOURCE_LOCATION 0' | Out-Null
}

$ProtoContext = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\compiler\java\context.h'
if (Test-Path $ProtoContext) {
  $ProtoLines = Get-Content $ProtoContext
  $ProtoLines = $ProtoLines | Where-Object { $_ -notmatch '^\s*#include\s*\\\s*$' }
  Set-Content -Path $ProtoContext -Value $ProtoLines
}
Ensure-Include $ProtoContext '#include "google/protobuf/compiler/java/helpers.h"' ("`n" + '#include "google/protobuf/compiler/java/field_common.h"') | Out-Null



$ProtoJavaBuild = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\compiler\java\BUILD.bazel'
if (Test-Path $ProtoJavaBuild) {
  $BuildText = Get-Content $ProtoJavaBuild -Raw
  $InsertLine = [Environment]::NewLine + '        "field_common.h",'
  $BuildText = $BuildText.Replace('"helpers.h",`n        "field_common.h",', '"helpers.h",' + $InsertLine)
  if ($BuildText -notmatch '"helpers.h",\s*"field_common.h"') {
    $BuildText = [regex]::Replace($BuildText, '"helpers.h",', '"helpers.h",' + $InsertLine, 1)
  }
  Set-Content -Path $ProtoJavaBuild -Value $BuildText
}

$UntypedHeader = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\json\internal\untyped_message.h'
if (Test-Path $UntypedHeader) {
  $Text = Get-Content $UntypedHeader -Raw
  $Text = $Text.Replace(', UntypedMessage,', ', std::unique_ptr<UntypedMessage>,')
  $Text = $Text.Replace('std::vector<UntypedMessage>>;', 'std::vector<std::unique_ptr<UntypedMessage>>>;')

  if ($Text -notmatch 'GetMessage\(int32_t field_number') {
    $Insert = @"
  const UntypedMessage* GetMessage(int32_t field_number, size_t idx = 0) const {
    auto it = fields_.find(field_number);
    ABSL_CHECK(it != fields_.end()) << "missing UntypedMessage field " << field_number;
    if (auto* val = std::get_if<std::unique_ptr<UntypedMessage>>(&it->second)) {
      return val->get();
    }
    if (auto* vec = std::get_if<std::vector<std::unique_ptr<UntypedMessage>>>(&it->second)) {
      ABSL_CHECK(idx < vec->size()) << "UntypedMessage index out of range for field " << field_number;
      return (*vec)[idx].get();
    }
    ABSL_CHECK(false) << "wrong type for UntypedMessage::GetMessage(" << field_number << ")";
    return nullptr;
  }

"@
    $Anchor = '  const ResolverPool::Message& desc() const { return *desc_; }'
    if ($Text -match [regex]::Escape($Anchor)) {
      $Text = $Text.Replace($Anchor, $Insert + $Anchor)
    }
  }
  Set-Content -Path $UntypedHeader -Value $Text
}

$UntypedCc = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\json\internal\untyped_message.cc'
if (Test-Path $UntypedCc) {
  $CcText = Get-Content $UntypedCc -Raw
  $CcText = $CcText.Replace('InsertField(*field, std::move(group))', 'InsertField(*field, std::make_unique<UntypedMessage>(std::move(group)))')
  $CcText = $CcText.Replace('InsertField(field, std::move(*inner))', 'InsertField(field, std::make_unique<UntypedMessage>(std::move(*inner)))')

  $InsertFieldPattern = '(?s)template <typename T>\\s+absl::Status UntypedMessage::InsertField\\([^\\{]*\\)\\s*\\{.*?\\n\\}'
  $InsertFieldReplacement = @"
template <typename T>
absl::Status UntypedMessage::InsertField(const ResolverPool::Field& field,
                                         T&& value) {
  using value_type = std::decay_t<T>;
  using stored_type = std::conditional_t<
      std::is_same_v<value_type, UntypedMessage>,
      std::unique_ptr<UntypedMessage>,
      value_type>;

  auto to_stored = [&](T&& v) -> stored_type {
    if constexpr (std::is_same_v<value_type, UntypedMessage>) {
      return std::make_unique<UntypedMessage>(std::forward<T>(v));
    } else {
      return std::forward<T>(v);
    }
  };

  int32_t number = field.proto().number();
  auto it = fields_.find(number);
  if (it == fields_.end()) {
    fields_.emplace(number,
                    Value(std::in_place_type<stored_type>,
                          to_stored(std::forward<T>(value))));
    return absl::OkStatus();
  }

  if (field.proto().cardinality() !=
      google::protobuf::Field::CARDINALITY_REPEATED) {
    return absl::InvalidArgumentError(
        absl::StrCat("repeated entries for singular field number ", number));
  }

  Value& slot = it->second;
  if (auto* extant = std::get_if<stored_type>(&slot)) {
    std::vector<stored_type> repeated;
    repeated.push_back(std::move(*extant));
    repeated.push_back(to_stored(std::forward<T>(value)));
    slot.emplace<std::vector<stored_type>>(std::move(repeated));
  } else if (auto* extant = std::get_if<std::vector<stored_type>>(&slot)) {
    extant->push_back(to_stored(std::forward<T>(value)));
  } else {
    absl::optional<absl::string_view> name =
        google::protobuf::internal::RttiTypeName<stored_type>();
    if (!name.has_value()) {
      name = "<unknown>";
    }

    return absl::InvalidArgumentError(
        absl::StrFormat("inconsistent types for field number %d: tried to "
                        "insert '%s', but index was %d",
                        number, *name, slot.index()));
  }

  return absl::OkStatus();
}
"@
  $NewCcText = [regex]::Replace($CcText, $InsertFieldPattern, $InsertFieldReplacement, 1)
  if ($NewCcText -eq $CcText -and $CcText -notmatch 'using value_type = std::decay_t<T>;') {
    $NewCcText = $CcText.Replace('T&& value) {', 'T&& value) {`n  using value_type = std::decay_t<T>;')
  }
  if ($NewCcText -notmatch 'using value_type = std::decay_t<T>;') {
    throw 'Failed to patch untyped_message.cc InsertField template for MSVC.'
  }
  Set-Content -Path $UntypedCc -Value $NewCcText
}

$UnparserTraits = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\json\internal\unparser_traits.h'
Replace-InFile $UnparserTraits 'return &msg.Get<Msg>(f->proto().number())[idx];' 'return msg.GetMessage(f->proto().number(), idx);' | Out-Null

$ProtoCompilerBuild = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\compiler\BUILD.bazel'
Replace-InFile $ProtoCompilerBuild '        "//src/google/protobuf/compiler/objectivec",' '' | Out-Null
Replace-InFile $ProtoCompilerBuild '        "//src/google/protobuf/compiler/rust",' '' | Out-Null

$ProtoMain = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\compiler\main.cc'
Replace-InFile $ProtoMain '#include "google/protobuf/compiler/objectivec/generator.h"' '' | Out-Null
Replace-InFile $ProtoMain '#include "google/protobuf/compiler/rust/generator.h"' '' | Out-Null
Replace-InFile $ProtoMain '  objectivec::ObjectiveCGenerator objc_generator;' '' | Out-Null
Replace-InFile $ProtoMain '  cli.RegisterGenerator("--objc_out", "--objc_opt", &objc_generator,' '' | Out-Null
Replace-InFile $ProtoMain '                        "Generate Objective-C header and source.");' '' | Out-Null
Replace-InFile $ProtoMain '  rust::RustGenerator rust_generator;' '' | Out-Null
Replace-InFile $ProtoMain '  cli.RegisterGenerator("--rust_out", "--rust_opt", &rust_generator,' '' | Out-Null
Replace-InFile $ProtoMain '                        "Generate Rust sources.");' '' | Out-Null

$BazelArgs = @(
  'build',
  '-c',
  'opt',
  '--config=windows',
  '--copt=-DLITERT_DISABLE_OPENCL_SUPPORT=1',
  '--repo_env=USE_PYWRAP_RULES=True',
  '--define=protobuf_allow_msvc=true',
  '--copt=/Iexternal\com_google_protobuf\src',
  '--host_copt=/Iexternal\com_google_protobuf\src'
)
if ($TargetArch -eq 'arm64') {
  $BazelArgs += $TargetConfigArgs
}
else {
  # x86-only ISA extensions.
  $BazelArgs += @('--copt=-mavx2', '--copt=-mfma', '--copt=-mf16c')
}
if ($env:HERMETIC_PYTHON_VERSION) { $BazelArgs += "--repo_env=HERMETIC_PYTHON_VERSION=$($env:HERMETIC_PYTHON_VERSION)" }
if ($env:BAZEL_CONFIG_FLAGS) { $BazelArgs += $env:BAZEL_CONFIG_FLAGS.Split(' ') }
if ($env:NIGHTLY_RELEASE_DATE) { $BazelArgs += "--//ci/tools/python/wheel:nightly_iso_date=$($env:NIGHTLY_RELEASE_DATE)" }
if ($env:CUSTOM_BAZEL_FLAGS) { $BazelArgs += $env:CUSTOM_BAZEL_FLAGS.Split(' ') }

Write-Host 'Starting bazel build...'
& $Bazel $BazelStartupOpts @BazelArgs @BuildTargets
if ($LASTEXITCODE -ne 0) {
  throw "Bazel build failed with exit code $LASTEXITCODE"
}

$DistDir = Join-Path $RepoRoot 'dist'
if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }
New-Item -ItemType Directory -Path $DistDir | Out-Null

if ($Mode -eq 'prebuilts') {
  # Stage the binaries in the same layout as litert_prebuilts.zip
  # (<platform>/<file>).
  $PlatformDir = Join-Path $DistDir 'prebuilts\windows_arm64'
  New-Item -ItemType Directory -Force -Path $PlatformDir | Out-Null
  $CqueryArgs = @('cquery') + $BazelArgs[1..($BazelArgs.Count - 1)] + @('--output=files')
  foreach ($Target in $PrebuiltTargets.Keys) {
    $FileName = $PrebuiltTargets[$Target]
    $Outputs = & $Bazel $BazelStartupOpts @CqueryArgs $Target
    if ($LASTEXITCODE -ne 0) {
      throw "bazel cquery failed for $Target with exit code $LASTEXITCODE"
    }
    $Output = $Outputs | Where-Object { $_ -match '\.(dll|exe)$' } | Select-Object -First 1
    if (-not $Output) {
      throw "No .dll/.exe output found for $Target"
    }
    Copy-Item -Path (Join-Path $RepoRoot $Output) -Destination (Join-Path $PlatformDir $FileName) -Force
    Write-Host "Staged $Target -> windows_arm64\$FileName"
  }
  $ZipPath = Join-Path $DistDir 'litert_prebuilts_windows_arm64.zip'
  # Use Python zipfile rather than Compress-Archive so ZIP entry paths always
  # use POSIX forward slashes even under Windows PowerShell 5.1.
  & python -c 'import pathlib, sys, zipfile; src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]; zf = zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED); [zf.write(p, f"windows_arm64/{p.name}") for p in sorted(src.iterdir()) if p.is_file()]; zf.close()' $PlatformDir $ZipPath
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to create $ZipPath with exit code $LASTEXITCODE"
  }
  # The binaries cannot run on the x86_64 host, so verify that every one of
  # them targets ARM64.
  & python (Join-Path $ScriptDir 'check_windows_wheel_arch.py') --arch arm64 $ZipPath
  if ($LASTEXITCODE -ne 0) {
    throw "Windows ARM64 prebuilts architecture check failed with exit code $LASTEXITCODE"
  }
  Get-ChildItem -Path $PlatformDir | ForEach-Object {
    Write-Host ('{0}  {1}' -f (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash, $_.Name)
  }
  Write-Host 'Output can be found here:' $DistDir
  return
}

Get-ChildItem -Path (Join-Path $RepoRoot 'bazel-bin\ci\tools\python\wheel\dist') -Filter *.whl | Move-Item -Destination $DistDir

if ($TargetArch -eq 'arm64') {
  # The cross-compiled wheel cannot be imported on the x86_64 host, so verify
  # that every native binary in it targets ARM64 instead.
  & python (Join-Path $ScriptDir 'check_windows_wheel_arch.py') --arch arm64 (Get-ChildItem -Path $DistDir -Filter *.whl).FullName
  if ($LASTEXITCODE -ne 0) {
    throw "Windows ARM64 wheel architecture check failed with exit code $LASTEXITCODE"
  }
  # Vendor SDK sdists are platform-independent and built by the x86_64 job.
  Write-Host 'Output can be found here:' $DistDir
  return
}

# Vendor SDK sdist packages
$SdkArgs = @(
  'build',
  '-c',
  'opt',
  '--config=windows',
  '--repo_env=USE_PYWRAP_RULES=True'
)
if ($env:HERMETIC_PYTHON_VERSION) { $SdkArgs += "--repo_env=HERMETIC_PYTHON_VERSION=$($env:HERMETIC_PYTHON_VERSION)" }
if ($env:NIGHTLY_RELEASE_DATE) { $SdkArgs += "--//ci/tools/python/wheel:nightly_iso_date=$($env:NIGHTLY_RELEASE_DATE)" }
if ($env:CUSTOM_BAZEL_FLAGS) { $SdkArgs += $env:CUSTOM_BAZEL_FLAGS.Split(' ') }

$SdkTargets = @(
  @{ Name = 'Qualcomm'; Target = '//ci/tools/python/vendor_sdk/qualcomm:ai_edge_litert_sdk_qualcomm_sdist'; Glob = 'qualcomm' },
  @{ Name = 'MediaTek'; Target = '//ci/tools/python/vendor_sdk/mediatek:ai_edge_litert_sdk_mediatek_sdist'; Glob = 'mediatek' },
  @{ Name = 'Intel';    Target = '//ci/tools/python/vendor_sdk/intel:ai_edge_litert_sdk_intel_sdist';       Glob = 'intel' },
  @{ Name = 'Samsung';  Target = '//ci/tools/python/vendor_sdk/samsung:ai_edge_litert_sdk_samsung_sdist';   Glob = 'samsung' }
)

foreach ($Sdk in $SdkTargets) {
  Write-Host "Building $($Sdk.Name) SDK sdist..."
  & $Bazel $BazelStartupOpts @SdkArgs $Sdk.Target
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "$($Sdk.Name) SDK build failed (non-fatal)"
  }
  else {
    $SdkDist = Join-Path $RepoRoot "bazel-bin\ci\tools\python\vendor_sdk\$($Sdk.Glob)"
    Get-ChildItem -Path $SdkDist -Filter *.tar.gz -ErrorAction SilentlyContinue | Move-Item -Destination $DistDir
  }
}

Write-Host 'Output can be found here:' $DistDir
