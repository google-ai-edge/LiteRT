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
# -Mode prebuilts builds prebuilt binaries into dist/prebuilts/windows_<arch>/
#   and dist/litert_prebuilts_windows_<arch>.zip.
# The target and mode can also be selected with the LITERT_TARGET_ARCH and
# LITERT_BUILD_MODE env variables.
param (
  [ValidateSet('x86_64', 'arm64')]
  [string]$TargetArch = $(if ($env:LITERT_TARGET_ARCH) { $env:LITERT_TARGET_ARCH } else { 'x86_64' }),
  [ValidateSet('wheel', 'prebuilts')]
  [string]$Mode = $(if ($env:LITERT_BUILD_MODE) { $env:LITERT_BUILD_MODE } else { 'wheel' })
)

$ErrorActionPreference = 'Stop'

function Write-GitHubError {
  param (
    [string]$Title,
    [string[]]$Lines
  )
  $Msg = ($Lines -join "`n") -replace '%', '%25' -replace "`r", '%0D' -replace "`n", '%0A'
  if ($Msg.Length -gt 3500) { $Msg = $Msg.Substring(0, 1700) + '%0A...%0A' + $Msg.Substring($Msg.Length - 1700) }
  Write-Host "::error title=${Title}::$Msg"
}

# On failure, surface the error and the Bazel errors as GitHub annotations.
trap {
  try {
    Write-GitHubError 'Script Error' @($_.Exception.Message, $_.InvocationInfo.PositionMessage)
    $CmdLog = if ($OutputBase) { Join-Path $OutputBase 'command.log' }
    if ($CmdLog -and (Test-Path -LiteralPath $CmdLog)) {
      $LogLines = @(Get-Content -LiteralPath $CmdLog)
      $ErrLines = @($LogLines | Where-Object { $_ -match '(ERROR:|FAILED:|fatal:|error C\d+|error LNK\d+|unresolved external|Exception:)' })
      if ($ErrLines) { Write-GitHubError 'Bazel Errors' $ErrLines }
      Write-GitHubError 'Bazel Log Tail' @($LogLines | Select-Object -Last 50)
    }
  }
  catch {}
  break
}

if ($TargetArch -notin @('x86_64', 'arm64')) {
  throw "Unsupported TargetArch '$TargetArch'. Expected 'x86_64' or 'arm64'."
}
if ($Mode -notin @('wheel', 'prebuilts')) {
  throw "Unsupported Mode '$Mode'. Expected 'wheel' or 'prebuilts'."
}

# Prebuilt binaries: Bazel target -> file name in the archive.
$PrebuiltTargets = [ordered]@{
  '//litert/c:litert_runtime_c_api_dll' = 'libLiteRt.dll'
  '//litert/runtime/accelerators/gpu:ml_drift_webgpu_accelerator_dll' = 'libLiteRtWebGpuAccelerator.dll'
  '//litert/tools:benchmark_model' = 'benchmark_model.exe'
  '//litert/tools:run_model' = 'run_model.exe'
  '//litert/tools:gpu_numerics_check' = 'gpu_numerics_check.exe'
  '//litert/tools:npu_numerics_check' = 'npu_numerics_check.exe'
  '//litert/tools:apply_plugin_main' = 'apply_plugin_main.exe'
  '//litert/tools:analyze_model_main' = 'analyze_model_main.exe'
}
if ($TargetArch -eq 'arm64') {
  $PrebuiltTargets['//litert/vendors/qualcomm/compiler:LiteRtCompilerPlugin_Qualcomm'] = 'LiteRtCompilerPlugin_Qualcomm.dll'
  $PrebuiltTargets['//litert/vendors/qualcomm/dispatch:LiteRtDispatch_Qualcomm'] = 'LiteRtDispatch_Qualcomm.dll'
}
else {
  $PrebuiltTargets['//litert/vendors/intel_openvino/compiler:LiteRtCompilerPlugin_IntelOpenvino'] = 'LiteRtCompilerPlugin_IntelOpenvino.dll'
  $PrebuiltTargets['//litert/vendors/intel_openvino/dispatch:LiteRtDispatch_IntelOpenvino'] = 'LiteRtDispatch_IntelOpenvino.dll'
}
$PrebuiltsPlatform = "windows_$TargetArch"
[string[]]$BuildTargets = if ($Mode -eq 'prebuilts') { @($PrebuiltTargets.Keys) } else { @('//ci/tools/python/wheel:litert_wheel') }

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

[string[]]$BazelStartupOpts = if ($env:BAZEL_OUTPUT_USER_ROOT) {
  @("--output_user_root=$($env:BAZEL_OUTPUT_USER_ROOT)")
}
else {
  @()
}

$GitHome = 'C:\t\_home'
$GitConfigDir = "$GitHome\.config\git"
$ShortTemp = 'C:\t\_tmp'
New-Item -ItemType Directory -Force -Path $GitConfigDir, $ShortTemp | Out-Null
New-Item -ItemType File -Force -Path "$GitConfigDir\ignore", "$GitConfigDir\attributes" | Out-Null
$GitConfigLines = @(
  '[core]',
  '	excludesFile = C:/t/_home/.config/git/ignore',
  '	attributesFile = C:/t/_home/.config/git/attributes'
)
Set-Content -LiteralPath "$GitHome\.gitconfig" -Value $GitConfigLines -Encoding ASCII
Set-Content -LiteralPath "$GitConfigDir\config" -Value $GitConfigLines -Encoding ASCII
$env:HOME = $GitHome
$env:XDG_CONFIG_HOME = 'C:/t/_home/.config'
$env:GIT_CONFIG_GLOBAL = 'C:/t/_home/.gitconfig'
$env:TMP = $ShortTemp
$env:TEMP = $ShortTemp

# The Qualcomm plugins only need the QNN headers: extract just those from the
# QAIRT SDK archive (CI caches C:\t\qairt_sdk) and point @qairt at them.
if ($TargetArch -eq 'arm64' -and -not $env:LITERT_QAIRT_SDK) {
  $QairtRoot = 'C:\t\qairt_sdk'
  $QairtOutDir = Join-Path $QairtRoot 'include\QNN'
  if (-not (Test-Path -LiteralPath (Join-Path $QairtOutDir 'QnnCommon.h'))) {
    $QairtWorkspace = Join-Path $RepoRoot 'third_party\qairt\workspace.bzl'
    $QairtText = Get-Content -LiteralPath $QairtWorkspace -Raw
    if ($QairtText -notmatch 'url\s*=\s*"([^"]+)"') {
      throw "Could not find QAIRT URL in $QairtWorkspace"
    }
    $QairtUrl = $Matches[1]
    $QairtSha256 = if ($env:LITERT_QAIRT_SHA256) {
      $env:LITERT_QAIRT_SHA256
    }
    elseif ($QairtText -match 'sha256\s*=\s*"([0-9a-fA-F]{64})"') {
      $Matches[1]
    }
    $TmpZip = Join-Path $env:TEMP 'qairt_sdk.zip'
    if (Test-Path -LiteralPath $QairtOutDir) {
      Remove-Item -LiteralPath $QairtOutDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $QairtOutDir | Out-Null
    Write-Host "Downloading QAIRT SDK archive from $QairtUrl ..."
    $PrevProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
      for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
        try {
          Invoke-WebRequest -Uri $QairtUrl -OutFile $TmpZip
          break
        }
        catch {
          if ($Attempt -eq 3) {
            throw
          }
          Start-Sleep -Seconds 5
        }
      }
      if ($QairtSha256) {
        $ActualHash = (Get-FileHash -Path $TmpZip -Algorithm SHA256).Hash
        if ($ActualHash -ne $QairtSha256) {
          throw "SHA256 mismatch for QAIRT archive: expected $QairtSha256, got $ActualHash"
        }
      }
      Add-Type -AssemblyName System.IO.Compression.FileSystem
      $Zip = [System.IO.Compression.ZipFile]::OpenRead($TmpZip)
      try {
        $Marker = '/include/QNN/'
        $Extracted = 0
        foreach ($Entry in $Zip.Entries) {
          $EntryName = $Entry.FullName.Replace('\', '/')
          if ($EntryName.EndsWith('/') -or (-not $EntryName.Contains($Marker))) {
            continue
          }
          $Idx = $EntryName.IndexOf($Marker)
          $RelPath = $EntryName.Substring($Idx + $Marker.Length)
          if (-not $RelPath) { continue }
          $Dest = Join-Path $QairtOutDir ($RelPath.Replace('/', '\'))
          $DestDir = Split-Path -Parent $Dest
          if (-not (Test-Path -LiteralPath $DestDir)) {
            New-Item -ItemType Directory -Force -Path $DestDir | Out-Null
          }
          [System.IO.Compression.ZipFileExtensions]::ExtractToFile($Entry, $Dest, $true)
          $Extracted++
        }
        if ($Extracted -eq 0 -or (-not (Test-Path -LiteralPath (Join-Path $QairtOutDir 'QnnCommon.h')))) {
          throw "Failed to extract QNN headers (extracted=$Extracted)"
        }
        Write-Host "Extracted $Extracted QNN header files into $QairtOutDir"
      }
      finally {
        $Zip.Dispose()
      }
    }
    finally {
      $ProgressPreference = $PrevProgress
      Remove-Item -LiteralPath $TmpZip -Force -ErrorAction SilentlyContinue
    }
  }
  else {
    Write-Host "Using cached QAIRT headers in $QairtOutDir"
  }
  $env:LITERT_QAIRT_SDK = $QairtRoot.Replace('\', '/')
}

$OutputBase = Get-BazelInfo 'output_base'
Write-Host "Bazel output_base: $OutputBase"

[string[]]$TargetConfigArgs = if ($TargetArch -eq 'arm64') {
  $ArgsList = @('--config=cross_compile_windows_arm64')
  if ($env:LITERT_QAIRT_SDK) {
    $ArgsList += "--repo_env=LITERT_QAIRT_SDK=$($env:LITERT_QAIRT_SDK)"
  }
  $ArgsList
}
else {
  @()
}
Write-Host "Target architecture: $TargetArch, mode: $Mode"

# Fetch dependencies first to ensure the Bazel server is running and external repos are populated.
Write-Host 'Fetching dependencies...'
$FetchArgs = @('fetch', '--config=windows') + $TargetConfigArgs + @('--repo_env=USE_PYWRAP_RULES=True') + $BuildTargets
& $Bazel $BazelStartupOpts $FetchArgs
if ($LASTEXITCODE -ne 0) {
  throw "Bazel fetch failed with exit code $LASTEXITCODE"
}

$ExecRoot = Get-BazelInfo 'execution_root'
Write-Host "Bazel execution_root: $ExecRoot"
if ($ExecRoot) {
  $ExecBazelOut = Join-Path $ExecRoot 'bazel-out'
  $WorkspaceBazelOut = Join-Path $RepoRoot 'bazel-out'
  # Skip if bazel-out is already Bazel's junction (e.g. on a second run), which
  # would otherwise become a junction cycle.
  if ((Test-Path $WorkspaceBazelOut) -and -not ((Get-Item $WorkspaceBazelOut -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
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
  Get-ChildItem -Path $ExternalDir -Include '*.bazel','*.bzl','*.rc','BUILD*' -Recurse -ErrorAction SilentlyContinue | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
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
  $FilteredProtoLines = $ProtoLines | Where-Object { $_ -notmatch '^\s*#include\s*\\\s*$' }
  if ($FilteredProtoLines.Count -ne $ProtoLines.Count) {
    Set-Content -Path $ProtoContext -Value $FilteredProtoLines
  }
}
Ensure-Include $ProtoContext '#include "google/protobuf/compiler/java/helpers.h"' ("`n" + '#include "google/protobuf/compiler/java/field_common.h"') | Out-Null



$ProtoJavaBuild = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\compiler\java\BUILD.bazel'
if (Test-Path $ProtoJavaBuild) {
  $OrigBuildText = Get-Content $ProtoJavaBuild -Raw
  $BuildText = $OrigBuildText
  $InsertLine = [Environment]::NewLine + '        "field_common.h",'
  $BuildText = $BuildText.Replace('"helpers.h",`n        "field_common.h",', '"helpers.h",' + $InsertLine)
  if ($BuildText -notmatch '"helpers.h",\s*"field_common.h"') {
    $BuildText = [regex]::Replace($BuildText, '"helpers.h",', '"helpers.h",' + $InsertLine, 1)
  }
  if ($BuildText -ne $OrigBuildText) {
    Set-Content -Path $ProtoJavaBuild -Value $BuildText
  }
}

$UntypedHeader = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\json\internal\untyped_message.h'
if (Test-Path $UntypedHeader) {
  $OrigText = Get-Content $UntypedHeader -Raw
  $Text = $OrigText
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
  if ($Text -ne $OrigText) {
    Set-Content -Path $UntypedHeader -Value $Text
  }
}

$UntypedCc = Join-Path $OutputBase 'external\com_google_protobuf\src\google\protobuf\json\internal\untyped_message.cc'
if (Test-Path $UntypedCc) {
  $OrigCcText = Get-Content $UntypedCc -Raw
  $CcText = $OrigCcText
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
  if ($NewCcText -ne $OrigCcText) {
    Set-Content -Path $UntypedCc -Value $NewCcText
  }
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
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

if ($Mode -eq 'prebuilts') {
  $PrebuiltsRoot = Join-Path $DistDir 'prebuilts'
  if (Test-Path $PrebuiltsRoot) { Remove-Item -Recurse -Force $PrebuiltsRoot }
  # Stage the binaries in the same layout as litert_prebuilts.zip
  # (<platform>/<file>).
  $PlatformDir = Join-Path $PrebuiltsRoot $PrebuiltsPlatform
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
    Write-Host "Staged $Target -> $PrebuiltsPlatform\$FileName"
  }
  $ZipPath = Join-Path $DistDir "litert_prebuilts_$PrebuiltsPlatform.zip"
  if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
  # Use Python zipfile rather than Compress-Archive so ZIP entry paths always
  # use POSIX forward slashes even under Windows PowerShell 5.1.
  & python -c "import pathlib, sys, zipfile; src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]; zf = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED); [zf.write(p, f'{src.name}/{p.name}') for p in sorted(src.iterdir()) if p.is_file()]; zf.close()" $PlatformDir $ZipPath
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to create $ZipPath with exit code $LASTEXITCODE"
  }
  & python (Join-Path $ScriptDir 'check_windows_wheel_arch.py') --arch $TargetArch $ZipPath
  if ($LASTEXITCODE -ne 0) {
    throw "Windows $TargetArch prebuilts architecture check failed with exit code $LASTEXITCODE"
  }
  Get-ChildItem -Path $PlatformDir | ForEach-Object {
    Write-Host ('{0}  {1}' -f (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash, $_.Name)
  }
  Write-Host 'Output can be found here:' $DistDir
  return
}

Get-ChildItem -Path $DistDir -Filter *.whl -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem -Path $DistDir -Filter *.tar.gz -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem -Path (Join-Path $RepoRoot 'bazel-bin\ci\tools\python\wheel\dist') -Filter *.whl | Move-Item -Destination $DistDir -Force

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
  @{ Name = 'Intel';    Target = '//ci/tools/python/vendor_sdk/intel:ai_edge_litert_sdk_intel_sdist';       Glob = 'intel' }
  # Samsung is intentionally not built: ai-edge-litert-sdk-samsung is not
  # published to PyPI.
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
