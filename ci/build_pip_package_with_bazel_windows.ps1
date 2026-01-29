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

$ErrorActionPreference = 'Stop'

function Convert-WslPathToWindows {
  param (
    [string]$Path
  )
  if ($Path -match '^/mnt/([a-zA-Z])/(.*)$') {
    return ("$($matches[1].ToUpper()):\" + ($matches[2] -replace '/', '\'))
  }
  if ($Path -match '^\\\\wsl\\\$\\[^\\]+\\mnt\\([a-zA-Z])\\(.*)$') {
    return ("$($matches[1].ToUpper()):\" + $matches[2])
  }
  return $Path
}

$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$RepoRoot = Split-Path -Parent $ScriptDir
$RepoRoot = Convert-WslPathToWindows $RepoRoot
Set-Location $RepoRoot

if (-not $env:TF_LOCAL_SOURCE_PATH) {
  $env:TF_LOCAL_SOURCE_PATH = Join-Path $RepoRoot "third_party\tensorflow"
}

# Get the first instance found in PATH. Prefer bazel.exe (Windows executable).
$Bazel = (Get-Command bazel.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $Bazel) {
  $Bazel = (Get-Command bazel -ErrorAction SilentlyContinue | Select-Object -First 1).Source
}

if (-not $Bazel) {
  throw "Bazel not found in system PATH"
}

if ($Bazel -notmatch '\.exe$') {
  if (Test-Path "$Bazel.exe") {
    $Bazel = "$Bazel.exe"
  }
}

if (-not (Test-Path $Bazel)) {
  throw "Bazel not found at $Bazel"
}

# Optional: Print it so you can see it in your GitHub Action logs
Write-Host "Using Bazel located at: $Bazel"

function Get-Bazel-Info {
  param (
    [string]$Key
  )
  Write-Host "Getting Bazel info: $Key..."
  $ProcInfo = New-Object System.Diagnostics.ProcessStartInfo
  $ProcInfo.FileName = $Bazel
  $ProcInfo.Arguments = "info $Key"
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

# Verify Bazel version
Write-Host "Checking Bazel version..."
& $Bazel --version
if ($LASTEXITCODE -ne 0) {
  Write-Warning "Failed to check Bazel version. Exit code: $LASTEXITCODE"
}

# Fetch dependencies first to ensure the Bazel server is running and external repos are populated.
Write-Host "Fetching dependencies..."
$FetchArgs = @("fetch", "--config=windows", "--repo_env=USE_PYWRAP_RULES=True", "//ci/tools/python/wheel:litert_wheel")
& $Bazel $FetchArgs
if ($LASTEXITCODE -ne 0) {
  throw "Bazel fetch failed with exit code $LASTEXITCODE"
}

$OutputBase = Get-Bazel-Info "output_base"
Write-Host "Bazel output_base: $OutputBase"

$ExecRoot = Get-Bazel-Info "execution_root"
Write-Host "Bazel execution_root: $ExecRoot"
if ($ExecRoot) {
  $ExecBazelOut = Join-Path $ExecRoot "bazel-out"
  $WorkspaceBazelOut = Join-Path $RepoRoot "bazel-out"
  Write-Host "Setting up bazel-out junction..."
  if (Test-Path $WorkspaceBazelOut) {
    if (Test-Path $ExecBazelOut) {
      & cmd.exe /c "rmdir /s /q `"$ExecBazelOut`"" | Out-Null
    }
    New-Item -ItemType Junction -Path $ExecBazelOut -Target $WorkspaceBazelOut | Out-Null
  }
}

$LocalXlaTsl = Join-Path $OutputBase "external\local_xla\xla\tsl\tsl.bzl"
Write-Host "Patching tsl.bzl..."
Replace-InFile $LocalXlaTsl 'clean_dep("//xla/tsl:windows"): get_win_copts(is_external, is_msvc = False),' 'clean_dep("//xla/tsl:windows"): get_win_copts(is_external, is_msvc = True),' | Out-Null

Write-Host "Patching Protobuf..."
$ProtoContext = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\compiler\java\context.h"
if (Test-Path $ProtoContext) {
  $ProtoLines = Get-Content $ProtoContext
  $ProtoLines = $ProtoLines | Where-Object { $_ -notmatch '^\s*#include\s*\\\s*$' }
  Set-Content -Path $ProtoContext -Value $ProtoLines
}
Ensure-Include $ProtoContext '#include "google/protobuf/compiler/java/helpers.h"' ("`n" + '#include "google/protobuf/compiler/java/field_common.h"') | Out-Null

$ProtoMessageSerialization = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\compiler\java\message_serialization.cc"
Write-Host "Patching message_serialization.cc..."
Replace-InFile $ProtoMessageSerialization '#include "google/protobuf/compiler/java/message_serialization.h"' '#include "message_serialization.h"' | Out-Null
$ProtoFullMessage = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\compiler\java\full\message.cc"
Write-Host "Patching full/message.cc..."
Replace-InFile $ProtoFullMessage '#include "google/protobuf/compiler/java/message_serialization.h"' '#include "../message_serialization.h"' | Out-Null

$ProtoJavaBuild = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\compiler\java\BUILD.bazel"
Write-Host "Patching compiler/java/BUILD.bazel..."
if (Test-Path $ProtoJavaBuild) {
  $BuildText = Get-Content $ProtoJavaBuild -Raw
  $InsertLine = [Environment]::NewLine + '        "field_common.h",'
  $BuildText = $BuildText.Replace('"helpers.h",`n        "field_common.h",', '"helpers.h",' + $InsertLine)
  if ($BuildText -notmatch '"helpers.h",\s*"field_common.h"') {
    $BuildText = [regex]::Replace($BuildText, '"helpers.h",', '"helpers.h",' + $InsertLine, 1)
  }
  Set-Content -Path $ProtoJavaBuild -Value $BuildText
}

$UntypedHeader = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\json\internal\untyped_message.h"
Write-Host "Patching untyped_message.h..."
if (Test-Path $UntypedHeader) {
  $Text = Get-Content $UntypedHeader -Raw
  $Text = $Text.Replace(", UntypedMessage,", ", std::unique_ptr<UntypedMessage>,")
  $Text = $Text.Replace("std::vector<UntypedMessage>>;", "std::vector<std::unique_ptr<UntypedMessage>>>;")

  if ($Text -notmatch "GetMessage\(int32_t field_number") {
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
    $Anchor = "  const ResolverPool::Message& desc() const { return *desc_; }"
    if ($Text -match [regex]::Escape($Anchor)) {
      $Text = $Text.Replace($Anchor, $Insert + $Anchor)
    }
  }
  Set-Content -Path $UntypedHeader -Value $Text
}

$UntypedCc = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\json\internal\untyped_message.cc"
Write-Host "Patching untyped_message.cc..."
if (Test-Path $UntypedCc) {
  $CcText = Get-Content $UntypedCc -Raw
  $CcText = $CcText.Replace("InsertField(*field, std::move(group))", "InsertField(*field, std::make_unique<UntypedMessage>(std::move(group)))")
  $CcText = $CcText.Replace("InsertField(field, std::move(*inner))", "InsertField(field, std::make_unique<UntypedMessage>(std::move(*inner)))")

  $InsertFieldPattern = "(?s)template <typename T>\\s+absl::Status UntypedMessage::InsertField\\([^\\{]*\\)\\s*\\{.*?\\n\\}"
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
  if ($NewCcText -eq $CcText -and $CcText -notmatch "using value_type = std::decay_t<T>;") {
    $NewCcText = $CcText.Replace("T&& value) {", "T&& value) {`n  using value_type = std::decay_t<T>;")
  }
  if ($NewCcText -notmatch "using value_type = std::decay_t<T>;") {
    throw "Failed to patch untyped_message.cc InsertField template for MSVC."
  }
  Set-Content -Path $UntypedCc -Value $NewCcText
}

$UnparserTraits = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\json\internal\unparser_traits.h"
Write-Host "Patching unparser_traits.h..."
Replace-InFile $UnparserTraits "return &msg.Get<Msg>(f->proto().number())[idx];" "return msg.GetMessage(f->proto().number(), idx);" | Out-Null

$ProtoCompilerBuild = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\compiler\BUILD.bazel"
Write-Host "Patching compiler/BUILD.bazel..."
Replace-InFile $ProtoCompilerBuild '        "//src/google/protobuf/compiler/objectivec",' '' | Out-Null
Replace-InFile $ProtoCompilerBuild '        "//src/google/protobuf/compiler/rust",' '' | Out-Null

$ProtoMain = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\compiler\main.cc"
Write-Host "Patching compiler/main.cc..."
Replace-InFile $ProtoMain '#include "google/protobuf/compiler/objectivec/generator.h"' '' | Out-Null
Replace-InFile $ProtoMain '#include "google/protobuf/compiler/rust/generator.h"' '' | Out-Null
Replace-InFile $ProtoMain '  objectivec::ObjectiveCGenerator objc_generator;' '' | Out-Null
Replace-InFile $ProtoMain '  cli.RegisterGenerator("--objc_out", "--objc_opt", &objc_generator,' '' | Out-Null
Replace-InFile $ProtoMain '                        "Generate Objective-C header and source.");' '' | Out-Null
Replace-InFile $ProtoMain '  rust::RustGenerator rust_generator;' '' | Out-Null
Replace-InFile $ProtoMain '  cli.RegisterGenerator("--rust_out", "--rust_opt", &rust_generator,' '' | Out-Null
Replace-InFile $ProtoMain '                        "Generate Rust sources.");' '' | Out-Null

Write-Host "Patching HighwayHash..."
$HighwayHashDir = Join-Path $OutputBase "external\highwayhash\highwayhash"
if (-not (Test-Path $HighwayHashDir)) {
  $HighwayHashDir = Join-Path $OutputBase "external\highwayhash"
}

$HighwayHash = Join-Path $HighwayHashDir "compiler_specific.h"
$OldAlign = "#if HH_GCC_VERSION && HH_GCC_VERSION < 408`n#define HH_ALIGNAS(multiple) __attribute__((aligned(multiple)))`n#else`n#define HH_ALIGNAS(multiple) alignas(multiple)  // C++11`n#endif"
$NewAlign = "#if HH_MSC_VERSION`n#define HH_ALIGNAS(multiple) __declspec(align(multiple))`n#elif HH_GCC_VERSION && HH_GCC_VERSION < 408`n#define HH_ALIGNAS(multiple) __attribute__((aligned(multiple)))`n#else`n#define HH_ALIGNAS(multiple) alignas(multiple)  // C++11`n#endif"
Replace-InFile $HighwayHash $OldAlign $NewAlign | Out-Null

$PostfixPattern = '^(\s*)([^;]*?)\s+HH_ALIGNAS\((\d+)\)\s*([^;]*;\s*)$'
Write-Host "Patching HighwayHash sources in $HighwayHashDir..."

if (Test-Path $HighwayHashDir) {
  Get-ChildItem -Path $HighwayHashDir -Recurse -Include *.h,*.cc | ForEach-Object {
    $Lines = Get-Content $_.FullName
    $Changed = $false
    for ($i = 0; $i -lt $Lines.Count; $i++) {
      $Line = $Lines[$i]
      if ($Line -notmatch 'HH_ALIGNAS\(') { continue }
      if ($Line -match '^\s*HH_ALIGNAS\(') { continue }
      if ($Line -match '^\s*(//|#)') { continue }
      if ($Line -match '^\s*(private|public|protected):') { continue }
      $AlignIndex = $Line.IndexOf('HH_ALIGNAS(')
      $BraceIndex = $Line.IndexOf('{')
      if ($BraceIndex -ge 0 -and $BraceIndex -lt $AlignIndex) { continue }
      $NewLine = $Line -replace $PostfixPattern, '$1HH_ALIGNAS($3) $2$4'
      if ($NewLine -ne $Line) {
        $Lines[$i] = $NewLine
        $Changed = $true
      }
    }
    if ($Changed) {
      Set-Content -Path $_.FullName -Value $Lines
    }
  }
} else {
  Write-Warning "HighwayHash directory not found: $HighwayHashDir"
}

$BazelArgs = @(
  "build",
  "-c",
  "opt",
  "--config=windows",
  "--copt=-DLITERT_DISABLE_OPENCL_SUPPORT=1",
  "--repo_env=USE_PYWRAP_RULES=True",
  "--define=protobuf_allow_msvc=true"
)
if ($env:BAZEL_CONFIG_FLAGS) { $BazelArgs += $env:BAZEL_CONFIG_FLAGS.Split(" ") }
if ($env:NIGHTLY_RELEASE_DATE) { $BazelArgs += "--//ci/tools/python/wheel:nightly_iso_date=$($env:NIGHTLY_RELEASE_DATE)" }
if ($env:USE_LOCAL_TF -eq "true") { $BazelArgs += "--config=use_local_tf" }
if ($env:CUSTOM_BAZEL_FLAGS) { $BazelArgs += $env:CUSTOM_BAZEL_FLAGS.Split(" ") }

Write-Host "Starting bazel build..."
& $Bazel @BazelArgs //ci/tools/python/wheel:litert_wheel
if ($LASTEXITCODE -ne 0) {
  throw "Bazel build failed with exit code $LASTEXITCODE"
}

$DistDir = Join-Path $RepoRoot "dist"
if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }
New-Item -ItemType Directory -Path $DistDir | Out-Null

Get-ChildItem -Path (Join-Path $RepoRoot "bazel-bin\ci\tools\python\wheel\dist") -Filter *.whl | Move-Item -Destination $DistDir
Write-Host "Output can be found here:" $DistDir
