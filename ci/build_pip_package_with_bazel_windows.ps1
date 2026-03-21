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

$ErrorActionPreference = "Stop"

# ---------------------------
# Utilities
# ---------------------------

function Convert-WslPathToWindows($Path) {
    if ($Path -match '^/mnt/([a-zA-Z])/(.*)$') {
        return "$($matches[1].ToUpper()):\" + ($matches[2] -replace '/', '\')
    }

    if ($Path -match '^\\\\wsl\\\$\\[^\\]+\\mnt\\([a-zA-Z])\\(.*)$') {
        return "$($matches[1].ToUpper()):\" + $matches[2]
    }

    return $Path
}

function Replace-InFile($Path,$Old,$New) {

    if(!(Test-Path $Path)){ return }

    $text = Get-Content $Path -Raw

    if($text.Contains($Old)){
        $text = $text.Replace($Old,$New)
        Set-Content $Path $text
    }
}

function Ensure-Include($Path,$Anchor,$IncludeLine){

    if(!(Test-Path $Path)){ return }

    $text = Get-Content $Path -Raw

    if($text.Contains($IncludeLine)){ return }

    if($text.Contains($Anchor)){
        $text = $text.Replace($Anchor,"$Anchor`n$IncludeLine")
        Set-Content $Path $text
    }
}

function Get-BazelInfo($Key){

    $result = & $Bazel info $Key

    if(!$result){
        throw "Bazel info $Key failed"
    }

    return $result.Trim()
}

# ---------------------------
# Repo paths
# ---------------------------

$ScriptDir = if($PSScriptRoot){$PSScriptRoot}else{Split-Path -Parent $MyInvocation.MyCommand.Path}

$RepoRoot = Convert-WslPathToWindows (Split-Path -Parent $ScriptDir)

Set-Location $RepoRoot

if(!$env:TF_LOCAL_SOURCE_PATH){
    $env:TF_LOCAL_SOURCE_PATH = Join-Path $RepoRoot "third_party\tensorflow"
}

# ---------------------------
# Locate Bazel
# ---------------------------

$Bazel = (Get-Command bazel -ErrorAction SilentlyContinue | Select-Object -First 1).Source

if(!$Bazel){
    throw "Bazel not found in PATH"
}

if($Bazel -notmatch '\.exe$'){

    if(Test-Path "$Bazel.exe"){
        $Bazel = "$Bazel.exe"
    }
    else{

        Write-Host "Creating bazel.exe wrapper"

        Copy-Item $Bazel "$Bazel.exe" -Force

        $Bazel = "$Bazel.exe"
    }
}

Write-Host "Using Bazel: $Bazel"

# ---------------------------
# Verify Bazel
# ---------------------------

Write-Host "Checking Bazel version..."
& $Bazel --version

# ---------------------------
# Fetch dependencies
# ---------------------------

Write-Host "Fetching dependencies..."

& $Bazel fetch `
    --config=windows `
    --repo_env=USE_PYWRAP_RULES=True `
    //ci/tools/python/wheel:litert_wheel

# ---------------------------
# Bazel info
# ---------------------------

$OutputBase = Get-BazelInfo "output_base"
$ExecRoot = Get-BazelInfo "execution_root"

Write-Host "OutputBase: $OutputBase"
Write-Host "ExecRoot: $ExecRoot"

# ---------------------------
# Patch external flags
# ---------------------------

$ExternalDir = Join-Path $OutputBase "external"

Write-Host "Patching external repositories..."

Get-ChildItem $ExternalDir -Recurse -Include *.bazel,*.bzl,*.rc,BUILD* |
Where-Object {!$_.PSIsContainer} |
ForEach-Object {

    Replace-InFile $_.FullName "-Wno-sign-compare" ""
}

# ---------------------------
# Proto fixes
# ---------------------------

$ProtoContext = Join-Path $OutputBase "external\com_google_protobuf\src\google\protobuf\compiler\java\context.h"

Ensure-Include `
$ProtoContext `
'#include "google/protobuf/compiler/java/helpers.h"' `
'#include "google/protobuf/compiler/java/field_common.h"'


# ---------------------------
# HighwayHash fix for MSVC
# ---------------------------

$HighwayHash = Join-Path $OutputBase "external\highwayhash\highwayhash\compiler_specific.h"

$Old = @"
#if HH_GCC_VERSION && HH_GCC_VERSION < 408
#define HH_ALIGNAS(multiple) __attribute__((aligned(multiple)))
#else
#define HH_ALIGNAS(multiple) alignas(multiple)
#endif
"@

$New = @"
#if HH_MSC_VERSION
#define HH_ALIGNAS(multiple) __declspec(align(multiple))
#elif HH_GCC_VERSION && HH_GCC_VERSION < 408
#define HH_ALIGNAS(multiple) __attribute__((aligned(multiple)))
#else
#define HH_ALIGNAS(multiple) alignas(multiple)
#endif
"@

Replace-InFile $HighwayHash $Old $New

# ---------------------------
# Bazel Build
# ---------------------------

$BazelArgs = @(
"build",
"-c","opt",
"--config=windows",
"--repo_env=USE_PYWRAP_RULES=True",
"--define=protobuf_allow_msvc=true",
"--copt=-DLITERT_DISABLE_OPENCL_SUPPORT=1",
"--copt=-mavx2",
"--copt=-mfma",
"--copt=-mf16c",
"--copt=/Iexternal\com_google_protobuf\src",
"--host_copt=/Iexternal\com_google_protobuf\src"
)

Write-Host "Starting build..."

& $Bazel $BazelArgs //ci/tools/python/wheel:litert_wheel

# ---------------------------
# Collect wheel
# ---------------------------

$DistDir = Join-Path $RepoRoot "dist"

Remove-Item $DistDir -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory $DistDir | Out-Null

Get-ChildItem "$RepoRoot\bazel-bin\ci\tools\python\wheel\dist" -Filter *.whl |
Move-Item -Destination $DistDir

Write-Host ""
Write-Host "Build complete"
Write-Host "Output:" $DistDir
