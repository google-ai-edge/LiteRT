# Copyright 2026 Google LLC
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

# Prepares an x86_64 Windows host to cross-compile LiteRT for Windows ARM64 with
# Bazel and MSVC (--config=cross_compile_windows_arm64): ensures the MSVC ARM64
# build tools (Hostx64\arm64 cl.exe, link.exe and the ARM64 CRT/SDK libraries)
# are installed. When running under GitHub Actions, BAZEL_VC is also written to
# $GITHUB_ENV for subsequent steps.

$ErrorActionPreference = 'Stop'

$VsBuildToolsUrl = 'https://download.visualstudio.microsoft.com/download/pr/2ae938ff-cbb6-4e4d-990c-7794a7a03745/15bc8cfc727e099aaba861eb21a811776bcbddc603c0a3f203eba5e5a7710421/vs_BuildTools.exe'
$VsBuildToolsSha256 = '15BC8CFC727E099AABA861EB21A811776BCBDDC603C0A3F203EBA5E5A7710421'

function Invoke-VerifiedDownload {
  param (
    [string]$Url,
    [string]$OutFile,
    [string]$Sha256
  )
  Write-Host "Downloading $Url ..."
  Invoke-WebRequest -Uri $Url -OutFile $OutFile
  $actual = (Get-FileHash -Path $OutFile -Algorithm SHA256).Hash
  if ($actual -ne $Sha256) {
    Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
    throw "SHA256 mismatch for ${OutFile}: expected $Sha256, got $actual"
  }
}

function Export-Env {
  param (
    [string]$Name,
    [string]$Value
  )
  Set-Item -Path "env:$Name" -Value $Value
  if ($env:GITHUB_ENV) {
    "$Name=$Value" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
  }
  Write-Host "$Name=$Value"
}

$VsPath = if ($env:BAZEL_VS -and (Test-Path $env:BAZEL_VS)) {
  $env:BAZEL_VS
}
else {
  (Get-ChildItem -Path 'C:\Program Files*\Microsoft Visual Studio\2022\*' -Directory -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
}
if (-not $VsPath) {
  throw 'Visual Studio 2022 installation not found.'
}
$Arm64ClGlob = "$VsPath\VC\Tools\MSVC\*\bin\Hostx64\arm64\cl.exe"
$Arm64Cl = Get-ChildItem -Path $Arm64ClGlob -ErrorAction SilentlyContinue | Select-Object -First 1
if ($Arm64Cl) {
  Write-Host "Found MSVC ARM64 build tools: $($Arm64Cl.FullName)"
}
else {
  Write-Host "Installing Microsoft.VisualStudio.Component.VC.Tools.ARM64 into $VsPath ..."
  $Installer = Join-Path $env:TEMP 'vs_BuildTools.exe'
  Invoke-VerifiedDownload -Url $VsBuildToolsUrl -OutFile $Installer -Sha256 $VsBuildToolsSha256
  $Proc = Start-Process -FilePath $Installer -ArgumentList @(
    'modify',
    '--installPath', $VsPath,
    '--add', 'Microsoft.VisualStudio.Component.VC.Tools.ARM64',
    '--quiet', '--wait', '--norestart', '--nocache'
  ) -Wait -PassThru
  Remove-Item -LiteralPath $Installer -Force -ErrorAction SilentlyContinue
  if ($Proc.ExitCode -ne 0 -and $Proc.ExitCode -ne 3010) {
    throw "vs_BuildTools.exe modify failed with exit code $($Proc.ExitCode)"
  }
  $Arm64Cl = Get-ChildItem -Path $Arm64ClGlob -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $Arm64Cl) {
    throw 'MSVC ARM64 build tools still not found after installation.'
  }
  Write-Host "Installed MSVC ARM64 build tools: $($Arm64Cl.FullName)"
}
Export-Env -Name 'BAZEL_VC' -Value (Join-Path $VsPath 'VC')
