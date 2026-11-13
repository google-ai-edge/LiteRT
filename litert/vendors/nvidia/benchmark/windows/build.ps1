# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
<#
build.sh on native Windows (x64 and ARM64): builds the NVIDIA compiler plugin, the dispatch
library and the LiteRT-LM advanced CLI with Bazel from the checkouts config.json names, and
stages them into the runtime directory with a BUILD_ID, the stamp the runners key their model
cache pools by.

  powershell -ExecutionPolicy Bypass -File windows\build.ps1 [-RuntimeDir DIR] [-Config FILE] [-NoStage]

config.json's "build" section (the machine's facts; the sequence is this script's):
  bazel        bazel.exe or bazelisk.exe, a path or a name on PATH
  msvc         {"script": vcvars64.bat or VsDevCmd.bat, "args": ["-arch=arm64", ...]}: run first,
               its environment taken over (the compiler, INCLUDE, LIB)
  cuda_home, tensorrt_rtx_root   the SDKs (CUDA_HOME, CUDA_PATH, TENSORRT_RTX_ROOT; their bin on PATH)
  env          other variables: BAZEL_VC, BAZEL_SH, BAZEL_VC_FULL_VERSION, GIT_LFS_SKIP_SMUDGE, ...
  subst        {"drive": "L:", "root": DIR}: a short build drive mapped to DIR first (optional)
  litert       {"startup_args": [...], "args": [...]}: the plugin build in litert_dir; null when the
               LiteRT-LM build builds the plugins too (@litert// targets through the override)
  litert_lm    {"startup_args": [...], "args": [...]}: the CLI build in litert_lm_dir; the script adds
               --override_repository=litert=<litert_dir> unless args name one
Startup args hold the output bases (--output_user_root, --output_base, --[no]windows_enable_symlinks);
args hold the build options (-c opt, --config=windows_arm64, --jobs, repository overrides, ...).
#>
param(
  [string]$Config,
  [string]$RuntimeDir,
  [switch]$NoStage
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
if (-not $Config) {
  $Config = if ($env:LITERT_BENCH_WINDOWS_CONFIG) { $env:LITERT_BENCH_WINDOWS_CONFIG } else { Join-Path $here 'config.json' }
}
$cfg = Get-Content -LiteralPath $Config -Raw | ConvertFrom-Json
$b = $cfg.build
if (-not $b) { throw "$Config has no 'build' section" }
if (-not $RuntimeDir) { $RuntimeDir = $cfg.runtime_dir }
foreach ($name in 'litert_dir', 'litert_lm_dir') {
  if (-not $cfg.$name -or -not (Test-Path -LiteralPath $cfg.$name)) { throw "config.json's $name is not a directory: $($cfg.$name)" }
}

function Slash([string]$path) { return $path.Replace('\', '/') }
# Strings: the non-empty values as strings; callers wrap the result in @(), since a function unrolls a one-item array.
function Strings($values) { return @(@($values) | Where-Object { "$_" -ne '' } | ForEach-Object { "$_" }) }
function Native {
  # Runs a native command, with Windows PowerShell's stderr-as-error behavior off, and returns
  # its stdout lines; $LASTEXITCODE tells the result.
  param([string]$exe, [string[]]$arguments, [string]$cwd)
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    if ($cwd) { Push-Location -LiteralPath $cwd }
    try { $output = & $exe @arguments } finally { if ($cwd) { Pop-Location } }
  } finally { $ErrorActionPreference = $previous }
  return @($output | ForEach-Object { "$_" })
}
function NativeQuery {
  # A read-only query (bazel info, cquery), tried again when Windows fails to start the process:
  # 0xC0000142 (-1073741502) happens to children of an SSH session after a long build.
  param([string]$exe, [string[]]$arguments, [string]$cwd)
  foreach ($attempt in 1..3) {
    $output = @(Native $exe $arguments $cwd)
    if ($LASTEXITCODE -ne -1073741502) { break }
    Start-Sleep -Seconds 5
  }
  return $output
}
function GitOutput([string]$dir, [string[]]$arguments) {
  $out = Native 'git.exe' (@('-C', $dir) + $arguments)
  if ($LASTEXITCODE -ne 0) { throw "git $($arguments -join ' ') failed in $dir" }
  return ($out -join "`n").Trim()
}

# ---------------------------------------------------------------- the environment
if ($b.subst) {
  $drive, $root = $b.subst.drive, (Resolve-Path -LiteralPath $b.subst.root).Path.TrimEnd('\')
  $mapped = @(& subst.exe) | Where-Object { $_ -like "$drive\: => *" }
  if ($mapped) {
    $current = ($mapped -replace "^$drive\\: => ", '').TrimEnd('\')
    if ($current -ine $root) { throw "$drive is mapped to $current; the build needs $root" }
  } elseif (Test-Path -LiteralPath "$drive\") {
    # Mapped by another logon session or token (subst tables are per session): accept it when it
    # shows the root's own top-level entries, refuse another volume.
    $same = -not (Compare-Object @(Get-ChildItem -LiteralPath "$drive\" -Force -Name) @(Get-ChildItem -LiteralPath $root -Force -Name))
    if (-not $same) { throw "$drive is another volume; choose a free build drive in config.json" }
  } else {
    & subst.exe $drive $root
    if ($LASTEXITCODE -ne 0) { throw "subst $drive $root failed" }
  }
}
if ($b.msvc) {
  $script = $b.msvc.script
  if (-not (Test-Path -LiteralPath $script)) { throw "MSVC environment script not found: $script" }
  $arguments = @(Strings $b.msvc.args) -join ' '
  $lines = Native $env:ComSpec @('/d', '/s', '/c', "`"$script`" $arguments >nul && set")
  if ($LASTEXITCODE -ne 0) { throw "MSVC environment script failed: $script $arguments" }
  foreach ($line in $lines) {
    $i = $line.IndexOf('=')
    if ($i -gt 0) { [Environment]::SetEnvironmentVariable($line.Substring(0, $i), $line.Substring($i + 1), 'Process') }
  }
}
if ($b.cuda_home) { $env:CUDA_HOME = $b.cuda_home; $env:CUDA_PATH = $b.cuda_home }
if ($b.tensorrt_rtx_root) { $env:TENSORRT_RTX_ROOT = $b.tensorrt_rtx_root }
if ($b.env) {
  foreach ($p in $b.env.PSObject.Properties) { [Environment]::SetEnvironmentVariable($p.Name, "$($p.Value)", 'Process') }
}
$env:PATH = (@($env:CUDA_HOME, $env:TENSORRT_RTX_ROOT | Where-Object { $_ } | ForEach-Object { Join-Path $_ 'bin' }) + $env:PATH) -join ';'
$bazel = (Get-Command $b.bazel).Source

# ---------------------------------------------------------------- the builds
$pluginTargets = @('//litert/vendors/nvidia/compiler:compiler_plugin_so', '//litert/vendors/nvidia/dispatch:dispatch_api_so')
$cliTarget = '//runtime/engine:litert_lm_advanced_main'
$steps = @()
if ($b.litert) {
  $steps += @{ name = 'litert'; cwd = $cfg.litert_dir; startup = @(Strings $b.litert.startup_args); args = @(Strings $b.litert.args); targets = $pluginTargets }
  $lmTargets = @($cliTarget)
} else {
  $lmTargets = @($cliTarget) + ($pluginTargets | ForEach-Object { "@litert$_" })
}
$lmArgs = @(Strings $b.litert_lm.args)
if (-not ($lmArgs | Where-Object { $_ -like '--override_repository=litert=*' })) {
  $lmArgs += "--override_repository=litert=$(Slash $cfg.litert_dir)"
}
$steps += @{ name = 'litert_lm'; cwd = $cfg.litert_lm_dir; startup = @(Strings $b.litert_lm.startup_args); args = $lmArgs; targets = $lmTargets }

$started = [DateTime]::UtcNow
$record = @{ started_utc = $started.ToString('o'); bazel = $bazel; config = (Resolve-Path -LiteralPath $Config).Path; steps = @() }
$outputs = @()
foreach ($step in $steps) {
  $command = @($step.startup) + @('build') + @($step.args) + @($step.targets)
  Write-Host "`n[$($step.name)] $bazel $($command -join ' ')`n  in $($step.cwd)"
  # A server left by an earlier session keeps that session's drive mappings and environment.
  $null = Native $bazel (@($step.startup) + @('shutdown')) $step.cwd
  $t0 = [DateTime]::UtcNow
  $null = Native $bazel $command $step.cwd
  $code = $LASTEXITCODE
  $record.steps += @{ name = $step.name; cwd = $step.cwd; command = $command; exit_code = $code; seconds = [math]::Round(([DateTime]::UtcNow - $t0).TotalSeconds, 1) }
  if ($code -ne 0) { throw "bazel build failed (exit $code) in $($step.cwd)" }
  if ($NoStage) { continue }
  # The outputs of this step's targets, by Bazel's own account.
  # info takes the build options too: the repository environment they set decides which external
  # repositories exist (without them a bare info can try to fetch others, offline).
  $info = @(NativeQuery $bazel (@($step.startup) + @('info') + @($step.args) + @('execution_root')) $step.cwd)
  $infoExit = $LASTEXITCODE
  $root = $info | Where-Object { $_ -and (Test-Path -LiteralPath $_.Trim()) } | Select-Object -Last 1
  if ($infoExit -ne 0 -or -not $root) { throw "bazel info execution_root failed in $($step.cwd) (exit $infoExit; output: $($info -join ' | '))" }
  $root = $root.Trim()
  $query = @($step.startup) + @('cquery') + @($step.args) + @("set($($step.targets -join ' '))", '--output=files')
  $files = NativeQuery $bazel $query $step.cwd
  if ($LASTEXITCODE -ne 0) { throw "bazel cquery failed in $($step.cwd)" }
  foreach ($f in $files) {
    $f = $f.Trim()
    if (-not $f) { continue }
    $outputs += (Join-Path $root $f)
  }
}

# ---------------------------------------------------------------- the runtime directory
if (-not $NoStage) {
  $exe = $outputs | Where-Object { $_ -like '*.exe' } | Select-Object -First 1
  if (-not $exe) { throw "no .exe among the build outputs: $($outputs -join ', ')" }
  # The CLI's companion DLLs (the constraint provider, the WebGPU stub): beside it in bazel-bin,
  # and the ones its runfiles manifest lists (prebuilt files the exe loads by name).
  $companions = @(Strings (Get-ChildItem -LiteralPath (Split-Path $exe) -Filter '*.dll' -File | ForEach-Object { $_.FullName }))
  $manifest = "$exe.runfiles_manifest"
  if (Test-Path -LiteralPath $manifest) {
    foreach ($line in Get-Content -LiteralPath $manifest) {
      $parts = $line.Trim() -split ' ', 2
      if ($parts.Count -eq 2 -and $parts[1] -like '*.dll') { $companions += $parts[1] }
    }
  }
  $null = New-Item -ItemType Directory -Force -Path $RuntimeDir
  $staged = @()
  $names = @{}
  foreach ($source in @(Strings $outputs) + @($companions)) {
    if (-not (Test-Path -LiteralPath $source)) { throw "build output missing: $source" }
    $leaf = Split-Path $source -Leaf
    if ($names.ContainsKey($leaf.ToLower())) { continue }
    $names[$leaf.ToLower()] = $true
    $target = Join-Path $RuntimeDir $leaf
    Copy-Item -LiteralPath $source -Destination $target -Force
    Set-ItemProperty -LiteralPath $target -Name IsReadOnly -Value $false
    $staged += @{ name = $leaf; bytes = (Get-Item -LiteralPath $target).Length
                  sha256 = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLower(); from = $source }
  }
  # BUILD_ID as env.sh's nb_build_id: the LiteRT commit outside the benchmark directory and the
  # LiteRT-LM commit, each with + when its tree has local changes.
  $exclude = @('--', '.', ':(exclude)litert/vendors/nvidia/benchmark')
  $rt = GitOutput $cfg.litert_dir (@('log', '-1', '--format=%h', '--abbrev=9', 'HEAD') + $exclude)
  if (GitOutput $cfg.litert_dir (@('status', '--porcelain', '--untracked-files=no') + $exclude)) { $rt += '+' }
  $lm = GitOutput $cfg.litert_lm_dir @('rev-parse', '--short=8', 'HEAD')
  if (GitOutput $cfg.litert_lm_dir @('status', '--porcelain', '--untracked-files=no')) { $lm += '+' }
  $buildId = "$rt-$lm"
  [IO.File]::WriteAllText((Join-Path $RuntimeDir 'BUILD_ID'), $buildId)
  $record.build_id = $buildId
  $record.runtime_dir = (Resolve-Path -LiteralPath $RuntimeDir).Path
  $record.files = $staged
  $record.sources = @{ litert = (GitOutput $cfg.litert_dir @('rev-parse', 'HEAD')); litert_lm = (GitOutput $cfg.litert_lm_dir @('rev-parse', 'HEAD')) }
  $record.finished_utc = [DateTime]::UtcNow.ToString('o')
  [IO.File]::WriteAllText((Join-Path $RuntimeDir 'build.json'), (($record | ConvertTo-Json -Depth 6) -replace "`r`n", "`n") + "`n")
  Write-Host "`nRuntime: $RuntimeDir ($buildId)"
  foreach ($s in $staged) { Write-Host ("  {0,-44} {1,12:n0} bytes  {2}" -f $s.name, $s.bytes, $s.sha256.Substring(0, 16)) }
}
Write-Host ("Build finished in {0:n0} s" -f ([DateTime]::UtcNow - $started).TotalSeconds)
