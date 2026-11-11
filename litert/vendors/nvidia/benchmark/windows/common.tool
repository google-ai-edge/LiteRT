#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""What the native Windows runners share: the machine's configuration, the host's state,
GPU sampling, process accounting and the build id.

Configuration: windows/config.json next to this file (config.example.json documents it), or
the file LITERT_BENCH_WINDOWS_CONFIG names. Run as a program for the host checks:

  common.tool check            the host's state as JSON; exit 1 when another benchmark runs
  common.tool cooldown C       wait until the GPU is at most C degrees (15 minutes at most)
"""
import ctypes
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MIB = 1 << 20


def utc():
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_config(path=None):
  path = pathlib.Path(path or os.environ.get('LITERT_BENCH_WINDOWS_CONFIG') or HERE / 'config.json')
  if not path.exists():
    raise SystemExit(f'No configuration at {path}; copy config.example.json and set this machine\'s paths')
  config = json.loads(path.read_text(encoding='utf-8-sig'))
  config['_path'] = str(path)
  return config


def registry(name):
  return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def sha256(path):
  with open(path, 'rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def run_text(command, **kw):
  return subprocess.run(command, capture_output=True, text=True, **kw).stdout


# ------------------------------------------------------------------------------- the GPU
def smi(fields, extra=()):
  out = run_text(['nvidia-smi', '-i', '0', f'--query-gpu={fields}', '--format=csv,noheader,nounits', *extra]).strip()
  return [v.strip() for v in out.split(',')] if out else []


def gpu_memory_used_mib():
  values = smi('memory.used')
  return int(values[0]) if values and values[0].isdigit() else 0


def gpu_state():
  names = 'name,driver_version,pstate,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory,utilization.gpu,memory.used,memory.total'
  values = smi(names)
  return dict(zip(names.split(','), values)) if values else {}


BENCHMARK_PROCESSES = ('litert_lm', 'llama-bench', 'llama-server', 'llama-cli')


def busy_processes():
  """Benchmark processes on the GPU (nvidia-smi lists every desktop process on WDDM; only
  another CLI or llama.cpp run counts)."""
  out = run_text(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name', '--format=csv,noheader'])
  busy = []
  for line in out.splitlines():
    if any(name in line.lower() for name in BENCHMARK_PROCESSES):
      busy.append(line.strip())
  return busy


class Sampler:
  """gpu.tsv: seconds since start and the device's memory.used in MiB, every `interval` s."""

  def __init__(self, path, interval=0.1):
    self.path, self.interval = pathlib.Path(path), interval
    self.stop_event = threading.Event()
    self.thread = threading.Thread(target=self.run, daemon=True)
    self.start = time.perf_counter()
    self.thread.start()

  def run(self):
    with self.path.open('w') as stream:
      while not self.stop_event.is_set():
        stream.write(f'{time.perf_counter() - self.start:.3f}\t{gpu_memory_used_mib()}\n')
        stream.flush()
        self.stop_event.wait(self.interval)

  def stop(self):
    self.stop_event.set()
    self.thread.join()


# ------------------------------------------------------------------------------- the host
class SystemPowerStatus(ctypes.Structure):
  _fields_ = [('ac', ctypes.c_ubyte), ('flags', ctypes.c_ubyte), ('percent', ctypes.c_ubyte),
              ('system', ctypes.c_ubyte), ('life', ctypes.c_uint32), ('full_life', ctypes.c_uint32)]


def power():
  status = SystemPowerStatus()
  ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
  scheme = run_text(['powercfg', '/getactivescheme']).strip()
  overlay = None
  try:
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes') as key:
      overlay = winreg.QueryValueEx(key, 'ActiveOverlayAcPowerScheme')[0]
  except OSError:
    pass
  return {'ac_connected': status.ac == 1, 'battery_percent': status.percent if status.percent != 255 else None,
          'active_scheme': scheme, 'overlay_ac_scheme': overlay}


def host_snapshot():
  return {'time': utc(), 'computer': os.environ.get('COMPUTERNAME'), 'power': power(), 'gpu': gpu_state(),
          'busy': busy_processes()}


def check(config):
  snapshot = host_snapshot()
  problems = list(snapshot['busy'])
  host = config.get('host', {})
  if host.get('require_ac') and not snapshot['power']['ac_connected']:
    problems.append('not on AC power')
  if host.get('overlay_ac_scheme') and snapshot['power']['overlay_ac_scheme'] != host['overlay_ac_scheme']:
    problems.append(f"power overlay {snapshot['power']['overlay_ac_scheme']} is not {host['overlay_ac_scheme']}")
  snapshot['problems'] = problems
  return snapshot


def cooldown(limit_c, timeout_s=900):
  waited = 0
  while waited < timeout_s:
    state = gpu_state()
    if int(float(state.get('temperature.gpu', 0))) <= limit_c:
      return {'target_c': limit_c, 'waited_s': waited, 'start_temperature_c': state.get('temperature.gpu')}
    time.sleep(5)
    waited += 5
  return {'target_c': limit_c, 'waited_s': waited, 'start_temperature_c': gpu_state().get('temperature.gpu'), 'timed_out': True}


# ------------------------------------------------------------------------------- processes
class ProcessMemoryCounters(ctypes.Structure):
  _fields_ = [('cb', ctypes.c_uint32), ('PageFaultCount', ctypes.c_uint32)] + [
      (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                           'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                                           'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]


def process_memory(process):
  counters = ProcessMemoryCounters()
  counters.cb = ctypes.sizeof(counters)
  if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(int(process._handle)), ctypes.byref(counters), counters.cb):
    return {'peak_working_set_bytes': counters.PeakWorkingSetSize, 'peak_pagefile_bytes': counters.PeakPagefileUsage}
  return {}


def run_process(command, log_path, env, cwd, sample_path=None, sample_interval=0.1, timeout=None, stdin_text=None):
  """One process with stdout and stderr merged into log_path, its peak memory read at exit
  (lifetime peaks, no polling), and optional GPU sampling. Returns time.json's content with
  the exit code."""
  sampler = Sampler(sample_path, sample_interval) if sample_path else None
  started = time.perf_counter()
  with open(log_path, 'wb') as log:
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL)
    try:
      process.communicate(stdin_text.encode() if stdin_text is not None else None, timeout=timeout)
    except subprocess.TimeoutExpired:
      process.kill()
      process.wait()
    memory = process_memory(process)
  wall = time.perf_counter() - started
  if sampler:
    sampler.stop()
  return dict({'wall_s': wall, 'exit_code': process.returncode, 'timer': 'QueryPerformanceCounter'}, **memory)


# ------------------------------------------------------------------------------- the build
def git(repo, *args):
  try:
    return run_text(['git', '-C', str(repo), *args]).strip()
  except FileNotFoundError:
    return ''


def checkout_build_id(config):
  """'<LiteRT sha9>[+]-<LiteRT-LM sha8>[+]' as env.sh's nb_build_id computes it from the
  configured checkouts (the LiteRT side: the last commit outside the benchmark directory), or
  None when they are not configured."""
  rt_dir, lm_dir = config.get('litert_dir'), config.get('litert_lm_dir')
  if not (rt_dir and lm_dir and git(rt_dir, 'rev-parse', 'HEAD')):
    return None
  exclude = ['--', '.', ':(exclude)litert/vendors/nvidia/benchmark']
  rt = git(rt_dir, 'log', '-1', '--format=%h', '--abbrev=9', 'HEAD', *exclude)
  if git(rt_dir, 'status', '--porcelain', '--untracked-files=no', *exclude):
    rt += '+'
  lm = git(lm_dir, 'rev-parse', '--short=8', 'HEAD')
  if git(lm_dir, 'status', '--porcelain', '--untracked-files=no'):
    lm += '+'
  return f'{rt}-{lm}'


def runtime_stamp(config, cli, libs):
  """The build the runtime directory's binaries came from: its BUILD_ID stamp (build.sh's,
  build.ps1's), else 'bin-' and a digest of the binaries. Model cache pools are keyed by it,
  since the binaries decide what the AOT and runtime caches hold."""
  stamp = pathlib.Path(config['runtime_dir']) / 'BUILD_ID'
  if stamp.exists():
    return stamp.read_text().strip()
  digest = hashlib.sha256(''.join(sha256(p) for p in [cli, *libs.values()]).encode()).hexdigest()[:16]
  return 'bin-' + digest


def runtime_files(config):
  """The CLI and the two NVIDIA plugin libraries in the runtime directory (.dll or .so names)."""
  runtime = pathlib.Path(config['runtime_dir'])
  cli = runtime / config.get('cli', 'litert_lm_advanced_main.exe')
  libs = {}
  for stem in ('libLiteRtCompilerPlugin_Nvidia', 'libLiteRtDispatch_Nvidia'):
    for suffix in ('.dll', '.so'):
      if (runtime / (stem + suffix)).exists():
        libs[stem] = runtime / (stem + suffix)
        break
  if not cli.exists() or len(libs) != 2:
    raise SystemExit(f'Runtime directory {runtime} needs the CLI and both NVIDIA plugin libraries')
  return cli, libs


def model_file(config, name):
  models = config.get('models', {})
  if name in models:
    return pathlib.Path(models[name])
  path = pathlib.Path(name)
  if path.is_file():
    return path
  raise SystemExit(f'{name} is neither a configured model nor a file')


def prompts_dir(config):
  for candidate in (config.get('prompts_dir'), HERE.parent / 'prompts',
                    config.get('litert_dir') and pathlib.Path(config['litert_dir']) / 'litert/vendors/nvidia/benchmark/prompts'):
    if candidate and pathlib.Path(candidate).is_dir():
      return pathlib.Path(candidate)
  raise SystemExit('No prompts directory: set prompts_dir or litert_dir in the configuration')


def main():
  if len(sys.argv) < 2 or sys.argv[1] not in ('check', 'cooldown'):
    raise SystemExit(__doc__)
  config = load_config()
  if sys.argv[1] == 'check':
    snapshot = check(config)
    print(json.dumps(snapshot, indent=1))
    sys.exit(1 if snapshot['problems'] else 0)
  print(json.dumps(cooldown(int(sys.argv[2]))))


if __name__ == '__main__':
  main()
