#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.XNNPACK file.

"""Deploys the standalone tensor CMake tests to an Android arm64 device.

Reads tensor-tests.json from --build-dir. Supports GoogleTest XML and native
self-check programs whose main returns nonzero on failure. The manifest contains a tests array
with name, absolute binary path, and optional labels for each executable.
Requires statically linked project/C++ dependencies; Android system libraries
remain dynamically linked. Results and a summary.json are written to a new
--output-dir. Device artifacts are retained unless --cleanup is requested.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET


SYSTEM_LIBRARIES = frozenset(
    ('libc.so', 'libm.so', 'libdl.so', 'liblog.so', 'libandroid.so', 'libz.so')
)


class RunError(Exception):
  """An actionable preflight, deployment, or result collection failure."""


def run_command(args, timeout=30):
  try:
    return subprocess.run(
        [str(arg) for arg in args], capture_output=True, text=True,
        errors='replace', timeout=timeout, check=False,
    )
  except (OSError, subprocess.TimeoutExpired) as error:
    raise RunError(f'Could not run {shlex.join(map(str, args))}: {error}') from error


def checked_command(args, timeout=30):
  result = run_command(args, timeout)
  if result.returncode:
    detail = (result.stderr + '\n' + result.stdout).strip()
    raise RunError(
        f'Command exited {result.returncode}: {shlex.join(map(str, args))}'
        f'\n{detail}'
    )
  return result.stdout.strip()


def find_adb():
  android_home = os.environ.get('ANDROID_HOME')
  if android_home:
    candidate = Path(android_home) / 'platform-tools' / 'adb'
    if candidate.is_file() and os.access(candidate, os.X_OK):
      return str(candidate)
  candidate = shutil.which('adb')
  if candidate:
    return candidate
  raise RunError('adb was not found in ANDROID_HOME/platform-tools or PATH.')


def read_cache(build_dir):
  path = build_dir / 'CMakeCache.txt'
  if not path.is_file():
    raise RunError(f'Missing CMake cache: {path}')
  cache = {}
  for line in path.read_text().splitlines():
    match = re.match(r'^([^/#][^:]*):[^=]+=(.*)$', line)
    if match:
      cache[match[1]] = match[2]
  return cache


def find_readelf(cache):
  # CMake's Android toolchain need not persist ANDROID_NDK itself in the cache.
  toolchain = Path(cache.get('CMAKE_TOOLCHAIN_FILE', ''))
  toolchain_ndk = None
  if toolchain.name == 'android.toolchain.cmake' and len(toolchain.parts) >= 4:
    toolchain_ndk = toolchain.parent.parent.parent
  candidates = [
      cache.get('CMAKE_ANDROID_NDK'), cache.get('ANDROID_NDK'),
      toolchain_ndk,
      os.environ.get('ANDROID_NDK'), os.environ.get('ANDROID_NDK_HOME'),
  ]
  for ndk in candidates:
    if not ndk:
      continue
    for candidate in sorted(
        (Path(ndk) / 'toolchains' / 'llvm' / 'prebuilt').glob('*/bin/llvm-readelf')
    ):
      if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
  raise RunError(
      'NDK llvm-readelf was not found. Configure the Android build with '
      'the NDK toolchain or CMAKE_ANDROID_NDK/ANDROID_NDK, or set ANDROID_NDK.'
  )


def load_tests(build_dir, pattern):
  path = build_dir / 'tensor-tests.json'
  try:
    manifest = json.loads(path.read_text())
  except (OSError, ValueError) as error:
    raise RunError(f'Cannot read test manifest {path}: {error}') from error
  tests = manifest.get('tests') if isinstance(manifest, dict) else None
  if not isinstance(tests, list) or not tests:
    raise RunError(f'{path} must contain a nonempty tests array.')
  try:
    expression = re.compile(pattern) if pattern else None
  except re.error as error:
    raise RunError(f'Invalid --filter regular expression: {error}') from error
  selected = []
  names = set()
  for item in tests:
    if not isinstance(item, dict):
      raise RunError('Every test manifest entry must be an object.')
    name, binary = item.get('name'), item.get('binary')
    if not isinstance(name, str) or not name or name in names:
      raise RunError(f'Test names must be nonempty and unique: {name!r}')
    names.add(name)
    if not isinstance(binary, str) or not Path(binary).is_absolute():
      raise RunError(f'{name}: binary must be an absolute path.')
    labels = item.get('labels', [])
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
      raise RunError(f'{name}: labels must be an array of strings.')
    framework = item.get('framework', 'gtest')
    if framework not in ('gtest', 'exit_code'):
      raise RunError(f'{name}: unsupported test framework: {framework}')
    if expression is None or expression.search(name):
      selected.append({'name': name, 'binary': binary, 'labels': labels,
                       'framework': framework})
  if not selected:
    raise RunError('No manifest tests match --filter.')
  return selected


def inspect_binary(binary, readelf):
  path = Path(binary)
  try:
    with path.open('rb') as source:
      header = source.read(64)
  except OSError as error:
    raise RunError(f'Cannot read test binary {path}: {error}') from error
  if len(header) < 64 or header[:6] != b'\x7fELF\x02\x01':
    raise RunError(f'{path}: expected a little-endian ELF64 executable.')
  elf_type, machine = struct.unpack_from('<HH', header, 16)
  if machine != 183 or elf_type not in (2, 3):
    raise RunError(f'{path}: expected an AArch64 executable; refusing host/other ELF.')
  details = checked_command([readelf, '--program-headers', '--dynamic', path])
  interpreters = re.findall(r'Requesting program interpreter:\s*([^\]]+)\]', details)
  if interpreters != ['/system/bin/linker64']:
    raise RunError(
        f'{path}: expected Android interpreter /system/bin/linker64, '
        f'found {interpreters!r}.'
    )
  needed = sorted(set(re.findall(r'\(NEEDED\).*?\[([^\]]+)\]', details)))
  unsupported = sorted(set(needed) - SYSTEM_LIBRARIES)
  if unsupported:
    raise RunError(
        f'{path}: shared dependencies are not Android system libraries: '
        f'{", ".join(unsupported)}. Rebuild with BUILD_SHARED_LIBS=OFF, '
        'XNNPACK_LIBRARY_TYPE=static and ANDROID_STL=c++_static. '
        'This runner does not deploy shared dependencies.'
    )
  return {'machine': 'AArch64', 'interpreter': interpreters[0], 'needed': needed}


def select_device(adb, requested):
  output = checked_command([adb, 'devices'])
  devices = {}
  for line in output.splitlines():
    fields = line.split()
    if len(fields) >= 2 and fields[0] != 'List' and not line.startswith('*'):
      devices[fields[0]] = fields[1]
  if requested:
    if devices.get(requested) != 'device':
      raise RunError(
          f'Device {requested!r} is not authorized and online '
          f'(state: {devices.get(requested, "not listed")}).'
      )
    return requested
  authorized = [serial for serial, state in devices.items() if state == 'device']
  if len(authorized) != 1:
    raise RunError(
        'Without --serial, exactly one authorized online device is required. '
        f'Found: {devices or "none"}. Use --serial to select a device.'
    )
  return authorized[0]


def parse_xml(path):
  try:
    root = ET.parse(path).getroot()
    if root.tag not in ('testsuites', 'testsuite'):
      raise ValueError(f'unexpected root element {root.tag!r}')
    cases = list(root.iter('testcase'))
    for key in ('tests', 'failures', 'errors', 'disabled', 'skipped'):
      if root.get(key) is not None and int(root.get(key)) < 0:
        raise ValueError(f'negative {key} count')
    if root.get('tests') is not None and int(root.get('tests')) != len(cases):
      raise ValueError('declared test count differs from the testcase records')
    skipped = []
    failures = errors = executed = disabled = 0
    for case in cases:
      if case.get('status') == 'notrun':
        disabled += 1
        continue
      executed += 1
      failures += bool(case.findall('failure'))
      errors += bool(case.findall('error'))
      skip = case.find('skipped')
      if skip is not None or case.get('result') == 'skipped':
        skipped.append({
            'name': f'{case.get("classname", "")}.{case.get("name", "")}',
            'message': (skip.get('message', '') or skip.text or '')
            if skip is not None else '',
        })
    failures = max(failures, int(root.get('failures', '0')))
    errors = max(errors, int(root.get('errors', '0')))
    return {
        'cases': len(cases), 'executed': executed, 'disabled': disabled,
        'passed': max(0, executed - failures - errors - len(skipped)),
        'failures': failures, 'errors': errors, 'skipped': len(skipped),
        'skipped_cases': skipped,
    }
  except (OSError, ET.ParseError, ValueError) as error:
    raise RunError(f'Missing or invalid GoogleTest XML {path}: {error}') from error


def run_test(adb_args, remote_dir, test, index, output_dir, timeout):
  stem = f'{index:02d}-' + re.sub(r'[^A-Za-z0-9_.-]', '_', test['name'])[:100]
  remote_binary = f'{remote_dir}/bin/{stem}'
  remote_result = f'{remote_dir}/results/{stem}'
  record = dict(test, artifact_prefix=stem, outcome='failed', issues=[])
  local_prefix = output_dir / stem
  started = time.monotonic()
  try:
    checked_command([*adb_args, 'push', test['binary'], remote_binary], timeout=120)
    checked_command([*adb_args, 'shell', shlex.join(['chmod', '700', remote_binary])])
    invocation = [
        'timeout', '-s', 'KILL', str(timeout), 'env',
        f'TMPDIR={remote_dir}/tmp', f'TEST_TMPDIR={remote_dir}/tmp',
        'GTEST_TOTAL_SHARDS=1', 'GTEST_SHARD_INDEX=0',
        remote_binary,
    ]
    if test.get('framework', 'gtest') == 'gtest':
      invocation.extend(['--gtest_color=no', '--gtest_filter=*', '--gtest_repeat=1',
                         f'--gtest_output=xml:{remote_result}.xml'])
    remote_command = (
        f'cd {shlex.quote(remote_dir)} && {shlex.join(invocation)} '
        f'> {shlex.quote(remote_result + ".log")} 2>&1; '
        'result=$?; '
        f'printf "%s\\n" "$result" > {shlex.quote(remote_result + ".exit")}; '
        'exit "$result"'
    )
    result = run_command([*adb_args, 'shell', remote_command], timeout=timeout + 30)
    record['adb_exit_code'] = result.returncode
    Path(f'{local_prefix}.transport.log').write_text(result.stdout + result.stderr)
  except RunError as error:
    record['issues'].append(str(error))
  record['duration_seconds'] = round(time.monotonic() - started, 3)
  suffixes = ('log', 'xml', 'exit') if test.get('framework', 'gtest') == 'gtest' else ('log', 'exit')
  for suffix in suffixes:
    try:
      checked_command([
          *adb_args, 'pull', f'{remote_result}.{suffix}', f'{local_prefix}.{suffix}'
      ])
    except RunError as error:
      record['issues'].append(str(error))
  try:
    record['exit_code'] = int(Path(f'{local_prefix}.exit').read_text().strip())
    if record['exit_code']:
      record['issues'].append(f'Test exited {record["exit_code"]}.')
    if record.get('adb_exit_code', 0) != record['exit_code']:
      record['issues'].append('adb exit status differs from the recorded test status.')
    record['timed_out'] = record['exit_code'] in (124, 137)
  except (OSError, ValueError) as error:
    record['issues'].append(f'Missing or invalid test exit status: {error}')
  if test.get('framework', 'gtest') == 'gtest':
    try:
      record.update(parse_xml(Path(f'{local_prefix}.xml')))
      if record['executed'] == 0:
        record['issues'].append('GoogleTest XML contains zero executed cases.')
      if record['failures'] or record['errors']:
        record['issues'].append('GoogleTest XML reports failures or errors.')
    except RunError as error:
      record['issues'].append(str(error))
  else:
    record.update(cases=1, executed=1, passed=int(not record['issues']),
                  failures=int(bool(record['issues'])), errors=0, skipped=0, disabled=0)
  if not record['issues']:
    record['outcome'] = 'passed'
  return record


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--build-dir', required=True, type=Path)
  parser.add_argument('--output-dir', required=True, type=Path,
                      help='New local directory for logs, XML, and summary.json.')
  parser.add_argument('--serial', help='Authorized adb device serial.')
  parser.add_argument('--filter', help='Regex selecting manifest test names.')
  parser.add_argument('--timeout-seconds', type=int, default=120,
                      help='Timeout for each test executable (default: 120).')
  parser.add_argument('--cleanup', action='store_true',
                      help='Remove only this run\'s remote directory after collection.')
  args = parser.parse_args(argv)
  if args.timeout_seconds <= 0:
    parser.error('--timeout-seconds must be positive')
  args.build_dir = args.build_dir.resolve()
  args.output_dir = args.output_dir.resolve()
  try:
    args.output_dir.mkdir(parents=True, exist_ok=False)
  except OSError as error:
    print(f'Output directory must be new: {args.output_dir}: {error}', file=sys.stderr)
    return 2
  summary = {
      'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'build_dir': str(args.build_dir), 'tests': [], 'issues': [],
      'outcome': 'failed', 'remote_cleaned': False,
  }
  remote_dir = None
  adb_args = None
  try:
    cache = read_cache(args.build_dir)
    readelf = find_readelf(cache)
    tests = load_tests(args.build_dir, args.filter)
    for test in tests:
      test['elf'] = inspect_binary(test['binary'], readelf)
    adb = find_adb()
    serial = select_device(adb, args.serial)
    adb_args = [adb, '-s', serial]
    summary.update(serial=serial, adb=adb, readelf=readelf, selected_tests=len(tests))
    abi = checked_command([*adb_args, 'shell', 'getprop ro.product.cpu.abilist'])
    sdk = checked_command([*adb_args, 'shell', 'getprop ro.build.version.sdk'])
    summary.update(device_abis=abi, device_api=sdk)
    if 'arm64-v8a' not in abi.split(','):
      raise RunError(f'Device does not advertise arm64-v8a: {abi!r}')
    platform = cache.get('ANDROID_PLATFORM') or cache.get('CMAKE_SYSTEM_VERSION', '')
    minimum_api = re.fullmatch(r'(?:android-)?(\d+)', platform)
    if not sdk.isdigit():
      raise RunError(f'Could not determine device API level: {sdk!r}')
    if minimum_api and int(sdk) < int(minimum_api[1]):
      raise RunError(f'Device API {sdk} is below build platform {platform}.')
    checked_command([*adb_args, 'shell', 'command -v timeout'])
    remote_dir = f'/data/local/tmp/litert-tensor-tests/{uuid.uuid4().hex}'
    summary['remote_dir'] = remote_dir
    checked_command([*adb_args, 'shell', shlex.join([
        'mkdir', '-p', f'{remote_dir}/bin', f'{remote_dir}/results', f'{remote_dir}/tmp'
    ])])
    print(f'Device: {serial}; artifacts: {remote_dir}', flush=True)
    for index, test in enumerate(tests, 1):
      print(f'[{index}/{len(tests)}] {test["name"]}', flush=True)
      record = run_test(
          adb_args, remote_dir, test, index, args.output_dir, args.timeout_seconds
      )
      summary['tests'].append(record)
      print(
          f'  {record["outcome"].upper()}: {record.get("passed", 0)} passed, '
          f'{record.get("skipped", 0)} skipped, '
          f'{record.get("failures", 0)} failures, {record.get("errors", 0)} errors',
          flush=True,
      )
      for issue in record['issues']:
        print(f'  {issue}', file=sys.stderr)
      (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if all(test['outcome'] == 'passed' for test in summary['tests']):
      summary['outcome'] = 'passed'
  except (RunError, OSError) as error:
    summary['issues'].append(str(error))
    print(f'ERROR: {error}', file=sys.stderr)
  except KeyboardInterrupt:
    summary['issues'].append('Interrupted; remote timeout bounds any active test.')
    print('Interrupted. Collected results will be preserved.', file=sys.stderr)
  finally:
    if args.cleanup and remote_dir and adb_args:
      try:
        checked_command([*adb_args, 'shell', shlex.join(['rm', '-rf', remote_dir])])
        summary['remote_cleaned'] = True
      except RunError as error:
        summary['issues'].append(f'Cleanup failed: {error}')
        summary['outcome'] = 'failed'
    summary['totals'] = {
        key: sum(test.get(key, 0) for test in summary['tests'])
        for key in ('cases', 'executed', 'passed', 'skipped', 'disabled', 'failures', 'errors')
    }
    summary['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Results: {args.output_dir / "summary.json"}', flush=True)
  return 0 if summary['outcome'] == 'passed' else 1


if __name__ == '__main__':
  sys.exit(main())
