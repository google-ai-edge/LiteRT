# Copyright 2026 Google LLC.
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
"""Resumable host/adb evaluation using LiteRT's C++ run_model input-list mode."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from litert.tools.dense_prediction_eval.evaluate import digest, load_manifest, read_outputs, write_json


@contextlib.contextmanager
def exclusive_lock(path):
  """Hold a nonblocking host lock until the file handle is closed."""
  with path.open('a+b') as lock:
    if os.name == 'nt':
      import msvcrt
      if lock.tell() == 0:
        lock.write(b'\0')
        lock.flush()
      lock.seek(0)
      msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
      import fcntl
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    yield


def execute(command, **kwargs):
  kwargs.setdefault('timeout', 300)
  return subprocess.run(command, check=True, **kwargs)


def environment_options(entries):
  result = {}
  for entry in entries:
    key, separator, value = entry.partition('=')
    if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
      raise ValueError('Environment options must use NAME=VALUE')
    if key == 'LD_LIBRARY_PATH' or key in result:
      raise ValueError(
          'Duplicate environment option; use --library-path for LD_LIBRARY_PATH'
      )
    result[key] = value
  return result


def terminate_remote(adb, remote):
  """Stop only the runner whose PID and executable match this invocation."""
  result = subprocess.run(adb + ['shell', 'cat', remote + '/run.pid'],
                          capture_output=True,
                          text=True,
                          timeout=15)
  pid = result.stdout.strip()
  if not pid.isdecimal():
    return
  command = subprocess.run(adb + ['shell', 'cat', f'/proc/{pid}/cmdline'],
                           capture_output=True,
                           timeout=15).stdout.split(b'\0')[0]
  if command.decode(errors='replace') == remote + '/run_model':
    subprocess.run(adb + ['shell', 'kill', '-TERM', pid],
                   check=False,
                   timeout=15)


def run(args):
  manifest = load_manifest(args.manifest)
  samples = manifest['samples'][:args.
                                limit] if args.limit else manifest['samples']
  if args.chunk_size <= 0 or args.chunk_timeout <= 0 or (args.limit is not None
                                                         and args.limit <= 0):
    raise ValueError('Positive chunk size, timeout and sample limit required')
  extra_environment = environment_options(args.env)
  reserved = ('--graph', '--input_list', '--output_dir', '--signature_index',
              '--write_sample_metrics', '--accelerator')
  if any(a.split('=')[0] in reserved for a in args.runner_arg):
    raise ValueError(
        'Runner arguments cannot override the recorded model/input/output contract'
    )
  args.output.mkdir(parents=True, exist_ok=True)
  provenance = dict(manifest_sha256=digest(args.manifest),
                    model_sha256=digest(args.model),
                    runner_sha256=digest(args.runner),
                    runtime_version=args.runtime_version,
                    accelerator=args.accelerator,
                    device=args.device,
                    signature_index=args.signature_index,
                    runner_args=args.runner_arg,
                    library_path=args.library_path,
                    environment=extra_environment,
                    sample_ids=[s['id'] for s in samples])
  provenance_path = args.output / 'provenance.json'
  if provenance_path.exists() and json.loads(
      provenance_path.read_text()) != provenance:
    raise ValueError('Run provenance changed; use a new output directory')
  write_json(provenance_path, provenance)
  pending = []
  for sample in samples:
    folder = args.output / sample['id']
    if (folder / 'complete.json').exists():
      expected = json.loads((folder / 'complete.json').read_text())
      if expected['input_sha256'] != sample['input_sha256']:
        raise ValueError('Completed sample input identity changed')
      read_outputs(folder,
                   {n: manifest['outputs'][n] for n in sample['targets']})
    else:
      pending.append(sample)
  if not pending:
    return
  adb = ['adb', '-s', args.device] if args.device else None
  remote = f'/data/local/tmp/litert_dense_prediction_eval/{provenance["model_sha256"][:16]}'
  if adb:
    busy = subprocess.run(adb + ['shell', 'pidof', 'run_model'],
                          capture_output=True,
                          text=True)
    if busy.stdout.strip():
      raise RuntimeError('A run_model process is already using the phone')
    execute(adb + ['shell', 'mkdir', '-p', remote])
    for path, name in ((args.runner, 'run_model'), (args.model,
                                                    'model.tflite')):
      target = remote + '/' + name
      execute(adb + ['push', str(path.resolve()), target],
              stdout=subprocess.DEVNULL)
      actual = execute(adb + ['shell', 'sha256sum', target],
                       capture_output=True,
                       text=True).stdout
      if actual.split()[0] != digest(path):
        raise ValueError('Phone file checksum mismatch: ' + target)
    execute(adb + ['shell', 'chmod', '755', remote + '/run_model'])
    properties = {}
    for key in ('ro.product.model', 'ro.soc.model', 'ro.build.fingerprint'):
      properties[key] = subprocess.check_output(adb + ['shell', 'getprop', key],
                                                text=True).strip()
    write_json(args.output / 'device.json', properties)
    for name, service in (('battery_before', 'battery'), ('thermal_before',
                                                          'thermalservice')):
      state = execute(adb + ['shell', 'dumpsys', service],
                      capture_output=True,
                      text=True)
      (args.output / (name + '.txt')).write_text(state.stdout)
  for begin in range(0, len(pending), args.chunk_size):
    batch = pending[begin:begin + args.chunk_size]
    staging = args.output / '_staging'
    if staging.exists():
      shutil.rmtree(staging)
    staging.mkdir()
    lines = []
    for i, sample in enumerate(batch):
      files = []
      if set(sample['inputs']) != {'pixel_values'}:
        raise ValueError(
            'The example orchestrator currently expects one pixel_values input')
      for name, relative in sample['inputs'].items():
        path = (args.manifest.parent / relative).resolve()
        if digest(path) != sample['input_sha256']:
          raise ValueError(f'Prepared input changed: {path}')
        if any(c.isspace() for c in str(path)) or ':=' in str(path):
          raise ValueError(
              'Input-list filenames cannot contain whitespace or :=')
        if adb:
          target = f'{remote}/{i}_{name}.raw'
          execute(adb + ['push', str(path), target], stdout=subprocess.DEVNULL)
        else:
          target = str(path)
        files.append(name + ':=' + target)
      lines.append(' '.join(files))
    input_list = staging / 'inputs.txt'
    input_list.write_text('\n'.join(lines) + '\n')
    binary = remote + '/run_model' if adb else str(args.runner.resolve())
    model = remote + '/model.tflite' if adb else str(args.model.resolve())
    output = remote + '/outputs' if adb else str(
        (staging / 'outputs').resolve())
    manifest_name = remote + '/inputs.txt' if adb else str(input_list.resolve())
    command = [
        binary, '--graph=' + model, '--accelerator=' + args.accelerator,
        '--signature_index=' + str(args.signature_index),
        '--input_list=' + manifest_name, '--output_dir=' + output,
        '--write_sample_metrics=true', *args.runner_arg
    ]
    if adb:
      execute(adb + ['push', str(input_list), manifest_name],
              stdout=subprocess.DEVNULL)
      execute(adb + ['shell', 'rm', '-rf', remote + '/outputs'])
      launch = shlex.join([
          'env', 'LD_LIBRARY_PATH=' + args.library_path,
          *[f'{key}={value}' for key, value in extra_environment.items()],
          *command
      ])
      shell = f'echo $$ > {shlex.quote(remote + "/run.pid")}; exec {launch}'
      command = adb + ['shell', shlex.join(['sh', '-c', shell])]
    environment = os.environ.copy()
    environment.update(extra_environment)
    if args.library_path and not adb:
      environment['LD_LIBRARY_PATH'] = args.library_path
    log_path = args.output / f'chunk_{batch[0]["id"]}.log'
    interrupted = None
    with log_path.open('w') as log:
      try:
        process = subprocess.run(command,
                                 env=environment,
                                 stdout=log,
                                 stderr=subprocess.STDOUT,
                                 timeout=args.chunk_timeout)
      except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
        interrupted = error
        process = subprocess.CompletedProcess(command, returncode=-1)
        if adb:
          terminate_remote(adb, remote)
    if adb:
      subprocess.run(adb +
                     ['pull', remote + '/outputs',
                      str(staging / 'outputs')],
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL,
                     check=False,
                     timeout=300)
    for i, sample in enumerate(batch):
      source = staging / 'outputs' / f'Result_{i}'
      if not (source / 'metrics.json').exists():
        continue
      # A completion marker alone is insufficient: check every output's size and finiteness.
      read_outputs(source, manifest['outputs'])
      destination = args.output / sample['id']
      destination.mkdir(exist_ok=True)
      for name in sample['targets']:
        shutil.move(str(source / (name + '.raw')),
                    destination / (name + '.raw'))
      shutil.copy2(source / 'metrics.json', destination / 'metrics.json')
      write_json(destination / 'complete.json',
                 dict(input_sha256=sample['input_sha256']))
    completed = sum(
        (args.output / s['id'] / 'complete.json').exists() for s in samples)
    write_json(
        args.output / 'progress.json',
        dict(completed=completed,
             requested=len(samples),
             exit_code=process.returncode))
    print(f'Completed {completed}/{len(samples)}', flush=True)
    if interrupted is not None:
      raise interrupted
    if process.returncode:
      raise RuntimeError(
          f'Runner failed; see {log_path}. Completed samples can be resumed.')
    if any(
        not (args.output / s['id'] / 'complete.json').exists() for s in batch):
      raise RuntimeError(
          'Runner exited successfully without completing all requested samples')
    shutil.rmtree(staging)
  if adb:
    for name, service in (('battery_after', 'battery'), ('thermal_after',
                                                         'thermalservice')):
      state = execute(adb + ['shell', 'dumpsys', service],
                      capture_output=True,
                      text=True)
      (args.output / (name + '.txt')).write_text(state.stdout)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--manifest', type=Path, required=True)
  parser.add_argument('--model', type=Path, required=True)
  parser.add_argument('--runner', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--runtime-version', required=True)
  parser.add_argument('--accelerator',
                      choices=('cpu', 'gpu', 'npu'),
                      default='cpu')
  parser.add_argument('--device', help='adb serial; omit for host execution')
  parser.add_argument('--library-path', default='')
  parser.add_argument('--env',
                      action='append',
                      default=[],
                      help='Additional runtime NAME=VALUE, repeatable')
  parser.add_argument('--signature-index', type=int, default=0)
  parser.add_argument('--runner-arg', action='append', default=[])
  parser.add_argument('--chunk-size', type=int, default=10)
  parser.add_argument('--chunk-timeout',
                      type=float,
                      default=900,
                      help='Seconds per runner invocation')
  parser.add_argument('--limit', type=int)
  args = parser.parse_args()
  lock_root = Path.home() / '.cache/litert_dense_prediction_eval'
  lock_root.mkdir(parents=True, exist_ok=True)
  # The lock identity is hashed so device serials cannot escape this directory.
  import hashlib
  identity = args.device or str(args.output.resolve())
  lock_name = hashlib.sha256(identity.encode()).hexdigest() + '.lock'
  with exclusive_lock(lock_root / lock_name):
    run(args)


if __name__ == '__main__':
  main()
