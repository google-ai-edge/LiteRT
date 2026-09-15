#!/usr/bin/env python3
# Copyright 2026 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare two native-runner binaries on one explicitly selected Android phone.

Weights must already be exported and deployed. Correctness mode checks all
full-vocabulary vectors bitwise on identical token histories. Performance mode
uses one warmup and three measured sessions, with no diagnostic dumps. It reports
paired measurements; it does not infer statistical significance or quality.
"""
import argparse
from array import array
import math
import re
import sys
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time
import uuid


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_fixtures(path, capacity):
    fixtures = {}
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        fields = line.split('\t')
        require(len(fields) == 3, 'Fixture rows need exactly three fields')
        name, prompt_text, forced_text = fields
        require(re.fullmatch(r'[A-Za-z0-9_-]+', name) is not None and name not in fixtures,
                'Unsafe or duplicate fixture ID')
        prompt = [int(x) for x in prompt_text.split(',')]
        forced = [] if forced_text in ('', '-') else [int(x) for x in forced_text.split(',')]
        require(prompt and prompt[0] == 2, 'Prompt must begin with BOS 2')
        require(all(0 <= token < 262144 for token in prompt + forced), 'Invalid token ID')
        require(len(prompt) + len(forced) <= capacity, 'Fixture exceeds capacity')
        fixtures[name] = (prompt, forced)
    require(bool(fixtures), 'Empty fixture file')
    return fixtures


def validate_raw_logits(directory, row):
    filename = row.get('logits_file')
    require(isinstance(filename, str) and Path(filename).name == filename,
            'Missing or unsafe logit filename')
    raw = (directory / filename).read_bytes()
    require(len(raw) == 262144 * 4, 'Wrong full-vocabulary logit size')
    values = array('f')
    values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    require(all(math.isfinite(v) for v in values), 'Nonfinite full logits')
    observed_argmax = max(range(len(values)), key=values.__getitem__)
    require(observed_argmax == row['argmax_id'], 'Raw logits disagree with recorded argmax')
    return hashlib.sha256(raw).hexdigest()


def validate_capture(directory, fixtures, threads, capacity, repeats, warmups, dump):
    """Validate retained files without device access; return measured summaries."""
    directory = Path(directory)
    index = json.loads((directory / 'run.json').read_text())
    require(index.get('status') == 'completed' and index.get('cases') == len(fixtures),
            'Runner completion/case count mismatch')
    expected_files = {f'{name}.{kind}_{i:03d}.json' for name in fixtures
                      for kind, count in [('warmup', warmups), ('run', repeats)]
                      for i in range(count)}
    actual_files = {p.name for p in directory.glob('*.json')
                    if re.search(r'\.(?:run|warmup)_\d+\.json$', p.name)}
    require(actual_files == expected_files, 'Missing or unexpected session records')
    measurements = {}
    session_number = 0
    for name, (prompt, forced) in fixtures.items():
        measured = []
        for kind, count in [('warmup', warmups), ('run', repeats)]:
            for i in range(count):
                record = json.loads((directory / f'{name}.{kind}_{i:03d}.json').read_text())
                expected = {'runner': 'xnnpack_tensor_live_int8', 'case_id': name,
                            'vocab_size': 262144, 'logits_dtype': 'float32-little-endian',
                            'run_index': i, 'warmup': kind == 'warmup',
                            'num_threads': threads, 'kv_capacity': capacity,
                            'prompt_token_ids': prompt, 'forced_decode_token_ids': forced,
                            'static_int2_tensor_count': 60, 'static_int2_operator_count': 60,
                            'static_int2_compact_bytes': 283115520,
                            'share_workspace': True, 'preserve_static_int2': True,
                            'kv_dtype': 'int8', 'kv_layout': 'owner_position_channel',
                            'kv_alignment': 32, 'prefill_rows': 128,
                            'fixed_attention_extent': False, 'memory_report': False,
                            'timings_valid_for_benchmark': True, 'reuse_runtimes': True,
                            'compiled_this_run': session_number == 0}
                for key, value in expected.items():
                    require(record.get(key) == value, f'{name}.{kind}_{i}: {key} mismatch')
                passes = record['passes']
                require(len(passes) == len(forced) + 1, 'Wrong prediction count')
                for j, row in enumerate(passes):
                    require(row.get('kind') == ('prefill' if j == 0 else 'decode') and
                            row.get('decode_index') == j, 'Prediction ordering mismatch')
                    require(row.get('input_ids') == (prompt if j == 0 else [forced[j - 1]]),
                            'Wrong per-pass consumed inputs')
                    require(row.get('context_length_after') == len(prompt) + j and
                            row.get('logits_position') == len(prompt) + j - 1,
                            'Wrong prediction history position')
                    require(type(row.get('argmax_id')) is int and 0 <= row['argmax_id'] < 262144,
                            'Invalid recorded argmax')
                    for key in ['elapsed_ms', 'forward_ms']:
                        value = row.get(key)
                        require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0,
                                'Missing/nonfinite/nonpositive timing')
                    if dump and kind == 'run' and i == 0:
                        validate_raw_logits(directory, row)
                    else:
                        require(row.get('logits_file') is None, 'Unexpected diagnostic dump')
                if kind == 'run':
                    measured.append(record)
                session_number += 1
        if dump:
            measurements[name] = {'vectors': len(measured[0]['passes'])}
        else:
            require(bool(forced), 'Performance fixtures must include forced decode steps')
            prefill = [r['passes'][0]['elapsed_ms'] / 1000 for r in measured]
            decode = [1000 * len(forced) / sum(p['elapsed_ms'] for p in r['passes'][1:])
                      for r in measured]
            measurements[name] = {'prefill_seconds': prefill, 'decode_tokens_per_second': decode,
                                  'prefill_median_seconds': statistics.median(prefill),
                                  'decode_median_tokens_per_second': statistics.median(decode)}
    return measurements


def compare_capture_predictions(reference, candidate, fixtures, repeats, warmups, dump):
    checked = 0
    predictions = 0
    for name in fixtures:
        for kind, count in [('warmup', warmups), ('run', repeats)]:
            for i in range(count):
                filename = f'{name}.{kind}_{i:03d}.json'
                ref = json.loads((Path(reference) / filename).read_text())
                got = json.loads((Path(candidate) / filename).read_text())
                require(len(ref['passes']) == len(got['passes']), 'Different prediction counts')
                for before, after in zip(ref['passes'], got['passes']):
                    require(before['argmax_id'] == after['argmax_id'], 'Different argmax predictions')
                    require(before['logits_position'] == after['logits_position'],
                            'Different prediction positions')
                    predictions += 1
                    if dump and kind == 'run' and i == 0:
                        a = Path(reference) / before['logits_file']
                        b = Path(candidate) / after['logits_file']
                        require(a.read_bytes() == b.read_bytes(),
                                f'Nonidentical full logits: {name}, position {after["logits_position"]}')
                        checked += 1
    return checked, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--affinity-mask', required=True)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--reference-binary', required=True, type=Path)
    parser.add_argument('--remote-bundle-dir', required=True)
    parser.add_argument('--cases-file', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--mode', choices=['correctness', 'performance'], default='correctness')
    parser.add_argument('--order', choices=['reference-first', 'candidate-first'], default='reference-first')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--capacity', type=int, default=8448)
    parser.add_argument('--cooldown-seconds', type=int, default=30)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    args = parser.parse_args()
    if args.threads < 1 or args.cooldown_seconds < 0 or args.timeout_seconds < 1:
        parser.error('Invalid thread count, cooldown, or timeout')
    if args.capacity < 32 or args.capacity > 8448 or args.capacity % 32:
        parser.error('Capacity must be a multiple of 32 in [32,8448]')
    if re.fullmatch(r'[0-9a-fA-F]+', args.affinity_mask) is None or int(args.affinity_mask, 16) == 0:
        parser.error('Affinity must be a nonzero hexadecimal mask')
    fixtures = read_fixtures(args.cases_file, args.capacity)
    if args.mode == 'performance' and any(not forced for _, forced in fixtures.values()):
        parser.error('Performance fixtures require forced decode inputs')
    adb = Path(os.environ['ANDROID_HOME']) / 'platform-tools/adb'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    remote = '/data/local/tmp/litert-native-compare/' + uuid.uuid4().hex
    base = [str(adb), '-s', args.serial]
    def call(argv, **kwargs):
        return subprocess.run(base + list(map(str, argv)), check=True, timeout=30,
                              capture_output=True, text=True, **kwargs)
    def shell(argv):
        return call(['shell', shlex.join(list(map(str, argv)))])
    summary = {'status': 'started', 'serial': args.serial, 'mode': args.mode,
               'remote_dir': remote, 'fixture_sha256': sha256(args.cases_file),
               'bundle_dir': args.remote_bundle_dir, 'runs': {},
               'script_sha256': sha256(__file__), 'threads': args.threads,
               'capacity': args.capacity, 'affinity_mask': args.affinity_mask,
               'order': args.order, 'cooldown_seconds': args.cooldown_seconds}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    shell(['test', '-d', args.remote_bundle_dir])
    shell(['mkdir', '-p', remote])
    call(['push', args.cases_file, remote + '/cases.tsv'])
    require(shell(['sha256sum', remote + '/cases.tsv']).stdout.split()[0] == summary['fixture_sha256'],
            'Deployed fixture SHA256 mismatch')
    summary['remote_fixture_sha256_verified'] = True
    summary['remote_bundle_manifest_sha256'] = shell(
        ['sha256sum', args.remote_bundle_dir.rstrip('/') + '/manifest.json']).stdout.split()[0]
    binaries = {'reference': args.reference_binary, 'candidate': args.binary}
    dump = args.mode == 'correctness'
    repeats, warmups = (1, 0) if dump else (3, 1)
    order = ['reference', 'candidate'] if args.order == 'reference-first' else ['candidate', 'reference']
    try:
        for role in order:
            processes = shell(['ps', '-A']).stdout
            if any(name in processes for name in ('native.android', 'litertlm.android', 'gemma4_native', 'llama-fixture',
                                                    'reference.android', 'candidate.android', 'memory-main.android')):
                raise RuntimeError('Another model runner is active on this phone')
            binary = binaries[role].resolve()
            remote_binary = remote + '/' + role + '.android'
            call(['push', binary, remote_binary])
            shell(['chmod', '700', remote_binary])
            local_binary_sha = sha256(binary)
            remote_binary_sha = shell(['sha256sum', remote_binary]).stdout.split()[0]
            require(remote_binary_sha == local_binary_sha, 'Deployed binary SHA256 mismatch')
            cmd = ['timeout', '-s', 'KILL', str(args.timeout_seconds), 'taskset',
                   args.affinity_mask, remote_binary,
                   '--bundle_dir=' + args.remote_bundle_dir,
                   '--cases_file=' + remote + '/cases.tsv',
                   '--output_dir=' + remote + '/' + role,
                   '--num_threads=' + str(args.threads), '--cache_capacity=' + str(args.capacity),
                   '--prefill_chunk_rows=128', '--kv_alignment=32',
                   '--preserve_static_int2=true', '--share_workspace=true',
                   '--memory_report=false', '--fixed_attention_extent=false', '--trace_position=-1',
                   '--weight_cache=true', '--consistent_arithmetic=false', '--dump_cache=false',
                   '--reuse_runtimes=true', '--warmup_runs=' + str(warmups),
                   '--measured_runs=' + str(repeats), '--dump_full_logits=' + str(dump).lower()]
            local = args.output_dir / role
            local.mkdir()
            (local / 'command.json').write_text(json.dumps({'argv': cmd, 'binary_sha256': local_binary_sha,
                                                              'remote_binary_sha256': remote_binary_sha,
                                                              'remote_binary_sha256_verified': True,
                                                              'serial': args.serial}, indent=2) + '\n')
            for suffix in ['before', 'after']:
                if suffix == 'after':
                    print(f'{args.serial}: {role} {args.mode} starting', flush=True)
                    with (local / 'process.log').open('w') as log:
                        started = time.time()
                        process = subprocess.run(base + ['shell', shlex.join(cmd)], stdout=log,
                                                 stderr=subprocess.STDOUT, check=False,
                                                 timeout=args.timeout_seconds + 30)
                    (local / 'process-status.json').write_text(json.dumps(
                        {'returncode': process.returncode, 'started_unix': started,
                         'finished_unix': time.time()}, indent=2) + '\n')
                    require(process.returncode == 0, f'Runner exited with {process.returncode}')
                snap = call(['shell', 'getprop ro.product.model; dumpsys battery; dumpsys thermalservice; ps -A'])
                (local / ('device-' + suffix + '.txt')).write_text(snap.stdout)
            subprocess.run(base + ['pull', remote + '/' + role + '/.', str(local)],
                           capture_output=True, check=True, timeout=120)
            measurements = validate_capture(local, fixtures, args.threads, args.capacity,
                                            repeats, warmups, dump)
            summary['runs'][role] = measurements
            if role != order[-1] and args.cooldown_seconds:
                print(f'{args.serial}: cooldown {args.cooldown_seconds}s', flush=True)
                time.sleep(args.cooldown_seconds)
        checked, predictions = compare_capture_predictions(
            args.output_dir / 'reference', args.output_dir / 'candidate',
            fixtures, repeats, warmups, dump)
        summary.update(status='passed', bitwise_full_vectors_checked=checked,
                       argmax_predictions_checked=predictions)
    except Exception as error:
        summary.update(status='failed', error=repr(error))
        raise
    finally:
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
