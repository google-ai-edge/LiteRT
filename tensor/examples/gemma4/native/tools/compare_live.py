#!/usr/bin/env python3
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

"""Compare frozen and live captures, including optional logical INT8 cache data.

No inference or device access. Default success means comparison completed, not
that the model is correct. --require-bitwise explicitly requests exact equality.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / 'compare_runs.py'


def load_reference():
    spec = importlib.util.spec_from_file_location('frozen_compare_runs', REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(directory, filename):
    require(isinstance(filename, str) and Path(filename).name == filename,
            'Unsafe or missing artifact filename')
    return directory / filename


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_view(directory, reports, keys, destination):
    destination.mkdir()
    (destination / 'run.json').write_text(json.dumps(dict(status='completed', cases=len(keys))))
    for key in keys:
        report = reports[key]
        safe_file(destination, key + '.run_000.json').write_text(json.dumps(report))
        for row in report['passes']:
            filename = row['logits_file']
            source = safe_file(directory, filename)
            require(source.is_file(), 'Missing full-logit dump: ' + str(source))
            safe_file(destination, filename).symlink_to(source.resolve())


def remove_threshold_counts(value):
    if isinstance(value, dict):
        return {k: remove_threshold_counts(v) for k, v in value.items()
                if k != 'strict_diagnostic_rows'}
    if isinstance(value, list):
        return [remove_threshold_counts(v) for v in value]
    return value


def logical_old_tensor(directory, entry, kind, length, dim, capacity):
    expected = [1, 1, capacity, dim] if kind == 'k' else [1, 1, dim, capacity]
    require(entry['shape'] == expected, 'Unexpected frozen cache tensor shape')
    path = safe_file(directory, entry['file'])
    require(path.stat().st_size == capacity * dim, 'Frozen cache byte count mismatch')
    raw = np.memmap(path, mode='r', dtype=np.int8, shape=tuple(expected))
    # K is [capacity,D]; old V is [D,capacity]. Compare active rows only.
    return raw[0, 0, :length, :] if kind == 'k' else raw[0, 0, :, :length].T


def cache_for_pass(directory, report, row, specs):
    """Return canonical [logical_tokens,D] arrays, or None if no dump exists."""
    case_id, index, length = report['case_id'], row['decode_index'], row['context_length_after']
    history = report['prompt_token_ids'] + report['forced_decode_token_ids'][:index]
    require(length == len(history), 'Cache history length disagrees with pass')
    logits_name = row['logits_file']
    if isinstance(logits_name, str):
        live_files = list(directory.glob(logits_name + '.owner*.i8'))
    else:
        live_files = []
    result = {}
    if live_files:
        require(report.get('kv_dtype') == 'int8' and
                report.get('kv_layout') == 'owner_position_channel',
                'Live cache layout/dtype metadata missing')
        require(type(report.get('kv_capacity')) is int and length <= report['kv_capacity'],
                'Live cache length exceeds declared capacity')
        expected_names = set()
        for owner, spec in specs.items():
            for kind in ['k', 'v']:
                name = f'{logits_name}.owner{owner}.{kind}.i8'
                expected_names.add(name)
                path = safe_file(directory, name)
                require(path.is_file() and path.stat().st_size == length * spec['head_dim'],
                        'Live cache missing or wrong byte count: ' + str(path))
                result[(owner, kind)] = np.fromfile(path, dtype=np.int8).reshape(length, spec['head_dim'])
        require({p.name for p in live_files} == expected_names, 'Unexpected live cache owner/kind files')
        return result, 'live token-major rows; runtime scales are not included in dump metadata'

    stem = case_id + ('.prefill' if index == 0 else f'.decode_{index:04d}')
    lm_path = safe_file(directory, stem + '.kv.json')
    native_path = safe_file(directory, f'{case_id}.pass_{index:04d}.json')
    if lm_path.is_file():
        meta = json.loads(lm_path.read_text())
        require(meta.get('status') == 'completed' and meta['active_tokens'] == length,
                'Frozen LiteRT-LM cache length or completion mismatch')
        require(meta['consumed_token_ids'] == history, 'Frozen cache token history mismatch')
        capacity = meta['capacity']
        for entry in meta['tensors']:
            owner = entry['owner']
            kind = {'key': 'k', 'value': 'v'}.get(entry['kind'])
            require(owner in specs and kind and (owner, kind) not in result,
                    'Unknown or duplicate frozen cache owner/kind')
            require(entry['dtype'] == 'int8' and entry['active_begin'] == 0 and
                    entry['active_end'] == length, 'Invalid frozen cache active interval/dtype')
            result[(owner, kind)] = logical_old_tensor(directory, entry, kind, length,
                                                      specs[owner]['head_dim'], capacity)
        note = 'historical LiteRT-LM dumps lack scale metadata; comparison is of exact INT8 codes'
    elif native_path.is_file():
        meta = json.loads(native_path.read_text())
        require(meta['case_id'] == case_id and meta['pass_index'] == index and
                meta['logical_valid_length'] == length and meta['dtype'] == 'int8',
                'Frozen native cache metadata mismatch')
        require(meta['input_token_ids'] == history, 'Frozen cache token history mismatch')
        capacity = meta['capacity']
        for entry in meta['owners']:
            owner = entry['owner']
            require(owner in specs and entry['head_dim'] == specs[owner]['head_dim'] and
                    entry['zero_point'] == 0, 'Frozen native owner metadata mismatch')
            for kind, field in [('k', 'key'), ('v', 'value')]:
                require((owner, kind) not in result, 'Duplicate frozen cache owner')
                require(np.float32(entry[field]['scale']) ==
                        np.float32(specs[owner][field + '_scale']), 'Frozen cache scale mismatch')
                result[(owner, kind)] = logical_old_tensor(directory, entry[field], kind, length,
                                                          specs[owner]['head_dim'], capacity)
        note = 'frozen native scales checked against published owner specs'
    else:
        return None
    require(set(result) == {(o, k) for o in specs for k in ['k', 'v']},
            'Frozen cache owner set is incomplete')
    return result, note


def compare_caches(baseline, candidate, a_reports, b_reports, keys, specs):
    rows, missing, notes = [], [], set()
    for key in keys:
        for left, right in zip(a_reports[key]['passes'], b_reports[key]['passes']):
            a = cache_for_pass(baseline, a_reports[key], left, specs)
            b = cache_for_pass(candidate, b_reports[key], right, specs)
            if a is None or b is None:
                missing.append(dict(case_id=key, decode_index=left['decode_index'],
                                    baseline_missing=a is None, candidate_missing=b is None))
                continue
            notes.update([a[1], b[1]])
            for owner, kind in sorted(a[0]):
                x, y = a[0][(owner, kind)], b[0][(owner, kind)]
                require(x.shape == y.shape, 'Logical cache shapes differ')
                delta = x.astype(np.int16) - y.astype(np.int16)
                locations = np.argwhere(delta != 0)
                rows.append(dict(case_id=key, decode_index=left['decode_index'], owner=owner,
                                 kind=kind, logical_shape=list(x.shape), compared_codes=x.size,
                                 different_codes=len(locations), bitwise_equal=len(locations) == 0,
                                 maximum_absolute_code_difference=int(np.abs(delta).max()) if delta.size else 0,
                                 first_different_position_channel=locations[0].tolist() if len(locations) else None))
    return dict(status='complete' if rows and not missing else 'partial' if rows else 'unavailable',
                interpretation='Compare logical INT8 codes only. Full-capacity inactive padding is excluded; old V is transposed to token-major. No quality acceptance claim.',
                compared_owner_tensors=len(rows), equal_owner_tensors=sum(r['bitwise_equal'] for r in rows),
                compared_codes=sum(r['compared_codes'] for r in rows),
                different_codes=sum(r['different_codes'] for r in rows),
                missing_passes=missing, quantization_notes=sorted(notes), rows=rows)


def compare(baseline, candidate, fixtures, cases=None, cache=True):
    baseline, candidate, fixtures = baseline.resolve(), candidate.resolve(), fixtures.resolve()
    reference = load_reference()
    a_reports, b_reports = reference.load(baseline), reference.load(candidate)
    keys = sorted(cases if cases else b_reports)
    require(keys and len(keys) == len(set(keys)), 'Empty or duplicate selected cases')
    require(set(keys) <= a_reports.keys() and set(keys) <= b_reports.keys(),
            'Requested case missing from a capture')
    # Isolated metadata views let an exact quick-case subset reuse a frozen
    # multi-case capture. Raw vectors are symlinked; source captures stay intact.
    with tempfile.TemporaryDirectory(prefix='comparison-view-', dir=HERE) as temp:
        a_view, b_view = Path(temp) / 'baseline', Path(temp) / 'candidate'
        selected_view(baseline, a_reports, keys, a_view)
        selected_view(candidate, b_reports, keys, b_view)
        result = remove_threshold_counts(reference.compare(a_view, b_view, fixtures))
    result.update(baseline_directory=str(baseline), candidate_directory=str(candidate),
                  selected_cases=keys, comparison_status='completed',
                  interpretation='Numerical diagnostics under verified identical histories and thread counts. No KL/RMSE threshold establishes quality or replacement readiness.',
                  comparison_implementation=dict(path=str(REFERENCE), sha256=sha(REFERENCE)),
                  fixture_sha256=sha(fixtures))
    result['capture_report_sha256'] = {
        side: {key: sha(safe_file(directory, key + '.run_000.json')) for key in keys}
        for side, directory in [('baseline', baseline), ('candidate', candidate)]}
    if cache:
        owner_path = HERE / 'owner_specs.json'
        specs = {s['owner']: s for s in json.loads(owner_path.read_text())['specs']}
        result['cache'] = compare_caches(baseline, candidate, a_reports, b_reports, keys, specs)
        result['owner_specs_sha256'] = sha(owner_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--fixtures', type=Path, default=HERE.parent / 'fixtures/manifest.json')
    parser.add_argument('--case', action='append', dest='cases')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--require-bitwise', action='store_true')
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate, args.fixtures, args.cases, not args.no_cache)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(logits=result['overall'], cache={k: v for k, v in result.get('cache', {}).items()
                                                           if k not in ['rows', 'missing_passes']}), indent=2))
    if args.require_bitwise and (result['overall']['bitwise_equal_rows'] != result['overall']['prediction_rows']
                                or result.get('cache', {}).get('different_codes', 0) != 0):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
