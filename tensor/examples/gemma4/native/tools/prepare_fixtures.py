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

"""Regenerate fixed Gemma4 histories without tokenization or model execution."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_SHA = "f484174da65870ae67ef883923146bc2e3b49e15c7240ead96aa71466e241347"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-manifest', type=Path,
                    default=HERE.parent / 'fixtures/source_manifest.json')
parser.add_argument('--output-dir', type=Path,
                    default=HERE.parent / 'fixtures')
parser.add_argument('--include-8192', action='store_true',
                    help='Also generate the optional 8192-token fixture; this does not run it.')
args = parser.parse_args()
raw = args.source_manifest.read_bytes()
if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
    raise ValueError('Source token manifest does not match the pinned validated seed')
source = json.loads(raw)
case = next(row for row in source['cases'] if row['id'] == 'performance_512')
original = case['prompt_token_ids'] + case['forced_token_ids']
assert len(original) == 544 and original[0] == 2
body = original[1:]
assert all(0 <= token < 262144 for token in body)
lengths = [128, 1024, 4096] + ([8192] if args.include_8192 else [])
stream_length = max(lengths) + 64
stream = ([2] + body * ((stream_length + len(body) - 1) // len(body)))[:stream_length]
assert len(stream) == stream_length and stream.count(2) == 1
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
rows = []
configs = [(f'performance_{length}_64', length, 64, 'benchmark') for length in lengths]
configs += [('capacity_smoke_8_1', 8, 1, 'capacity_smoke'),
            ('native_boundary_4096_2', 4096, 2, 'boundary')]
for case_id, length, decode, kind in configs:
    prompt, forced = stream[:length], stream[length:length + decode]
    assert len(prompt) == length and len(forced) == decode
    content = case_id + '\t' + ','.join(map(str, prompt)) + '\t' + ','.join(map(str, forced)) + '\n'
    path = out / (case_id + '.tsv')
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f'Refusing to overwrite differing fixture: {path}')
    path.write_text(content)
    rows.append({'id': case_id, 'kind': kind, 'fixture_file': path.name,
                 'prompt_token_ids': prompt, 'forced_token_ids': forced,
                 'prompt_length': length, 'forced_decode_steps': decode,
                 'total_logit_predictions': decode + 1,
                 'required_context_capacity': length + decode,
                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
manifest = {'schema_version': 1,
            'source_manifest': args.source_manifest.name,
            'source_manifest_sha256': SOURCE_SHA,
            'source_case': case['id'], 'source_tokens': len(original),
            'tokenizer_sha256': source['tokenizer_sha256'],
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol': 'Single BOS followed by repeated Wikitext-derived body IDs. Shared stream prefixes; fixed teacher-forced continuation. Synthetic throughput workload, not quality evaluation. No tokenization, sampling, chat template, or EOS stopping.',
            'timing_contract': 'Prompt measurement includes the first full-vocabulary prediction after the entire prompt. Then time exactly 64 forced one-token steps separately for benchmark cases; smoke and boundary use the recorded shorter continuation.',
            'minimum_common_capacity': max(row['required_context_capacity'] for row in rows),
            'recommended_common_capacity': 8448,
            'cases': rows}
manifest_path = out / 'manifest.json'
if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
    raise RuntimeError(f'Refusing to overwrite differing manifest: {manifest_path}')
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({row['id']: {'prompt': row['prompt_length'], 'decode': row['forced_decode_steps'], 'sha256': row['sha256']} for row in rows}, indent=2))
