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

"""Validate identical forced histories and compare actual runner full logits.

Reports differences without treating a chosen numerical threshold or mean NLL
as evidence of model-quality equivalence. Requires first measured run dumps.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(directory):
    require(not (directory / 'INVALID_CONTROL.md').exists(),
            'Capture explicitly invalidated: '+str(directory))
    completion_path = directory / 'run.json'
    require(completion_path.is_file(), 'Missing completed-run marker: '+str(directory))
    completion = json.loads(completion_path.read_text())
    require(completion.get('status') == 'completed', 'Run is incomplete: '+str(directory))
    result = {}
    for path in sorted(directory.glob('*.run_000.json')):
        report = json.loads(path.read_text())
        key = report['case_id']
        require(key not in result, 'Duplicate case: '+key)
        require(report['run_index'] == 0 and report['warmup'] is False, 'Wrong repetition: '+str(path))
        require(report['vocab_size'] == 262144, 'Vocabulary size mismatch: '+key)
        require(report.get('logits_dtype') == 'float32-little-endian', 'Unexpected dtype: '+key)
        require(len(report['passes']) == len(report['forced_decode_token_ids'])+1, 'Pass count mismatch: '+key)
        result[key] = report
    require(result, 'No first measured runs in '+str(directory))
    case_count = completion.get('cases', completion.get('completed_case_count'))
    require(type(case_count) is int and case_count == len(result),
            'Completed case count disagrees with available first runs: '+str(directory))
    return result


def vector(directory, filename):
    require(isinstance(filename, str) and Path(filename).name == filename, 'Missing or unsafe raw logit file')
    path = directory / filename
    require(path.stat().st_size == 262144*4, 'Wrong raw length: '+str(path))
    values = np.fromfile(path, dtype='<f4').astype(np.float64)
    require(np.isfinite(values).all(), 'Nonfinite logits: '+str(path))
    return values


def distribution(values):
    z = float(values.max()) + math.log(float(np.exp(values-values.max()).sum()))
    return values-z, z


def summarize(rows):
    if not rows:
        return None
    scored = [r for r in rows if 'baseline_nll' in r]
    out = dict(prediction_rows=len(rows), argmax_matches=sum(r['argmax_match'] for r in rows),
               mean_kl_baseline_to_candidate=float(np.mean([r['kl_baseline_to_candidate'] for r in rows])),
               max_kl_baseline_to_candidate=max(r['kl_baseline_to_candidate'] for r in rows),
               mean_rmse=float(np.mean([r['rmse'] for r in rows])),
               max_rmse=max(r['rmse'] for r in rows),
               minimum_cosine=min(r['cosine'] for r in rows),
               bitwise_equal_rows=sum(r['bitwise_equal'] for r in rows),
               strict_diagnostic_rows=sum(r['argmax_match'] and r['cosine'] >= .999 and r['rmse'] <= .05 for r in rows))
    if scored:
        out.update(scored_targets=len(scored),baseline_mean_nll=float(np.mean([r['baseline_nll'] for r in scored])),
                   candidate_mean_nll=float(np.mean([r['candidate_nll'] for r in scored])),
                   mean_nll_delta_candidate_minus_baseline=float(np.mean([r['candidate_nll']-r['baseline_nll'] for r in scored])),
                   max_absolute_nll_delta=max(abs(r['candidate_nll']-r['baseline_nll']) for r in scored),
                   baseline_target_top1=sum(r['baseline_argmax']==r['target_id'] for r in scored),
                   candidate_target_top1=sum(r['candidate_argmax']==r['target_id'] for r in scored))
    return out


def compare(baseline_dir, candidate_dir, fixture_file):
    baseline, candidate = load(baseline_dir), load(candidate_dir)
    require(baseline.keys() == candidate.keys(), 'Case sets differ')
    fixture_rows = json.loads(fixture_file.read_text())['cases']
    fixtures = {r['id']:r for r in fixture_rows}
    require(len(fixtures) == len(fixture_rows), 'Duplicate fixture IDs')
    require(baseline.keys() <= fixtures.keys(), 'Unknown case in run reports')
    for key in baseline:
        fixture = fixtures[key]
        prompt, forced = fixture['prompt_token_ids'], fixture['forced_token_ids']
        targets = fixture.get('score_target_ids', [])
        require(isinstance(prompt, list) and prompt and prompt[0] == 2 and
                isinstance(forced, list) and isinstance(targets, list), key+': invalid fixture history')
        require(all(type(token) is int and 0 <= token < 262144
                    for token in prompt+forced+targets), key+': invalid fixture token or target ID')
        require(len(targets) <= len(forced)+1, key+': more targets than prediction rows')
        shared = min(len(targets), len(forced))
        require(targets[:shared] == forced[:shared], key+': scored targets do not follow forced history')
    rows = []
    for key in baseline:
        a, b = baseline[key], candidate[key]
        fixture = fixtures[key]
        for report in [a,b]:
            require(report['prompt_token_ids']==fixture['prompt_token_ids'], key+': prompt mismatch')
            require(report['forced_decode_token_ids']==fixture['forced_token_ids'], key+': forced history mismatch')
        require(a['num_threads']==b['num_threads'], key+': thread count mismatch')
        prompt=fixture['prompt_token_ids']
        forced=fixture['forced_token_ids']
        for i,(pa,pb) in enumerate(zip(a['passes'],b['passes'])):
            expected = {'decode_index':i,'input_ids':prompt if i==0 else [forced[i-1]],
                        'context_length_after':len(prompt)+i,'logits_position':len(prompt)+i-1,
                        'kind':'prefill' if i==0 else 'decode'}
            for pass_report in [pa,pb]:
                for field,value in expected.items():
                    require(pass_report[field]==value, f'{key} row{i}: {field} mismatch')
            x,y=vector(baseline_dir,pa['logits_file']),vector(candidate_dir,pb['logits_file'])
            px,zx=distribution(x); py,zy=distribution(y)
            ax,ay=int(np.argmax(x)),int(np.argmax(y))
            require(ax==pa['argmax_id'] and ay==pb['argmax_id'],key+': raw/JSON argmax disagreement')
            top2x=np.sort(np.partition(x,-2)[-2:]); top2y=np.sort(np.partition(y,-2)[-2:])
            norm_x, norm_y = float(np.linalg.norm(x)), float(np.linalg.norm(y))
            cosine = (float(np.dot(x,y)/(norm_x*norm_y)) if norm_x and norm_y
                      else (1.0 if norm_x == norm_y else 0.0))
            row=dict(case_id=key,decode_index=i,logits_position=len(prompt)+i-1,
                     baseline_argmax=ax,candidate_argmax=ay,argmax_match=ax==ay,
                     baseline_top2_margin=float(top2x[-1]-top2x[-2]),candidate_top2_margin=float(top2y[-1]-top2y[-2]),
                     rmse=float(np.sqrt(np.mean((x-y)**2))),max_absolute_error=float(np.max(np.abs(x-y))),
                     cosine=cosine,
                     kl_baseline_to_candidate=max(0.,float(np.dot(np.exp(px),px-py))),
                     bitwise_equal=np.array_equal(x.astype('<f4').view('<u4'),y.astype('<f4').view('<u4')),
                     baseline_argmax_probability=float(np.exp(px[ax])),candidate_probability_for_baseline_argmax=float(np.exp(py[ax])))
            targets=fixture.get('score_target_ids',[])
            if i<len(targets):
                target=targets[i]
                row.update(target_id=target,baseline_nll=float(-px[target]),candidate_nll=float(-py[target]))
            rows.append(row)
    return dict(baseline_directory=str(baseline_dir),candidate_directory=str(candidate_dir),fixture_file=str(fixture_file),
                integrity='Full histories, row positions, byte lengths, finite values and JSON argmax verified',
                interpretation='Actual LiteRT-LM CPU vs tensor runner. Model representation differences must be read with model-audit report. Numerical diagnostic thresholds are not quality acceptance criteria.',
                overall=summarize(rows),by_case={key:summarize([r for r in rows if r['case_id']==key]) for key in baseline},
                by_category={key:summarize([r for r in rows if r['case_id'].startswith(key+'_')]) for key in ['prose','code','arithmetic']},
                worst_kl_rows=sorted(rows,key=lambda x:x['kl_baseline_to_candidate'],reverse=True)[:10],
                worst_target_nll_rows=sorted([r for r in rows if 'baseline_nll' in r],key=lambda x:abs(x['candidate_nll']-x['baseline_nll']),reverse=True)[:10],rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--fixtures',type=Path,default=Path(__file__).parent.parent/'fixtures/manifest.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=compare(args.baseline,args.candidate,args.fixtures)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['overall'],indent=2))
