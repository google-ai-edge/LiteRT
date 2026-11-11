#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""The evaluation tables of one platform from its evaluation runs.

usage: eval.tool --platform ID [--lcb DIR...] [--gpqa DIR...] [--out DIR]

LiveCodeBench (eval/lcb): every run directory under the --lcb roots that holds grades.json
(grade.tool) and run.json (run.sh) is a candidate; the newest complete run per (model,
runtime, mode, output limit) makes the summary table, and the per-problem table lists those
runs' verdicts. Written to --out/lcb/<platform>.md.

GPQA Diamond (eval/gpqa): every run directory under the --gpqa roots with results.json;
the newest run per (model, output limit, question count) makes the table, pass@1 as
Artificial Analysis counts it. Written to --out/gpqa/<platform>.md.

Only what the runs say is written: a provenance line and the tables.
"""
import argparse
import json
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
  return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))


def fmt(value, digits=1):
  return '—' if value is None else f'{value:,.{digits}f}'


def mean(rows, key, percent=False, digits=1):
  values = [row[key] for row in rows if row.get(key) is not None]
  if not values:
    return '—'
  value = statistics.mean(values)
  return f'{100 * value:.0f}%' if percent else f'{value:,.{digits}f}'


def table(header, rows):
  return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]


# ------------------------------------------------------------------------------------- LCB
def lcb_runs(roots, selection):
  runs = []
  for root in roots:
    for grades_path in sorted(pathlib.Path(root).rglob('grades.json')):
      run = grades_path.parent
      grades = read(grades_path)
      config = grades.get('config') or (read(run / 'run.json') if (run / 'run.json').exists() else {})
      rows = [r for r in grades['rows'] if r['status'] != 'missing']
      if not rows or not config:
        continue
      runtime = 'llama.cpp' if any(r.get('runtime') == 'llama.cpp' for r in rows) else 'TensorRT-RTX'
      start = max((r.get('start', '') for r in rows), default='') or ''
      stamp = re.search(r'(\d{8}T\d{6})Z', run.name)
      runs.append({'run': run, 'label': run.name, 'rows': rows, 'config': config, 'runtime': runtime,
                   'model': config.get('profile', run.name.split('_')[0]), 'mode': config.get('mode', 'plain'),
                   'budget': int(config.get('max_output_tokens', 0)), 'context': int(config.get('context', 0)),
                   'complete': {r['question_id'] for r in rows} >= set(selection),
                   'when': stamp.group(1) if stamp else grades_path.stat().st_mtime})
  return runs


def sampler(run, models):
  """The sampler of a run: the explicit --sampler_* arguments its run.json records, else the
  model's policy (models.json: greedy, or top-k 64 / top-p 0.95 / temperature 1.0)."""
  explicit = run['config'].get('sampler_arguments') or []
  if explicit:
    values = dict(a.lstrip('-').split('=', 1) for a in explicit if '=' in a)
    if values.get('sampler_top_k') == '1':
      return 'greedy'
    return (f"top-k {values.get('sampler_top_k', '?')}, top-p {values.get('sampler_top_p', '?')}, "
            f"T {values.get('sampler_temperature', '?')}, seed {values.get('sampler_seed', '?')}")
  policy = models.get(run['model'], {}).get('sampler', 'model')
  return 'greedy' if policy == 'greedy' else 'top-k 64, top-p 0.95, T 1.0'


def lcb_tables(roots, platform, models):
  selection = {s['question_id']: s for s in read(ROOT / 'eval/lcb/selection.json')}
  runs = lcb_runs(roots, selection)
  chosen = {}
  for run in sorted(runs, key=lambda r: str(r['when'])):
    if run['complete']:
      chosen[(run['model'], run['runtime'], run['mode'], run['budget'], sampler(run, models))] = run
  chosen = [chosen[k] for k in sorted(chosen, key=lambda k: (list(models).index(k[0]) if k[0] in models else 99, k[1], k[3], k[2]))]
  if not chosen:
    return None
  rows = []
  for run in chosen:
    r = run['rows']
    rows.append([models.get(run['model'], {}).get('name', run['model']), run['runtime'], 'On' if run['mode'] == 'mtp' else 'Off',
                 sampler(run, models), f"{run['budget']:,}", f"{run['context']:,}" if run['context'] else '—',
                 f"{sum(x['status'] == 'pass' for x in r)}/{len(r)}",
                 str(sum(x['status'] == 'no code' for x in r)), mean(r, 'decode_tokens_per_second'),
                 mean(r, 'decode_tokens', digits=0), mean(r, 'acceptance', percent=True)])
  summary = table(['Model', 'Runtime', 'MTP', 'Sampler', 'Output limit', 'Context capacity', 'Passed / attempted', 'No code',
                   'Decode (tokens/s)', 'Mean output tokens', 'Draft acceptance'], rows)
  columns = [f"{models.get(run['model'], {}).get('name', run['model'])} {run['runtime']} {run['budget']:,} {run['mode']} ({sampler(run, models)})" for run in chosen]
  by_run = [{r['question_id']: r for r in run['rows']} for run in chosen]
  per_problem = []
  for qid, item in selection.items():
    cells = []
    for rows_of in by_run:
      row = rows_of.get(qid, {})
      cells.append({'pass': 'pass', 'fail': f"fail ({row.get('passed', 0)}/{row.get('total', 0)})",
                    'no code': 'no code', 'missing': '—'}[row.get('status', 'missing')])
    per_problem.append([qid, item['platform'], item['difficulty'], str(item['tests'])] + cells)
  problems = table(['Problem', 'Platform', 'Difficulty', 'Tests'] + columns, per_problem)
  tests = sum(s['tests'] for s in selection.values())
  intro = (f"{platform['name']}. LiveCodeBench v6 code generation, {len(selection)} problems, {tests:,} public and hidden "
           f"tests; one answer per problem, pass when every test passes. Runs: " + ', '.join(f"`{run['label']}`" for run in chosen)
           + '. Method and reproduction: README.md.')
  return '\n'.join(['# LiveCodeBench v6', '', intro, '', '## Summary', '', *summary, '', '## Per problem', '', *problems, ''])


# ------------------------------------------------------------------------------------ GPQA
def gpqa_tables(roots, platform, models):
  runs = []
  for root in roots:
    for results_path in sorted(pathlib.Path(root).rglob('results.json')):
      run = results_path.parent
      if not (run / 'run.json').exists():
        continue
      results, config = read(results_path), read(run / 'run.json')
      if results:
        runs.append({'label': run.name, 'results': results, 'config': config, 'when': results_path.stat().st_mtime})
  chosen = {}
  for run in sorted(runs, key=lambda r: r['when']):
    chosen[(run['config'].get('model'), run['config'].get('max_output_tokens'), len(run['results']))] = run
  if not chosen:
    return None
  rows = []
  for key, run in sorted(chosen.items(), key=lambda kv: str(kv[0])):
    r = run['results']
    right = sum(bool(x.get('is_correct')) for x in r)
    rows.append([models.get(run['config'].get('model'), {}).get('name', run['config'].get('model')), str(len(r)),
                 f"{right}/{len(r)} ({100 * right / len(r):.0f}%)", str(sum(x.get('chosen') is None for x in r)),
                 str(sum(x.get('exit_code') != 0 for x in r)), f"{run['config'].get('max_output_tokens', 0):,}",
                 mean(r, 'output_tokens', digits=0), mean(r, 'decode_tokens_per_second'),
                 fmt(sum(x.get('wall_s') or 0 for x in r) / 60, 0), f"`{run['label']}`"])
  summary = table(['Model', 'Questions', 'Correct (pass@1)', 'No answer', 'Failed runs', 'Output limit', 'Mean output tokens',
                   'Decode (tokens/s)', 'Total minutes', 'Run'], rows)
  intro = (f"{platform['name']}. GPQA Diamond by Artificial Analysis's method (its prompt, answer extraction, "
           "zero-shot pass@1, temperature 0.6), through LiteRT-LM on TensorRT-RTX. Method and reproduction: README.md.")
  return '\n'.join(['# GPQA Diamond', '', intro, '', *summary, ''])


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--platform', required=True)
  parser.add_argument('--lcb', nargs='*', default=[], metavar='DIR')
  parser.add_argument('--gpqa', nargs='*', default=[], metavar='DIR')
  parser.add_argument('--out', default=str(ROOT / 'google'))
  args = parser.parse_args()
  platforms = read(ROOT / 'platforms.json')
  models = read(ROOT / 'models.json')
  platform = platforms[args.platform]
  written = []
  for name, roots, render in (('lcb', args.lcb, lcb_tables), ('gpqa', args.gpqa, gpqa_tables)):
    if not roots:
      continue
    text = render(roots, platform, models)
    if text is None:
      print(f'{name}: no complete runs under {roots}')
      continue
    out = pathlib.Path(args.out) / name / f'{args.platform}.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, newline='\n')
    written.append(str(out))
  print('wrote ' + ', '.join(written) if written else 'nothing written')


if __name__ == '__main__':
  main()
