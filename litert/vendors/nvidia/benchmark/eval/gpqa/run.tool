#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""GPQA Diamond through LiteRT-LM on TensorRT-RTX, by Artificial Analysis's published method:
its four-option prompt, its answer-extraction regexes tried in order with the last match
taken, zero-shot pass@1, temperature 0.6 (a reasoning model), the model's output limit.

usage: run.tool --model PROFILE|FILE --label LABEL [--count N | --question ID|INDEX ...]
                [--max-output-tokens 32768] [--runs DIR] [--rescore] [--one-process]
                [--env NAME=VALUE ...] [--selected-signatures NAME,NAME]

The data: GPQA's official release (fetch.sh: data/gpqa_diamond.csv). Each question is one CLI
run through the platform's bench runner (bench.sh on Linux, windows/bench.tool on Windows),
sampled with top-k 20, top-p 0.95, temperature 0.6 and a fixed seed per question, on the
pool of the measured runs. The questions stay out of printed output, as the authors ask.
A fixed sample of --count questions (seeded), or the questions named; --question takes a
Record ID or a 1-based index into the CSV. Results: RUNS/LABEL/results.json (per question:
the letters, tokens, rates, wall time) and RUNS/LABEL/<n>-<record>/ (the CLI run).

--one-process (or EVAL_ONE_PROCESS=1; Linux) runs the pending questions in one process
(../one_process.sh, ../engine_runner.tool): the model loads once and each question gets a fresh
conversation configured as the CLI's, its cli.log and time.txt written as the CLI's run would
have them; LITERT_LM_PYTHON names the directory of LiteRT-LM's Python package. --env sets a
runtime variable (a partition policy, say) in either mode; --selected-signatures the main
model's signatures to load (default: bench.sh's rule, prefill_1024 and decode for 12b and 26b).
"""
import argparse
import csv
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = HERE / 'data' / 'gpqa_diamond.csv'

PROMPT = ("Answer the following multiple choice question. The last line of your response should be in the "
          "following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n{question}\n\n"
          "A) {A}\nB) {B}\nC) {C}\nD) {D}")
PATTERNS = [r'(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])',
            r'\\boxed\{[^}]*([A-Z])[^}]*\}', r'answer is ([a-zA-Z])', r'answer is \\\(([a-zA-Z])',
            r'([A-Z])\)\s*[^A-Z]*', r'([A-Z])\s+is\s+the\s+correct\s+answer', r'([A-Z])\s*$',
            r'([A-Z])\s*\.', r'([A-Z])\s*[^\w]']
LOG_LINE = re.compile(r'^(INFO: \[|[IWEF]\d{4} \d\d:\d\d:\d\d|[IWEF]0000 )')
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def extract(response):
  text = response.strip()
  if len(text) == 1 and text.isalpha():
    return text.upper()
  for pattern in PATTERNS:
    matches = re.findall(pattern, text)
    if matches:
      return matches[-1].upper()
  return None


def model_output(log):
  """(reasoning, final answer) from the CLI's log: the message printed just before the
  'Captured model output:' line (which absl truncates), its reasoning inside [thought]."""
  lines = [ANSI.sub('', line) for line in log.splitlines()]
  end = next((i for i, line in enumerate(lines) if 'Captured model output: ' in line), None)
  if end is None:
    return None, None
  start = end
  while start > 0 and not LOG_LINE.match(lines[start - 1]):
    start -= 1
  block = lines[start:end]
  closes = [i for i, line in enumerate(block) if line.startswith('[/thought]')]
  if block and block[0].startswith('[thought]') and not closes:
    return '\n'.join(block)[len('[thought]'):].strip(), None
  if not closes:
    return '', '\n'.join(block)
  reasoning = '\n'.join(block[:closes[-1]])
  reasoning = reasoning[len('[thought]'):] if reasoning.startswith('[thought]') else reasoning
  final = '\n'.join([block[closes[-1]][len('[/thought]'):]] + block[closes[-1] + 1:])
  return reasoning.strip(), final.strip()


def score(entry, log):
  reasoning, final = model_output(log)
  entry['chosen'] = extract(final) if final else None
  entry['is_correct'] = entry['chosen'] == entry['correct']
  entry['has_final_answer'] = final is not None
  entry['reasoning_chars'] = len(reasoning or '')
  entry['response_chars'] = len(final or '')
  decode = re.findall(r'Decode Speed: ([\d.]+) tokens/sec', log)
  prefill = re.findall(r'Prefill Speed: ([\d.]+) tokens/sec', log)
  decoded = [int(t) for t in re.findall(r'Decode Turn \d+: Processed (\d+) tokens', log)]
  prompt_tokens = re.findall(r'Prefill Turn 1: Processed (\d+) tokens', log)
  failure = re.search(r'Failed to run litert_lm: (.*)', log)
  entry.update(prompt_tokens=int(prompt_tokens[0]) if prompt_tokens else None,
               output_tokens=sum(decoded) if decoded else None,
               decode_tokens_per_second=float(decode[0]) if decode else None,
               prefill_tokens_per_second=float(prefill[0]) if prefill else None,
               error=failure.group(1).strip() if failure else entry.get('error'),
               nvidia=bool(re.search(r'NVIDIA dispatch context \S+ (?:uses|shares) ', log)))
  return entry


# The method's sampler (the CLI's --sampler_* flags); the seed is the question's number.
SAMPLER = {'top_k': 20, 'top_p': 0.95, 'temperature': 0.6}


def bench_command(args, label, prompt_file, out, context, n):
  """The platform's bench runner, one iteration on the prompt file, benchmark mode on real
  text (actual token counts), the sampler of the method, the output limit."""
  cli = [f'--cli-arg=--sampler_top_k={SAMPLER["top_k"]}', f'--cli-arg=--sampler_top_p={SAMPLER["top_p"]}',
         f'--cli-arg=--sampler_temperature={SAMPLER["temperature"]}', f'--cli-arg=--sampler_seed={n}',
         f'--cli-arg=--max_output_tokens={args.max_output_tokens}']
  if context > 32771:
    cli.append('--cli-arg=--allow_context_length_expansion=true')
  if args.selected_signatures:
    cli.append(f'--cli-arg=--selected_signatures={args.selected_signatures}')
  cli += [f'--env={assignment}' for assignment in args.env]
  common = ['--model', args.model, '--out', str(out), '--label', label, '--workload', 'short', '--metric', 'latency',
            '--prompt-file', str(prompt_file), '--iterations', '1', '--context', str(context), *cli]
  if os.name == 'nt':
    return [sys.executable, str(ROOT / 'windows' / 'bench.tool'), *common]
  return ['bash', str(ROOT / 'bench.sh'), *common]


def run_one_process(args, pending, out, context):
  """The pending questions, (n, label, prompt file, ...), in one engine_runner.tool process
  with bench.sh's settings for them: each question's cli.log as `> cli.log 2>&1` would have it
  and time.txt as GNU time's. Returns {label: (exit status, wall seconds)}."""
  if context > 32771:
    raise SystemExit('--one-process cannot expand the context (LiteRT-LM\'s C API has no setting for it); '
                     'lower --max-output-tokens or run the CLI per question')
  selected = (args.selected_signatures.split(',') if args.selected_signatures
              else ['prefill_1024', 'decode'] if args.model in ('12b', '26b') else None)
  jobs = []
  for n, label, prompt_file, *_ in pending:
    (out / label).mkdir(exist_ok=True)
    log = str(out / label / 'cli.log')
    jobs.append({'name': label, 'prompt_file': str(prompt_file), 'max_output_tokens': args.max_output_tokens,
                 'sampler': dict(SAMPLER, seed=n), 'stdout': log, 'stderr': log, 'time': str(out / label / 'time.txt')})
  spec = {'engine': {'max_num_tokens': context, 'selected_signatures': selected, 'speculative_decoding': False,
                     'min_log_severity': 0},
          'jobs': jobs, 'summary': str(out / 'one_process.json')}
  (out / 'jobs.json').write_text(json.dumps(spec, indent=1) + '\n')
  command = ['bash', str(HERE.parent / 'one_process.sh'), args.model, str(out / 'jobs.json')]
  for assignment in args.env:
    command += ['--env', assignment]
  subprocess.run(command, stdin=subprocess.DEVNULL)
  timings = {}
  for _, label, *_ in pending:
    text = (out / label / 'time.txt').read_text() if (out / label / 'time.txt').exists() else ''
    status = re.search(r'Exit status: (\d+)', text)
    wall = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)', text)
    seconds = sum(float(p) * 60 ** i for i, p in enumerate(reversed(wall[1].split(':')))) if wall else None
    timings[label] = (int(status[1]) if status else 1, seconds)
  return timings


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--model', required=True)
  parser.add_argument('--label', required=True)
  parser.add_argument('--count', type=int, default=10)
  parser.add_argument('--question', action='append', default=[], help='a Record ID or 1-based CSV index (repeatable)')
  parser.add_argument('--max-output-tokens', type=int, default=32768)
  parser.add_argument('--runs', default=os.environ.get('EVAL_RUNS') or str(pathlib.Path(
      os.environ.get('LITERT_BENCH_ROOT', pathlib.Path.home() / '.local/state/litert-nvidia-benchmark-wl')) / 'eval' / 'gpqa' / 'runs'))
  parser.add_argument('--rescore', action='store_true', help='re-read the label\'s logs; run nothing')
  parser.add_argument('--one-process', action='store_true', default=os.environ.get('EVAL_ONE_PROCESS') == '1',
                      help='run the pending questions in one process: the model loads once (Linux)')
  parser.add_argument('--env', action='append', default=[], metavar='NAME=VALUE',
                      help='a runtime environment variable (bench.sh --env; repeatable)')
  parser.add_argument('--selected-signatures', metavar='NAME,NAME',
                      help='the main model signatures to load (default: bench.sh\'s rule)')
  args = parser.parse_args()
  if not DATA.exists():
    raise SystemExit(f'{DATA} is missing: run eval/gpqa/fetch.sh')
  rows = list(csv.DictReader(DATA.open(encoding='utf-8')))
  if args.question:
    ids = [r['Record ID'] for r in rows]
    picked = []
    for q in args.question:
      if q in ids:
        picked.append(ids.index(q))
      elif q.isdigit() and 1 <= int(q) <= len(rows):
        picked.append(int(q) - 1)
      else:
        raise SystemExit(f'--question {q} is neither a Record ID nor an index 1..{len(rows)}')
  else:
    picked = random.Random(0).sample(range(len(rows)), args.count)  # the fixed sample
  context = (args.max_output_tokens + 2048 + 1023) // 1024 * 1024
  out = pathlib.Path(args.runs) / args.label
  out.mkdir(parents=True, exist_ok=True)
  results_path = out / 'results.json'
  results = json.loads(results_path.read_text()) if results_path.exists() else []
  if args.one_process and os.name == 'nt':
    raise SystemExit('--one-process runs on Linux (one_process.sh); on Windows, run the CLI per question')
  config = {'model': args.model, 'max_output_tokens': args.max_output_tokens, 'context': context,
            'sampler': {'top_k': 20, 'top_p': 0.95, 'temperature': 0.6, 'seed': 'the question number'},
            'method': 'Artificial Analysis intelligence-benchmarking methodology (GPQA prompt, extraction, pass@1)',
            'data': 'idavidrein/gpqa at 56686c0, gpqa_diamond.csv', 'questions': len(picked)}
  if args.env:
    config['environment'] = args.env
  if args.selected_signatures:
    config['selected_signatures'] = args.selected_signatures
  if args.one_process:
    config['one_process'] = True
  (out / 'run.json').write_text(json.dumps(config, indent=1) + '\n')
  if args.rescore:
    for entry in results:
      log = pathlib.Path(entry['run']) / 'cli.log'
      score(entry, log.read_text(encoding='utf-8', errors='replace') if log.exists() else '')
    results_path.write_text(json.dumps(results, indent=1) + '\n')
    print(json.dumps(summary(results)))
    return
  done = {r['record_id'] for r in results if r.get('exit_code') == 0}
  pending = []
  for n, index in enumerate(picked, 1):
    row = rows[index]
    rid = row['Record ID']
    if rid in done:
      continue
    choices = [row['Correct Answer'], row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3']]
    order = list(range(4))
    random.Random(rid).shuffle(order)  # a fixed shuffle of the choices per question
    letters = 'ABCD'
    correct = letters[order.index(0)]
    prompt_dir = out / 'prompts'
    prompt_dir.mkdir(exist_ok=True)
    prompt_file = prompt_dir / f'{rid}.txt'
    prompt_file.write_text(PROMPT.format(question=row['Question'].strip(),
                                         **{letters[i]: choices[order[i]].strip() for i in range(4)}),
                           encoding='utf-8', newline='\n')
    label = f'{n:03d}-{rid}'
    if args.one_process:
      pending.append((n, label, prompt_file, row, correct))
      continue
    started = time.time()
    result = subprocess.run(bench_command(args, label, prompt_file, out, context, n), stdin=subprocess.DEVNULL)
    results = record(args, results, results_path, out, n, row, correct, label, result.returncode,
                     round(time.time() - started, 1))
  if pending:
    timings = run_one_process(args, pending, out, context)
    for n, label, _, row, correct in pending:
      exit_code, wall = timings[label]
      results = record(args, results, results_path, out, n, row, correct, label, exit_code,
                       round(wall, 1) if wall is not None else None)
  print(json.dumps(summary(results)))


def record(args, results, results_path, out, n, row, correct, label, exit_code, wall):
  """Scores question `n` from its run's log into the results, and saves them."""
  rid = row['Record ID']
  log_path = out / label / 'cli.log'
  log = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else ''
  entry = {'n': n, 'record_id': rid, 'domain': row.get('High-level domain'), 'subdomain': row.get('Subdomain'),
           'correct': correct, 'exit_code': exit_code, 'wall_s': wall, 'run': str(out / label)}
  score(entry, log)
  entry['hit_output_limit'] = bool(entry.get('output_tokens')) and entry['output_tokens'] >= args.max_output_tokens
  results = [r for r in results if r['record_id'] != rid] + [entry]
  results_path.write_text(json.dumps(results, indent=1) + '\n')
  print(json.dumps({k: entry[k] for k in ('n', 'domain', 'correct', 'chosen', 'is_correct', 'exit_code',
                                           'output_tokens', 'decode_tokens_per_second', 'wall_s')}), flush=True)
  return results


def summary(results):
  """Artificial Analysis's pass@1 over every question: a failed or unanswered one is wrong."""
  right = sum(bool(r.get('is_correct')) for r in results)
  return {'questions': len(results), 'correct': right, 'accuracy': right / len(results) if results else None,
          'failed_runs': sum(r.get('exit_code') != 0 for r in results),
          'unextracted': sum(r.get('chosen') is None for r in results),
          'hit_output_limit': sum(bool(r.get('hit_output_limit')) for r in results)}


if __name__ == '__main__':
  main()
