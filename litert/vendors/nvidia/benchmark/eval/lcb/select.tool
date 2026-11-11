#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""The LiveCodeBench code-generation problems of the evaluation.

Reads the release file (data/test6.jsonl = the problems LiveCodeBench added in
release v6, January-April 2025, fetched with fetch.sh) and writes
  selection.json  the problems' ids, platform, difficulty, test type, and
                  whether the problem belongs to the 18-problem smoke sample
  prompts/<id>.txt the generation prompt of each, in LiveCodeBench's generic
                   format (question, format instructions, "### Answer")
  problems/<id>.json the problem with its public and private tests decoded

--all takes every problem of the release (175). Otherwise the selection is a
stratified sample: --per-cell problems from each platform x difficulty cell,
in a fixed seeded order, so reruns pick the same problems; the 3-per-cell
sample with the default seed is the smoke sample either way.

usage: select.py [--all] [--per-cell N] [--seed S] [--data FILE]
"""
import argparse
import base64
import io
import json
import pathlib
import pickle
import random
import zlib

HERE = pathlib.Path(__file__).resolve().parent

SYSTEM = ('You are an expert Python programmer. You will be given a question '
          '(problem specification) and will generate a correct Python program '
          'that matches the specification and passes all tests.')
WITH_STARTER = ('You will use the following starter code to write the solution '
                'to the problem and enclose your code within delimiters.')
WITHOUT_STARTER = ('Read the inputs from stdin solve the problem and write the '
                   'answer to stdout (do not directly test on the sample '
                   'inputs). Enclose your code within delimiters as follows. '
                   'Ensure that when the python program runs, it reads the '
                   'inputs, runs the algorithm and writes output to STDOUT.')


class SafeUnpickler(pickle.Unpickler):
  """The private tests are a pickled str; nothing else may be loaded."""

  def find_class(self, module, name):
    raise pickle.UnpicklingError(f'refusing to load {module}.{name}')


def decode_private_tests(blob):
  raw = zlib.decompress(base64.b64decode(blob))
  text = SafeUnpickler(io.BytesIO(raw)).load()
  if not isinstance(text, str):
    raise ValueError('private tests are not a string')
  return json.loads(text)


def newlines(text):
  """The dataset's texts mix CRLF and LF line endings (70 of the 175 prompts);
  a runtime that reads the prompt file raw would see stray \r tokens."""
  return text.replace('\r\n', '\n').replace('\r', '\n')


def prompt_of(problem):
  """LiveCodeBench's generic prompt (lcb_runner/prompts/code_generation.py)."""
  text = SYSTEM + '\n\n'
  text += f"### Question:\n{newlines(problem['question_content'])}\n\n"
  if problem['starter_code']:
    text += f'### Format: {WITH_STARTER}\n'
    text += f"```python\n{newlines(problem['starter_code'])}\n```\n\n"
  else:
    text += f'### Format: {WITHOUT_STARTER}\n'
    text += '```python\n# YOUR CODE HERE\n```\n\n'
  text += '### Answer: (use the provided format with backticks)\n\n'
  return text


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--all', action='store_true')
  parser.add_argument('--per-cell', type=int, default=3)
  parser.add_argument('--seed', type=int, default=20261013)
  parser.add_argument('--data', default=str(HERE / 'data' / 'test6.jsonl'))
  parser.add_argument('--resolve', metavar='QUESTION',
                      help='print the question id of QUESTION (an id, or a 1-based index into selection.json) and exit')
  args = parser.parse_args()
  if args.resolve:
    ids = [s['question_id'] for s in json.loads((HERE / 'selection.json').read_text())]
    if args.resolve in ids:
      print(args.resolve)
    elif args.resolve.isdigit() and 1 <= int(args.resolve) <= len(ids):
      print(ids[int(args.resolve) - 1])
    else:
      raise SystemExit(f'{args.resolve} is neither a question id nor an index 1..{len(ids)}')
    return

  rows = [json.loads(line) for line in open(args.data)]
  cells = {}
  for row in rows:
    cells.setdefault((row['platform'], row['difficulty']), []).append(row)
  rng = random.Random(args.seed)
  selected, smoke = [], []
  for key in sorted(cells):
    pool = sorted(cells[key], key=lambda r: r['question_id'])
    rng.shuffle(pool)
    smoke.extend(r['question_id'] for r in pool[:3])
    selected.extend(pool if args.all else pool[:args.per_cell])
  smoke = set(smoke)

  (HERE / 'prompts').mkdir(exist_ok=True)
  (HERE / 'problems').mkdir(exist_ok=True)
  index = []
  for problem in selected:
    qid = problem['question_id']
    metadata = json.loads(problem['metadata']) if problem['metadata'] else {}
    tests = json.loads(problem['public_test_cases'])
    tests += decode_private_tests(problem['private_test_cases'])
    record = {
        'question_id': qid, 'title': problem['question_title'],
        'platform': problem['platform'], 'difficulty': problem['difficulty'],
        'contest_date': problem['contest_date'],
        'starter_code': problem['starter_code'],
        'func_name': metadata.get('func_name'),
        'tests': tests,
    }
    (HERE / 'problems' / f'{qid}.json').write_text(json.dumps(record))
    (HERE / 'prompts' / f'{qid}.txt').write_text(prompt_of(problem))
    index.append({k: record[k] for k in ('question_id', 'title', 'platform',
                                         'difficulty', 'contest_date')}
                 | {'tests': len(tests),
                    'test_type': 'functional' if problem['starter_code'] else 'stdin',
                    'smoke': qid in smoke})
  (HERE / 'selection.json').write_text(json.dumps(index, indent=1) + '\n')
  for item in index:
    print(f"{item['question_id']:28} {item['platform']:9} {item['difficulty']:6} "
          f"{item['test_type']:10} {item['tests']:4d} tests  {item['title']}")


if __name__ == '__main__':
  main()
