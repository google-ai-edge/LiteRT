#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""Grades model responses to the selected LiveCodeBench problems.

For every response file RUN_DIR/<question_id>.out the code is extracted the
way LiveCodeBench does (the text between the last two ``` fences), prefixed
with LiveCodeBench's import header (testing_util.py's import_string: the
LeetCode starter code assumes a platform where typing's List, collections,
math and the rest are imported, and a program may rely on them too) and run
against the problem's tests (problems/<id>.json from select.py):
  stdin problems      the program reads the test input on stdin; its stdout
                      must equal the expected output line by line (trailing
                      whitespace ignored)
  functional problems Solution().<func_name>(*args) must return the expected
                      value (JSON-compared, tuples as lists)
Each test runs in its own python3 -I subprocess with a wall-clock timeout
(LiveCodeBench's 6 s per test) and an address-space limit, in an empty
scratch directory; a problem's tests stop at the first failure. A problem
passes when every test passes (pass@1); "passed" counts the tests before
the first failure.

usage: grade.tool RUN_DIR... [--timeout S] [--json FILE [--merge]]
Writes each run's grades to RUN_DIR/grades.json (what report/eval.tool reads); --json also
collects them into one file (--merge keeps the runs already in it). Prints one line per
problem and run, then the pass rate per run.
"""
import argparse
import json
import os
import pathlib
import resource
import signal
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent

# LiveCodeBench's import_string (lcb_runner/evaluation/testing_util.py), put
# before the code of every response as LiveCodeBench does.
IMPORT_HEADER = (
    'from string import *\nfrom re import *\nfrom datetime import *\n'
    'from collections import *\nfrom heapq import *\nfrom bisect import *\n'
    'from copy import *\nfrom math import *\nfrom random import *\n'
    'from statistics import *\nfrom itertools import *\nfrom functools import *\n'
    'from operator import *\nfrom io import *\nfrom sys import *\n'
    'from json import *\nfrom builtins import *\nfrom typing import *\n'
    'import string\nimport re\nimport datetime\nimport collections\nimport heapq\n'
    'import bisect\nimport copy\nimport math\nimport random\nimport statistics\n'
    'import itertools\nimport functools\nimport operator\nimport io\nimport sys\n'
    'import json\nsys.setrecursionlimit(50000)\n')

HARNESS = r'''
import json, signal, sys
sys.setrecursionlimit(10000)
code = open(sys.argv[1]).read()
tests = json.load(open(sys.argv[2]))
func_name = sys.argv[3]
timeout = float(sys.argv[4])
class TestTimeout(Exception):
    pass
def on_alarm(signum, frame):
    raise TestTimeout()
signal.signal(signal.SIGALRM, on_alarm)
namespace = {}
exec(compile(code, "solution.py", "exec"), namespace)
solution = namespace["Solution"]()
method = getattr(solution, func_name)
def normalize(value):
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value
results = []
for test in tests:
    args = [json.loads(line) for line in test["input"].split("\n") if line.strip()]
    expected = json.loads(test["output"])
    signal.setitimer(signal.ITIMER_REAL, timeout)  # LiveCodeBench's per-test limit
    try:
        got = method(*args)
        results.append(normalize(got) == normalize(expected))
    except BaseException:
        results.append(False)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if not results[-1]:
        break  # a problem passes only when every test passes
print(json.dumps(results))
'''


def extract_code(text):
  lines = text.split('\n')
  fences = [i for i, line in enumerate(lines) if '```' in line]
  if len(fences) < 2:
    return ''
  return '\n'.join(lines[fences[-2] + 1:fences[-1]])


def limits():
  resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
  # RLIMIT_NPROC counts every thread of the real UID, including unrelated host
  # applications. It cannot enforce a per-solution budget and rejects valid
  # threaded solutions on busy hosts. This runner is not a process sandbox.


def run_solution(command, timeout, workdir, input_text=None):
  # Own a fresh process group, so ordinary descendants cannot outlive grading.
  # This bounds their lifetime; it is not isolation from hostile code.
  with subprocess.Popen(command, stdin=subprocess.PIPE if input_text is not None
                        else subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True, cwd=workdir,
                        preexec_fn=limits, start_new_session=True) as proc:
    try:
      stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
      return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    finally:
      try:
        os.killpg(proc.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass


def run_stdin(code, tests, timeout, workdir):
  solution = workdir / 'solution.py'
  solution.write_text(code)
  results = []
  for test in tests:
    try:
      proc = run_solution([sys.executable, '-I', str(solution)], timeout,
                          workdir, input_text=test['input'])
      got = [line.rstrip() for line in proc.stdout.rstrip('\n').split('\n')]
      want = [line.rstrip() for line in test['output'].rstrip('\n').split('\n')]
      results.append(proc.returncode == 0 and got == want)
    except subprocess.TimeoutExpired:
      results.append(False)
    if not results[-1]:
      break  # a problem passes only when every test passes
  return results


def run_functional(code, tests, func_name, timeout, workdir):
  solution = workdir / 'solution.py'
  solution.write_text(code)
  tests_path = workdir / 'tests.json'
  tests_path.write_text(json.dumps(tests))
  harness = workdir / 'harness.py'
  harness.write_text(HARNESS)
  try:
    proc = run_solution([sys.executable, '-I', str(harness), str(solution),
                         str(tests_path), func_name, str(timeout)],
                        timeout * len(tests) + 30, workdir)
    if proc.returncode != 0:
      return [False] * len(tests)
    return json.loads(proc.stdout.strip().split('\n')[-1])
  except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
    return [False] * len(tests)


def grade_response(problem, text, timeout):
  code = extract_code(text)
  if not code.strip():
    return {'status': 'no code', 'passed': 0, 'total': len(problem['tests'])}
  code = IMPORT_HEADER + code
  with tempfile.TemporaryDirectory(prefix='lcb-') as tmp:
    workdir = pathlib.Path(tmp)
    if problem['starter_code']:
      results = run_functional(code, problem['tests'], problem['func_name'],
                               timeout, workdir)
    else:
      results = run_stdin(code, problem['tests'], timeout, workdir)
  passed = sum(1 for r in results if r)
  total = len(problem['tests'])
  return {'status': 'pass' if passed == total else 'fail',
          'passed': passed, 'total': total}


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('runs', nargs='+')
  parser.add_argument('--timeout', type=float, default=6.0)
  parser.add_argument('--json')
  parser.add_argument('--merge', action='store_true',
                      help='retain other runs already saved in --json')
  args = parser.parse_args()
  if args.merge and not args.json:
    parser.error('--merge requires --json')
  if args.json and pathlib.Path(args.json).exists() and not args.merge:
    parser.error('existing results require --merge; use a new file for a separate report')
  selection = json.load(open(HERE / 'selection.json'))
  report = (json.loads(pathlib.Path(args.json).read_text())
            if args.merge and pathlib.Path(args.json).exists() else {})
  for run in args.runs:
    run_dir = pathlib.Path(run)
    rows = []
    for item in selection:
      qid = item['question_id']
      problem = json.load(open(HERE / 'problems' / f'{qid}.json'))
      out = run_dir / f'{qid}.out'
      if not out.exists():
        rows.append({'question_id': qid, 'status': 'missing', 'passed': 0,
                     'total': len(problem['tests'])})
        continue
      text = out.read_text(errors='replace')
      result = grade_response(problem, text, args.timeout)
      meta_path = run_dir / f'{qid}.json'
      meta = json.load(open(meta_path)) if meta_path.exists() else {}
      row = {'question_id': qid, 'platform': item['platform'],
             'difficulty': item['difficulty'], **result,
             'response_chars': len(text), **meta}
      rows.append(row)
      print(f"{run_dir.name:24} {qid:10} {item['platform']:9} {item['difficulty']:6} "
            f"{row['status']:8} {row['passed']:3d}/{row['total']:<3d} "
            f"{meta.get('decode_tokens', '?'):>5} tok "
            f"{meta.get('decode_tokens_per_second', 0) or 0:6.1f} tok/s "
            f"{(meta.get('acceptance') or 0) * 100:3.0f}%")
    passed = sum(1 for r in rows if r['status'] == 'pass')
    print(f"{run_dir.name}: {passed}/{len(rows)} problems pass")
    report[run_dir.name] = {'rows': rows, 'pass': passed, 'total': len(rows)}
    config = run_dir / 'run.json'
    if config.exists():
      report[run_dir.name]['config'] = json.loads(config.read_text())
    (run_dir / 'grades.json').write_text(json.dumps(report[run_dir.name], indent=1) + '\n')
  if args.json:
    pathlib.Path(args.json).write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
  main()
