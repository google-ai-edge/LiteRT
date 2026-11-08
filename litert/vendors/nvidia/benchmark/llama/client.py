#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""One completion against a running llama-server, with the server's timings.

usage: client.py --prompt-file FILE --out PREFIX [--n-predict N] [--port P]
                 [--sampling greedy|model] [--seed S] [--raw] [--ignore-eos]
                 [--timeout S]

Posts the prompt to /completion and writes PREFIX.txt (the generated text) and
PREFIX.json (the server's timings: prompt_n / prompt_ms, predicted_n /
predicted_ms and, with speculative decoding, draft_n / draft_n_accepted; the
stop reason; the request's sampling). Prints one summary line.

The prompt is wrapped in Gemma 4's user turn the way LiteRT-LM's conversation
renders a single user message (no system turn, no thinking; the server adds the
BOS token); --raw sends the file's text as it is.

--sampling greedy: temperature 0 (the top token every step), what LiteRT-LM
does for the E2B and E4B files on this route; model: top-k 64, top-p 0.95,
temperature 1.0 with --seed, the sampler the 12B and 26B-A4B files ask for.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main():
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument('--prompt-file', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--n-predict', type=int, default=256)
  parser.add_argument('--port', type=int, default=8089)
  parser.add_argument(
      '--sampling', choices=('greedy', 'model'), default='greedy'
  )
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--raw', action='store_true')
  parser.add_argument('--ignore-eos', action='store_true')
  parser.add_argument('--timeout', type=float, default=7200)
  args = parser.parse_args()

  prompt = open(args.prompt_file, errors='replace').read()
  if not args.raw:
    prompt = '<|turn>user\n' + prompt.strip() + '<turn|>\n<|turn>model\n'
  body = {
      'prompt': prompt,
      'n_predict': args.n_predict,
      'cache_prompt': False,
      'stream': False,
      'seed': args.seed,
  }
  if args.sampling == 'greedy':
    body.update(temperature=0.0, top_k=1)
  else:
    body.update(temperature=1.0, top_k=64, top_p=0.95, min_p=0.0)
  if args.ignore_eos:
    body['ignore_eos'] = True
  request = urllib.request.Request(
      f'http://127.0.0.1:{args.port}/completion',
      data=json.dumps(body).encode(),
      headers={'Content-Type': 'application/json'},
  )
  start = time.monotonic()
  try:
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
      result = json.load(response)
  except (urllib.error.URLError, TimeoutError) as e:
    json.dump(
        {
            'error': str(e),
            'wall_s': time.monotonic() - start,
            'request': {k: v for k, v in body.items() if k != 'prompt'},
        },
        open(args.out + '.json', 'w'),
        indent=1,
    )
    print(f'{args.out}: request failed: {e}', file=sys.stderr)
    return 1
  wall = time.monotonic() - start
  timings = result.get('timings', {})
  open(args.out + '.txt', 'w').write(result.get('content', ''))
  json.dump(
      {
          'wall_s': wall,
          'timings': timings,
          'tokens_predicted': result.get('tokens_predicted'),
          'tokens_evaluated': result.get('tokens_evaluated'),
          'stop_type': result.get('stop_type'),
          'request': {k: v for k, v in body.items() if k != 'prompt'},
      },
      open(args.out + '.json', 'w'),
      indent=1,
  )
  draft = ''
  if timings.get('draft_n'):
    rate = timings.get('draft_n_accepted', 0) / timings['draft_n']
    draft = f" draft {timings['draft_n']} accepted {rate:.0%}"
  print(
      f"{args.out}: prompt {timings.get('prompt_n')} tok at"
      f" {timings.get('prompt_per_second', 0):.0f} tok/s, generated"
      f" {timings.get('predicted_n')} tok at"
      f" {timings.get('predicted_per_second', 0):.1f} tok/s"
      f" ({result.get('stop_type')}){draft}"
  )
  return 0


if __name__ == '__main__':
  sys.exit(main())
