#!/usr/bin/env python3
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
"""Runs evaluation prompts through one LiteRT-LM engine: the model loads once.

The evaluations used to start the LiteRT-LM CLI once per problem, and every problem paid the
process start and the model load (about 12 s on Gemma 4 26B-A4B). This runner loads the model
once through LiteRT-LM's C API (its Python package, litert_lm) and gives every prompt a fresh
conversation, configured the way the CLI configures its single conversation in benchmark mode:
the same engine settings (backend, dispatch libraries, cache directory, context capacity,
signatures, speculative decoding, serial execution), the same user message (one text part), the
same sampler parameters and output limit. Each prompt's files are what the CLI would have
written: its stdout (the response, printed as the CLI prints a streamed response), its log (the
engine's native log while the prompt ran, the captured output and the benchmark report in the
CLI's format) and a GNU-time-style accounting of the prompt's wall time and exit status.

usage: engine_runner.tool JOBS.json
JOBS.json: {"engine": {"model": FILE, "dispatch_lib_dir": DIR, "cache_dir": DIR,
                       "max_num_tokens": N, "selected_signatures": [NAME, ...] | null,
                       "speculative_decoding": BOOL, "min_log_severity": N},
            "jobs": [{"name": ID, "prompt_file": FILE, "max_output_tokens": N,
                      "sampler": {"top_k": K, "top_p": P, "temperature": T, "seed": S} | null,
                      "stdout": FILE, "stderr": FILE, "time": FILE}, ...],
            "summary": FILE}
A job whose stdout and stderr name the same file gets both there, as `> cli.log 2>&1` does.
The sampler entry follows the CLI's flags: null keeps the model's own sampler; otherwise
missing fields take the CLI's defaults (top-k 40, top-p 1, temperature 1, no seed).
The summary records the engine's load time, every job's wall time and exit status.

Environment: PYTHONPATH (or LITERT_LM_PYTHON) names the directory holding the litert_lm
package; the dispatch's runtime environment (env.sh's nb_runtime_env) as for the CLI.
"""
import ctypes
import json
import os
import queue
import sys
import threading
import time

if os.environ.get('LITERT_LM_PYTHON'):
  sys.path.insert(0, os.environ['LITERT_LM_PYTHON'])
from litert_lm import _ffi  # pylint: disable=g-import-not-at-top

# LiteRtLmSamplerType's TOP_P: the CLI's --sampler_* flags ask for it.
TOP_P = 2
SPECULATIVE_ON = 1


class StreamError(RuntimeError):
  """An error the turn's stream reported; the CLI prints its message as the response."""


def log_line(text):
  """An absl-style INFO line, so the evaluations' log parsers read it as a log line."""
  now = time.time()
  return f'I0000 00:00:{now:.6f} {os.getpid():7d} engine_runner.tool] {text}\n'


def format_duration(seconds):
  """absl::Duration's text form for the benchmark report (e.g. 169.13ms, 2m45.28s)."""
  if seconds < 1:
    return f'{seconds * 1000:.6f}ms'
  minutes, rest = divmod(seconds, 60)
  if minutes >= 60:
    hours, minutes = divmod(minutes, 60)
    return f'{int(hours)}h{int(minutes)}m{rest:.9f}s'
  return f'{int(minutes)}m{rest:.9f}s' if minutes else f'{rest:.9f}s'


def format_wall(seconds):
  """GNU time's elapsed wall clock form (h:mm:ss or m:ss)."""
  minutes, rest = divmod(seconds, 60)
  hours, minutes = divmod(int(minutes), 60)
  return f'{hours}:{minutes:02d}:{rest:05.2f}' if hours else f'{minutes}:{rest:05.2f}'


class Printer:
  """The CLI's streamed printing of a response (litert_lm_lib.cc, CreatePrintMessageCallback
  and PrintMessage with streaming): a channel's text after `[name] `, `[/name]` and a newline
  when the channel ends, the content's text as it comes, a newline at the end of the turn.
  Not a terminal, so no colors."""

  def __init__(self, write):
    self.write = write
    self.active = ''
    self.captured = []

  def message(self, msg):
    out = []
    printed = False
    channels = msg.get('channels')
    if isinstance(channels, dict):
      for name, text in channels.items():
        if isinstance(text, str):
          if self.active != name:
            if self.active:
              out.append(f'[/{self.active}]\n')
            out.append(f'[{name}] ')
            self.active = name
          out.append(text)
          self.captured.append(text)
          printed = True
    if 'content' in msg:
      if self.active:
        out.append(f'[/{self.active}]\n')
        self.active = ''
      content = msg['content']
      texts = []
      if isinstance(content, list):
        texts = [c['text'] for c in content
                 if isinstance(c, dict) and c.get('type') == 'text' and 'text' in c]
      elif isinstance(content, dict) and isinstance(content.get('text'), str):
        texts = [content['text']]
      elif isinstance(content, str):
        texts = [content]
      for text in texts:
        out.append(text)
        self.captured.append(text)
        printed = True
    if printed:
      self.write(''.join(out))
    elif not ('tool_calls' in msg or msg.get('type') == 'function'):
      raise RuntimeError(f'Invalid message: {json.dumps(msg)}')

  def end_of_turn(self):
    self.write(f'[/{self.active}]\n' if self.active else '\n')


class Engine:
  """One LiteRT-LM engine through the C API, configured like the CLI's benchmark mode."""

  def __init__(self, spec):
    lib = _ffi._get_lib()  # pylint: disable=protected-access
    self.lib = lib
    # Not declared by the package: serial execution (the CLI's benchmark mode), and the wait
    # that runs a serial engine's tasks.
    lib.litert_lm_engine_settings_set_single_threaded_execution.restype = ctypes.c_int
    lib.litert_lm_engine_settings_set_single_threaded_execution.argtypes = [
        ctypes.c_void_p, ctypes.c_bool]
    lib.litert_lm_conversation_wait_until_done.restype = ctypes.c_int
    lib.litert_lm_conversation_wait_until_done.argtypes = [ctypes.c_void_p]
    if spec.get('min_log_severity') is not None:
      lib.litert_lm_set_min_log_level(int(spec['min_log_severity']))
    settings = _ffi.create_checked(lib, 'litert_lm_engine_settings_create', spec['model'], 'npu',
                                   None, None)
    try:
      call = lambda name, *args: _ffi.call_checked(lib, name, settings, *args)
      call('litert_lm_engine_settings_enable_benchmark')
      call('litert_lm_engine_settings_set_single_threaded_execution', True)
      call('litert_lm_engine_settings_set_litert_dispatch_lib_dir', spec['dispatch_lib_dir'])
      call('litert_lm_engine_settings_set_cache_dir', spec['cache_dir'])
      call('litert_lm_engine_settings_set_max_num_tokens', int(spec['max_num_tokens']))
      signatures = spec.get('selected_signatures')
      if signatures:
        array = (ctypes.c_char_p * len(signatures))(*(s.encode() for s in signatures))
        call('litert_lm_engine_settings_set_selected_signatures', array, len(signatures))
      if spec.get('speculative_decoding'):
        # The CLI's --enable_speculative_decoding=true: mode on. Off leaves the mode unset,
        # as the CLI does.
        config = _ffi.create_checked(lib, 'litert_lm_speculative_decoding_config_create')
        try:
          _ffi.call_checked(lib, 'litert_lm_speculative_decoding_config_set_mode', config,
                            SPECULATIVE_ON)
          call('litert_lm_engine_settings_set_speculative_decoding_config', config)
        finally:
          lib.litert_lm_speculative_decoding_config_delete(config)
      started = time.monotonic()
      self.ptr = _ffi.create_checked(lib, 'litert_lm_engine_create', settings)
      self.load_s = time.monotonic() - started
    finally:
      lib.litert_lm_engine_settings_delete(settings)

  def close(self):
    if self.ptr:
      self.lib.litert_lm_engine_delete(self.ptr)
      self.ptr = None

  def conversation(self, sampler):
    """A conversation with the CLI's session config (litert_lm_lib.cc, CreateSessionConfig)."""
    lib = self.lib
    session = _ffi.create_checked(lib, 'litert_lm_session_config_create')
    try:
      if sampler is not None:
        params = _ffi.create_checked(lib, 'litert_lm_sampler_params_create', TOP_P)
        try:
          _ffi.call_checked(lib, 'litert_lm_sampler_params_set_top_k', params,
                            int(sampler.get('top_k', 40)))
          _ffi.call_checked(lib, 'litert_lm_sampler_params_set_top_p', params,
                            float(sampler.get('top_p', 1.0)))
          _ffi.call_checked(lib, 'litert_lm_sampler_params_set_temperature', params,
                            float(sampler.get('temperature', 1.0)))
          # The CLI leaves the seed alone without --sampler_seed: SessionConfig's default, 0.
          _ffi.call_checked(lib, 'litert_lm_sampler_params_set_seed', params,
                            int(sampler.get('seed', 0)))
          _ffi.call_checked(lib, 'litert_lm_session_config_set_sampler_params', session, params)
        finally:
          lib.litert_lm_sampler_params_delete(params)
      config = _ffi.create_checked(lib, 'litert_lm_conversation_config_create')
      try:
        _ffi.call_checked(lib, 'litert_lm_conversation_config_set_session_config', config,
                          session)
        return _ffi.create_checked(lib, 'litert_lm_conversation_create', self.ptr, config)
      finally:
        lib.litert_lm_conversation_config_delete(config)
    finally:
      lib.litert_lm_session_config_delete(session)

  def send(self, conversation, prompt, max_output_tokens, printer):
    """Sends the CLI's single-turn message and prints the streamed response."""
    lib = self.lib
    chunks = queue.Queue()

    def on_chunk(unused_data, chunk):
      try:
        error = _ffi.get_checked(lib, 'litert_lm_stream_chunk_get_error', ctypes.c_char_p, chunk)
        if error:
          chunks.put(StreamError(error.decode('utf-8', 'replace')))
          return
        text = _ffi.get_checked(lib, 'litert_lm_stream_chunk_get_text', ctypes.c_char_p, chunk)
        final = _ffi.get_checked(lib, 'litert_lm_stream_chunk_is_final', ctypes.c_bool, chunk)
        chunks.put((text.decode('utf-8') if text else '', bool(final)))
      except Exception as e:  # pylint: disable=broad-exception-caught
        chunks.put(e)

    callback = _ffi.STREAM_CALLBACK_TYPE(on_chunk)
    args = _ffi.create_checked(lib, 'litert_lm_conversation_optional_args_create')
    try:
      if max_output_tokens > 0:
        _ffi.call_checked(lib, 'litert_lm_conversation_optional_args_set_max_output_tokens', args,
                          int(max_output_tokens))
      message = {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}
      status = lib.litert_lm_conversation_send_message_stream(conversation, json.dumps(message),
                                                             None, args, callback, None)
      _ffi.check_status(lib, 'litert_lm_conversation_send_message_stream', status)
    finally:
      lib.litert_lm_conversation_optional_args_delete(args)
    # A serial engine runs the turn's tasks inside this wait, in this thread, as the CLI's
    # does. The wait gives up after LiteRT-LM's default 10 minutes, between two tasks, with
    # the rest still queued: waiting again resumes them, so a long answer is not cut off.
    while True:
      status = lib.litert_lm_conversation_wait_until_done(conversation)
      if status != _ffi.StatusCode.DEADLINE_EXCEEDED:
        _ffi.check_status(lib, 'litert_lm_conversation_wait_until_done', status)
        break
      os.write(2, log_line('Still generating after the 10-minute wait; waiting again').encode())
    while True:
      try:
        item = chunks.get_nowait()
      except queue.Empty:
        raise RuntimeError('The turn ended without its end-of-turn message') from None
      if isinstance(item, StreamError):
        printer.write(f'{item}\n')
        raise item
      if isinstance(item, Exception):
        raise item
      text, final = item
      if final:
        printer.end_of_turn()
        return
      if text:
        printer.message(json.loads(text))

  def benchmark_report(self, conversation):
    """The CLI's BenchmarkInfo block, from what the C API reports of it."""
    lib = self.lib
    info = _ffi.create_checked(lib, 'litert_lm_conversation_get_benchmark_info', conversation)
    try:
      get = lambda name, kind, *args: _ffi.get_checked(lib, name, kind, info, *args)
      init_s = get('litert_lm_benchmark_info_get_total_init_time_in_second', ctypes.c_double)
      ttft = get('litert_lm_benchmark_info_get_time_to_first_token', ctypes.c_double)
      turns = {}
      for kind in ('prefill', 'decode'):
        count = get(f'litert_lm_benchmark_info_get_num_{kind}_turns', ctypes.c_size_t)
        turns[kind] = [(get(f'litert_lm_benchmark_info_get_{kind}_token_count_at', ctypes.c_size_t, i),
                        get(f'litert_lm_benchmark_info_get_{kind}_tokens_per_sec_at', ctypes.c_double, i))
                       for i in range(count)]
    finally:
      lib.litert_lm_benchmark_info_delete(info)
    lines = [log_line('BenchmarkInfo:').rstrip('\n'),
             '  Init Phases (1):', f'    - Init Total: {init_s * 1000:.2f} ms',
             '--------------------------------------------------',
             f'  Time to first token: {ttft:.2f} s']
    for kind, title in (('prefill', 'Prefill'), ('decode', 'Decode')):
      lines += ['--------------------------------------------------',
                f'  {title} Turns (Total {len(turns[kind])} turns):']
      if not turns[kind]:
        lines.append(f'    No {kind} turns recorded.')
      for i, (tokens, rate) in enumerate(turns[kind], 1):
        lines += [f'    {title} Turn {i}: Processed {tokens} tokens in '
                  f'{format_duration(tokens / rate if rate else 0.0)} duration.',
                  f'      {title} Speed: {rate:.2f} tokens/sec.']
    lines.append('--------------------------------------------------')
    return '\n'.join(lines) + '\n', turns


def redirect(path, fd, opened):
  """Points `fd` at `path` (appending; one open file per path), returning the saved fd."""
  sys.stdout.flush()
  sys.stderr.flush()
  saved = os.dup(fd)
  if path not in opened:
    opened[path] = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
  os.dup2(opened[path], fd)
  return saved


def run_job(engine, job, progress):
  opened = {}
  saved_out = redirect(job['stdout'], 1, opened)
  saved_err = redirect(job['stderr'], 2, opened)
  started = time.monotonic()
  exit_code = 0
  write_out = lambda text: os.write(1, text.encode('utf-8'))
  write_err = lambda text: os.write(2, text.encode('utf-8'))
  result = {'name': job['name']}
  try:
    with open(job['prompt_file'], 'rb') as f:
      prompt = f.read().decode('utf-8')
    write_err(log_line(f'Running single-turn conversation: {job["name"]}'))
    conversation = engine.conversation(job.get('sampler'))
    try:
      printer = Printer(write_out)
      engine.send(conversation, prompt, int(job.get('max_output_tokens', -1)), printer)
      write_err(log_line('Captured model output: ' + ''.join(printer.captured)))
      report, turns = engine.benchmark_report(conversation)
      write_err(report)
      if turns['decode']:
        result['decode_tokens'] = turns['decode'][-1][0]
        result['decode_tokens_per_second'] = round(turns['decode'][-1][1], 2)
    finally:
      engine.lib.litert_lm_conversation_delete(conversation)
  except Exception as e:  # pylint: disable=broad-exception-caught
    # The CLI's failure line (litert_lm_advanced_main.cc) and a status of 13 (INTERNAL).
    write_err(f'E0000 00:00:{time.time():.6f} {os.getpid():7d} engine_runner.tool] '
              f'Failed to run litert_lm: {e}\n')
    exit_code = 13
  wall = time.monotonic() - started
  sys.stdout.flush()
  os.dup2(saved_out, 1)
  os.dup2(saved_err, 2)
  os.close(saved_out)
  os.close(saved_err)
  for fd in opened.values():
    os.close(fd)
  with open(job['time'], 'w') as f:
    f.write(f'\tCommand being timed: "engine_runner.tool {job["name"]}"\n'
            f'\tElapsed (wall clock) time (h:mm:ss or m:ss): {format_wall(wall)}\n'
            f'\tExit status: {exit_code}\n')
  result.update(exit_code=exit_code, wall_s=round(wall, 3))
  progress(result)
  return result


def main():
  if len(sys.argv) != 2:
    raise SystemExit(__doc__)
  spec = json.loads(open(sys.argv[1]).read())
  summary = {'engine': spec['engine'], 'jobs': []}
  started = time.monotonic()
  report = lambda result: print(json.dumps(result), flush=True)
  engine = Engine(spec['engine'])
  summary['load_s'] = round(engine.load_s, 3)
  report({'engine_load_s': summary['load_s']})
  try:
    for job in spec['jobs']:
      summary['jobs'].append(run_job(engine, job, report))
      summary['total_s'] = round(time.monotonic() - started, 3)
      if spec.get('summary'):
        with open(spec['summary'], 'w') as f:
          json.dump(summary, f, indent=1)
  finally:
    engine.close()
  summary['total_s'] = round(time.monotonic() - started, 3)
  if spec.get('summary'):
    with open(spec['summary'], 'w') as f:
      json.dump(summary, f, indent=1)
  sys.exit(1 if any(j['exit_code'] for j in summary['jobs']) else 0)


if __name__ == '__main__':
  main()
