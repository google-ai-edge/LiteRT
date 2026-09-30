#!/usr/bin/env python3
"""Focused tests for the embedded collector and public shell interface (Linux)."""

import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name('run_head.sh')
SOURCE = SCRIPT.read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
H = types.ModuleType('embedded_helper')
exec(compile(SOURCE, str(SCRIPT), 'exec'), H.__dict__)
# Native output shape captured from the successful E2B 20260926 short run.
TURN = """  Time to first token: 0.07 s
    Prefill Turn 1: Processed 1024 tokens in 67.507363ms duration.
      Prefill Speed: 15168.72 tokens/sec.
    Decode Turn 1: Processed 256 tokens in 914.8416ms duration.
      Decode Speed: 279.83 tokens/sec.
"""


class TestRunner(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory(prefix='benchmark test ')
    self.base = Path(self.temp.name).resolve()
    self.root = H.root_open(self.base / 'owned root')
    self.report = self.root / 'reports' / 'test'
    (self.report / 'logs').mkdir(parents=True)
    self.env = patch.dict(
        os.environ,
        {
            'NB_REPORT': str(self.report),
            'NB_CWD': str(self.base),
            'NB_MEMORY_LIMIT': '',
            'NB_INPUT': '1024',
            'NB_OUTPUT': '256',
            'NB_MTP': 'false',
        },
        clear=False,
    )
    self.env.start()
    H.save_result(
        {'status': 'running', 'invocation': 'fixture', 'paths': {}, 'jobs': []}
    )

  def tearDown(self):
    self.env.stop()
    self.temp.cleanup()

  def entry(self, profile='e2b', kind='aot'):
    p = self.root / 'cache' / 'models' / profile / ('a' * 64) / kind
    H.entry(self.root, p, profile, kind)
    (p / 'data.bin').write_bytes(b'x' * 8192)
    return p

  def cli(self, *args, root=None, extra=None):
    env = {
        **os.environ,
        'LITERT_BENCH_ROOT': str(root or self.root),
        **(extra or {}),
    }
    return subprocess.run(
        ['bash', str(SCRIPT), *args],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

  def test_syntax_and_help(self):
    self.assertEqual(subprocess.run(['bash', '-n', str(SCRIPT)]).returncode, 0)
    self.assertIn('Native TTFT is an estimate', self.cli('--help').stdout)

  def test_invalid_arguments_precede_mutation(self):
    cases = [
        ['--profile', 'e2b,12b'],
        ['--workload', 'bad'],
        ['--profile', 'e2b', '--workload', '32k'],
        ['--prefill', '129'],
        ['--metrics', ''],
        ['--metrics', 'build'],
        ['--cache-state', 'bad'],
        ['--residency', 'bad'],
        ['--profile', 'e2b', '--mtp'],
    ]
    for old in (
        '--config',
        '--build-state',
        '--aot-state',
        '--memory-cache-states',
        '--results-root',
        '--cache-root',
    ):
      cases.append([old, 'unused'])
    for args in cases:
      with self.subTest(args=args):
        root = self.base / 'must not exist'
        p = self.cli('report', *args, root=root)
        self.assertNotEqual(p.returncode, 0)
        self.assertFalse(root.exists())

  def test_valid_presets_reach_prerequisites(self):
    for profile, workload, prefill in [
        ('e2b', 'short', '128'),
        ('e2b', 'short', '1024'),
        ('12b', 'short', '128'),
        ('12b', 'short', '1024'),
        ('12b', '32k', '128'),
        ('12b', '32k', '1024'),
        ('12b', '128k', '128'),
        ('12b', '128k', '1024'),
    ]:
      for state in ('warm', 'runtime-cold', 'cold', 'all'):
        p = self.cli(
            'report',
            '--profile',
            profile,
            '--workload',
            workload,
            '--prefill',
            prefill,
            '--cache-state',
            state,
            '--model-file',
            str(self.base / 'missing model'),
        )
        self.assertNotEqual(p.returncode, 0)
        self.assertRegex(p.stderr, 'Missing LiteRT|Model missing')

  def test_residency_presets_and_explicit_overrides(self):
    policy = (
        'NB_LAZY=0\n'
        + SCRIPT.read_text()
        .split('NB_LAZY=0\n', 1)[1]
        .split('export NB_INPUT', 1)[0]
    )
    for workload, prefill in [
        ('short', '128'),
        ('short', '1024'),
        ('32k', '128'),
        ('32k', '1024'),
        ('128k', '128'),
        ('128k', '1024'),
    ]:
      for mtp in ('false', 'true'):
        auto = '0' if workload == 'short' or mtp == 'false' else '1'
        for residency, expected in [
            ('auto', auto),
            ('resident', '0'),
            ('lazy', '1'),
        ]:
          with self.subTest(
              workload=workload, prefill=prefill, mtp=mtp, residency=residency
          ):
            env = {
                **os.environ,
                'workload': workload,
                'prefill': prefill,
                'mtp': mtp,
                'residency': residency,
            }
            p = subprocess.run(
                ['bash', '-c', policy + 'printf "%s" "$NB_LAZY"'],
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(p.stdout, expected)

  def test_native_count_and_warmup(self):
    data = H.parse_native(
        TURN * 3 + 'Prefill Speed: 999.99 tokens/sec.\n', 3, 1, 1024, 256
    )
    self.assertEqual(
        data['summary']['decode_tokens_per_second']['mean'], 279.83
    )
    self.assertAlmostEqual(
        data['estimated_ttft_seconds'][0], 0.067507363 + 0.9148416 / 256
    )
    self.assertNotIn('first_delivery_seconds', data)

  def test_missing_checkout_model_sdk_before_mutation(self):
    rt = self.base / 'source with spaces'
    script = rt / 'litert/vendors/nvidia/run_head.sh'
    script.parent.mkdir(parents=True)
    script.write_text(SCRIPT.read_text())
    (rt / 'WORKSPACE').touch()
    lm = self.base / 'lm'
    marker = lm / 'runtime/engine/litert_lm_advanced_main.cc'
    marker.parent.mkdir(parents=True)
    marker.touch()
    model = self.base / 'model'
    model.write_text('fixture')
    root = self.base / 'never created'
    env = {
        **os.environ,
        'LITERT_BENCH_ROOT': str(root),
        'LITERT_LM_DIR': str(lm),
        'TENSORRT_RTX_ROOT': str(self.base / 'missing SDK'),
    }
    for target, message in (
        (model, 'Missing SDK files'),
        (self.base / 'absent', 'Model missing'),
    ):
      p = subprocess.run(
          ['bash', str(script), 'report', '--model-file', str(target)],
          env=env,
          text=True,
          capture_output=True,
      )
      self.assertNotEqual(p.returncode, 0)
      self.assertIn(message, p.stderr)
      self.assertFalse(root.exists())
    marker.unlink()
    for metrics in (
        'verify',
        'latency',
        'memory',
        'verify,latency',
        'latency,memory',
        'all',
    ):
      p = subprocess.run(
          ['bash', str(script), 'report', '--metrics', metrics],
          env=env,
          text=True,
          capture_output=True,
      )
      self.assertIn('Missing LiteRT or LiteRT-LM checkout', p.stderr)
      self.assertFalse(root.exists())

  def test_duration_units(self):
    self.assertAlmostEqual(H.duration('1m2.5s'), 62.5)
    self.assertAlmostEqual(H.duration('1.25ms'), 0.00125)
    for bad in ('nan', '0s', '1bad', 'infs'):
      with self.assertRaises(ValueError):
        H.duration(bad)

  def test_truncated_wrong_count_and_missing_metrics(self):
    for text in (TURN[:-30], TURN.replace('1024 tokens', '1023 tokens'), ''):
      with self.assertRaises(ValueError):
        H.parse_native(text, 1, 0, 1024, 256)

  def test_compiler_events_and_reuse(self):
    text = (
        'component=compiler phase=compile_begin context=0'
        ' monotonic_ns=100\ncomponent=compiler phase=compile_end context=0'
        ' monotonic_ns=200\nAOT cache hit:'
    )
    d = H.cache_events(text)
    self.assertEqual(d['aot_hits'], 1)
    self.assertEqual(d['compiler_event_seconds'], 1e-7)
    self.assertIsNone(
        H.cache_events('AOT cache hit:')['compiler_event_seconds']
    )

  def test_preview_apply_idempotence_and_spaces(self):
    p = self.entry()
    model = self.base / 'model.litertlm'
    model.write_text('keep')
    preview = self.cli('cache', 'clear', '--profile', 'e2b', '--kind', 'aot')
    self.assertEqual(preview.returncode, 0, preview.stderr)
    self.assertTrue(p.exists())
    result = self.cli(
        'cache', 'clear', '--profile', 'e2b', '--kind', 'aot', '--yes'
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertFalse(p.exists())
    self.assertTrue(model.exists())
    self.assertTrue((self.report / 'results.json').exists())
    self.assertEqual(self.cli('cache', 'clear', '--all', '--yes').returncode, 0)

  def test_shared_build_not_selected_by_profile(self):
    self.entry()
    build = self.root / 'cache/build/litert'
    H.entry(self.root, build, 'shared', 'build')
    p = self.cli('cache', 'clear', '--profile', 'e2b', '--yes')
    self.assertEqual(p.returncode, 0, p.stderr)
    self.assertTrue(build.exists())

  def test_build_readonly_and_external_symlink(self):
    build = self.root / 'cache/build/litert'
    H.entry(self.root, build, 'shared', 'build')
    keep = self.base / 'external'
    keep.mkdir()
    (keep / 'file').write_text('keep')
    (build / 'linked').symlink_to(keep, target_is_directory=True)
    readonly = build / 'readonly'
    readonly.mkdir()
    (readonly / 'x').write_text('x')
    readonly.chmod(0o555)
    p = self.cli('cache', 'clear', '--kind', 'build', '--yes')
    self.assertEqual(p.returncode, 0, p.stderr)
    self.assertTrue((keep / 'file').exists())

  def test_symlinks_traversal_and_unowned_refused(self):
    existing = self.base / 'unowned'
    existing.mkdir()
    (existing / 'file').touch()
    with self.assertRaises(ValueError):
      H.root_open(existing)
    link = self.base / 'link'
    link.symlink_to(self.root)
    with self.assertRaises(ValueError):
      H.root_open(link)
    with self.assertRaises(ValueError):
      H.root_open(str(self.root) + '/../escape')
    p = self.entry()
    (p / 'outside').symlink_to(existing)
    result = self.cli('cache', 'clear', '--all', '--yes')
    self.assertNotEqual(result.returncode, 0)
    self.assertTrue(p.exists())

  def test_mount_refused(self):
    p = self.entry()
    original = Path.is_mount
    with patch.object(
        Path, 'is_mount', lambda q: q == p / 'data.bin' or original(q)
    ):
      with self.assertRaises(ValueError):
        H.storage([p], True)

  def test_same_filesystem_bind_mount_refused_before_delete(self):
    p = self.entry()
    bind = p / 'bind with spaces'
    bind.mkdir()
    protected = bind / 'keep'
    protected.write_text('external data')
    original = Path.read_text
    mount_path = str(bind).replace(' ', r'\040')

    def read(path, *args, **kwargs):
      if str(path) == '/proc/self/mountinfo':
        return f'36 25 8:1 /elsewhere {mount_path} rw - ext4 /dev/sda rw\n'
      return original(path, *args, **kwargs)

    with patch.object(Path, 'read_text', read):
      with self.assertRaisesRegex(ValueError, '[Mm]ounted'):
        H.cache_command(self.root, 'clear', '', '', True)
    self.assertTrue(protected.exists())

  def test_active_lock(self):
    with (self.root / '.lock').open('w') as lock:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      p = self.cli('cache', 'clear', '--all', '--yes')
      self.assertNotEqual(p.returncode, 0)
      self.assertIn('in use', p.stderr)

  def test_stale_legacy_environment_ignored(self):
    p = self.cli(
        'cache',
        'list',
        extra={
            'RUN_ROOT': '/do/not/touch',
            'LITERT_LM_G3_HEAD': '/bad_trt_rtx',
            'G4MODEL': '/bad',
            'LITERT_G3_HEAD': '/wrong',
        },
    )
    self.assertEqual(p.returncode, 0, p.stderr)
    self.assertEqual(json.loads(p.stdout)['entries'], [])

  def test_nonzero_command_retains_failure(self):
    with self.assertRaises(RuntimeError):
      H.collect(
          'exit',
          'verify',
          'warm',
          '0',
          '0',
          'none',
          'none',
          ['/bin/sh', '-c', 'echo failed; exit 7'],
      )
    j = H.load_result()['jobs'][-1]
    self.assertEqual(j['status'], 'failed')
    self.assertEqual(j['exit_code'], 7)
    self.assertIn('failed', Path(j['log']).read_text())

  def test_missing_metric_fails_report(self):
    with self.assertRaises(ValueError):
      H.collect(
          'missing',
          'latency',
          'warm',
          '1',
          '0',
          'none',
          'none',
          ['/bin/echo', 'nothing'],
      )
    self.assertEqual(H.load_result()['jobs'][-1]['status'], 'failed')

  def test_sampler_failure_fails_report(self):
    with patch.object(
        H.subprocess, 'check_output', side_effect=OSError('sampler unavailable')
    ):
      with self.assertRaises(ValueError):
        H.collect(
            'memory',
            'memory',
            'warm',
            '1',
            '0',
            'none',
            'none',
            ['/bin/sleep', '.2'],
        )
    j = H.load_result()['jobs'][-1]
    self.assertIn('sampler unavailable', j['error'])
    self.assertTrue(Path(j['samples']).exists())

  def test_interrupt_retains_report(self):
    marker = self.base / 'started'
    command = ['/bin/sh', '-c', 'touch "$1"; exec sleep 60', 'sh', str(marker)]
    p = subprocess.Popen(
        [
            sys.executable,
            '-c',
            SOURCE,
            'collect',
            'interrupt',
            'verify',
            'warm',
            '0',
            '0',
            'none',
            'none',
            *command,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    while not marker.exists() and p.poll() is None:
      time.sleep(0.02)
    p.send_signal(signal.SIGTERM)
    p.communicate()
    self.assertNotEqual(p.returncode, 0)
    self.assertEqual(H.load_result()['jobs'][-1]['status'], 'failed')

  def test_compatibility_uses_actual_artifacts_and_settings(self):
    rt = self.base / 'rt'
    rt.mkdir()
    sdk = self.base / 'sdk'
    (sdk / 'lib').mkdir(parents=True)
    (sdk / 'lib/libtensorrt_rtx.so').write_text('SDK')
    build = self.base / 'lm output'
    (build / 'external').mkdir(parents=True)
    (build / 'external/litert').symlink_to(rt)
    binary = self.base / 'binary'
    binary.write_text('initial')
    model_path = self.base / 'model file'
    model_path.write_text('model')
    data = H.load_result()
    data.update(
        model={'path': str(model_path), 'sha256': 'model one'},
        settings={
            'profile': '12b',
            'context': '2048',
            'metrics': 'all',
            'state': 'warm',
        },
    )
    H.save_result(data)
    env = {
        'NB_PROFILE': '12b',
        'NB_PREFILL': '128',
        'NB_MTP': 'false',
        'NB_RT': str(rt),
        'NB_BUILD_LM': str(build),
        'NB_SDK': str(sdk),
        'NB_CUDA': '/fake cuda',
    }

    def key():
      with contextlib.redirect_stdout(io.StringIO()):
        H.main(['identity', str(binary)])
      return H.load_result()['compatibility_key']

    gpu = ({'name': 'GPU', 'uuid': 'uuid', 'driver': 'driver'}, '')
    with patch.dict(os.environ, env), patch.object(
        H, 'gpu_query', return_value=gpu
    ), patch.object(H.subprocess, 'check_output', return_value='CUDA version'):
      baseline = key()
      self.assertEqual(key(), baseline)
      before = model_path.stat()
      os.utime(
          model_path,
          ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
      )
      touched = key()
      self.assertNotEqual(touched, baseline)
      baseline = touched
      data = H.load_result()
      data['settings'].update(metrics='memory', state='cold')
      H.save_result(data)
      self.assertEqual(key(), baseline)
      binary.write_text('changed')
      changed = key()
      self.assertNotEqual(changed, baseline)
      data = H.load_result()
      data['model']['sha256'] = 'model two'
      H.save_result(data)
      model = key()
      self.assertNotEqual(model, changed)
      data = H.load_result()
      data['settings']['context'] = '34818'
      H.save_result(data)
      context = key()
      self.assertNotEqual(context, model)
      with patch.dict(os.environ, {'NB_PREFILL': '1024'}):
        self.assertNotEqual(key(), context)
      with patch.dict(os.environ, {'NB_MTP': 'true'}):
        self.assertNotEqual(key(), context)
      with patch.dict(os.environ, {'NB_PROFILE': 'e2b'}):
        self.assertNotEqual(key(), context)
      (sdk / 'lib/libtensorrt_rtx.so').write_text('SDK changed')
      self.assertNotEqual(key(), context)

  def test_unknown_gpu_process_and_heat_block(self):
    gpu = {'compute_percent': '0', 'temperature_c': '39'}
    with patch.object(H, 'gpu_query', return_value=(gpu, '123 unknown')):
      with self.assertRaises(RuntimeError):
        H.safe_idle()
    with patch.object(
        H, 'gpu_query', return_value=({**gpu, 'temperature_c': '80'}, '')
    ):
      with self.assertRaises(RuntimeError):
        H.safe_idle()

  def test_e2b_explicit_128_requires_selected_signature(self):
    # The static executor ignores the batch hint when both prefill graphs exist.
    rt = self.base / 'rt'
    rt.mkdir()
    sdk = self.base / 'sdk'
    (sdk / 'lib').mkdir(parents=True)
    (sdk / 'lib/libtensorrt_rtx.so').write_text('sdk')
    build = self.base / 'build'
    (build / 'external').mkdir(parents=True)
    (build / 'external/litert').symlink_to(rt)
    binary = self.base / 'binary'
    binary.write_text('binary')
    model = self.base / 'model'
    model.write_text('model')
    data = H.load_result()
    data.update(
        model={'path': str(model), 'sha256': 'model'},
        settings={'profile': 'e2b'},
    )
    H.save_result(data)
    env = {
        'NB_PROFILE': 'e2b',
        'NB_PREFILL': '128',
        'NB_MTP': 'false',
        'NB_RT': str(rt),
        'NB_BUILD_LM': str(build),
        'NB_SDK': str(sdk),
        'NB_CUDA': '/cuda',
    }
    with patch.dict(os.environ, env), patch.object(
        H,
        'gpu_query',
        return_value=({'name': 'GPU', 'uuid': 'u', 'driver': 'd'}, ''),
    ), patch.object(H.subprocess, 'check_output', return_value='cuda'):
      with contextlib.redirect_stdout(io.StringIO()):
        H.main(['identity', str(binary)])
      self.assertEqual(
          H.load_result()['cache_identity']['selected_signatures'],
          ['prefill_128', 'decode'],
      )
      with patch.dict(
          os.environ, {'NB_PREFILL': '1024'}
      ), contextlib.redirect_stdout(io.StringIO()):
        H.main(['identity', str(binary)])
      self.assertEqual(
          H.load_result()['cache_identity']['selected_signatures'], []
      )

  def test_generic_mtp_requires_execution_evidence(self):
    env = {'NB_MTP': 'true'}
    raw = 'AOT cache hit: fixture\nPARIS\nMTP Drafter - Success rate: 1e-05\n'
    with patch.dict(os.environ, env), patch.object(
        H, 'safe_idle', return_value={'used_mib': '0'}
    ):
      H.collect(
          'mtp_ok', 'verify', 'warm', '0', '0', 'hit', 'npu', ['/bin/echo', raw]
      )
      j = H.load_result()['jobs'][-1]
      self.assertEqual(
          j['mtp_execution_evidence']['generic_drafter_success_rates'], [1e-5]
      )
      with self.assertRaises(ValueError):
        H.collect(
            'mtp_missing',
            'verify',
            'warm',
            '0',
            '0',
            'hit',
            'npu',
            ['/bin/echo', 'AOT cache hit: fixture\nPARIS\n'],
        )

  def test_native_mtp_backend_rejection_is_actionable(self):
    with patch.dict(os.environ, {'NB_MTP': 'true'}), patch.object(
        H, 'safe_idle', return_value={'used_mib': '0'}
    ):
      with self.assertRaisesRegex(
          RuntimeError, 'MTP unsupported.*Unsupported backend: 6'
      ):
        H.collect(
            'mtp_rejected',
            'verify',
            'prepare',
            '0',
            '0',
            'any',
            'npu',
            ['/bin/sh', '-c', 'echo "Unsupported backend: 6"; exit 3'],
        )
    j = H.load_result()['jobs'][-1]
    self.assertEqual(j['status'], 'failed')
    self.assertEqual(j['exit_code'], 3)

  def host_fixture(self, flag=None, rc=1, log=None, gpu=None):
    if log is None:
      log = (
          'High Performance\nASUS Profile 1\nno stable 10-second idle window\n'
      )
    if gpu is None:
      gpu = {
          'pstate': 'P0',
          'compute_percent': '0',
          'temperature_c': '50',
          'memory_controller_percent': '65',
      }

    def prepare(command, **kwargs):
      self.assertEqual(command, ['zsh', '-lic', 'benchmark'])
      kwargs['stdout'].write(log)
      return types.SimpleNamespace(returncode=rc)

    env = {
        k: v
        for k, v in os.environ.items()
        if k != 'LITERT_BENCH_ALLOW_BACKGROUND_ACTIVITY'
    }
    if flag is not None:
      env['LITERT_BENCH_ALLOW_BACKGROUND_ACTIVITY'] = flag
    output = io.StringIO()
    with patch.dict(os.environ, env, clear=True), patch.object(
        H.os, 'uname', return_value=types.SimpleNamespace(nodename='Cuda')
    ), patch.object(H, 'gpu_query', return_value=(gpu, '')), patch.object(
        H.subprocess, 'run', side_effect=prepare
    ), contextlib.redirect_stdout(
        output
    ), contextlib.redirect_stderr(
        output
    ):
      H.main(['host'])
    return H.load_result()['host'], output.getvalue()

  def test_background_activity_defaults_to_warning(self):
    for flag in (None, '1'):
      with self.subTest(flag=flag):
        host, output = self.host_fixture(flag)
        self.assertEqual(
            host['decision'], 'diagnostic_background_activity_exception'
        )
        self.assertFalse(host['clean_idle_passed'])
        self.assertTrue(host['allow_background_activity'])
        self.assertIn('WARNING', output)
        self.assertIn(
            'diagnostic_background_activity_exception',
            (self.report / 'report.md').read_text(),
        )
        self.assertIn(
            'no stable 10-second idle window', Path(host['log']).read_text()
        )

  def test_background_activity_can_be_disabled(self):
    with self.assertRaisesRegex(RuntimeError, 'Host preflight failed'):
      self.host_fixture('0')
    self.assertEqual(H.load_result()['host']['decision'], 'blocked')

  def test_background_activity_does_not_hide_preparation_failures(self):
    for log in (
        'benchmark: command not found',
        'High Performance\nno stable 10-second idle window',
        'ASUS Profile 1\nno stable 10-second idle window',
        'High Performance\nASUS Profile 1\nservice readiness failed',
    ):
      with self.subTest(log=log):
        with self.assertRaisesRegex(RuntimeError, 'Host preflight failed'):
          self.host_fixture(log=log)

  def test_clean_preflight_remains_clean(self):
    host, output = self.host_fixture(rc=0)
    self.assertEqual(host['decision'], 'clean_idle_passed')
    self.assertTrue(host['clean_idle_passed'])
    self.assertNotIn('WARNING', output)

  def test_background_activity_does_not_allow_contention(self):
    for gpu in (
        {'compute_percent': '6', 'temperature_c': '40'},
        {'compute_percent': '0', 'temperature_c': '70'},
    ):
      with self.subTest(gpu=gpu):
        with self.assertRaisesRegex(RuntimeError, 'GPU busy or hot'):
          self.host_fixture(gpu=gpu)

  def test_invalid_background_activity_flag_precedes_mutation(self):
    root = self.base / 'must not exist'
    p = self.cli(
        'report',
        root=root,
        extra={'LITERT_BENCH_ALLOW_BACKGROUND_ACTIVITY': 'yes'},
    )
    self.assertNotEqual(p.returncode, 0)
    self.assertIn(
        'LITERT_BENCH_ALLOW_BACKGROUND_ACTIVITY must be 0 or 1', p.stderr
    )
    self.assertFalse(root.exists())


if __name__ == '__main__':
  unittest.main(verbosity=2)
