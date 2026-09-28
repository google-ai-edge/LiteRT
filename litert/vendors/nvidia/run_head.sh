#!/usr/bin/env bash
# Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
# NVIDIA benchmarks using the unchanged LiteRT-LM advanced CLI. See run_head.md.
# A subshell also prevents accidental sourcing from changing the caller's shell.
(
set -euo pipefail

helper() {
  python3 - "$@" <<'PY'
import datetime, fcntl, hashlib, json, math, os, re, shutil, signal, stat
import statistics, subprocess, sys, threading, time
from pathlib import Path

FORMAT = 'litert-nvidia-benchmark-v2'
KINDS = {'build', 'aot', 'runtime', 'compiler', 'cpu'}
MARKER = '.benchmark-entry.json'

def write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def checked_path(path):
    p = Path(path).expanduser().absolute()
    if '..' in p.parts or p.resolve() != p or p in (Path('/'), Path.home()):
        raise ValueError('Unsafe or symlinked managed path: ' + str(p))
    return p

def root_open(path):
    p = checked_path(path)
    marker = p / '.benchmark-root'
    if not marker.exists():
        if p.exists() and any(p.iterdir()): raise ValueError('Refusing to adopt nonempty root: ' + str(p))
        p.mkdir(parents=True, exist_ok=True)
        marker.write_text(FORMAT + '\n')
    if marker.is_symlink() or marker.read_text().strip() != FORMAT:
        raise ValueError('Invalid benchmark root marker')
    for name in ('cache', 'reports'):
        checked_path(p/name).mkdir(exist_ok=True)
    if (p/'.lock').is_symlink(): raise ValueError('Symlinked lock')
    return p

def entry(root, path, profile, kind):
    root, path = checked_path(root), checked_path(path)
    if kind not in KINDS or profile not in ('e2b', '12b', 'shared'):
        raise ValueError('Invalid cache ownership')
    relative = path.relative_to(root/'cache')
    parts = relative.parts
    valid = ((kind == 'build' and profile == 'shared' and parts in (('build','litert'), ('build','litert-lm')))
             or (len(parts) == 4 and parts[0] == 'models' and parts[1] == profile
                 and re.fullmatch('[0-9a-f]{64}', parts[2]) and parts[3] == kind)
             or (len(parts) == 4 and parts[0] == 'scratch' and parts[3] == kind
                 and re.fullmatch('[A-Za-z0-9_.-]+', parts[1]) and re.fullmatch('[A-Za-z0-9_.-]+', parts[2])))
    if not valid: raise ValueError('Invalid cache layout: ' + str(relative))
    record = {'format': FORMAT, 'path': str(relative), 'profile': profile, 'kind': kind}
    marker = path/MARKER
    if marker.exists():
        if marker.is_symlink() or json.loads(marker.read_text()) != record:
            raise ValueError('Invalid cache entry marker: ' + str(path))
    else:
        if path.exists() and any(path.iterdir()): raise ValueError('Unowned cache entry: ' + str(path))
        path.mkdir(parents=True, exist_ok=True)
        write(marker, record)
    return path

def reject_mounts(path):
    # is_mount() alone misses bind mounts on the same filesystem.
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        name = re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1],8)), line.split()[4])
        mount = Path(name)
        if mount == path or path in mount.parents:
            raise ValueError('Mounted cache path: ' + str(mount))

def inventory(root):
    reject_mounts(root/'cache')
    rows = []
    for d, dirs, files in os.walk(root/'cache', followlinks=False):
        p = Path(d)
        if p.is_mount(): raise ValueError('Mounted cache path: ' + str(p))
        if MARKER in files:
            marker = p/MARKER
            if marker.is_symlink(): raise ValueError('Symlinked cache marker')
            row = json.loads(marker.read_text())
            entry(root, p, row['profile'], row['kind'])
            rows.append({**row, 'path': str(p)})
            dirs[:] = []
        elif any((p/n).is_symlink() for n in dirs+files):
            raise ValueError('Symlink in cache ownership path')
        elif files:
            raise ValueError('Unowned files in cache hierarchy: ' + str(p))
    return rows

def storage(paths, validate=False):
    inodes, logical, links = {}, 0, 0
    for p in map(Path, paths):
        reject_mounts(p)
        for d, dirs, files in os.walk(p, followlinks=False):
            for name in dirs + files:
                q = Path(d)/name
                s = q.lstat()
                if stat.S_ISLNK(s.st_mode):
                    links += 1
                    if validate and p.name != 'litert' and p.name != 'litert-lm':
                        raise ValueError('Symlink inside model cache: ' + str(q))
                    continue  # Bazel creates legitimate symlinks; never follow them.
                if q.is_mount(): raise ValueError('Refusing mounted cache: ' + str(q))
                if stat.S_ISREG(s.st_mode):
                    logical += s.st_size
                    item = inodes.setdefault((s.st_dev,s.st_ino), [s.st_blocks*512,s.st_nlink,0])
                    item[2] += 1
    return {'logical_bytes': logical, 'allocated_bytes': sum(x[0] for x in inodes.values()),
            'estimated_reclaimable_bytes': sum(x[0] for x in inodes.values() if x[1] == x[2]),
            'files': len(inodes), 'symlinks_not_followed': links}

def delete_owned(path):
    reject_mounts(Path(path))
    # Bazel's extracted repositories may contain read-only directories.
    for d, dirs, _ in os.walk(path, followlinks=False):
        dirs[:] = [n for n in dirs if not (Path(d)/n).is_symlink()]
        s = os.stat(d)
        if s.st_uid != os.getuid(): raise ValueError('Cache not owned by current user: ' + d)
        os.chmod(d, stat.S_IMODE(s.st_mode) | stat.S_IWUSR)
    shutil.rmtree(path)

def cache_command(root, action, profile, kind, apply):
    rows = [r for r in inventory(root) if (not profile or r['profile'] == profile)
            and (not kind or r['kind'] == kind)]
    paths = [r['path'] for r in rows]
    data = {'action': action, 'apply': apply, 'entries': rows, 'storage': storage(paths, True),
            'preserved': 'Models, source, reports and unowned data. Cleared artifact paths in older reports become unavailable.'}
    for r in rows: r['storage'] = storage([r['path']])
    if action == 'clear':
        dest = root/'reports'/('cleanup-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.json')
        data['manifest'] = str(dest)
        data['free_bytes_before'] = shutil.disk_usage(root).free
        write(dest, data)  # Persist the intended deletion even if later deletion fails.
        if apply:
            for p in paths: delete_owned(p)
            data['free_bytes_after'] = shutil.disk_usage(root).free
            data['deleted_paths'] = paths
            write(dest, data)
    print(json.dumps(data, indent=2))

def duration(text):
    units = {'h':3600, 'm':60, 's':1, 'ms':.001, 'us':.000001, 'µs':.000001, 'ns':1e-9}
    matches = list(re.finditer(r'([0-9.]+)(ms|us|µs|ns|h|m|s)', text))
    if not matches or ''.join(m.group() for m in matches) != text:
        raise ValueError('Unrecognized native duration: ' + text)
    value = sum(float(m[1])*units[m[2]] for m in matches)
    if not math.isfinite(value) or value <= 0: raise ValueError('Invalid native duration')
    return value

def parse_native(text, iterations, warmups, prefill, decode):
    result = {}
    for phase, count in (('Prefill',prefill), ('Decode',decode)):
        rows = re.findall(phase + r' Turn \d+: Processed (\d+) tokens in (\S+) duration\.\s*' + phase + r' Speed: ([0-9.]+) tokens/sec', text)
        if len(rows) != iterations or any(int(r[0]) != count for r in rows):
            raise ValueError(f'{phase}: expected {iterations} turns of {count}, got {[r[0] for r in rows]}')
        result[phase.lower()] = [{'native_count':int(n), 'seconds':duration(t), 'tokens_per_second':float(v)} for n,t,v in rows]
    if not 0 <= warmups < iterations: raise ValueError('Invalid warmup count')
    result['warmups_excluded'] = warmups
    result['iterations'] = iterations
    result['estimated_ttft_seconds'] = [p['seconds']+d['seconds']/d['native_count'] for p,d in zip(result['prefill'],result['decode'])]
    result['summary'] = {}
    for phase in ('prefill','decode'):
        for field in ('seconds','tokens_per_second'):
            values = [r[field] for r in result[phase][warmups:]]
            result['summary'][phase+'_'+field] = {'mean':statistics.mean(values), 'min':min(values), 'max':max(values),
                                                'stdev':statistics.stdev(values) if len(values)>1 else None}
    result['note'] = 'Native synthetic counts; decode counts are runtime steps. TTFT is prefill plus average decode, not first delivery.'
    return result

def cache_events(text):
    starts, intervals = {}, []
    for phase, context, ns in re.findall(r'component=compiler phase=(compile_begin|compile_end) context=(\S+) monotonic_ns=(\d+)', text):
        if phase == 'compile_begin': starts[context] = int(ns)
        elif context in starts: intervals.append((int(ns)-starts.pop(context))/1e9)
    return {'aot_hits':text.count('AOT cache hit:'), 'aot_misses':text.count('AOT cache miss:'),
            'compiled_partitions':len(re.findall(r'compiling partition \d+/\d+',text)),
            'compilation_failures':text.count('failed to compile partition'),
            'runtime_cache_loads':text.count('loaded runtime cache for'),
            'invalid_runtime_cache_loads':text.count('ignored invalid runtime cache'),
            'runtime_cache_save_warnings':text.count('failed to save runtime cache for'),
            'compiler_event_seconds':sum(intervals) if intervals and not starts else None,
            'engine_build_seconds':sum(map(float,re.findall(r'Engine generation completed in ([0-9.]+) seconds', text))) or None}

def gpu_query():
    exe = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    rows = subprocess.check_output([exe,'--query-gpu=name,uuid,driver_version,pstate,utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(rows) != 1: raise RuntimeError('Exactly one visible NVIDIA GPU is required')
    v = [x.strip() for x in rows[0].split(',')]
    apps = subprocess.check_output([exe,'--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    return dict(zip(('name','uuid','driver','pstate','compute_percent','memory_controller_percent','used_mib','temperature_c','power_w'),v)), apps

def safe_idle():
    gpu, apps = gpu_query()
    if apps: raise RuntimeError('GPU compute process present; it was not interrupted: ' + apps)
    if float(gpu['compute_percent']) > 5 or float(gpu['temperature_c']) >= 70:
        raise RuntimeError('GPU busy or hot: ' + str(gpu))
    return gpu

def result_path(): return Path(os.environ['NB_REPORT'])/'results.json'
def load_result(): return json.loads(result_path().read_text())

def save_result(data):
    write(result_path(), data)
    lines = ['# NVIDIA benchmark report', '', 'Status: '+data['status'], '',
             'Invocation: `'+data['invocation']+'`', '',
             'Resolved engine residency: '+('lazy' if data.get('settings',{}).get('lazy')=='1' else 'resident')+'.',
             'Host decision: '+data.get('host',{}).get('decision','not prepared')+'.',
             *['WARNING: '+warning for warning in data.get('host',{}).get('warnings',[])], '',
             'Native synthetic benchmark; estimated TTFT is not measured first delivery. Memory passes are separate from throughput.',
             'CPU HWM is GNU time kernel maximum RSS (runtime process including initialization; for builds, largest child, not concurrent sum).',
             'GPU peaks are sampled device-wide values including desktop use; sampling can miss brief peaks.', '',
             '## Paths', '', '```json', json.dumps(data.get('paths',{}),indent=2), '```', '',
             '| Job | Status | Wall s | CPU HWM GiB | GPU peak GiB | PP tok/s | Decode steps/s |',
             '|---|---|---:|---:|---:|---:|---:|']
    for j in data.get('jobs',[]):
        stats=j.get('native',{}).get('summary',{})
        def val(x): return '—' if x is None else f'{x:.3f}'
        lines.append('| '+ ' | '.join([j['name'],j['status'],val(j.get('wall_seconds')),
             val(j.get('cpu_max_rss_bytes',0)/2**30) if 'cpu_max_rss_bytes' in j else '—',
             val(j.get('gpu_sampled_peak_bytes',0)/2**30) if 'gpu_sampled_peak_bytes' in j else '—',
             val(stats.get('prefill_tokens_per_second',{}).get('mean')),
             val(stats.get('decode_tokens_per_second',{}).get('mean'))])+' |')
        if j.get('error'): lines.append('\n'+j['name']+': '+j['error']+'\n')
    if data.get('host'): lines += ['', 'Host preparation: `'+json.dumps(data['host'])+'`']
    lines += ['', 'Full commands, environments, cache observations, iteration spread, artifact hashes and raw-log paths: [results.json](results.json).']
    (result_path().parent/'report.md').write_text('\n'.join(lines)+'\n')

def collect(name, kind, cache_state, iterations, warmups, expectation, backend, command):
    data = load_result()
    logdir = result_path().parent/'logs'
    logfile, timefile = logdir/(name+'.log'), logdir/(name+'.time')
    env = os.environ.copy()
    env['LC_ALL'] = 'C'  # GNU time field names and decimal separators are parsed below.
    effective = {k:v for k,v in env.items() if k.startswith('LITERT_NVIDIA_') or k in
                 ('LC_ALL','LD_LIBRARY_PATH','CUDA_CACHE_PATH','CUDA_CACHE_DISABLE','CUDA_HOME','TENSORRT_RTX_ROOT','CUDA_VISIBLE_DEVICES')}
    job = {'name':name, 'kind':kind, 'cache_state':cache_state, 'status':'running',
           'command':command, 'cwd':env['NB_CWD'], 'environment':effective,
           'log':str(logfile), 'gnu_time':str(timefile), 'memory_sampling':kind=='memory',
           'scope':'source build' if kind=='build' else ('cold compile/startup' if cache_state=='cold' else 'runtime invocation')}
    if backend in ('npu','cpu'):
        cache_dirs = {'cli_cache':next((a.split('=',1)[1] for a in command if a.startswith('--cache_dir=')),None)}
        if backend == 'npu': cache_dirs.update(aot=env.get('LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR'),runtime=env.get('LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR'),cuda=env.get('CUDA_CACHE_PATH'))
        job['cache_contents_before'] = {}
        for label,path in cache_dirs.items():
            files = [p for p in Path(path).rglob('*') if p.is_file() and p.name not in (MARKER,'.warm-ready')] if path else []
            job['cache_contents_before'][label] = {'path':path,'payload_files':len(files),'bytes':sum(p.stat().st_size for p in files)}
    if kind == 'build':
        base = next(Path(a.split('=',1)[1]) for a in command if a.startswith('--output_base='))
        job['initial_output_state'] = 'incremental' if (base/'execroot').exists() else 'empty'
    data['jobs'].append(job)
    save_result(data)
    start = time.monotonic()
    samples, errors, stop = [], [], threading.Event()
    proc = None
    def sample():
        try:
            while not stop.is_set():
                # Only the launched runtime's RSS; no aggregate process-tree accounting.
                pending, rss = [proc.pid], None
                while pending:
                    pid = pending.pop()
                    try:
                        exe = Path(f'/proc/{pid}/exe').resolve()
                        if exe == Path(command[0]).resolve():
                            status = Path(f'/proc/{pid}/status').read_text()
                            m = re.search(r'^VmRSS:\s*(\d+) kB',status,re.M)
                            if m: rss = int(m[1])*1024
                        pending += list(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
                    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
                smi = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
                used = float(subprocess.check_output([smi,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())*1024**2
                samples.append({'elapsed_seconds':time.monotonic()-start,'cpu_rss_bytes':rss,'device_used_bytes':int(used)})
                stop.wait(.1)
        except Exception as exc: errors.append(str(exc))
    def interrupted(signum, frame):
        if proc and proc.poll() is None: os.killpg(proc.pid, signal.SIGTERM)
        raise KeyboardInterrupt('Signal '+str(signum))
    old_handlers = {s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    thread = None
    try:
        if backend == 'npu':
            job['gpu_before'] = safe_idle()
            job['gpu_baseline_bytes'] = int(float(job['gpu_before']['used_mib'])*1024**2)
        launch = ['/usr/bin/time','-v','-o',str(timefile)] + command
        if env.get('NB_MEMORY_LIMIT'):
            launch = ['systemd-run','--user','--scope','--quiet','-p','MemoryMax='+env['NB_MEMORY_LIMIT'],'-p','MemorySwapMax=8G'] + launch
        job['launch_command'] = launch
        with logfile.open('w') as log:
            start = time.monotonic()
            proc = subprocess.Popen(launch,cwd=env['NB_CWD'],env=env,stdin=subprocess.DEVNULL,
                                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if kind == 'memory':
                thread = threading.Thread(target=sample,daemon=True)
                thread.start()
            job['exit_code'] = proc.wait()
            ended = time.monotonic()
        stop.set()
        if thread: thread.join()
        job['wall_seconds'] = ended-start
        raw = logfile.read_text(errors='replace')
        job['cache'] = cache_events(raw)
        hwm = re.search(r'Maximum resident set size \(kbytes\): (\d+)',timefile.read_text())
        if not hwm: raise ValueError('Missing kernel maximum RSS from GNU time')
        job['cpu_max_rss_bytes'] = int(hwm[1])*1024
        if kind == 'memory':
            samplefile = logdir/(name+'.samples.jsonl')
            samplefile.write_text(''.join(json.dumps(r)+'\n' for r in samples))
            job.update(samples=str(samplefile),sampling_interval_seconds=.1,sampler_errors=errors)
            if errors or not samples or not any(r['cpu_rss_bytes'] is not None for r in samples):
                raise ValueError('Memory sampler failed or missed the runtime: '+str(errors))
            job['gpu_sampled_peak_bytes'] = max(r['device_used_bytes'] for r in samples)
            checkpoints = list(map(int,re.findall(r'cuda_available=1 cuda_device_used_bytes=(\d+)',raw)))
            job['gpu_backend_checkpoint_peak_bytes'] = max(checkpoints) if checkpoints else None
        if job['exit_code']:
            unsupported = re.search(r'Unsupported backend: [^\n]+', raw)
            if backend == 'npu' and env.get('NB_MTP') == 'true' and unsupported:
                raise RuntimeError('MTP unsupported by the native runtime ('+unsupported[0]+'); see '+str(logfile))
            raise RuntimeError('Command exited '+str(job['exit_code'])+'; see '+str(logfile))
        if backend == 'npu':
            cache = job['cache']
            if cache['compilation_failures']: raise RuntimeError('NVIDIA compilation failed; refusing CPU fallback')
            if not (cache['aot_hits'] or cache['aot_misses']): raise ValueError('No NVIDIA AOT execution evidence')
            if cache['invalid_runtime_cache_loads']: raise ValueError('Runtime cache load was invalid; refusing a misleading cache-state result')
            if expectation == 'miss' and (not cache['aot_misses'] or cache['aot_hits'] or not cache['compiled_partitions']):
                raise ValueError('Cold AOT miss/compilation evidence missing')
            if expectation == 'hit' and (not cache['aot_hits'] or cache['compiled_partitions']):
                raise ValueError('Expected AOT reuse but logs disagree')
        if kind in ('latency','memory','prepare'):
            job['native'] = parse_native(raw,int(iterations),int(warmups),int(env['NB_INPUT']),int(env['NB_OUTPUT']))
        if kind == 'verify':
            match = re.search(r'^\s*(Paris)\s*$',re.sub(r'\x1b\[[0-9;]*m','',raw),re.M|re.I)
            if not match: raise ValueError('Missing expected-answer response')
            job['expected_answer'] = {'response':match[1],'passed':True,'note':'CPU/NVIDIA answer smoke test, not logits equivalence'}
        if env.get('NB_MTP') == 'true' and backend == 'npu' and kind == 'verify':
            evidence = {'npu_cycles':raw.count('Starting MTP Speculative Cycle'),
                        'generic_drafter_success_rates':list(map(float,re.findall(r'MTP Drafter - Success rate: ([0-9.eE+-]+)',raw)))}
            job['mtp_execution_evidence'] = evidence
            if not (evidence['npu_cycles'] or evidence['generic_drafter_success_rates']):
                raise ValueError('MTP requested but execution was not confirmed; unsupported capability')
        job['status'] = 'passed'
    except BaseException as exc:
        job.update(status='failed',error=str(exc),wall_seconds=time.monotonic()-start)
        if proc and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            proc.wait()
        raise
    finally:
        stop.set()
        if thread: thread.join()
        if kind == 'memory':
            samplefile = logdir/(name+'.samples.jsonl')
            samplefile.write_text(''.join(json.dumps(r)+'\n' for r in samples))
            job.update(samples=str(samplefile),sampler_errors=errors,sampling_interval_seconds=.1)
        for s,h in old_handlers.items(): signal.signal(s,h)
        save_result(data)

def provenance(repo):
    def git(*a): return subprocess.check_output(['git','-C',repo,*a])
    other = {}
    for name in git('ls-files','--others','--exclude-standard','-z').decode().split('\0'):
        if name and Path(name).suffix in ('.cc','.h','.py','.sh','.bzl','.md','.json') and '/results/' not in '/'+name:
            p = Path(repo)/name
            if p.is_file(): other[name] = digest(p)
    return {'path':repo,'head':git('rev-parse','HEAD').decode().strip(),
            'status':git('status','--short').decode(),'diff_sha256':hashlib.sha256(git('diff','--binary','HEAD')).hexdigest(),
            'untracked_source_sha256':other}

def main(args):
    action, *args = args
    if action == 'root': root_open(args[0])
    elif action == 'entry': entry(*args)
    elif action == 'cache': cache_command(root_open(args[0]),args[1],args[2],args[3],args[4]=='yes')
    elif action == 'init':
        report = Path(os.environ['NB_REPORT']); (report/'logs').mkdir(parents=True)
        keys = ('PROFILE','WORKLOAD','PREFILL','INPUT','OUTPUT','CONTEXT','LAZY','MTP','METRICS','STATE')
        paths = {k.lower():os.environ['NB_'+k] for k in ('RT','LM','MODEL','CUDA','SDK','BUILD_RT','BUILD_LM','REPORT')}
        data = {'schema_version':2,'status':'running','invocation':args[0], 'paths':paths,
                'settings':{k.lower():os.environ['NB_'+k] for k in keys},'jobs':[],
                'sources':{k:provenance(os.environ['NB_'+k]) for k in ('RT','LM')},
                'model':{'path':paths['model'],'bytes':Path(paths['model']).stat().st_size,'sha256':digest(paths['model'])},
                'runner_sha256':digest(os.environ['NB_SCRIPT']),
                'build_policy':{'jobs':os.environ['NB_JOBS'],'ram_mb':os.environ['NB_RAM'],'memory_limit':os.environ.get('NB_MEMORY_LIMIT'),
                                'action_cache':'local incremental only; remote/disk action caches disabled',
                                'dependency_distdir':os.environ.get('LITERT_BENCH_DISTDIR'),
                                'os_and_dependency_caches':'may be warm; no global cache drop'}}
        save_result(data)
    elif action == 'host':
        data = load_result(); before = safe_idle()
        receipt = {'before':before,'decision':'local_readiness_check',
                   'allow_background_activity':os.environ.get('LITERT_BENCH_ALLOW_BACKGROUND_ACTIVITY','1')=='1'}
        if os.uname().nodename.lower() in ('cuda','cuda-wsl'):
            path = result_path().parent/'logs'/'host-preparation.log'
            with path.open('w') as log:
                rc = subprocess.run(['zsh','-lic','benchmark'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT).returncode
            receipt.update(command=['zsh','-lic','benchmark'],exit_code=rc,log=str(path),clean_idle_passed=rc==0)
            receipt['decision'] = 'clean_idle_passed' if rc==0 else 'blocked'
            after = safe_idle(); receipt['after'] = after
            raw = path.read_text()
            if rc and receipt['allow_background_activity'] and 'no stable 10-second idle window' in raw and 'Profile 1' in raw and re.search('High [Pp]erformance',raw):
                receipt['decision'] = 'diagnostic_background_activity_exception'
                receipt['warnings'] = ['Strict idle preflight failed; background activity may affect timings. '
                                       'Continuing with diagnostic measurements. GPU state: '+json.dumps(after)]
        data['host'] = receipt; save_result(data)
        if receipt['decision']=='blocked': raise RuntimeError('Host preflight failed; see retained log')
        for warning in receipt.get('warnings',[]): print('WARNING: '+warning,file=sys.stderr,flush=True)
        print('Host: '+receipt['decision'],flush=True)
    elif action == 'collect': collect(*args[:7],args[7:])
    elif action == 'identity':
        data=load_result(); binaries=[]
        for p in args:
            p=Path(p).resolve(); binaries.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
        gpu,_=gpu_query()
        sdk=Path(os.environ['NB_SDK'])
        model_path=Path(data['model']['path']).resolve(); model_stat=model_path.stat()
        identity={'model_file_identity':{'path':str(model_path),'device':model_stat.st_dev,'inode':model_stat.st_ino,
                                        'bytes':model_stat.st_size,'mtime_ns':model_stat.st_mtime_ns,'ctime_ns':model_stat.st_ctime_ns},
                  'binaries':[b['sha256'] for b in binaries], 'model':data['model']['sha256'],
                  'settings':{k:v for k,v in data['settings'].items() if k not in ('metrics','state')},
                  'selected_signatures':([] if os.environ['NB_PROFILE']=='e2b' and os.environ['NB_PREFILL']!='128' else ['prefill_'+os.environ['NB_PREFILL'],'decode']+(['verify'] if os.environ['NB_MTP']=='true' else [])),
                  'backend':{'precision':'bf16','gemv':'cuda_gemv','shared_weights':1,'jit_handle':0,
                             'fc_cap':536870912 if os.environ['NB_PROFILE']=='12b' else None},
                  'device':{k:gpu[k] for k in ('name','uuid','driver')},
                  'sdk_sha256':digest(sdk/'lib/libtensorrt_rtx.so'),
                  'cuda_version':subprocess.check_output([os.environ['NB_CUDA']+'/bin/nvcc','--version'],text=True)}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        data.update(binaries=binaries,cache_identity=identity,compatibility_key=key)
        override=Path(os.environ['NB_BUILD_LM'])/'external/litert'
        if override.resolve()!=Path(os.environ['NB_RT']): raise RuntimeError('Local LiteRT override did not resolve correctly')
        data['resolved_litert_override']=str(override.resolve()); save_result(data)
        print(key)
    elif action == 'finish':
        data=load_result()
        data['status']='passed' if args[0]=='0' and all(j['status']=='passed' for j in data['jobs']) else 'failed'
        changed=[b['path'] for b in data.get('binaries',[]) if not Path(b['path']).is_file() or digest(b['path'])!=b['sha256']]
        data['binary_identity_verified_at_finish']=not changed
        if changed: data.update(status='failed',error='Binary changed during run: '+str(changed))
        data['cache_paths']={k:os.environ.get(k) for k in ('NB_POOL','LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR','LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR','CUDA_CACHE_PATH')}
        save_result(data)
        if changed: raise RuntimeError(data['error'])
    else: raise ValueError('Unknown helper operation')

if __name__ == '__main__':
    try: main(sys.argv[1:])
    except (Exception,KeyboardInterrupt) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        sys.exit(1)
PY
}

usage() {
  cat <<'HELP'
Usage:
  run_head.sh report [--profile e2b|12b] [--model-file PATH]
    [--workload short|32k|128k] [--prefill 128|1024] [--mtp]
    [--residency auto|lazy|resident] [--metrics verify,latency,memory|all]
    [--cache-state warm|runtime-cold|cold|all]
  run_head.sh cache list [--profile e2b|12b] [--kind build|aot|runtime|compiler|cpu]
  run_head.sh cache clear (--all|--profile e2b|12b|--kind KIND) [--yes]

Defaults: 12b, short, prefill 1024 (long: 128), MTP off, all metrics, warm caches.
Auto residency: resident for short and non-MTP 32k/prefill 128; otherwise lazy.
One model/workload per invocation.
Environment: LITERT_LM_DIR, TENSORRT_RTX_ROOT, CUDA_HOME, LITERT_BENCH_ROOT.
LITERT_BENCH_ALLOW_BACKGROUND_ACTIVITY defaults to 1 (warn); set 0 for strict idle.
Root defaults to ~/.local/state/litert-nvidia-benchmark; no input JSON.
Clear previews by default. Models, source and reports are preserved.
Native TTFT is an estimate. Long presets use synthetic counts, not old ID fixtures.
See run_head.md for build limits, diagnostic host exceptions and migration examples.
HELP
}
die() { echo "ERROR: $*" >&2; exit 2; }
invocation=$(printf '%q ' "$0" "$@")
action=${1:---help}; shift || true
profile=12b; workload=short; prefill=; model=; residency=auto; mtp=false
metrics=all; cache_state=warm; kind=; apply=no; all=no; profile_filter=; cache_action=
case "$action" in
  -h|--help|help) usage; exit 0 ;;
  report) ;;
  cache) cache_action=${1:-}; shift || true; [[ $cache_action == list || $cache_action == clear ]] || die 'Use cache list or cache clear.' ;;
  *) die "Obsolete or unknown action '$action'. Use report; select --metrics verify,latency,memory. See --help." ;;
esac
while (($#)); do
  option=$1; shift
  case "$option" in
    -h|--help) usage; exit 0 ;;
    --mtp) [[ $action == report ]] || die 'Only report accepts --mtp'; mtp=true ;;
    --all|--yes) [[ $action == cache ]] || die "$option is a cache option"; if [[ $option == --all ]]; then all=yes; else apply=yes; fi ;;
    --profile|--model-file|--workload|--prefill|--residency|--metrics|--cache-state|--kind)
      (($#)) && [[ $1 != --* ]] || die "Missing value for $option"
      value=$1; shift
      if [[ $action == cache && $option != --profile && $option != --kind ]]; then die "$option is a report option"; fi
      case "$option" in
        --profile) profile=$value; profile_filter=$value ;;
        --model-file) model=$value ;;
        --workload) workload=$value ;;
        --prefill) prefill=$value ;;
        --residency) residency=$value ;;
        --metrics) metrics=$value ;;
        --cache-state) cache_state=$value ;;
        --kind) [[ $action == cache ]] || die '--kind is a cache option'; kind=$value ;;
      esac ;;
    *) die "Unknown/retired option '$option'. Use --cache-state and --metrics; configure paths through environment. See --help." ;;
  esac
done
[[ $profile == e2b || $profile == 12b ]] || die '--profile must be one of e2b or 12b'
[[ -z $kind || $kind =~ ^(build|aot|runtime|compiler|cpu)$ ]] || die 'Invalid cache kind'
if [[ $action == cache ]]; then
  [[ $cache_action != list || $apply == no ]] || die '--yes is only for cache clear'
  [[ $cache_action != clear || $all == yes || -n $profile_filter || -n $kind ]] || die 'Cache clear requires a selector or --all'
  [[ $all == no || ( -z $profile_filter && -z $kind ) ]] || die 'Use --all by itself, or intersect --profile and --kind'
else
  [[ ${LITERT_BENCH_ALLOW_BACKGROUND_ACTIVITY-1} =~ ^[01]$ ]] || die 'LITERT_BENCH_ALLOW_BACKGROUND_ACTIVITY must be 0 or 1'
  [[ $workload =~ ^(short|32k|128k)$ ]] || die 'Invalid workload'
  [[ $residency =~ ^(auto|lazy|resident)$ ]] || die 'Invalid residency'
  [[ $cache_state =~ ^(warm|runtime-cold|cold|all)$ ]] || die 'Invalid cache state'
  [[ $metrics == all || $metrics =~ ^(verify|latency|memory)(,(verify|latency|memory))*$ ]] || die 'Metrics must be a nonempty subset of verify,latency,memory'
  [[ $profile != e2b || ( $workload == short && $mtp == false ) ]] || die 'E2B supports short, non-MTP presets only'
  prefill=${prefill:-$([[ $workload == short ]] && echo 1024 || echo 128)}
  [[ $prefill == 128 || $prefill == 1024 ]] || die 'Prefill must be 128 or 1024'
  [[ $workload != 128k || $prefill == 128 ]] || die '128k requires prefill 128'
fi

resolve_paths() {
  export NB_SCRIPT=$(realpath "${BASH_SOURCE[0]}")
  export NB_RT=$(dirname "$(dirname "$(dirname "$(dirname "$NB_SCRIPT")")")")
  export NB_LM=$(realpath -m "${LITERT_LM_DIR:-$NB_RT/../../llm/LiteRT-LM}")
  export NB_ROOT=${LITERT_BENCH_ROOT:-$HOME/.local/state/litert-nvidia-benchmark}
  [[ $NB_ROOT == /* ]] || die 'LITERT_BENCH_ROOT must be absolute'
  export NB_SDK=${TENSORRT_RTX_ROOT:-$HOME/opt/tensorrt-rtx-sdk} NB_CUDA=${CUDA_HOME:-/usr/local/cuda}
  NB_SDK=$(realpath -m "$NB_SDK"); NB_CUDA=$(realpath -m "$NB_CUDA")
  local model_name=gemma-4-12B-it
  [[ $profile != e2b ]] || model_name=gemma-4-E2B-it
  export NB_MODEL=$(realpath -m "${model:-$NB_LM/../models/$model_name-litert-lm/$model_name.litertlm}")
  export NB_BUILD_RT=$NB_ROOT/cache/build/litert NB_BUILD_LM=$NB_ROOT/cache/build/litert-lm
}
prerequisites() {
  local command
  for command in python3 flock realpath git; do command -v "$command" >/dev/null || die "Missing prerequisite: $command"; done
  [[ $action == report ]] || return 0
  [[ ( -f $NB_RT/WORKSPACE || -f $NB_RT/MODULE.bazel ) && -f $NB_LM/runtime/engine/litert_lm_advanced_main.cc ]] || die 'Missing LiteRT or LiteRT-LM checkout; set LITERT_LM_DIR'
  [[ -f $NB_MODEL ]] || die "Model missing: $NB_MODEL (use --model-file)"
  [[ -f $NB_SDK/include/NvInfer.h && -f $NB_SDK/lib/libtensorrt_rtx.so && -x $NB_CUDA/bin/nvcc ]] || die 'Missing SDK files; set TENSORRT_RTX_ROOT and CUDA_HOME'
  for command in bazel clang clang++; do command -v "$command" >/dev/null || die "Missing prerequisite: $command"; done
  [[ -x /usr/bin/time ]] || die 'GNU time is required at /usr/bin/time'
  export NB_JOBS=${LITERT_BENCH_JOBS:-8} NB_RAM=${LITERT_BENCH_BUILD_RAM_MB:-8192}
  [[ $NB_JOBS =~ ^[1-9][0-9]*$ && $NB_RAM =~ ^[1-9][0-9]*$ ]] || die 'Invalid build resource limits'
  export NB_MEMORY_LIMIT=${LITERT_BENCH_MEMORY_LIMIT:-}
  if [[ $(hostname | tr '[:upper:]' '[:lower:]') =~ ^cuda(-wsl)?$ ]]; then NB_MEMORY_LIMIT=${NB_MEMORY_LIMIT:-25G}; fi
  if [[ -n $NB_MEMORY_LIMIT ]]; then
    [[ $NB_MEMORY_LIMIT =~ ^[1-9][0-9]*[MGT]$ ]] || die 'Memory limit must be a positive size such as 25G'
    command -v systemd-run >/dev/null || die 'systemd-run required for LITERT_BENCH_MEMORY_LIMIT'
  fi
}
resolve_paths
prerequisites
helper root "$NB_ROOT"
exec 9>"$NB_ROOT/.lock"
flock -n 9 || die 'Benchmark cache is in use; no active run was interrupted'
if [[ $action == cache ]]; then
  helper cache "$NB_ROOT" "$cache_action" "$profile_filter" "$kind" "$apply"
  exit 0
fi

export NB_PROFILE=$profile NB_WORKLOAD=$workload NB_PREFILL=$prefill NB_MTP=$mtp NB_METRICS=$metrics NB_STATE=$cache_state
case "$workload:$prefill" in
  short:*) NB_INPUT=1024; NB_OUTPUT=256; NB_CONTEXT=2048; iterations=8; warmups=2; processes=1 ;;
  32k:128) NB_INPUT=32768; NB_OUTPUT=256; NB_CONTEXT=34818; iterations=4; warmups=1; processes=2 ;;
  32k:1024) NB_INPUT=32768; NB_OUTPUT=256; NB_CONTEXT=33792; iterations=4; warmups=1; processes=2 ;;
  128k:128) NB_INPUT=128001; NB_OUTPUT=128; NB_CONTEXT=130816; iterations=4; warmups=1; processes=1 ;;
esac
NB_LAZY=0
[[ $residency != lazy && ( $residency != auto || $workload == short || ( $workload == 32k && $prefill == 128 && $mtp == false ) ) ]] || NB_LAZY=1
export NB_INPUT NB_OUTPUT NB_CONTEXT NB_LAZY
export NB_REPORT=$NB_ROOT/reports/$(date -u +%Y%m%dT%H%M%S)-$$ NB_CWD=$NB_RT
helper init "$invocation"
finalize() { local rc=$?; trap - EXIT; if ! helper finish "$rc"; then rc=1; fi; echo "Report: $NB_REPORT/report.md"; exit "$rc"; }
trap finalize EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'LiteRT: %s\nLiteRT-LM: %s\nModel: %s\nCUDA: %s\nTensorRT: %s\nBuild outputs: %s, %s\nLogs: %s/logs\n' "$NB_RT" "$NB_LM" "$NB_MODEL" "$NB_CUDA" "$NB_SDK" "$NB_BUILD_RT" "$NB_BUILD_LM" "$NB_REPORT"

build() {
  local label=$1 repo=$2 base=$3; shift 3
  helper entry "$NB_ROOT" "$base" shared build
  export NB_CWD=$repo
  local args=("$(command -v bazel)" --batch "--output_base=$base" build -c opt
    --noincompatible_enable_android_toolchain_resolution "--action_env=CC=$(command -v clang)"
    "--action_env=CXX=$(command -v clang++)" "--jobs=$NB_JOBS" "--local_ram_resources=$NB_RAM"
    --disk_cache= --remote_cache= --remote_executor= --noremote_accept_cached)
  [[ -z ${LITERT_BENCH_DISTDIR:-} ]] || args+=("--distdir=$LITERT_BENCH_DISTDIR")
  echo "Building $label (normal incremental Bazel; empty output base compiles from scratch)"
  helper collect "build_$label" build none 0 0 none none "${args[@]}" "$@"
}
export CUDA_HOME=$NB_CUDA TENSORRT_RTX_ROOT=$NB_SDK
build litert "$NB_RT" "$NB_BUILD_RT" "--repo_env=CUDA_HOME=$NB_CUDA" "--repo_env=TENSORRT_RTX_ROOT=$NB_SDK" \
  //litert/vendors/nvidia/compiler:compiler_plugin_so //litert/vendors/nvidia/dispatch:dispatch_api_so
build litert_lm "$NB_LM" "$NB_BUILD_LM" "--override_repository=litert=$NB_RT" //runtime/engine:litert_lm_advanced_main
compiler=$(realpath "$NB_RT/bazel-bin/litert/vendors/nvidia/compiler/libLiteRtCompilerPlugin_Nvidia.so")
dispatch=$(realpath "$NB_RT/bazel-bin/litert/vendors/nvidia/dispatch/libLiteRtDispatch_Nvidia.so")
engine=$(realpath "$NB_LM/bazel-bin/runtime/engine/litert_lm_advanced_main")
runtime_libs=$NB_BUILD_RT/runtime-libs
mkdir -p "$runtime_libs"
ln -sfn "$compiler" "$runtime_libs/libLiteRtCompilerPlugin_Nvidia.so"
ln -sfn "$dispatch" "$runtime_libs/libLiteRtDispatch_Nvidia.so"
key=$(helper identity "$compiler" "$dispatch" "$engine")
export NB_POOL=$NB_ROOT/cache/models/$profile/$key
printf 'Executable: %s\nNVIDIA libraries: %s\nAOT: %s/aot\nRuntime cache: %s/runtime\nCompiler cache: %s/compiler\nCPU cache: %s/cpu\n' "$engine" "$runtime_libs" "$NB_POOL" "$NB_POOL" "$NB_POOL" "$NB_POOL"
helper host
# Ignore stale legacy tuning variables; every relevant setting is explicit below.
for variable in ${!LITERT_NVIDIA_@}; do unset "$variable"; done
unset LITERT_LM_EXCLUDE_PREFILL_SIGNATURES
export LD_LIBRARY_PATH=$runtime_libs:$NB_LM/prebuilt/linux_x86_64:$NB_SDK/lib:$NB_CUDA/lib64:/usr/lib/wsl/lib
export LITERT_NVIDIA_TENSORRT_PARTITION_POLICY=gemma4 LITERT_NVIDIA_TENSORRT_FP16_ACTIVATIONS=bf16
export LITERT_NVIDIA_TENSORRT_PREDEQUANTIZE_FC_WEIGHTS=cuda_gemv LITERT_NVIDIA_TENSORRT_SHARED_WEIGHTS=1
export LITERT_NVIDIA_TENSORRT_JIT_HANDLE=0 LITERT_NVIDIA_TENSORRT_AOT_MODEL_PATH=$NB_MODEL
export LITERT_NVIDIA_DISPATCH_LAZY_AOT_ENGINES=$NB_LAZY LITERT_NVIDIA_MTP_GPU_SAMPLING=$([[ $mtp == true ]] && echo 1 || echo 0)
export LITERT_NVIDIA_MEMORY_PROFILE=0 LITERT_NVIDIA_DISPATCH_PROFILE=0 LITERT_NVIDIA_DISPATCH_LAYER_PROFILE=0 LITERT_NVIDIA_DISPATCH_DUMP_IO=0
[[ $profile != 12b ]] || export LITERT_NVIDIA_TENSORRT_MAX_FC_WEIGHT_BYTES=536870912
export CUDA_CACHE_DISABLE=0 NB_CWD=$NB_LM
selected=()
# The static executor otherwise chooses E2B's larger graph despite the batch hint.
# Keep the default E2B baseline unfiltered; select graphs for explicit prefill 128.
if [[ $profile == 12b || $prefill == 128 ]]; then
  signatures=prefill_$prefill,decode
  [[ $mtp != true ]] || signatures+=,verify
  selected=("--selected_signatures=$signatures")
fi

cache_paths() {
  local base=$1 aot_override=${2:-} part
  for part in aot runtime compiler cpu; do helper entry "$NB_ROOT" "$base/$part" "$profile" "$part"; done
  export LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR=${aot_override:-$base/aot}
  export LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR=$base/runtime CUDA_CACHE_PATH=$base/compiler/cuda
  compiler_cache=$base/compiler; cpu_cache=$base/cpu
  printf "Cache paths: AOT=%s runtime=%s compiler=%s cpu=%s\n" "$LITERT_NVIDIA_TENSORRT_AOT_CACHE_DIR" "$LITERT_NVIDIA_DISPATCH_RUNTIME_CACHE_DIR" "$compiler_cache" "$cpu_cache"
}
launch() {
  local name=$1 mode=$2 state=$3 count=$4 skip=$5 expected=$6 backend=${7:-npu}
  local context=$NB_CONTEXT cache=$compiler_cache
  if [[ $backend == cpu ]]; then context=2048; cache=$cpu_cache; fi
  local args=("$engine" "--model_path=$NB_MODEL" "--backend=$backend" "--prefill_batch_sizes=$prefill"
    "--max_num_tokens=$context" "--cache_dir=$cache" "--litert_dispatch_lib_dir=$runtime_libs" --min_log_severity=0)
  if [[ $backend == npu ]]; then args+=("${selected[@]}" "--enable_speculative_decoding=$mtp"); fi
  if [[ $mode == verify ]]; then
    args+=(--benchmark=false --benchmark_prefill_tokens=0 --benchmark_decode_tokens=0 --max_output_tokens=16
      '--input_prompt=Answer with only the capital city: What is the capital of France?' --expected_output=Paris)
    if [[ $backend == cpu ]]; then args+=(--enable_speculative_decoding=false); fi
    if [[ $mtp == true && $backend == npu ]]; then args+=(--enable_npu_debug_logging=true); fi
  else
    args+=(--benchmark=true "--benchmark_prefill_tokens=$NB_INPUT" "--benchmark_decode_tokens=$NB_OUTPUT"
      "--max_output_tokens=$NB_OUTPUT" "--num_iterations=$count"
      '--input_prompt=Write one sentence explaining why CUDA is useful for neural network inference:')
  fi
  export LITERT_NVIDIA_MEMORY_PROFILE=$([[ $mode == memory ]] && echo 1 || echo 0)
  echo "$profile $workload prefill=$prefill MTP=$mtp: $name"
  helper collect "$name" "$mode" "$state" "$count" "$skip" "$expected" "$backend" "${args[@]}"
}
prepare_pool() {
  cache_paths "$NB_POOL"
  if [[ ! -f $NB_POOL/compiler/.warm-ready ]] || ! compgen -G "$NB_POOL/aot/tensorrt_aot*.bin" >/dev/null || ! compgen -G "$NB_POOL/runtime/*.trt_rtx_runtime_cache" >/dev/null; then
    launch prepare_warm prepare prepare 1 0 any
    touch "$NB_POOL/compiler/.warm-ready"
  fi
}
if [[ $mtp == true ]]; then
  cache_paths "$NB_POOL"
  launch mtp_capability verify prepare 0 0 any
fi
metrics=${metrics/all/verify,latency,memory}
states=($cache_state)
[[ $cache_state != all ]] || states=(warm runtime-cold cold)
if [[ $cache_state == all ]]; then echo 'Cold latency, verification and memory use separate empty caches and each compiles AOT. Warm preparation may also compile on a miss.'; fi
for state in "${states[@]}"; do
  [[ $state == cold ]] || prepare_pool
  for metric in verify latency memory; do
    [[ ,$metrics, == *,$metric,* ]] || continue
    repeats=1; [[ $metric != latency ]] || repeats=$processes
    for ((repeat=0; repeat<repeats; repeat++)); do
      name=${state}_${metric}_${repeat}
      expected=hit
      if [[ $state == warm ]]; then cache_paths "$NB_POOL"
      else
        scratch=$NB_ROOT/cache/scratch/$(basename "$NB_REPORT")/$name
        if [[ $state == cold ]]; then cache_paths "$scratch"; expected=miss
        else cache_paths "$scratch" "$NB_POOL/aot"; fi
      fi
      case "$metric" in
        verify) launch "${name}_npu" verify "$state" 0 0 "$expected"; launch "${name}_cpu" verify "$state" 0 0 none cpu ;;
        latency) launch "$name" latency "$state" "$iterations" "$warmups" "$expected" ;;
        memory) launch "$name" memory "$state" 1 0 "$expected" ;;
      esac
    done
  done
done
)

