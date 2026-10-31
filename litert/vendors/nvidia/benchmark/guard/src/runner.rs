// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! Runs the probes: the LiteRT-LM CLI on a prompt with the dispatch's logits
//! dump (numerics), and bench.sh for a workload (performance). Both run in
//! the model's standard cache pool of the current build, so a pool compiled
//! by one probe serves the other.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::config::PromptSpec;
use crate::env::BenchEnv;
use crate::numerics::{loopiness, NumericsRun, StepLogits};
use crate::parse::{parse_cli_log, parse_logits_jsonl};
use crate::perf::{summarize_run, PerfPoint};

/// Plain runs of the larger models compile only the signatures they run; an
/// MTP run compiles every signature (bench.sh's rule).
fn selected_signatures(model: &str, mtp: bool) -> Option<&'static str> {
    if !mtp && (model == "12b" || model == "26b") {
        Some("--selected_signatures=prefill_1024,decode")
    } else {
        None
    }
}

fn fnv1a(text: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

pub struct NumericsProbe<'a> {
    pub model: &'a str,
    pub prompt: &'a PromptSpec,
    pub prompt_path: &'a Path,
    pub mtp: bool,
    pub top_k: u32,
    pub out_dir: &'a Path,
}

/// Runs one numerics probe and summarizes it (summary.json in out_dir).
pub fn run_numerics_probe(env: &BenchEnv, probe: &NumericsProbe) -> Result<NumericsRun> {
    if !env.built() {
        bail!("no built CLI under {} (run build.sh)", env.runtime_libs.display());
    }
    if !probe.prompt_path.is_file() {
        bail!("prompt file missing: {}", probe.prompt_path.display());
    }
    if env.gpu_busy()? {
        bail!("another GPU compute process is running");
    }
    std::fs::create_dir_all(probe.out_dir)?;
    let logits_path = probe.out_dir.join("logits.jsonl");
    let _ = std::fs::remove_file(&logits_path);
    let model_file = env.model_file(probe.model)?;
    let pool = env.pool(probe.model);
    let mut envs = env.runtime_env(&pool, &model_file)?;
    envs.retain(|(k, _)| {
        k != "LITERT_NVIDIA_DISPATCH_DUMP_IO" && k != "LITERT_NVIDIA_MEMORY_PROFILE"
    });
    envs.push((
        "LITERT_NVIDIA_DISPATCH_DUMP_IO".into(),
        format!("top{}:{}", probe.top_k, logits_path.display()),
    ));
    envs.push(("LITERT_NVIDIA_MEMORY_PROFILE".into(), "0".into()));
    let mut args: Vec<String> = vec![
        format!("--model_path={}", model_file.display()),
        "--backend=npu".into(),
        "--prefill_batch_sizes=1024".into(),
        format!("--max_num_tokens={}", probe.prompt.context),
        format!("--cache_dir={}", pool.join("compiler").display()),
        format!("--litert_dispatch_lib_dir={}", env.runtime_libs.display()),
        "--min_log_severity=0".into(),
    ];
    if let Some(selected) = selected_signatures(probe.model, probe.mtp) {
        args.push(selected.into());
    }
    args.extend([
        format!("--enable_speculative_decoding={}", probe.mtp),
        "--benchmark=true".into(),
        "--benchmark_prefill_tokens=0".into(),
        "--benchmark_decode_tokens=0".into(),
        "--num_iterations=1".into(),
        format!("--max_output_tokens={}", probe.prompt.max_output_tokens),
        format!("--input_prompt_file={}", probe.prompt_path.display()),
        // Greedy decoding whatever the model's own sampler: the probe must be
        // deterministic, and k = 1 with p = 0 is the fixed greedy sampler the
        // NVIDIA MTP path keeps on the device.
        "--sampler_top_k=1".into(),
        "--sampler_top_p=0".into(),
        "--sampler_temperature=1".into(),
    ]);
    let compiles = !env.pool_has_engines(&pool);
    if compiles {
        eprintln!("  (the pool {} has no engines yet: this run compiles)", pool.display());
    }
    let stdout = File::create(probe.out_dir.join("response.txt"))?;
    let stderr = File::create(probe.out_dir.join("cli.log"))?;
    let start = Instant::now();
    let status = Command::new(env.cli())
        .args(&args)
        .current_dir(&env.lm)
        .env_clear()
        .envs(envs.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr)
        .status()
        .with_context(|| format!("running {}", env.cli().display()))?;
    let wall_s = start.elapsed().as_secs_f64();
    let exit_code = status.code().unwrap_or(-1);

    let log_text = std::fs::read_to_string(probe.out_dir.join("cli.log")).unwrap_or_default();
    let log = parse_cli_log(&log_text);
    let text = std::fs::read_to_string(probe.out_dir.join("response.txt")).unwrap_or_default();
    let lines = match std::fs::read_to_string(&logits_path) {
        Ok(t) => parse_logits_jsonl(&t)?,
        Err(_) => Vec::new(),
    };
    let mut steps = Vec::new();
    let mut nonfinite = false;
    for line in &lines {
        for (row, top) in line.top.iter().enumerate() {
            nonfinite |= top.iter().any(|(_, v)| !v.is_finite());
            steps.push(StepLogits {
                context: line.context,
                function: line.function.clone(),
                step: line.step,
                row,
                top: top.clone(),
            });
        }
    }
    let iteration = log.iterations.last();
    let run = NumericsRun {
        model: probe.model.to_string(),
        prompt: probe.prompt.name.clone(),
        mtp: probe.mtp,
        build_id: env.build_id.clone(),
        exit_code,
        wall_s,
        tokens: steps.iter().filter_map(|s| s.argmax()).collect(),
        steps,
        decode_tokens: iteration.and_then(|i| i.decode.first()).map(|t| t.tokens).unwrap_or(0),
        decode_tps: iteration.and_then(|i| i.decode.first()).map(|t| t.tokens_per_second),
        prefill_tokens: iteration.and_then(|i| i.prefill.first()).map(|t| t.tokens).unwrap_or(0),
        prefill_tps: iteration.and_then(|i| i.prefill.first()).map(|t| t.tokens_per_second),
        acceptance: iteration.and_then(|i| i.mtp.as_ref()).map(|m| m.acceptance),
        text_len: text.chars().count(),
        text_sha256: fnv1a(&text),
        loopiness: loopiness(&text),
        nonfinite,
    };
    let meta = json!({
        "command": std::iter::once(env.cli().display().to_string()).chain(args.iter().cloned()).collect::<Vec<_>>(),
        "environment": envs.iter().filter(|(k, _)| k.starts_with("LITERT_NVIDIA_")).cloned().collect::<std::collections::BTreeMap<_, _>>(),
        "build_id": env.build_id,
        "pool": pool.display().to_string(),
        "compiled": compiles,
        "exit_code": exit_code,
        "wall_s": wall_s,
        "sampler_lines": log.sampler_lines,
        "aot_hits": log.aot_hits,
        "aot_misses": log.aot_misses,
    });
    std::fs::write(probe.out_dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
    std::fs::write(probe.out_dir.join("summary.json"), serde_json::to_string_pretty(&run)?)?;
    Ok(run)
}

pub fn load_numerics_run(out_dir: &Path) -> Option<NumericsRun> {
    let text = std::fs::read_to_string(out_dir.join("summary.json")).ok()?;
    serde_json::from_str(&text).ok()
}

pub struct PerfProbe<'a> {
    pub model: &'a str,
    pub workload: &'a str,
    pub mtp: bool,
    pub prompt_file: Option<&'a Path>,
    pub iterations: u32,
    pub out_dir: &'a Path,
    pub label: &'a str,
}

/// Runs one bench.sh latency measurement and summarizes its run directory.
pub fn run_perf_probe(env: &BenchEnv, probe: &PerfProbe) -> Result<PerfPoint> {
    std::fs::create_dir_all(probe.out_dir)?;
    let mut args: Vec<String> = vec![
        "--model".into(),
        probe.model.into(),
        "--out".into(),
        probe.out_dir.display().to_string(),
        "--workload".into(),
        probe.workload.into(),
        "--metric".into(),
        "latency".into(),
        "--iterations".into(),
        probe.iterations.to_string(),
        "--label".into(),
        probe.label.into(),
    ];
    if probe.mtp {
        args.push("--mtp".into());
    }
    if let Some(prompt) = probe.prompt_file {
        args.push("--prompt-file".into());
        args.push(prompt.display().to_string());
    }
    let bench_log = File::create(probe.out_dir.join(format!("{}.bench.log", probe.label)))?;
    let status = Command::new(env.bench_dir.join("bench.sh"))
        .args(&args)
        .env("LITERT_RT_DIR", &env.rt)
        .stdin(Stdio::null())
        .stdout(bench_log.try_clone()?)
        .stderr(bench_log)
        .status()
        .context("running bench.sh")?;
    let run_dir = probe.out_dir.join(probe.label);
    if !status.success() {
        bail!("bench.sh failed for {} ({}); see {}", probe.label, status, run_dir.display());
    }
    summarize_run(&run_dir)
}

pub fn perf_run_complete(run_dir: &Path) -> Option<PerfPoint> {
    let point = summarize_run(run_dir).ok()?;
    (point.exit_code == 0 && point.measured_iterations > 0 && point.decode_tps.is_some())
        .then_some(point)
}

pub fn perf_label(model: &str, workload: &str, mtp: bool, text: bool) -> String {
    let mode = if mtp { "mtp" } else { "plain" };
    let kind = if text { "text" } else { "latency" };
    format!("{model}_{workload}_{mode}_{kind}")
}

pub fn numerics_dir(capture_dir: &Path, model: &str, prompt: &str, mtp: bool) -> PathBuf {
    let suffix = if mtp { "_mtp" } else { "" };
    capture_dir.join("numerics").join(model).join(format!("{prompt}{suffix}"))
}
