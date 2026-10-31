// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! A capture: every probe of a build, under $LITERT_BENCH_ROOT/guard/<name>,
//! resumable (a probe that already ran is kept) and summarized in capture.json.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::{workload_prompt, Config};
use crate::env::BenchEnv;
use crate::numerics::NumericsRun;
use crate::perf::PerfPoint;
use crate::runner::{
    load_numerics_run, numerics_dir, perf_label, perf_run_complete, run_numerics_probe,
    run_perf_probe, NumericsProbe, PerfProbe,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub created: String,
    pub updated: String,
    pub build_id: String,
    pub rt_head: String,
    pub lm_head: String,
    pub tier: String,
    pub models: Vec<String>,
    pub mtp: bool,
    /// CI_HOST_MODE of the capture (formal after the host's benchmark
    /// preparation, else diagnostic): timings compare within one mode.
    #[serde(default = "default_host_mode")]
    pub host_mode: String,
}

fn default_host_mode() -> String {
    std::env::var("CI_HOST_MODE").unwrap_or_else(|_| "diagnostic".to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capture {
    pub manifest: Manifest,
    pub numerics: Vec<NumericsRun>,
    pub perf: Vec<PerfPoint>,
    #[serde(skip)]
    pub dir: PathBuf,
}

pub struct CaptureOptions {
    pub name: String,
    pub models: Vec<String>,
    pub tier: String,
    pub numerics: bool,
    pub perf: bool,
    pub mtp: bool,
}

pub fn guard_root(env: &BenchEnv) -> PathBuf {
    env.root.join("guard")
}

pub fn capture_dir(env: &BenchEnv, name: &str) -> PathBuf {
    guard_root(env).join(name)
}

fn now() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

pub fn load_capture(env: &BenchEnv, name: &str) -> Result<Capture> {
    let dir = capture_dir(env, name);
    let path = dir.join("capture.json");
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("no capture {name} ({})", path.display()))?;
    let mut capture: Capture = serde_json::from_str(&text).context("parsing capture.json")?;
    capture.dir = dir;
    Ok(capture)
}

pub fn list_captures(env: &BenchEnv) -> Vec<Manifest> {
    let mut manifests = Vec::new();
    if let Ok(entries) = std::fs::read_dir(guard_root(env)) {
        for entry in entries.flatten() {
            if let Ok(text) = std::fs::read_to_string(entry.path().join("capture.json")) {
                if let Ok(capture) = serde_json::from_str::<Capture>(&text) {
                    manifests.push(capture.manifest);
                }
            }
        }
    }
    manifests.sort_by(|a, b| a.updated.cmp(&b.updated));
    manifests
}

fn save(capture: &Capture) -> Result<()> {
    std::fs::create_dir_all(&capture.dir)?;
    std::fs::write(capture.dir.join("capture.json"), serde_json::to_string_pretty(capture)?)?;
    Ok(())
}

/// Runs (or resumes) a capture of the current build.
pub fn run_capture(
    env: &BenchEnv,
    config: &Config,
    guard_dir: &Path,
    opts: &CaptureOptions,
) -> Result<Capture> {
    if !env.built() {
        bail!("no built CLI under {} (run build.sh first)", env.runtime_libs.display());
    }
    let dir = capture_dir(env, &opts.name);
    let mut capture = match load_capture(env, &opts.name) {
        Ok(existing) => {
            if existing.manifest.build_id != env.build_id {
                bail!(
                    "capture {} holds build {} but the trees are at {}; use another --name",
                    opts.name,
                    existing.manifest.build_id,
                    env.build_id
                );
            }
            existing
        }
        Err(_) => Capture {
            manifest: Manifest {
                name: opts.name.clone(),
                created: now(),
                updated: now(),
                build_id: env.build_id.clone(),
                rt_head: env.git_head(&env.rt),
                lm_head: env.git_head(&env.lm),
                tier: opts.tier.clone(),
                models: opts.models.clone(),
                mtp: opts.mtp,
                host_mode: default_host_mode(),
            },
            numerics: Vec::new(),
            perf: Vec::new(),
            dir: dir.clone(),
        },
    };
    for model in &opts.models {
        if !capture.manifest.models.contains(model) {
            capture.manifest.models.push(model.clone());
        }
    }
    capture.manifest.mtp |= opts.mtp;
    if crate::config::tier_rank(&opts.tier)? > crate::config::tier_rank(&capture.manifest.tier)? {
        capture.manifest.tier = opts.tier.clone();
    }
    save(&capture)?;

    for model in &opts.models {
        if opts.numerics {
            for prompt in config.prompts_for(model, &opts.tier)? {
                let modes: Vec<bool> = if opts.mtp { vec![false, true] } else { vec![false] };
                for mtp in modes {
                    let out_dir = numerics_dir(&dir, model, &prompt.name, mtp);
                    let existing = load_numerics_run(&out_dir)
                        .filter(|r| r.exit_code == 0 && !r.steps.is_empty());
                    let run = match existing {
                        Some(run) => {
                            eprintln!(
                                "numerics {model} {}{}: kept",
                                prompt.name,
                                if mtp { " (mtp)" } else { "" }
                            );
                            run
                        }
                        None => {
                            eprintln!(
                                "numerics {model} {}{}: running",
                                prompt.name,
                                if mtp { " (mtp)" } else { "" }
                            );
                            let prompt_path = config.prompt_path(guard_dir, prompt);
                            let run = run_numerics_probe(
                                env,
                                &NumericsProbe {
                                    model,
                                    prompt,
                                    prompt_path: &prompt_path,
                                    mtp,
                                    top_k: config.numerics.top_k,
                                    out_dir: &out_dir,
                                },
                            )?;
                            eprintln!(
                                "  exit {} in {:.0} s: {} steps, {} decode tokens at {:.1} tok/s{}",
                                run.exit_code,
                                run.wall_s,
                                run.steps.len(),
                                run.decode_tokens,
                                run.decode_tps.unwrap_or(0.0),
                                if run.loopiness >= 0.25 { ", LOOPS" } else { "" }
                            );
                            run
                        }
                    };
                    capture.numerics.retain(|r| {
                        !(r.model == run.model && r.prompt == run.prompt && r.mtp == run.mtp)
                    });
                    capture.numerics.push(run);
                    capture.manifest.updated = now();
                    save(&capture)?;
                }
            }
        }
        if opts.perf {
            let perf_dir = dir.join("perf").join(model);
            let mut plan: Vec<(String, bool, Option<PathBuf>, u32)> = Vec::new();
            for workload in &config.perf.workloads {
                let iterations = *config.perf.iterations.get(workload).unwrap_or(&2);
                plan.push((workload.clone(), false, None, iterations));
                if opts.mtp {
                    if let Some(file) = workload_prompt(workload) {
                        let prompt = env.bench_dir.join("prompts").join(file);
                        plan.push((
                            workload.clone(),
                            false,
                            Some(prompt.clone()),
                            iterations.min(3),
                        ));
                        plan.push((workload.clone(), true, Some(prompt), iterations.min(3)));
                    }
                }
            }
            for (workload, mtp, prompt_file, iterations) in plan {
                let label = perf_label(model, &workload, mtp, prompt_file.is_some());
                let run_dir = perf_dir.join(&label);
                let point = match perf_run_complete(&run_dir) {
                    Some(point) => {
                        eprintln!("perf {label}: kept");
                        point
                    }
                    None => {
                        eprintln!("perf {label}: running ({iterations} iterations)");
                        let point = run_perf_probe(
                            env,
                            &PerfProbe {
                                model,
                                workload: &workload,
                                mtp,
                                prompt_file: prompt_file.as_deref(),
                                iterations,
                                out_dir: &perf_dir,
                                label: &label,
                            },
                        )?;
                        eprintln!(
                            "  decode {:.1} tok/s, prefill {:.0} tok/s{}",
                            point.decode_tps.unwrap_or(0.0),
                            point.prefill_tps.unwrap_or(0.0),
                            point
                                .acceptance
                                .map(|a| format!(", acceptance {:.0}%", a * 100.0))
                                .unwrap_or_default()
                        );
                        point
                    }
                };
                capture.perf.retain(|p| !(p.model == point.model && p.label == point.label));
                capture.perf.push(point);
                capture.manifest.updated = now();
                save(&capture)?;
            }
        }
    }
    Ok(capture)
}
