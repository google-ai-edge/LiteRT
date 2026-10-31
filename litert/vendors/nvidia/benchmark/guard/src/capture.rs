// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! A capture: every probe of a build, under $LITERT_BENCH_ROOT/guard/<name>,
//! resumable (a probe that already ran is kept) and summarized in capture.json.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::autopsy::Autopsy;
use crate::config::{workload_prompt, Config};
use crate::env::BenchEnv;
use crate::numerics::NumericsRun;
use crate::perf::PerfPoint;
use crate::runner::{
    autopsy_dir, load_autopsy, load_numerics_run, load_text_run, numerics_dir, perf_label,
    perf_run_complete, rescan_numerics_run, run_autopsy_probe, run_numerics_probe, run_perf_probe,
    run_text_probe, text_dir, AutopsyProbe, NumericsProbe, PerfProbe, TextProbe,
};
use crate::texts::TextRun;

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
    /// The extra environment of every probe (capture --env), and the pool
    /// suffix its probes compiled under.
    #[serde(default)]
    pub env: Vec<(String, String)>,
    #[serde(default)]
    pub pool_suffix: String,
    /// clock_check.py's report when the capture last ran (null: not checked).
    #[serde(default)]
    pub host_clock: Option<serde_json::Value>,
    /// Whether that report found the clock fit for timings.
    #[serde(default)]
    pub clock_fit: Option<bool>,
    /// The build the binaries came from (runtime-libs/BUILD_ID), when stamped.
    #[serde(default)]
    pub binaries_build_id: Option<String>,
    /// What the capture did differently and why (a skipped probe family).
    #[serde(default)]
    pub notes: Vec<String>,
}

fn default_host_mode() -> String {
    std::env::var("CI_HOST_MODE").unwrap_or_else(|_| "diagnostic".to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capture {
    pub manifest: Manifest,
    pub numerics: Vec<NumericsRun>,
    pub perf: Vec<PerfPoint>,
    /// The sampled and constrained decodes, device and host.
    #[serde(default)]
    pub texts: Vec<TextRun>,
    /// The per-kernel autopsies, one per model.
    #[serde(default)]
    pub autopsies: Vec<Autopsy>,
    #[serde(skip)]
    pub dir: PathBuf,
}

pub struct CaptureOptions {
    pub name: String,
    pub models: Vec<String>,
    pub tier: String,
    pub numerics: bool,
    pub perf: bool,
    pub texts: bool,
    pub autopsy: bool,
    pub mtp: bool,
    /// Run the timed probes (performance, autopsy) on a clock clock_check.py
    /// finds unfit; they are noted as such.
    pub allow_unfit_clock: bool,
}

pub fn guard_root(env: &BenchEnv) -> PathBuf {
    env.root.join("guard")
}

pub fn capture_dir(env: &BenchEnv, name: &str) -> PathBuf {
    guard_root(env).join(name)
}

/// A build id with '+' is a dirty tree's: env.sh marks any uncommitted
/// tracked change so, hashing nothing of the diff, so the id is the same
/// string for every edit of the same heads and cannot name a capture by
/// itself.
pub fn dirty_build_id(build_id: &str) -> bool {
    build_id.contains('+')
}

/// A capture name is one path component of letters, digits, '.', '_', '-'
/// and '+' (a dirty tree's build id carries one), not starting with '.': it
/// becomes the capture's directory and the dumps' `<dir>:<steps>:<mb>` spec,
/// where a ':' or a '/' would misroute.
pub fn validate_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' || c == '+');
    if !ok {
        bail!("capture name {name:?}: use letters, digits, '.', '_', '-' and '+' only, not starting with '.'");
    }
    Ok(())
}

fn now() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

pub fn load_capture(env: &BenchEnv, name: &str) -> Result<Capture> {
    validate_name(name)?;
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
    validate_name(&opts.name)?;
    // The binaries must be the trees' build: a capture names its build id.
    let stamp = env.binaries_build_id();
    match &stamp {
        Some(stamp) if stamp != &env.build_id => bail!(
            "the binaries in {} are build {stamp} but the trees are at {}: run build.sh, or capture from the trees they were built from",
            env.runtime_libs.display(),
            env.build_id
        ),
        None => eprintln!("warning: no BUILD_ID stamp in {} (a build.sh before the stamp): the binaries' build is not checked", env.runtime_libs.display()),
        _ => {}
    }
    let dir = capture_dir(env, &opts.name);
    let mut capture = match load_capture(env, &opts.name) {
        Ok(existing) => {
            if dirty_build_id(&env.build_id) {
                eprintln!(
                    "capture {} resumes under the dirty build id {}: an edit since its probes ran is not told apart (use a new name after each edit)",
                    opts.name, env.build_id
                );
            }
            if existing.manifest.env != env.extra_env {
                bail!(
                    "capture {} ran with the extra environment {:?}, this run has {:?}; use another --name",
                    opts.name,
                    existing.manifest.env,
                    env.extra_env
                );
            }
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
                env: env.extra_env.clone(),
                pool_suffix: env.pool_suffix.clone(),
                host_clock: None,
                clock_fit: None,
                binaries_build_id: stamp.clone(),
                notes: Vec::new(),
            },
            numerics: Vec::new(),
            perf: Vec::new(),
            texts: Vec::new(),
            autopsies: Vec::new(),
            dir: dir.clone(),
        },
    };
    for model in &opts.models {
        if !capture.manifest.models.contains(model) {
            capture.manifest.models.push(model.clone());
        }
    }
    capture.manifest.mtp |= opts.mtp;
    capture.manifest.binaries_build_id = stamp;
    // The host's clock: the timed probes (performance, autopsy) need it fit.
    let mut timed_ok = true;
    match env.host_clock(3) {
        Ok((report, fit)) => {
            if !fit {
                let reasons = report.get("reasons").map(|r| r.to_string()).unwrap_or_default();
                if opts.allow_unfit_clock {
                    let note = format!("timed probes ran on an unfit clock ({reasons}): their numbers are diagnostic");
                    eprintln!("warning: {note}");
                    capture.manifest.notes.push(note);
                } else if opts.perf || opts.autopsy {
                    let note = format!("timed probes skipped: the host's clock is unfit ({reasons}); --allow-unfit-clock runs them as diagnostic");
                    eprintln!("{note}");
                    capture.manifest.notes.push(note);
                    timed_ok = false;
                }
            }
            capture.manifest.host_clock = Some(report);
            capture.manifest.clock_fit = Some(fit);
        }
        Err(error) => {
            let note = format!("the host's clock was not checked: {error}");
            eprintln!("warning: {note}");
            capture.manifest.notes.push(note);
        }
    }
    capture.manifest.notes.dedup();
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
                                    full_steps: config.numerics.full_steps,
                                    outputs_max_mb: config.numerics.outputs_max_mb,
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
        if opts.texts {
            let sampled = &config.probes.sampled;
            let constraint = &config.probes.constraint;
            let sampled_prompt = config.prompt_path(guard_dir, config.prompt(&sampled.prompt)?);
            let constraint_prompt = dir.join("texts").join("constraint_prompt.txt");
            std::fs::create_dir_all(constraint_prompt.parent().unwrap())?;
            std::fs::write(&constraint_prompt, &constraint.prompt_text)?;
            for kind in ["sampled", "constraint"] {
                // The constraint also runs the device verifier in the
                // immediate-upload mode: the fused argmax after a host write
                // is what that mode must get right.
                let variants: &[&str] = if kind == "constraint" {
                    &["device", "host", "immediate"]
                } else {
                    &["device", "host"]
                };
                for &variant in variants {
                    let out_dir = text_dir(&dir, model, kind, variant);
                    let run = match load_text_run(&out_dir)
                        .filter(|r| r.exit_code == 0 && r.build_id == env.build_id)
                    {
                        Some(run) => {
                            eprintln!("text {model} {kind} {variant}: kept");
                            run
                        }
                        None => {
                            eprintln!("text {model} {kind} {variant}: running");
                            let probe = if kind == "sampled" {
                                TextProbe {
                                    model,
                                    kind,
                                    variant,
                                    prompt_path: &sampled_prompt,
                                    max_output_tokens: sampled.max_output_tokens,
                                    context: sampled.context,
                                    top_k: sampled.top_k,
                                    top_p: sampled.top_p,
                                    temperature: sampled.temperature,
                                    seed: sampled.seed,
                                    regex: None,
                                    repetition_penalty: None,
                                    out_dir: &out_dir,
                                }
                            } else {
                                TextProbe {
                                    model,
                                    kind,
                                    variant,
                                    prompt_path: &constraint_prompt,
                                    max_output_tokens: constraint.max_output_tokens,
                                    context: constraint.context,
                                    top_k: constraint.top_k,
                                    top_p: constraint.top_p,
                                    temperature: constraint.temperature,
                                    seed: constraint.seed,
                                    regex: Some(&constraint.regex),
                                    repetition_penalty: Some(constraint.repetition_penalty),
                                    out_dir: &out_dir,
                                }
                            };
                            let run = run_text_probe(env, &probe)?;
                            eprintln!(
                                "  exit {} in {:.0} s: {} chars, device samplers {}{}",
                                run.exit_code,
                                run.wall_s,
                                run.text_len,
                                run.device_samplers,
                                match run.regex_ok {
                                    Some(true) => ", regex ok",
                                    Some(false) => ", REGEX VIOLATED",
                                    None => "",
                                }
                            );
                            run
                        }
                    };
                    capture.texts.retain(|r| {
                        !(r.model == run.model && r.kind == run.kind && r.variant == run.variant)
                    });
                    capture.texts.push(run);
                    capture.manifest.updated = now();
                    save(&capture)?;
                }
            }
        }
        if opts.autopsy && timed_ok {
            let out_dir = autopsy_dir(&dir, model);
            let autopsy = match load_autopsy(&out_dir) {
                Some(a) => {
                    eprintln!("autopsy {model}: kept");
                    a
                }
                None => {
                    let spec = &config.probes.autopsy;
                    eprintln!(
                        "autopsy {model}: running ({}{})",
                        spec.workload,
                        if spec.mtp { ", mtp" } else { "" }
                    );
                    let prompt_file = if spec.mtp {
                        workload_prompt(&spec.workload)
                            .map(|f| env.bench_dir.join("prompts").join(f))
                    } else {
                        None
                    };
                    let autopsy = run_autopsy_probe(
                        env,
                        &AutopsyProbe {
                            model,
                            workload: &spec.workload,
                            mtp: spec.mtp,
                            prompt_file: prompt_file.as_deref(),
                            out_dir: &out_dir,
                        },
                    )?;
                    for f in &autopsy.functions {
                        eprintln!(
                            "  {}: {} kernels, warm {:.0} us, cold {:.0} us",
                            f.function,
                            f.kernels.len(),
                            f.warm_us,
                            f.cold_us
                        );
                    }
                    autopsy
                }
            };
            capture.autopsies.retain(|a| a.model != autopsy.model);
            capture.autopsies.push(autopsy);
            capture.manifest.updated = now();
            save(&capture)?;
        }
        if opts.perf && timed_ok {
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

#[cfg(test)]
mod tests {
    use super::{dirty_build_id, validate_name};

    #[test]
    fn a_dirty_build_id_is_told_apart() {
        assert!(dirty_build_id("44cdec619+-95cd9c0b"));
        assert!(dirty_build_id("44cdec619-95cd9c0b+"));
        assert!(!dirty_build_id("44cdec619-95cd9c0b"));
    }

    #[test]
    fn capture_names_are_one_plain_path_component() {
        for ok in [
            "3c590a6e1-95cd9c0b",
            "44cdec619+-95cd9c0b",
            "1261dad0a+-75a399d7+",
            "scratch_div",
            "rebuild.2",
            "A-b_c",
        ] {
            assert!(validate_name(ok).is_ok(), "{ok}");
        }
        for bad in ["", "a:b", "a/b", "../x", ".hidden", "a b", "2026-10-27T21:42"] {
            assert!(validate_name(bad).is_err(), "{bad}");
        }
    }
}

/// Recomputes every numerics summary of a capture from its files.
pub fn rescan_capture(env: &BenchEnv, name: &str) -> Result<Capture> {
    let mut capture = load_capture(env, name)?;
    let mut rescanned = Vec::new();
    for run in &capture.numerics {
        let dir = numerics_dir(&capture.dir, &run.model, &run.prompt, run.mtp);
        match rescan_numerics_run(&dir)? {
            Some(updated) => rescanned.push(updated),
            None => rescanned.push(run.clone()),
        }
    }
    capture.numerics = rescanned;
    capture.manifest.updated = now();
    save(&capture)?;
    Ok(capture)
}
