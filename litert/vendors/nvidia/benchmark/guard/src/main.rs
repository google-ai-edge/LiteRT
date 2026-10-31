// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! nvguard: does a change to LiteRT's NVIDIA backend regress numerics or
//! performance? A capture runs cheap probes of the current build (greedy
//! decodes with per-step logits; short benchmark runs) into
//! $LITERT_BENCH_ROOT/guard/<name>; a check captures and compares against a
//! baseline capture, with a verdict per cell and an exit code.

mod autopsy;
mod capture;
mod config;
mod dumps;
mod env;
mod numerics;
mod parse;
mod perf;
mod report;
mod runner;
mod texts;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use crate::capture::{list_captures, load_capture, run_capture, CaptureOptions};
use crate::config::Config;
use crate::env::BenchEnv;

#[derive(Parser)]
#[command(
    name = "nvguard",
    version,
    about = "Numerics and performance guard of LiteRT's NVIDIA backend"
)]
struct Cli {
    /// The benchmark directory (litert/vendors/nvidia/benchmark of the LiteRT
    /// tree); default: the one this crate sits in.
    #[arg(long, global = true)]
    bench: Option<PathBuf>,
    /// A configuration file replacing the built-in nvguard.toml.
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Args)]
struct ProbeArgs {
    /// Capture name (default: the build id).
    #[arg(long)]
    name: Option<String>,
    /// Models to probe (comma separated; default: the configuration's).
    #[arg(long, value_delimiter = ',')]
    models: Option<Vec<String>>,
    /// quick | standard | long (cumulative prompt tiers).
    #[arg(long, default_value = "quick")]
    tier: String,
    /// Skip the numerics probes.
    #[arg(long)]
    no_numerics: bool,
    /// Skip the performance probes.
    #[arg(long)]
    no_perf: bool,
    /// Skip the text probes (the sampled and the constrained decode, with the
    /// MTP verifier sampling on the device and on the host).
    #[arg(long)]
    no_texts: bool,
    /// Skip the per-kernel autopsy.
    #[arg(long)]
    no_autopsy: bool,
    /// Also probe speculative decoding (MTP): greedy MTP numerics, and the
    /// plain and MTP text runs of each workload.
    #[arg(long)]
    mtp: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Run the probes of the current build into a capture (resumable).
    Capture(ProbeArgs),
    /// Capture the current build and compare it with a baseline capture.
    Check {
        #[command(flatten)]
        probe: ProbeArgs,
        /// The baseline capture's name.
        #[arg(long)]
        baseline: String,
        /// A record of full measurements (measurements' bench/results.json or
        /// a report.py --json file) for the predictions.
        #[arg(long)]
        record: Option<PathBuf>,
    },
    /// Compare two existing captures.
    Compare {
        baseline: String,
        candidate: String,
        #[arg(long)]
        record: Option<PathBuf>,
    },
    /// A capture's tables, cost models and predictions; with --record, the
    /// predictor's error on the record.
    Predict {
        name: String,
        #[arg(long)]
        record: Option<PathBuf>,
    },
    /// Where a candidate capture parts from a baseline: the first output
    /// tensor that differs (step, partition, output), the KL of the first
    /// logits rows over the whole vocabulary, the kernels whose times moved,
    /// and the sampled texts' parting. Reads the captures' dumps; runs nothing.
    Localize {
        /// The baseline capture's name.
        #[arg(long)]
        baseline: String,
        /// The candidate capture's name.
        name: String,
    },
    /// Recompute a capture's numerics summaries from its files (after a
    /// change of the metrics).
    Rescan { name: String },
    /// The captures under $LITERT_BENCH_ROOT/guard.
    List,
    /// The resolved benchmark environment.
    Env,
    /// The built-in configuration.
    Config,
}

fn guard_dir_default() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn resolve(cli: &Cli) -> Result<(BenchEnv, Config, PathBuf)> {
    let guard_dir = guard_dir_default();
    let bench_dir = cli.bench.clone().unwrap_or_else(|| {
        guard_dir.parent().map(Path::to_path_buf).unwrap_or_else(|| guard_dir.clone())
    });
    let env = BenchEnv::resolve(&bench_dir)?;
    let config = Config::load(cli.config.as_deref())?;
    Ok((env, config, guard_dir))
}

fn capture_options(probe: &ProbeArgs, env: &BenchEnv, config: &Config) -> Result<CaptureOptions> {
    let models = probe.models.clone().unwrap_or_else(|| config.models.clone());
    if models.is_empty() {
        bail!("no models to probe");
    }
    config::tier_rank(&probe.tier)?;
    let name = match &probe.name {
        Some(name) => name.clone(),
        None if capture::dirty_build_id(&env.build_id) => bail!(
            "the build id {} is a dirty tree's (uncommitted tracked changes): it does not identify the code, so name the capture with --name, a new name after each edit",
            env.build_id
        ),
        None => env.build_id.clone(),
    };
    capture::validate_name(&name)?;
    Ok(CaptureOptions {
        name,
        models,
        tier: probe.tier.clone(),
        numerics: !probe.no_numerics,
        perf: !probe.no_perf,
        texts: !probe.no_texts,
        autopsy: !probe.no_autopsy,
        mtp: probe.mtp,
    })
}

/// The comparison's markdown with the localization appended when the
/// verdict is not PASS (or the reason it could not be built), written as
/// against_<baseline>.md beside localize_<baseline>.md.
fn comparison_with_localization(
    base: &capture::Capture,
    cand: &capture::Capture,
    comparison: &report::Comparison,
) -> Result<String> {
    let mut markdown = report::render_comparison(comparison);
    if comparison.verdict != numerics::Verdict::Pass {
        match report::localize_captures(base, cand) {
            Ok(localization) => {
                let local = report::render_localization(&localization);
                write_report(
                    &cand.dir,
                    &format!("localize_{}", base.manifest.name),
                    &local,
                    &localization,
                )?;
                markdown.push_str(&local);
            }
            Err(error) => {
                let line = format!("The localization could not be built: {error:#}\n\n");
                eprintln!("{line}");
                markdown.push_str(&line);
            }
        }
    }
    write_report(&cand.dir, &format!("against_{}", base.manifest.name), &markdown, comparison)?;
    Ok(markdown)
}

fn write_report(
    dir: &Path,
    stem: &str,
    markdown: &str,
    json: &impl serde::Serialize,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(format!("{stem}.md")), markdown)?;
    std::fs::write(dir.join(format!("{stem}.json")), serde_json::to_string_pretty(json)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Command::Config => {
            print!("{}", config::DEFAULT_CONFIG);
            Ok(())
        }
        Command::Env => {
            let (env, _, guard_dir) = resolve(&cli)?;
            println!("guard directory: {}", guard_dir.display());
            println!("benchmark directory: {}", env.bench_dir.display());
            println!("LiteRT: {} ({})", env.rt.display(), env.git_head(&env.rt));
            println!("LiteRT-LM: {} ({})", env.lm.display(), env.git_head(&env.lm));
            println!("benchmark root: {}", env.root.display());
            println!(
                "runtime libraries: {} ({})",
                env.runtime_libs.display(),
                if env.built() { "built" } else { "NOT BUILT" }
            );
            println!("models: {}", env.models_dir.display());
            println!("build id: {}", env.build_id);
            println!("captures: {}", capture::guard_root(&env).display());
            Ok(())
        }
        Command::Localize { baseline, name } => {
            let (env, _, _) = resolve(&cli)?;
            let base = load_capture(&env, baseline)?;
            let cand = load_capture(&env, name)?;
            let localization = report::localize_captures(&base, &cand)?;
            let markdown = report::render_localization(&localization);
            write_report(
                &cand.dir,
                &format!("localize_{}", base.manifest.name),
                &markdown,
                &localization,
            )?;
            print!("{markdown}");
            println!(
                "Report: {}",
                cand.dir.join(format!("localize_{}.md", base.manifest.name)).display()
            );
            Ok(())
        }
        Command::Rescan { name } => {
            let (env, _, _) = resolve(&cli)?;
            let capture = capture::rescan_capture(&env, name)?;
            let markdown = report::render_capture(&capture);
            write_report(&capture.dir, "capture", &markdown, &capture)?;
            print!("{markdown}");
            Ok(())
        }
        Command::List => {
            let (env, _, _) = resolve(&cli)?;
            for m in list_captures(&env) {
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}{}",
                    m.name,
                    m.build_id,
                    m.tier,
                    m.host_mode,
                    m.updated,
                    m.models.join(","),
                    if m.mtp { "\tmtp" } else { "" }
                );
            }
            Ok(())
        }
        Command::Capture(probe) => {
            let (env, config, guard_dir) = resolve(&cli)?;
            let opts = capture_options(probe, &env, &config)?;
            eprintln!(
                "capture {} of build {} ({:?}, tier {})",
                opts.name, env.build_id, opts.models, opts.tier
            );
            let capture = run_capture(&env, &config, &guard_dir, &opts)?;
            let markdown = report::render_capture(&capture);
            write_report(&capture.dir, "capture", &markdown, &capture)?;
            print!("{markdown}");
            println!("Capture: {}", capture.dir.display());
            Ok(())
        }
        Command::Check { probe, baseline, record } => {
            let (env, config, guard_dir) = resolve(&cli)?;
            let base = load_capture(&env, baseline)?;
            let opts = capture_options(probe, &env, &config)?;
            if opts.name == base.manifest.name {
                bail!("the candidate needs a name other than the baseline's ({})", opts.name);
            }
            eprintln!(
                "check {} of build {} against {} ({})",
                opts.name, env.build_id, base.manifest.name, base.manifest.build_id
            );
            let cand = run_capture(&env, &config, &guard_dir, &opts)?;
            write_report(&cand.dir, "capture", &report::render_capture(&cand), &cand)?;
            let record_points = match record {
                Some(path) => report::load_record(path)
                    .with_context(|| format!("reading {}", path.display()))?,
                None => Vec::new(),
            };
            let comparison = report::compare(&base, &cand, &config, &record_points);
            let markdown = comparison_with_localization(&base, &cand, &comparison)?;
            print!("{markdown}");
            println!(
                "Report: {}",
                cand.dir.join(format!("against_{}.md", base.manifest.name)).display()
            );
            std::process::exit(comparison.verdict.exit_code());
        }
        Command::Compare { baseline, candidate, record } => {
            let (env, config, _) = resolve(&cli)?;
            let base = load_capture(&env, baseline)?;
            let cand = load_capture(&env, candidate)?;
            let record_points = match record {
                Some(path) => report::load_record(path)?,
                None => Vec::new(),
            };
            let comparison = report::compare(&base, &cand, &config, &record_points);
            let markdown = comparison_with_localization(&base, &cand, &comparison)?;
            print!("{markdown}");
            std::process::exit(comparison.verdict.exit_code());
        }
        Command::Predict { name, record } => {
            let (env, config, _) = resolve(&cli)?;
            let capture = load_capture(&env, name)?;
            let mut markdown = report::render_capture(&capture);
            let record_points = match record {
                Some(path) => report::load_record(path)?,
                None => Vec::new(),
            };
            let predicted = report::predictions(&capture, &capture, &config, &record_points);
            if !predicted.is_empty() {
                markdown.push_str("## Predictions from the cost models\n\n| Model | Workload | Metric | Model | Record |\n|---|---|---|---:|---:|\n");
                for p in &predicted {
                    markdown.push_str(&format!(
                        "| {} | {} | {} | {} | {} |\n",
                        p.model,
                        p.workload,
                        p.metric,
                        p.cand_model.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into()),
                        p.record.map(|v| format!("{v:.1}")).unwrap_or_else(|| "-".into())
                    ));
                }
                markdown.push('\n');
            }
            if !record_points.is_empty() {
                markdown
                    .push_str(&report::validate_predictor(&record_points, &config.perf.workloads));
            }
            print!("{markdown}");
            Ok(())
        }
    }
}
