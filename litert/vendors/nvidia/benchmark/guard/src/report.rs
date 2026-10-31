// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The comparison of two captures and its rendering (markdown and JSON), and
//! the cost-model predictions of a capture.

use std::collections::BTreeSet;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::autopsy::{diff_autopsies, AutopsyDiff};
use crate::capture::Capture;
use crate::config::{workload, Config, WORKLOADS};
use crate::dumps::{full_row_kls, localize, Localization, RowKl, TensorDiff};
use crate::numerics::{compare_runs, NumericsComparison, Verdict};
use crate::perf::{
    compare_points, decode_context, interpolate, CostModel, PerfComparison, PerfPoint,
};
use crate::runner::{numerics_dir, outputs_dir};
use crate::texts::{common_words, compare_texts, parted_at, TextComparison, TextRun};

/// A text probe of the candidate judged: the device and the host must agree,
/// the constrained text must match its regex, and both runs must end well.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextCheck {
    pub model: String,
    pub kind: String,
    /// The device-side variant judged against the host run: device, or
    /// immediate (the immediate-upload mode).
    #[serde(default)]
    pub variant: String,
    pub comparison: TextComparison,
    /// Whether the candidate's text of this variant equals the baseline's.
    pub same_as_baseline: Option<bool>,
    pub verdict: Verdict,
    pub reasons: Vec<String>,
}

/// What a workload name means, in the measurements record's language: its
/// name there (1k = 1024 input tokens, 8k = 8192, 16k = 16384, 30k = 30720,
/// 128k = 128001), the requested output tokens and the runtime capacity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadLegend {
    pub name: String,
    pub record_name: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub capacity: u32,
}

pub fn record_workload_name(name: &str) -> &str {
    match name {
        "short" => "1k",
        other => other,
    }
}

pub fn workload_legend(names: &[String]) -> Vec<WorkloadLegend> {
    WORKLOADS
        .iter()
        .filter(|w| names.iter().any(|n| n == w.name))
        .map(|w| WorkloadLegend {
            name: w.name.to_string(),
            record_name: record_workload_name(w.name).to_string(),
            input_tokens: w.input,
            output_tokens: w.output,
            capacity: w.context,
        })
        .collect()
}

/// What the probes do not cover, printed under every comparison.
pub const NOT_PROBED: &str = "Not probed: a grammar that ends inside an MTP round; the samplers at exact ties on the k-th logit; the device's memory; contexts beyond the measured workloads other than by the cost models; prefill chunks past the dumped steps (numerics.full_steps); outputs above the dump's cap (numerics.outputs_max_mb: the caches of a long context, which the localization reports as above the cap).";

fn text_run<'a>(
    runs: &'a [TextRun],
    model: &str,
    kind: &str,
    variant: &str,
) -> Option<&'a TextRun> {
    runs.iter().find(|r| r.model == model && r.kind == kind && r.variant == variant)
}

/// Every device-side text run of the candidate judged against its host run:
/// the texts must agree, the device run must show the verifier's device path
/// (the candidate selection at k > 1, the greedy sampler at k = 1), the host
/// run must show none, the constrained texts must match the regex, and both
/// runs must end well.
pub fn check_texts(base: &Capture, cand: &Capture) -> Vec<TextCheck> {
    let mut out = Vec::new();
    let keys: BTreeSet<(String, String, String)> = cand
        .texts
        .iter()
        .filter(|r| r.variant != "host")
        .map(|r| (r.model.clone(), r.kind.clone(), r.variant.clone()))
        .collect();
    for (model, kind, variant) in keys {
        let (Some(device), Some(host)) = (
            text_run(&cand.texts, &model, &kind, &variant),
            text_run(&cand.texts, &model, &kind, "host"),
        ) else {
            continue;
        };
        let comparison = compare_texts(device, host);
        let mut verdict = Verdict::Pass;
        let mut reasons = Vec::new();
        let mut raise = |v: Verdict, reason: String| {
            verdict = verdict.max(v);
            reasons.push(reason);
        };
        if device.exit_code != 0 || host.exit_code != 0 {
            raise(
                Verdict::Fail,
                format!("exit codes {variant} {} host {}", device.exit_code, host.exit_code),
            );
        }
        if device.parser_errors > 0 || host.parser_errors > 0 {
            raise(
                Verdict::Fail,
                format!(
                    "parser errors {variant} {} host {}",
                    device.parser_errors, host.parser_errors
                ),
            );
        }
        if device.regex_ok == Some(false) {
            raise(Verdict::Fail, format!("the {variant} text violates the regex"));
        }
        if host.regex_ok == Some(false) {
            raise(Verdict::Fail, "the host text violates the regex".to_string());
        }
        if device.top_k > 1 && device.device_candidates == 0 {
            raise(
                Verdict::Warn,
                format!("the {variant} run shows no candidate selection at k {}: the verifier sampled on the host", device.top_k),
            );
        } else if device.device_samplers == 0 {
            raise(Verdict::Warn, format!("the {variant} run reports no device sampler"));
        }
        if host.device_samplers > 0 {
            raise(
                Verdict::Warn,
                format!("the host run reports {} device sampler line(s): LITERT_NVIDIA_MTP_GPU_SAMPLING=0 was not honored", host.device_samplers),
            );
        }
        if !comparison.identical {
            raise(
                Verdict::Warn,
                format!(
                    "{variant} and host texts part at char {} ({} words in common; a tie at the k-th logit, or a sampler change)",
                    comparison.parted_char.unwrap_or(0),
                    comparison.common_words
                ),
            );
        }
        let same_as_baseline =
            text_run(&base.texts, &model, &kind, &variant).map(|b| b.text == device.text);
        out.push(TextCheck {
            model,
            kind,
            variant,
            comparison,
            same_as_baseline,
            verdict,
            reasons,
        });
    }
    out
}

pub fn autopsy_diffs(base: &Capture, cand: &Capture) -> Vec<AutopsyDiff> {
    cand.autopsies
        .iter()
        .filter_map(|c| {
            base.autopsies.iter().find(|b| b.model == c.model).map(|b| diff_autopsies(b, c))
        })
        .collect()
}

/// A context the captures did not measure, predicted from what they did.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub model: String,
    pub workload: String,
    pub metric: String,
    /// The baseline's cost model at that context.
    pub base_model: Option<f64>,
    /// The candidate's cost model at that context.
    pub cand_model: Option<f64>,
    /// The measured candidate/baseline ratios, interpolated in context, times
    /// the record's value there (when a record is given), else the baseline's
    /// model: what the candidate would measure.
    pub cand_expected: Option<f64>,
    pub record: Option<f64>,
    pub ratio: Option<f64>,
}

/// One prompt's MTP acceptance in both captures (greedy MTP numerics probes).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceProbe {
    pub prompt: String,
    pub base: f64,
    pub cand: f64,
    pub base_drafted: u64,
    pub cand_drafted: u64,
    pub text_identical: bool,
}

/// A model's MTP acceptance across the paired prompts: the pooled rates
/// (accepted over drafted), how many prompts fell and rose, the verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceCheck {
    pub model: String,
    pub probes: Vec<AcceptanceProbe>,
    pub base_pooled: f64,
    pub cand_pooled: f64,
    pub fell: usize,
    pub rose: usize,
    pub verdict: Verdict,
    pub reasons: Vec<String>,
}

/// The MTP acceptance of each model's greedy MTP numerics probes, paired by
/// prompt. A run without the counts (a summary from before they were kept)
/// is left out of the pool.
pub fn check_acceptance(
    base: &Capture,
    cand: &Capture,
    t: &crate::config::AcceptanceThresholds,
) -> Vec<AcceptanceCheck> {
    let mut models: Vec<String> = Vec::new();
    for c in cand.numerics.iter().filter(|r| r.mtp) {
        if !models.contains(&c.model) {
            models.push(c.model.clone());
        }
    }
    let mut out = Vec::new();
    for model in models {
        let mut probes = Vec::new();
        let (mut base_acc, mut base_draft, mut cand_acc, mut cand_draft) = (0u64, 0u64, 0u64, 0u64);
        for c in cand.numerics.iter().filter(|r| r.mtp && r.model == model) {
            let Some(b) =
                base.numerics.iter().find(|b| b.mtp && b.model == model && b.prompt == c.prompt)
            else {
                continue;
            };
            let (Some(ba), Some(bd), Some(ca), Some(cd)) =
                (b.accepted, b.drafted, c.accepted, c.drafted)
            else {
                continue;
            };
            if bd == 0 || cd == 0 || b.exit_code != 0 || c.exit_code != 0 {
                continue;
            }
            base_acc += ba;
            base_draft += bd;
            cand_acc += ca;
            cand_draft += cd;
            probes.push(AcceptanceProbe {
                prompt: c.prompt.clone(),
                base: ba as f64 / bd as f64,
                cand: ca as f64 / cd as f64,
                base_drafted: bd,
                cand_drafted: cd,
                text_identical: b.text_sha256 == c.text_sha256,
            });
        }
        if probes.is_empty() {
            continue;
        }
        let base_pooled = base_acc as f64 / base_draft.max(1) as f64;
        let cand_pooled = cand_acc as f64 / cand_draft.max(1) as f64;
        let fell = probes.iter().filter(|p| p.cand < p.base).count();
        let rose = probes.iter().filter(|p| p.cand > p.base).count();
        let drop = base_pooled - cand_pooled;
        let mut verdict = Verdict::Pass;
        let mut reasons = Vec::new();
        if probes.len() < t.min_probes {
            reasons.push(format!(
                "{} prompts paired, fewer than {}: reported, not judged",
                probes.len(),
                t.min_probes
            ));
        } else if drop >= t.fail_drop && 3 * fell >= 2 * probes.len() {
            verdict = Verdict::Fail;
            reasons.push(format!(
                "FAIL: pooled acceptance {:.1}% against {:.1}% (a drop of {:.1} points, at least {:.0}), {fell} of {} prompts lower",
                cand_pooled * 100.0, base_pooled * 100.0, drop * 100.0, t.fail_drop * 100.0, probes.len()
            ));
        } else if drop >= t.warn_drop && fell > rose {
            verdict = Verdict::Warn;
            reasons.push(format!(
                "WARN: pooled acceptance {:.1}% against {:.1}% (a drop of {:.1} points, at least {:.0}), {fell} prompts lower, {rose} higher",
                cand_pooled * 100.0, base_pooled * 100.0, drop * 100.0, t.warn_drop * 100.0
            ));
        }
        out.push(AcceptanceCheck {
            model,
            probes,
            base_pooled,
            cand_pooled,
            fell,
            rose,
            verdict,
            reasons,
        });
    }
    out
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    pub base_name: String,
    pub cand_name: String,
    pub base_build: String,
    pub cand_build: String,
    pub base_host_mode: String,
    pub cand_host_mode: String,
    pub numerics: Vec<NumericsComparison>,
    pub perf: Vec<PerfComparison>,
    pub predictions: Vec<Prediction>,
    #[serde(default)]
    pub texts: Vec<TextCheck>,
    #[serde(default)]
    pub autopsies: Vec<AutopsyDiff>,
    #[serde(default)]
    pub workloads: Vec<WorkloadLegend>,
    #[serde(default)]
    pub acceptance: Vec<AcceptanceCheck>,
    /// The extra environment of each capture (capture --env).
    #[serde(default)]
    pub base_env: Vec<(String, String)>,
    #[serde(default)]
    pub cand_env: Vec<(String, String)>,
    /// Whether each capture's clock was fit for timings (None: not checked).
    #[serde(default)]
    pub base_clock_fit: Option<bool>,
    #[serde(default)]
    pub cand_clock_fit: Option<bool>,
    /// Probes of either capture without a counterpart in the other.
    #[serde(default)]
    pub not_compared: Vec<String>,
    /// Why the verdict is not PASS for reasons outside the tables.
    #[serde(default)]
    pub reasons: Vec<String>,
    pub verdict: Verdict,
}

/// The probes of one capture without a counterpart in the other, by section.
fn not_compared(base: &Capture, cand: &Capture) -> Vec<String> {
    let mut out = Vec::new();
    let mode = |mtp: bool| if mtp { "mtp" } else { "plain" };
    for b in &base.numerics {
        if !cand
            .numerics
            .iter()
            .any(|c| c.model == b.model && c.prompt == b.prompt && c.mtp == b.mtp)
        {
            out.push(format!(
                "numerics {} {} {}: not in the candidate",
                b.model,
                b.prompt,
                mode(b.mtp)
            ));
        }
    }
    for c in &cand.numerics {
        if !base
            .numerics
            .iter()
            .any(|b| b.model == c.model && b.prompt == c.prompt && b.mtp == c.mtp)
        {
            out.push(format!(
                "numerics {} {} {}: not in the baseline",
                c.model,
                c.prompt,
                mode(c.mtp)
            ));
        }
    }
    for b in &base.perf {
        if !cand.perf.iter().any(|c| c.model == b.model && c.label == b.label) {
            out.push(format!("performance {}: not in the candidate", b.label));
        }
    }
    for c in &cand.perf {
        if !base.perf.iter().any(|b| b.model == c.model && b.label == c.label) {
            out.push(format!("performance {}: not in the baseline", c.label));
        }
    }
    for b in &base.texts {
        if text_run(&cand.texts, &b.model, &b.kind, &b.variant).is_none() {
            out.push(format!("text {} {} {}: not in the candidate", b.model, b.kind, b.variant));
        }
    }
    for c in &cand.texts {
        if text_run(&base.texts, &c.model, &c.kind, &c.variant).is_none() {
            out.push(format!("text {} {} {}: not in the baseline", c.model, c.kind, c.variant));
        }
    }
    for b in &base.autopsies {
        if !cand.autopsies.iter().any(|c| c.model == b.model) {
            out.push(format!("autopsy {}: not in the candidate", b.model));
        }
    }
    for c in &cand.autopsies {
        if !base.autopsies.iter().any(|b| b.model == c.model) {
            out.push(format!("autopsy {}: not in the baseline", c.model));
        }
    }
    out
}

/// A recorded data point of a full matrix (measurements' bench/results.json
/// or a report.py JSON), for the predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordPoint {
    pub model: String,
    pub workload: String,
    pub mtp: bool,
    pub text: bool,
    pub decode_tps: Option<f64>,
    pub prefill_tps: Option<f64>,
}

/// Reads measurements' bench/results.json (`records[].parts.main`) or a
/// report.py --json document (a list of run reports).
pub fn load_record(path: &std::path::Path) -> Result<Vec<RecordPoint>> {
    let text = std::fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&text)?;
    let mut points = Vec::new();
    let runs: Vec<&serde_json::Value> = match &value {
        serde_json::Value::Object(map) if map.contains_key("records") => map["records"]
            .as_array()
            .map(|a| a.iter().filter_map(|r| r.get("parts").and_then(|p| p.get("main"))).collect())
            .unwrap_or_default(),
        serde_json::Value::Array(items) => items.iter().collect(),
        _ => Vec::new(),
    };
    for run in runs {
        let (Some(meta), Some(summary)) = (run.get("meta"), run.get("summary")) else { continue };
        if meta.get("metric").and_then(|m| m.as_str()) != Some("latency") {
            continue;
        }
        let tps = |key: &str| {
            summary.get(key).and_then(|v| v.get("mean").or(Some(v))).and_then(|v| v.as_f64())
        };
        points.push(RecordPoint {
            model: meta.get("profile").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            workload: meta.get("workload").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            mtp: meta.get("mtp").and_then(|v| v.as_bool()).unwrap_or(false),
            text: meta.get("prompt_file").map(|v| !v.is_null()).unwrap_or(false),
            decode_tps: tps("decode_tokens_per_second"),
            prefill_tps: tps("prefill_tokens_per_second"),
        });
    }
    Ok(points)
}

fn plain_points<'a>(points: &'a [PerfPoint], model: &str) -> Vec<&'a PerfPoint> {
    points.iter().filter(|p| p.model == model && !p.mtp && !p.text).collect()
}

pub fn predictions(
    base: &Capture,
    cand: &Capture,
    config: &Config,
    record: &[RecordPoint],
) -> Vec<Prediction> {
    let mut out = Vec::new();
    let models: BTreeSet<&str> = cand.perf.iter().map(|p| p.model.as_str()).collect();
    for model in models {
        let base_model = CostModel::fit(model, &base.perf);
        let cand_model = CostModel::fit(model, &cand.perf);
        let base_plain = plain_points(&base.perf, model);
        let cand_plain = plain_points(&cand.perf, model);
        for metric in ["decode", "prefill"] {
            // Measured ratios by context, for the interpolation.
            let mut ratios: Vec<(f64, f64)> = Vec::new();
            for c in &cand_plain {
                if let Some(b) = base_plain.iter().find(|b| b.label == c.label) {
                    let (bv, cv) = if metric == "decode" {
                        (b.decode_tps, c.decode_tps)
                    } else {
                        (b.prefill_tps, c.prefill_tps)
                    };
                    if let (Some(bv), Some(cv)) = (bv, cv) {
                        if bv > 0.0 {
                            let x = if metric == "decode" {
                                decode_context(c.input_tokens, c.output_tokens)
                            } else {
                                c.input_tokens as f64
                            };
                            ratios.push((x, cv / bv));
                        }
                    }
                }
            }
            for name in &config.perf.predict {
                let Ok(w) = workload(name) else { continue };
                let x = if metric == "decode" {
                    decode_context(w.input, w.output)
                } else {
                    w.input as f64
                };
                let (bm, cm) = if metric == "decode" {
                    (base_model.predict_decode_tps(name), cand_model.predict_decode_tps(name))
                } else {
                    (base_model.predict_prefill_tps(name), cand_model.predict_prefill_tps(name))
                };
                let ratio = interpolate(&ratios, x);
                let recorded = record
                    .iter()
                    .find(|r| r.model == model && r.workload == *name && !r.mtp && !r.text)
                    .and_then(|r| if metric == "decode" { r.decode_tps } else { r.prefill_tps });
                let expected = ratio.and_then(|r| recorded.or(bm).map(|v| v * r));
                out.push(Prediction {
                    model: model.to_string(),
                    workload: name.clone(),
                    metric: metric.to_string(),
                    base_model: bm,
                    cand_model: cm,
                    cand_expected: expected,
                    record: recorded,
                    ratio,
                });
            }
        }
    }
    out
}

pub fn compare(
    base: &Capture,
    cand: &Capture,
    config: &Config,
    record: &[RecordPoint],
) -> Comparison {
    let mut numerics = Vec::new();
    for c in &cand.numerics {
        if let Some(b) = base
            .numerics
            .iter()
            .find(|b| b.model == c.model && b.prompt == c.prompt && b.mtp == c.mtp)
        {
            numerics.push(compare_runs(b, c, &config.numerics.thresholds));
        }
    }
    let perf = compare_points(&base.perf, &cand.perf, &config.perf.thresholds);
    let predictions = predictions(base, cand, config, record);
    let texts = check_texts(base, cand);
    let autopsies = autopsy_diffs(base, cand);
    let acceptance = check_acceptance(base, cand, &config.acceptance);
    let mut names: Vec<String> = config.perf.workloads.clone();
    names.extend(config.perf.predict.iter().cloned());
    names.push(config.probes.autopsy.workload.clone());
    let workloads = workload_legend(&names);
    let not_compared = not_compared(base, cand);
    let mut reasons = Vec::new();
    if !base.numerics.is_empty() && numerics.is_empty() {
        reasons.push(
            "no numerics probe compared: the candidate ran none of the baseline's".to_string(),
        );
    }
    if !base.perf.is_empty() && perf.is_empty() {
        reasons.push(
            "no performance run compared: the candidate ran none of the baseline's".to_string(),
        );
    }
    if !base.texts.is_empty() && texts.is_empty() {
        reasons
            .push("no text probe compared: the candidate ran none of the baseline's".to_string());
    }
    if base.numerics.is_empty() && base.perf.is_empty() && base.texts.is_empty() {
        reasons.push("the baseline holds no probe".to_string());
    }
    if !perf.is_empty()
        && (base.manifest.clock_fit == Some(false) || cand.manifest.clock_fit == Some(false))
    {
        reasons.push("a capture's timings ran on a clock clock_check.py found unfit: the performance rows are diagnostic".to_string());
    }
    let verdict = numerics
        .iter()
        .map(|n| n.verdict)
        .chain(perf.iter().map(|p| p.verdict))
        .chain(texts.iter().map(|t| t.verdict))
        .chain(acceptance.iter().map(|a| a.verdict))
        .chain(reasons.iter().map(|_| Verdict::Warn))
        .max()
        .unwrap_or(Verdict::Pass);
    Comparison {
        base_name: base.manifest.name.clone(),
        cand_name: cand.manifest.name.clone(),
        base_build: base.manifest.build_id.clone(),
        cand_build: cand.manifest.build_id.clone(),
        base_host_mode: base.manifest.host_mode.clone(),
        cand_host_mode: cand.manifest.host_mode.clone(),
        numerics,
        perf,
        predictions,
        texts,
        autopsies,
        workloads,
        acceptance,
        base_env: base.manifest.env.clone(),
        cand_env: cand.manifest.env.clone(),
        base_clock_fit: base.manifest.clock_fit,
        cand_clock_fit: cand.manifest.clock_fit,
        not_compared,
        reasons,
        verdict,
    }
}

fn opt(value: Option<f64>, digits: usize) -> String {
    value.map(|v| format!("{v:.digits$}")).unwrap_or_else(|| "-".to_string())
}

fn pct(ratio: f64) -> String {
    format!("{:+.1}%", (ratio - 1.0) * 100.0)
}

pub fn render_comparison(c: &Comparison) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# Guard: {} ({}) against {} ({}): {}\n\n",
        c.cand_name,
        c.cand_build,
        c.base_name,
        c.base_build,
        c.verdict.label()
    ));
    let samplers = |cap: &[NumericsComparison]| cap.len();
    let _ = samplers;
    let env_text = |env: &[(String, String)]| {
        if env.is_empty() {
            "none".to_string()
        } else {
            env.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(" ")
        }
    };
    if !c.base_env.is_empty() || !c.cand_env.is_empty() {
        s.push_str(&format!(
            "Extra environment: baseline {}, candidate {}.\n\n",
            env_text(&c.base_env),
            env_text(&c.cand_env)
        ));
    }
    let clock_text = |fit: Option<bool>| match fit {
        Some(true) => "fit",
        Some(false) => "UNFIT",
        None => "not checked",
    };
    s.push_str(&format!(
        "Host clock (clock_check.py): baseline {}, candidate {}.\n\n",
        clock_text(c.base_clock_fit),
        clock_text(c.cand_clock_fit)
    ));
    if c.base_host_mode != c.cand_host_mode {
        s.push_str(&format!("Host modes differ: baseline {}, candidate {}; timings are not comparable across them.\n\n", c.base_host_mode, c.cand_host_mode));
    }
    for reason in &c.reasons {
        s.push_str(&format!("- WARN: {reason}\n"));
    }
    if !c.reasons.is_empty() {
        s.push('\n');
    }
    if !c.acceptance.is_empty() {
        s.push_str(
            "## MTP acceptance (greedy MTP numerics probes, accepted over drafted tokens)\n\n",
        );
        s.push_str("| Model | Prompts | Pooled (base, cand) | Lower, higher | Per prompt (base -> cand, same text) | Verdict |\n|---|---:|---|---|---|---|\n");
        for a in &c.acceptance {
            let per = a
                .probes
                .iter()
                .map(|p| {
                    format!(
                        "{} {:.0}->{:.0}%{}",
                        p.prompt,
                        p.base * 100.0,
                        p.cand * 100.0,
                        if p.text_identical { " =" } else { "" }
                    )
                })
                .collect::<Vec<_>>()
                .join("; ");
            s.push_str(&format!(
                "| {} | {} | {:.1}%, {:.1}% | {}, {} | {} | {} |\n",
                a.model,
                a.probes.len(),
                a.base_pooled * 100.0,
                a.cand_pooled * 100.0,
                a.fell,
                a.rose,
                per,
                a.verdict.label()
            ));
        }
        for a in &c.acceptance {
            for r in &a.reasons {
                s.push_str(&format!("- {}: {r}\n", a.model));
            }
        }
        s.push('\n');
    }
    if !c.numerics.is_empty() {
        s.push_str("## Numerics (greedy decode, logits per step)\n\n");
        s.push_str("| Model | Prompt | Mode | Steps compared | Parted at | First-steps KL | max abs delta | mean KL | top-K overlap | Text | Decode tok/s (base, cand) | Verdict |\n|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|\n");
        for n in &c.numerics {
            s.push_str(&format!(
                "| {} | {} | {} | {} / {} | {} | {:.2e} | {:.4} | {:.2e} | {:.3} | {}{} | {}, {} | {} |\n",
                n.model,
                n.prompt,
                if n.mtp { "mtp" } else { "plain" },
                n.steps_compared,
                n.steps_base.min(n.steps_cand),
                n.diverged_at.map(|d| d.to_string()).unwrap_or_else(|| "-".into()),
                n.first_kl,
                n.max_abs_delta,
                n.mean_kl,
                n.mean_overlap,
                if n.text_identical { "identical" } else { "differs" },
                if n.cand_loopiness >= 0.25 { " (loops)" } else { "" },
                opt(n.base_decode_tps, 1),
                opt(n.cand_decode_tps, 1),
                n.verdict.label()
            ));
        }
        s.push('\n');
        for n in c.numerics.iter().filter(|n| n.verdict != Verdict::Pass) {
            s.push_str(&format!(
                "- {} {}{}: {}\n",
                n.model,
                n.prompt,
                if n.mtp { " (mtp)" } else { "" },
                n.reasons.join("; ")
            ));
        }
        s.push('\n');
    }
    if !c.texts.is_empty() {
        s.push_str("## Texts (sampled and constrained decodes, MTP verifier on the device vs the host)\n\nVariants: device (the verifier's candidates on the device, the default upload mode), immediate (the same in the immediate-upload mode, LITERT_NVIDIA_DISPATCH_DEFERRED_UPLOAD=0), each against the host run (LITERT_NVIDIA_MTP_GPU_SAMPLING=0) of the same seed.\n\n| Model | Probe | Variant | Vs host | Regex (variant, host) | Device samplers (candidate selections) | Chars (variant, host) | Same as baseline | Verdict |\n|---|---|---|---|---|---:|---|---|---|\n");
        for t in &c.texts {
            let cmp = &t.comparison;
            let regex = |v: Option<bool>| match v {
                Some(true) => "ok",
                Some(false) => "VIOLATED",
                None => "-",
            };
            s.push_str(&format!(
                "| {} | {} | {} | {} | {}, {} | {} ({}) | {}, {} | {} | {} |\n",
                t.model,
                t.kind,
                t.variant,
                if cmp.identical {
                    "identical".to_string()
                } else {
                    format!(
                        "part at char {} ({} words in common)",
                        cmp.parted_char.unwrap_or(0),
                        cmp.common_words
                    )
                },
                regex(cmp.device_regex_ok),
                regex(cmp.host_regex_ok),
                cmp.device_samplers,
                cmp.device_candidates,
                cmp.device_text_len,
                cmp.host_text_len,
                match t.same_as_baseline {
                    Some(true) => "yes",
                    Some(false) => "no",
                    None => "-",
                },
                t.verdict.label()
            ));
        }
        s.push('\n');
        for t in c.texts.iter().filter(|t| t.verdict != Verdict::Pass) {
            s.push_str(&format!(
                "- {} {} {}: {}\n",
                t.model,
                t.kind,
                t.variant,
                t.reasons.join("; ")
            ));
        }
        s.push('\n');
    }
    if !c.not_compared.is_empty() {
        s.push_str("## Not compared\n\n");
        for line in &c.not_compared {
            s.push_str(&format!("- {line}\n"));
        }
        s.push('\n');
    }
    if !c.workloads.is_empty() {
        s.push_str(&render_legend(&c.workloads));
    }
    if !c.perf.is_empty() {
        s.push_str("## Performance (measured)\n\nSpread: (max - min) / mean of the measured iterations; a change within the spread is noise.\n\n| Model | Workload | Mode | Metric | Baseline | Candidate | Change | Spread (base, cand) | Verdict |\n|---|---|---|---|---:|---:|---:|---|---|\n");
        for p in &c.perf {
            let mode = format!(
                "{}{}",
                if p.mtp { "mtp" } else { "plain" },
                if p.text { " text" } else { "" }
            );
            let digits = if p.metric == "acceptance" {
                3
            } else if p.metric == "tokens/round" {
                2
            } else {
                1
            };
            let spread = |v: Option<f64>| {
                v.map(|x| format!("{:.1}%", x * 100.0)).unwrap_or_else(|| "-".into())
            };
            s.push_str(&format!(
                "| {} | {} | {} | {} | {:.digits$} | {:.digits$} | {} | {}, {} | {} |\n",
                p.model,
                p.workload,
                mode,
                p.metric,
                p.base,
                p.cand,
                pct(p.ratio),
                spread(p.base_spread),
                spread(p.cand_spread),
                p.verdict.label()
            ));
        }
        s.push('\n');
    }
    if !c.predictions.is_empty() {
        s.push_str("## Performance (predicted at unmeasured contexts)\n\nCost models: decode step time a + b·context, prefill time c0 + c1·N + c2·N², fitted on the measured workloads; the expected candidate value applies the measured change ratio, interpolated in context, to the record where one is given (else to the baseline's model).\n\n| Model | Workload | Metric | Baseline model | Candidate model | Record | Expected candidate | Change |\n|---|---|---|---:|---:|---:|---:|---:|\n");
        for p in &c.predictions {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                p.model,
                p.workload,
                p.metric,
                opt(p.base_model, 1),
                opt(p.cand_model, 1),
                opt(p.record, 1),
                opt(p.cand_expected, 1),
                p.ratio.map(pct).unwrap_or_else(|| "-".into())
            ));
        }
        s.push('\n');
    }
    s.push_str(&render_autopsy_diffs(&c.autopsies));
    s.push_str(NOT_PROBED);
    s.push_str("\n\n");
    s
}

/// The workload legend in the measurements record's language.
pub fn render_legend(workloads: &[WorkloadLegend]) -> String {
    let legend: Vec<String> = workloads
        .iter()
        .map(|w| {
            format!(
                "{}{} = {} input tokens, {} requested output tokens, runtime capacity {}",
                w.name,
                if w.record_name != w.name {
                    format!(" (the record's {})", w.record_name)
                } else {
                    String::new()
                },
                w.input_tokens,
                w.output_tokens,
                w.capacity
            )
        })
        .collect();
    format!(
        "Workloads: {}. tok/s are tokens per second of the measured phase, over few iterations in the manifest's host mode: diagnostic numbers, never a cell of the measurements' tables.\n\n",
        legend.join("; ")
    )
}

/// The kernels whose warm times moved, per model: the attribution of a
/// performance change (the autopsy times every kernel node of a step graph
/// alone, so the sums are not the step's time).
pub fn render_autopsy_diffs(diffs: &[AutopsyDiff]) -> String {
    let mut s = String::new();
    if diffs.is_empty() {
        return s;
    }
    s.push_str("## Kernels (autopsy of the step graphs, warm times per kernel)\n\nEvery kernel node of a function's CUDA graph timed alone; a kernel's time is summed over its launches in the step. The sums are the kernels' own times, not the step's. The sums are noisy: between captures of one build, kernels the change did not touch moved by up to about 85 µs over a step's launches on the E2B (a few µs per launch), so a Δ of that size says nothing; attribute a change to a kernel only when its Δ is well above what a same-build pair shows and agrees with the measured cells.\n\n");
    for d in diffs {
        s.push_str(&format!("### {} ({}{})\n\n| Function | Kernels (base, cand) | Base us | Cand us | Change |\n|---|---|---:|---:|---:|\n", d.model, d.workload, if d.mtp { ", mtp" } else { "" }));
        for f in &d.functions {
            s.push_str(&format!(
                "| {} | {}, {} | {:.0} | {:.0} | {} |\n",
                f.function,
                f.base_kernels,
                f.cand_kernels,
                f.base_us,
                f.cand_us,
                if f.base_us > 0.0 { pct(f.cand_us / f.base_us) } else { "-".into() }
            ));
        }
        s.push('\n');
        let moved: Vec<&crate::autopsy::KernelDelta> = d
            .kernels
            .iter()
            .filter(|k| k.delta_us.abs() >= 1.0 || k.base_count != k.cand_count)
            .take(16)
            .collect();
        if moved.is_empty() {
            s.push_str("No kernel moved by a microsecond or more.\n\n");
        } else {
            s.push_str("| Function | Kernel | Launches (base, cand) | Base us | Cand us | Δ us | In step (base, cand) |\n|---|---|---|---:|---:|---:|---|\n");
            for k in moved {
                s.push_str(&format!(
                    "| {} | `{}` | {}, {} | {:.1} | {:.1} | {:+.1} | {}, {} |\n",
                    k.function,
                    k.short,
                    k.base_count,
                    k.cand_count,
                    k.base_us,
                    k.cand_us,
                    k.delta_us,
                    opt(k.base_graph_us, 0),
                    opt(k.cand_graph_us, 0)
                ));
            }
            s.push('\n');
        }
    }
    s
}

/// Where one numerics probe's whole-output dumps part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeLocalization {
    pub model: String,
    pub prompt: String,
    pub mtp: bool,
    /// The index of the first emitted token that differs: the argmax rows of
    /// a plain greedy run are its emitted tokens. None for an MTP run, whose
    /// rows are the verifier's; its texts' parting stands for it.
    pub parted_token: Option<usize>,
    /// The emitted tokens of both runs (plain runs).
    pub tokens_base: usize,
    pub tokens_cand: usize,
    /// Where the two runs' texts part (a character offset) and the words in
    /// common before it; None when the texts are identical.
    pub parted_char: Option<usize>,
    pub common_words: usize,
    pub localization: Localization,
    /// The whole-vocabulary KL of the first logits rows, in production order.
    pub row_kls: Vec<RowKl>,
    pub max_row_kl: Option<f64>,
}

/// Where a candidate parts from a baseline, as far as the captures' dumps say.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationReport {
    pub base_name: String,
    pub cand_name: String,
    pub base_build: String,
    pub cand_build: String,
    pub probes: Vec<ProbeLocalization>,
    /// Probes without dumps on one side.
    pub skipped: Vec<String>,
    /// Probes, texts and autopsies of either capture without a counterpart
    /// in the other: what this localization does not cover.
    #[serde(default)]
    pub not_compared: Vec<String>,
    pub texts: Vec<TextCheck>,
    /// The candidate's texts against the baseline's, per variant.
    pub text_partings: Vec<TextParting>,
    pub autopsies: Vec<AutopsyDiff>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextParting {
    pub model: String,
    pub kind: String,
    pub variant: String,
    pub parted_char: Option<usize>,
    pub base_len: usize,
    pub cand_len: usize,
}

pub fn localize_captures(base: &Capture, cand: &Capture) -> Result<LocalizationReport> {
    let mut probes = Vec::new();
    let mut skipped = Vec::new();
    for c in &cand.numerics {
        let label = format!("{} {}{}", c.model, c.prompt, if c.mtp { " (mtp)" } else { "" });
        let Some(b) = base
            .numerics
            .iter()
            .find(|b| b.model == c.model && b.prompt == c.prompt && b.mtp == c.mtp)
        else {
            skipped.push(format!("{label}: not in the baseline"));
            continue;
        };
        let base_dumps = outputs_dir(&numerics_dir(&base.dir, &b.model, &b.prompt, b.mtp));
        let cand_dumps = outputs_dir(&numerics_dir(&cand.dir, &c.model, &c.prompt, c.mtp));
        if !base_dumps.join("index.jsonl").is_file() || !cand_dumps.join("index.jsonl").is_file() {
            skipped.push(format!(
                "{label}: no whole-output dumps on both sides (numerics.full_steps)"
            ));
            continue;
        }
        let localization = localize(&base_dumps, &cand_dumps)?;
        let row_kls = full_row_kls(&base_dumps, &cand_dumps)?;
        let max_row_kl = row_kls
            .iter()
            .map(|r| r.kl)
            .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
        let parted_token = if c.mtp { None } else { parted_tokens(&b.tokens, &c.tokens) };
        let base_text = std::fs::read_to_string(
            numerics_dir(&base.dir, &b.model, &b.prompt, b.mtp).join("response.txt"),
        )
        .unwrap_or_default();
        let cand_text = std::fs::read_to_string(
            numerics_dir(&cand.dir, &c.model, &c.prompt, c.mtp).join("response.txt"),
        )
        .unwrap_or_default();
        probes.push(ProbeLocalization {
            model: c.model.clone(),
            prompt: c.prompt.clone(),
            mtp: c.mtp,
            parted_token,
            tokens_base: b.tokens.len(),
            tokens_cand: c.tokens.len(),
            parted_char: parted_at(&base_text, &cand_text),
            common_words: common_words(&base_text, &cand_text),
            localization,
            row_kls,
            max_row_kl,
        });
    }
    let mut text_partings = Vec::new();
    for c in &cand.texts {
        if let Some(b) = text_run(&base.texts, &c.model, &c.kind, &c.variant) {
            text_partings.push(TextParting {
                model: c.model.clone(),
                kind: c.kind.clone(),
                variant: c.variant.clone(),
                parted_char: parted_at(&b.text, &c.text),
                base_len: b.text_len,
                cand_len: c.text_len,
            });
        }
    }
    Ok(LocalizationReport {
        base_name: base.manifest.name.clone(),
        cand_name: cand.manifest.name.clone(),
        base_build: base.manifest.build_id.clone(),
        cand_build: cand.manifest.build_id.clone(),
        probes,
        skipped,
        not_compared: not_compared(base, cand),
        texts: check_texts(base, cand),
        text_partings,
        autopsies: autopsy_diffs(base, cand),
    })
}

/// The index of the first emitted token that differs (the shorter run's
/// length when one is a prefix of the other); None when they are the same.
pub fn parted_tokens(base: &[u32], cand: &[u32]) -> Option<usize> {
    let n = base.iter().zip(cand.iter()).take_while(|(a, b)| a == b).count();
    if n == base.len() && n == cand.len() {
        None
    } else {
        Some(n)
    }
}

/// The differing outputs of the first differing step (one function: an
/// invocation context runs one), the first few by name and the rest counted.
fn reached(step: &[TensorDiff], outputs: usize) -> String {
    const SHOWN: usize = 6;
    let states = step.iter().filter(|d| d.state).count();
    let mut parts: Vec<String> = step
        .iter()
        .take(SHOWN)
        .map(|d| {
            format!("out {}{} ({:.1e})", d.output, if d.state { " state" } else { "" }, d.relative)
        })
        .collect();
    if step.len() > SHOWN {
        parts.push(format!("and {} more", step.len() - SHOWN));
    }
    format!(
        "{} of its {} outputs differ ({} state): {}",
        step.len(),
        outputs,
        states,
        parts.join(", ")
    )
}

/// The step's input tokens tell the signatures apart: a prefill chunk's
/// count, a decode step's one, a verify step's few.
fn tokens_cell(rows: i64) -> String {
    match rows {
        0 => String::new(),
        1 => " (1 input token)".to_string(),
        n => format!(" ({n} input tokens)"),
    }
}

fn tensor_cell(d: &TensorDiff) -> String {
    format!(
        "c{} {}{} step {} out {} `{}`{} {:?}",
        d.context,
        d.function,
        tokens_cell(d.input_rows),
        d.step,
        d.output,
        d.name,
        if d.state { " (state)" } else { "" },
        d.dims
    )
}

pub fn render_localization(r: &LocalizationReport) -> String {
    let mut s = format!(
        "# Localization: {} ({}) against {} ({})\n\n",
        r.cand_name, r.cand_build, r.base_name, r.base_build
    );
    if r.probes.is_empty() && r.skipped.is_empty() {
        s.push_str("No numerics probe to localize.\n\n");
    }
    if !r.probes.is_empty() {
        s.push_str("## Where the outputs part (whole-output dumps of the first steps)\n\nTensors are compared in production order; the first that differs names the step and the partition (function) the change first reaches. A relative move is rms(Δ) / rms(baseline). The emitted token where the texts part is counted on a plain run's emitted tokens (an MTP run's rows are the verifier's, so its texts' character offset stands instead).\n\n| Model | Prompt | Mode | Emitted token parts at | Text parts at (char, words in common) | Tensors paired | Differing | First differing tensor | Rel. move | max abs delta | Largest move | Rel. | max row KL |\n|---|---|---|---:|---|---:|---:|---|---:|---:|---|---:|---:|\n");
        for p in &r.probes {
            let l = &p.localization;
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                p.model,
                p.prompt,
                if p.mtp { "mtp" } else { "plain" },
                match (p.mtp, p.parted_token) {
                    (true, _) => "- (verify rows)".to_string(),
                    (false, Some(t)) => format!("{t} of {}/{}", p.tokens_base, p.tokens_cand),
                    (false, None) => "same".to_string(),
                },
                p.parted_char
                    .map(|c| format!("{c}, {}", p.common_words))
                    .unwrap_or_else(|| "identical".into()),
                l.pairs,
                l.differing,
                l.first.as_ref().map(tensor_cell).unwrap_or_else(|| "none: identical".into()),
                l.first
                    .as_ref()
                    .map(|d| format!("{:.2e}", d.relative))
                    .unwrap_or_else(|| "-".into()),
                l.first
                    .as_ref()
                    .map(|d| format!("{:.3e}", d.max_abs_delta))
                    .unwrap_or_else(|| "-".into()),
                l.largest.as_ref().map(tensor_cell).unwrap_or_else(|| "-".into()),
                l.largest
                    .as_ref()
                    .map(|d| format!("{:.2e}", d.relative))
                    .unwrap_or_else(|| "-".into()),
                p.max_row_kl.map(|v| format!("{v:.2e}")).unwrap_or_else(|| "-".into())
            ));
        }
        s.push('\n');
        for p in &r.probes {
            let l = &p.localization;
            let Some(first) = &l.first else { continue };
            s.push_str(&format!(
                "- {} {}{}: the first difference is at step {} of `{}` (context {}{}), output {} `{}`: {} of {} elements differ, max |Δ| {:.3e}, rms Δ {:.3e} (relative {:.2e}){}{}{}. At that step {}.\n",
                p.model,
                p.prompt,
                if p.mtp { " (mtp)" } else { "" },
                first.step,
                first.function,
                first.context,
                if first.input_rows > 0 { format!(", {} input tokens", first.input_rows) } else { String::new() },
                first.output,
                first.name,
                first.differing,
                first.elements,
                first.max_abs_delta,
                first.rms_delta,
                first.relative,
                first.first_index.map(|i| format!(", first at element {i}")).unwrap_or_default(),
                if first.nonfinite { ", NON-FINITE values" } else { "" },
                if !first.base_top.is_empty() {
                    format!(
                        "; top logits base {} vs cand {}",
                        first.base_top.iter().take(4).map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>().join(" "),
                        first.cand_top.iter().take(4).map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>().join(" ")
                    )
                } else {
                    String::new()
                },
                reached(&l.first_step, l.first_step_outputs)
            ));
            if l.unpaired > 0 {
                s.push_str(&format!(
                    "  ({} tensors without a counterpart in the other run)\n",
                    l.unpaired
                ));
            }
        }
        for p in &r.probes {
            let l = &p.localization;
            if l.above_cap > 0 {
                s.push_str(&format!(
                    "- {} {}{}: {} outputs above the dump's cap{}: the steps they belong to are not localized below their dumped outputs (numerics.outputs_max_mb).\n",
                    p.model,
                    p.prompt,
                    if p.mtp { " (mtp)" } else { "" },
                    l.above_cap,
                    match l.above_cap_state {
                        0 => String::new(),
                        n if n == l.above_cap => " (all state, the caches)".to_string(),
                        n => format!(" ({n} of them state, the caches)"),
                    }
                ));
            }
        }
        s.push('\n');
        let with_rows: Vec<&ProbeLocalization> =
            r.probes.iter().filter(|p| !p.row_kls.is_empty()).collect();
        if !with_rows.is_empty() {
            s.push_str("## KL over the whole vocabulary (the first logits rows)\n\n| Model | Prompt | Mode | Row (context, function, step) | KL | max abs delta logit | Same argmax |\n|---|---|---|---|---:|---:|---|\n");
            for p in with_rows {
                for k in p.row_kls.iter().take(12) {
                    s.push_str(&format!(
                        "| {} | {} | {} | c{} {} step {} row {} | {:.2e} | {:.3e} | {} |\n",
                        p.model,
                        p.prompt,
                        if p.mtp { "mtp" } else { "plain" },
                        k.context,
                        k.function,
                        k.step,
                        k.row,
                        k.kl,
                        k.max_abs_delta,
                        if k.same_argmax { "yes" } else { "NO" }
                    ));
                }
            }
            s.push('\n');
        }
    }
    for line in &r.skipped {
        s.push_str(&format!("- skipped {line}\n"));
    }
    if !r.skipped.is_empty() {
        s.push('\n');
    }
    if !r.not_compared.is_empty() {
        s.push_str("## Not compared\n\n");
        for line in &r.not_compared {
            s.push_str(&format!("- {line}\n"));
        }
        s.push('\n');
    }
    if !r.text_partings.is_empty() {
        s.push_str("## Texts against the baseline's\n\n| Model | Probe | Variant | Parted at char | Chars (base, cand) |\n|---|---|---|---:|---|\n");
        for t in &r.text_partings {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {}, {} |\n",
                t.model,
                t.kind,
                t.variant,
                t.parted_char.map(|c| c.to_string()).unwrap_or_else(|| "identical".into()),
                t.base_len,
                t.cand_len
            ));
        }
        s.push('\n');
        for t in r.texts.iter().filter(|t| t.verdict != Verdict::Pass) {
            s.push_str(&format!(
                "- {} {} ({} vs host): {}\n",
                t.model,
                t.kind,
                t.variant,
                t.reasons.join("; ")
            ));
        }
        s.push('\n');
    }
    s.push_str(&render_autopsy_diffs(&r.autopsies));
    s
}

pub fn render_capture(capture: &Capture) -> String {
    let m = &capture.manifest;
    let mut s = format!(
        "# Capture {} (build {}, tier {}, host {}, {})\n\nLiteRT {}\nLiteRT-LM {}\n\n",
        m.name, m.build_id, m.tier, m.host_mode, m.updated, m.rt_head, m.lm_head
    );
    if !capture.numerics.is_empty() {
        s.push_str("## Numerics probes\n\n| Model | Prompt | Mode | Exit | Steps | Prefill tokens | Decode tokens | Decode tok/s | Loopiness | Non-finite | Device samplers | Wall s |\n|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|\n");
        for n in &capture.numerics {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {:.2} | {} | {} | {:.0} |\n",
                n.model,
                n.prompt,
                if n.mtp { "mtp" } else { "plain" },
                n.exit_code,
                n.steps.len(),
                n.prefill_tokens,
                n.decode_tokens,
                opt(n.decode_tps, 1),
                n.loopiness,
                if n.nonfinite { "yes" } else { "no" },
                n.device_samplers,
                n.wall_s
            ));
        }
        s.push('\n');
        let compiles: Vec<&crate::numerics::NumericsRun> =
            capture.numerics.iter().filter(|n| n.weight_store.is_some()).collect();
        if !compiles.is_empty() {
            s.push_str("Compiles seen by the probes: ");
            let sdks: BTreeSet<String> =
                compiles.iter().filter_map(|n| n.sdk_version.clone()).collect();
            s.push_str(&format!(
                "SDK {}; ",
                if sdks.is_empty() {
                    "unknown".to_string()
                } else {
                    sdks.into_iter().collect::<Vec<_>>().join(", ")
                }
            ));
            let without_store: Vec<String> = compiles
                .iter()
                .filter(|n| n.weight_store == Some(false))
                .map(|n| n.model.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            if without_store.is_empty() {
                s.push_str("weight store used by every compile.\n\n");
            } else {
                s.push_str(&format!("WITHOUT the weight store for {} (plans carry their weights: check the SDK the plugin was built against).\n\n", without_store.join(", ")));
            }
        }
    }
    if !capture.perf.is_empty() {
        s.push_str("## Performance probes\n\n| Model | Workload | Mode | Iterations | Decode tok/s | Prefill tok/s | Acceptance | Tokens/round | Init s |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n");
        for p in &capture.perf {
            let mode = format!(
                "{}{}",
                if p.mtp { "mtp" } else { "plain" },
                if p.text { " text" } else { "" }
            );
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                p.model,
                p.workload,
                mode,
                p.measured_iterations,
                opt(p.decode_tps, 1),
                opt(p.prefill_tps, 0),
                p.acceptance.map(|a| format!("{:.0}%", a * 100.0)).unwrap_or_else(|| "-".into()),
                opt(p.tokens_per_round, 2),
                opt(p.init_total_s, 1)
            ));
        }
        s.push('\n');
        let models: BTreeSet<&str> = capture.perf.iter().map(|p| p.model.as_str()).collect();
        s.push_str("## Cost models\n\n| Model | Decode step ms = a + b·context | Prefill s = c0 + c1·N + c2·N² | Fit points |\n|---|---|---|---:|\n");
        for model in models {
            let cm = CostModel::fit(model, &capture.perf);
            s.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                model,
                cm.decode_ms
                    .map(|(a, b)| format!("{a:.3} + {b:.3e}·c"))
                    .unwrap_or_else(|| "-".into()),
                cm.prefill_s
                    .map(|(c0, c1, c2)| format!("{c0:.4} + {c1:.3e}·N + {c2:.3e}·N²"))
                    .unwrap_or_else(|| "-".into()),
                cm.decode_fit_points.len()
            ));
        }
        s.push_str("\nWith two measured contexts the prefill model has no attention term (c2 = 0) and long contexts come out optimistic; measure a third (perf.workloads) for it. The measured change ratios are the regression yardstick.\n");
        s.push('\n');
    }
    if !capture.texts.is_empty() {
        s.push_str("## Text probes (session mode, MTP verifier sampling on the device or on the host)\n\n| Model | Probe | Variant | Exit | Chars | Regex | Device samplers | Parser errors | Wall s |\n|---|---|---|---:|---:|---|---:|---:|---:|\n");
        for t in &capture.texts {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {:.0} |\n",
                t.model,
                t.kind,
                t.variant,
                t.exit_code,
                t.text_len,
                match t.regex_ok {
                    Some(true) => "ok",
                    Some(false) => "VIOLATED",
                    None => "-",
                },
                t.device_samplers,
                t.parser_errors,
                t.wall_s
            ));
        }
        s.push('\n');
        let keys: BTreeSet<(String, String)> =
            capture.texts.iter().map(|t| (t.model.clone(), t.kind.clone())).collect();
        for (model, kind) in keys {
            if let (Some(d), Some(h)) = (
                text_run(&capture.texts, &model, &kind, "device"),
                text_run(&capture.texts, &model, &kind, "host"),
            ) {
                let c = compare_texts(d, h);
                s.push_str(&format!(
                    "- {model} {kind}: device and host texts {}\n",
                    if c.identical {
                        "identical".to_string()
                    } else {
                        format!(
                            "part at char {} ({} words in common)",
                            c.parted_char.unwrap_or(0),
                            c.common_words
                        )
                    }
                ));
            }
        }
        s.push('\n');
    }
    if !capture.autopsies.is_empty() {
        s.push_str("## Autopsies (every kernel node of a step graph timed alone, and the step's timeline)\n\nWarm and cold: each kernel node replayed alone (the sums are not the step's time). Step and kernels in step: the step graph replayed with an event after every node, the real data and launch gaps included.\n\n| Model | Workload | Mode | Function | Kernels | Warm us | Cold us | Flush us | Step us | Kernels in step us |\n|---|---|---|---|---:|---:|---:|---:|---:|---:|\n");
        for a in &capture.autopsies {
            for f in &a.functions {
                s.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {:.0} | {:.0} | {:.0} | {} | {} |\n",
                    a.model,
                    a.workload,
                    if a.mtp { "mtp" } else { "plain" },
                    f.function,
                    f.kernels.len(),
                    f.warm_us,
                    f.cold_us,
                    f.flush_us,
                    opt(f.graph_total_us, 0),
                    opt(f.graph_kernel_us, 0)
                ));
            }
        }
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture_of(
        name: &str,
        texts: Vec<TextRun>,
        numerics_probes: &[(&str, &str, bool)],
    ) -> Capture {
        let mut capture: Capture = serde_json::from_value(serde_json::json!({
            "manifest": {"name": name, "created": "", "updated": "", "build_id": "b", "rt_head": "", "lm_head": "", "tier": "quick", "models": ["e2b"], "mtp": false, "host_mode": "diagnostic"},
            "numerics": [], "perf": []
        }))
        .unwrap();
        for (model, prompt, mtp) in numerics_probes {
            let run: crate::numerics::NumericsRun = serde_json::from_value(serde_json::json!({
                "model": model, "prompt": prompt, "mtp": mtp, "build_id": "b", "exit_code": 0, "wall_s": 1.0, "steps": [], "tokens": [1, 2, 3],
                "decode_tokens": 3, "decode_tps": 1.0, "prefill_tokens": 1, "prefill_tps": 1.0, "acceptance": null, "text_len": 3, "text_sha256": "x",
                "loopiness": 0.0, "nonfinite": false, "sdk_version": null, "weight_store": null, "device_samplers": 0
            }))
            .unwrap();
            capture.numerics.push(run);
        }
        capture.texts = texts;
        capture
    }

    /// A capture of MTP numerics runs with these (prompt, accepted, drafted, text).
    fn mtp_capture(name: &str, runs: &[(&str, u64, u64, &str)]) -> Capture {
        let mut capture = capture_of(name, Vec::new(), &[]);
        for (prompt, accepted, drafted, text) in runs {
            let run: crate::numerics::NumericsRun = serde_json::from_value(serde_json::json!({
                "model": "e4b", "prompt": prompt, "mtp": true, "build_id": "b", "exit_code": 0, "wall_s": 1.0, "steps": [], "tokens": [1],
                "decode_tokens": 256, "decode_tps": 1.0, "prefill_tokens": 1, "prefill_tps": 1.0,
                "acceptance": *accepted as f64 / *drafted as f64, "accepted": accepted, "drafted": drafted, "rounds": 90,
                "text_len": 3, "text_sha256": text, "loopiness": 0.0, "nonfinite": false, "sdk_version": null, "weight_store": null, "device_samplers": 0
            }))
            .unwrap();
            capture.numerics.push(run);
        }
        capture
    }

    #[test]
    fn acceptance_falling_on_every_prompt_fails_and_a_mixed_one_passes() {
        let t = crate::config::AcceptanceThresholds::default();
        // The INT8 GEMM's E4B 16k case: 52.5% against 44.2%, and two prompts lower too.
        let base = mtp_capture(
            "fp16",
            &[("doc_16k", 147, 280, "a"), ("code_grid", 160, 240, "b"), ("reason", 70, 120, "c")],
        );
        let cand = mtp_capture(
            "int8",
            &[("doc_16k", 124, 280, "x"), ("code_grid", 150, 240, "y"), ("reason", 64, 120, "z")],
        );
        let checks = check_acceptance(&base, &cand, &t);
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0].fell, 3);
        assert_eq!(checks[0].verdict, Verdict::Fail, "{:?}", checks[0].reasons);
        // Moves of both signs with the pool within two points: text-dependent, not judged down.
        let mixed = mtp_capture(
            "mixed",
            &[("doc_16k", 150, 280, "x"), ("code_grid", 150, 240, "y"), ("reason", 75, 120, "z")],
        );
        let checks = check_acceptance(&base, &mixed, &t);
        assert_eq!(checks[0].verdict, Verdict::Pass, "{:?}", checks[0].reasons);
        // The same build twice: identical, PASS.
        let checks = check_acceptance(&base, &base, &t);
        assert_eq!(checks[0].verdict, Verdict::Pass);
        assert!(checks[0].probes.iter().all(|p| p.text_identical && p.base == p.cand));
        // Too few prompts paired: reported, not judged.
        let one = mtp_capture("one", &[("doc_16k", 100, 280, "x")]);
        let checks = check_acceptance(&base, &one, &t);
        assert_eq!(checks[0].verdict, Verdict::Pass);
        assert_eq!(checks[0].probes.len(), 1);
    }

    #[test]
    fn acceptance_without_counts_is_left_out() {
        let t = crate::config::AcceptanceThresholds::default();
        let base = capture_of("old", Vec::new(), &[("e4b", "doc_16k", true)]);
        let cand = mtp_capture("new", &[("doc_16k", 100, 280, "x")]);
        assert!(check_acceptance(&base, &cand, &t).is_empty());
    }

    fn text(variant: &str, text: &str, samplers: usize, candidates: usize, top_k: u32) -> TextRun {
        TextRun {
            model: "e2b".into(),
            kind: "constraint".into(),
            variant: variant.into(),
            build_id: "b".into(),
            exit_code: 0,
            wall_s: 1.0,
            text_len: text.chars().count(),
            text: text.into(),
            regex_ok: Some(true),
            device_samplers: samplers,
            device_candidates: candidates,
            top_k,
            immediate_upload: variant == "immediate",
            parser_errors: 0,
        }
    }

    #[test]
    fn text_checks_judge_every_device_variant_against_the_host() {
        let host = text("host", "abc", 0, 0, 64);
        let device = text("device", "abc", 2, 1, 64);
        let immediate = text("immediate", "abd", 2, 1, 64);
        let base = capture_of("base", vec![host.clone(), device.clone()], &[]);
        let cand = capture_of("cand", vec![host.clone(), device.clone(), immediate], &[]);
        let checks = check_texts(&base, &cand);
        assert_eq!(checks.len(), 2);
        let device_check = checks.iter().find(|c| c.variant == "device").unwrap();
        assert_eq!(device_check.verdict, Verdict::Pass);
        assert_eq!(device_check.same_as_baseline, Some(true));
        let immediate_check = checks.iter().find(|c| c.variant == "immediate").unwrap();
        assert_eq!(immediate_check.verdict, Verdict::Warn, "{:?}", immediate_check.reasons);
        assert!(immediate_check.reasons[0].contains("part at char 2"));
        assert_eq!(immediate_check.same_as_baseline, None);
        // The verifier's device path missing at k > 1, and a host run that is not on the host.
        let fallen = capture_of(
            "fallen",
            vec![text("host", "abc", 1, 0, 64), text("device", "abc", 1, 0, 64)],
            &[],
        );
        let checks = check_texts(&base, &fallen);
        assert_eq!(checks[0].verdict, Verdict::Warn);
        assert!(
            checks[0].reasons.iter().any(|r| r.contains("no candidate selection at k 64")),
            "{:?}",
            checks[0].reasons
        );
        assert!(
            checks[0].reasons.iter().any(|r| r.contains("host run reports 1 device sampler")),
            "{:?}",
            checks[0].reasons
        );
    }

    #[test]
    fn a_comparison_says_what_it_did_not_compare() {
        let config = Config::load(None).unwrap();
        let base = capture_of(
            "base",
            vec![text("host", "abc", 0, 0, 64), text("device", "abc", 2, 1, 64)],
            &[("e2b", "code_grid", false), ("e2b", "code_grid", true)],
        );
        let cand = capture_of("cand", vec![], &[("e2b", "reason", false)]);
        let c = compare(&base, &cand, &config, &[]);
        assert_eq!(c.verdict, Verdict::Warn, "{:?}", c.reasons);
        assert!(c.reasons.iter().any(|r| r.contains("no numerics probe compared")));
        assert!(c.reasons.iter().any(|r| r.contains("no text probe compared")));
        assert!(
            c.not_compared.iter().any(|l| l == "numerics e2b code_grid mtp: not in the candidate"),
            "{:?}",
            c.not_compared
        );
        assert!(c
            .not_compared
            .iter()
            .any(|l| l == "numerics e2b reason plain: not in the baseline"));
        assert!(c
            .not_compared
            .iter()
            .any(|l| l == "text e2b constraint device: not in the candidate"));
        let rendered = render_comparison(&c);
        assert!(rendered.contains("## Not compared"));
        assert!(rendered.contains("- WARN: no numerics probe compared"));
    }

    #[test]
    fn the_localization_says_how_many_outputs_were_above_the_cap() {
        let probe = |above_cap: usize, above_cap_state: usize| ProbeLocalization {
            model: "e2b".into(),
            prompt: "doc_8k".into(),
            mtp: false,
            parted_token: None,
            tokens_base: 3,
            tokens_cand: 3,
            parted_char: None,
            common_words: 0,
            localization: Localization { above_cap, above_cap_state, ..Localization::default() },
            row_kls: Vec::new(),
            max_row_kl: None,
        };
        let report = |p: ProbeLocalization| LocalizationReport {
            base_name: "a".into(),
            cand_name: "b".into(),
            base_build: "x".into(),
            cand_build: "y".into(),
            probes: vec![p],
            skipped: Vec::new(),
            not_compared: Vec::new(),
            texts: Vec::new(),
            text_partings: Vec::new(),
            autopsies: Vec::new(),
        };
        let none = render_localization(&report(probe(0, 0)));
        assert!(!none.contains("above the dump's cap"), "{none}");
        let logits = render_localization(&report(probe(8, 0)));
        assert!(
            logits.contains("- e2b doc_8k: 8 outputs above the dump's cap: the steps"),
            "{logits}"
        );
        let all = render_localization(&report(probe(270, 270)));
        assert!(
            all.contains("270 outputs above the dump's cap (all state, the caches): the steps"),
            "{all}"
        );
        let some = render_localization(&report(probe(308, 300)));
        assert!(
            some.contains(
                "308 outputs above the dump's cap (300 of them state, the caches): the steps"
            ),
            "{some}"
        );
    }

    #[test]
    fn parted_tokens_and_the_legend() {
        assert_eq!(parted_tokens(&[1, 2, 3], &[1, 2, 3]), None);
        assert_eq!(parted_tokens(&[1, 2, 3], &[1, 2, 4]), Some(2));
        assert_eq!(parted_tokens(&[1, 2], &[1, 2, 3]), Some(2));
        let legend = workload_legend(&["short".into(), "8k".into(), "nope".into()]);
        assert_eq!(legend.len(), 2);
        assert_eq!(
            (
                legend[0].record_name.as_str(),
                legend[0].input_tokens,
                legend[0].output_tokens,
                legend[0].capacity
            ),
            ("1k", 1024, 256, 2048)
        );
        let text = render_legend(&legend);
        assert!(text.contains("short (the record's 1k) = 1024 input tokens, 256 requested output tokens, runtime capacity 2048"), "{text}");
        assert!(text.contains("8k = 8192 input tokens"));
    }
}

/// The predictor's own error on a record: fitted on the record's 2k and 8k
/// points, predicting the others.
pub fn validate_predictor(record: &[RecordPoint], fit_workloads: &[String]) -> String {
    let mut s = String::from("## Predictor validation on the record\n\nFitted on the record's own measured workloads given, predicting the record's other contexts.\n\n| Model | Workload | Metric | Recorded | Predicted | Error |\n|---|---|---|---:|---:|---:|\n");
    let models: BTreeSet<&str> = record.iter().map(|r| r.model.as_str()).collect();
    for model in models {
        let as_points: Vec<PerfPoint> = record
            .iter()
            .filter(|r| {
                r.model == model && !r.mtp && !r.text && fit_workloads.contains(&r.workload)
            })
            .filter_map(|r| {
                let w = workload(&r.workload).ok()?;
                Some(PerfPoint {
                    model: r.model.clone(),
                    workload: r.workload.clone(),
                    mtp: false,
                    text: false,
                    label: String::new(),
                    input_tokens: w.input,
                    output_tokens: w.output,
                    context: w.context,
                    iterations: 0,
                    warmups: 0,
                    measured_iterations: 0,
                    prompt_tokens: None,
                    decode_tps: r.decode_tps,
                    prefill_tps: r.prefill_tps,
                    decode_tps_values: Vec::new(),
                    prefill_tps_values: Vec::new(),
                    acceptance: None,
                    tokens_per_round: None,
                    round_ms: None,
                    draft_ms: None,
                    verify_ms: None,
                    init_total_s: None,
                    max_rss_gib: None,
                    exit_code: 0,
                })
            })
            .collect();
        let cm = CostModel::fit(model, &as_points);
        for r in record.iter().filter(|r| {
            r.model == model && !r.mtp && !r.text && !fit_workloads.contains(&r.workload)
        }) {
            for (metric, recorded, predicted) in [
                ("decode", r.decode_tps, cm.predict_decode_tps(&r.workload)),
                ("prefill", r.prefill_tps, cm.predict_prefill_tps(&r.workload)),
            ] {
                if let (Some(rec), Some(pred)) = (recorded, predicted) {
                    s.push_str(&format!(
                        "| {} | {} | {} | {rec:.1} | {pred:.1} | {} |\n",
                        model,
                        r.workload,
                        metric,
                        pct(pred / rec)
                    ));
                }
            }
        }
    }
    s
}
