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
use crate::texts::{compare_texts, parted_at, TextComparison, TextRun};

/// A text probe of the candidate judged: the device and the host must agree,
/// the constrained text must match its regex, and both runs must end well.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextCheck {
    pub model: String,
    pub kind: String,
    pub comparison: TextComparison,
    /// Whether the candidate's device text equals the baseline's.
    pub same_as_baseline: Option<bool>,
    pub verdict: Verdict,
    pub reasons: Vec<String>,
}

/// What a workload name means (the report's measurement language).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadLegend {
    pub name: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub context: u32,
}

pub fn workload_legend(names: &[String]) -> Vec<WorkloadLegend> {
    WORKLOADS
        .iter()
        .filter(|w| names.iter().any(|n| n == w.name))
        .map(|w| WorkloadLegend {
            name: w.name.to_string(),
            input_tokens: w.input,
            output_tokens: w.output,
            context: w.context,
        })
        .collect()
}

fn text_run<'a>(
    runs: &'a [TextRun],
    model: &str,
    kind: &str,
    variant: &str,
) -> Option<&'a TextRun> {
    runs.iter().find(|r| r.model == model && r.kind == kind && r.variant == variant)
}

pub fn check_texts(base: &Capture, cand: &Capture) -> Vec<TextCheck> {
    let mut out = Vec::new();
    let keys: BTreeSet<(String, String)> =
        cand.texts.iter().map(|r| (r.model.clone(), r.kind.clone())).collect();
    for (model, kind) in keys {
        let (Some(device), Some(host)) = (
            text_run(&cand.texts, &model, &kind, "device"),
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
                format!("exit codes device {} host {}", device.exit_code, host.exit_code),
            );
        }
        if device.parser_errors > 0 || host.parser_errors > 0 {
            raise(
                Verdict::Fail,
                format!(
                    "parser errors device {} host {}",
                    device.parser_errors, host.parser_errors
                ),
            );
        }
        if device.regex_ok == Some(false) {
            raise(Verdict::Fail, "the device text violates the regex".to_string());
        }
        if host.regex_ok == Some(false) {
            raise(Verdict::Fail, "the host text violates the regex".to_string());
        }
        if device.device_samplers == 0 {
            raise(Verdict::Warn, "the device run reports no device sampler".to_string());
        }
        if !comparison.identical {
            raise(
                Verdict::Warn,
                format!(
                    "device and host texts part at char {} ({} words in common; a tie at the k-th logit, or a sampler change)",
                    comparison.parted_char.unwrap_or(0),
                    comparison.common_words
                ),
            );
        }
        let same_as_baseline =
            text_run(&base.texts, &model, &kind, "device").map(|b| b.text == device.text);
        out.push(TextCheck { model, kind, comparison, same_as_baseline, verdict, reasons });
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
    pub verdict: Verdict,
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
    let mut names: Vec<String> = config.perf.workloads.clone();
    names.extend(config.perf.predict.iter().cloned());
    names.push(config.probes.autopsy.workload.clone());
    let workloads = workload_legend(&names);
    let verdict = numerics
        .iter()
        .map(|n| n.verdict)
        .chain(perf.iter().map(|p| p.verdict))
        .chain(texts.iter().map(|t| t.verdict))
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
    if c.base_host_mode != c.cand_host_mode {
        s.push_str(&format!("Host modes differ: baseline {}, candidate {}; timings are not comparable across them.\n\n", c.base_host_mode, c.cand_host_mode));
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
        s.push_str("## Texts (sampled and constrained decodes, MTP verifier on the device vs the host)\n\n| Model | Probe | Device vs host | Regex (device, host) | Device samplers | Chars (device, host) | Same as baseline | Verdict |\n|---|---|---|---|---:|---|---|---|\n");
        for t in &c.texts {
            let cmp = &t.comparison;
            let regex = |v: Option<bool>| match v {
                Some(true) => "ok",
                Some(false) => "VIOLATED",
                None => "-",
            };
            s.push_str(&format!(
                "| {} | {} | {} | {}, {} | {} | {}, {} | {} | {} |\n",
                t.model,
                t.kind,
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
            s.push_str(&format!("- {} {}: {}\n", t.model, t.kind, t.reasons.join("; ")));
        }
        s.push('\n');
    }
    if !c.workloads.is_empty() {
        s.push_str("Workloads: ");
        let legend: Vec<String> = c
            .workloads
            .iter()
            .map(|w| {
                format!(
                    "{} = {} input tokens, {} output tokens (context {})",
                    w.name, w.input_tokens, w.output_tokens, w.context
                )
            })
            .collect();
        s.push_str(&legend.join("; "));
        s.push_str(". tok/s are tokens per second of the measured phase.\n\n");
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
    s
}

/// The kernels whose warm times moved, per model: the attribution of a
/// performance change (the autopsy times every kernel node of a step graph
/// alone, so the sums are not the step's time).
pub fn render_autopsy_diffs(diffs: &[AutopsyDiff]) -> String {
    let mut s = String::new();
    if diffs.is_empty() {
        return s;
    }
    s.push_str("## Kernels (autopsy of the step graphs, warm times per kernel)\n\nEvery kernel node of a function's CUDA graph timed alone; a kernel's time is summed over its launches in the step. The sums are the kernels' own times, not the step's.\n\n");
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
            s.push_str("| Function | Kernel | Launches (base, cand) | Base us | Cand us | Δ us |\n|---|---|---|---:|---:|---:|\n");
            for k in moved {
                s.push_str(&format!(
                    "| {} | `{}` | {}, {} | {:.1} | {:.1} | {:+.1} |\n",
                    k.function,
                    k.short,
                    k.base_count,
                    k.cand_count,
                    k.base_us,
                    k.cand_us,
                    k.delta_us
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
    /// The step at which the greedy tokens part (the top-K logits).
    pub parted_token: Option<usize>,
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

pub fn localize_captures(
    base: &Capture,
    cand: &Capture,
    config: &Config,
) -> Result<LocalizationReport> {
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
        let comparison = compare_runs(b, c, &config.numerics.thresholds);
        let localization = localize(&base_dumps, &cand_dumps)?;
        let row_kls = full_row_kls(&base_dumps, &cand_dumps)?;
        let max_row_kl = row_kls
            .iter()
            .map(|r| r.kl)
            .fold(None, |m: Option<f64>, v| Some(m.map_or(v, |m| m.max(v))));
        probes.push(ProbeLocalization {
            model: c.model.clone(),
            prompt: c.prompt.clone(),
            mtp: c.mtp,
            parted_token: comparison.diverged_at,
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
        texts: check_texts(base, cand),
        text_partings,
        autopsies: autopsy_diffs(base, cand),
    })
}

fn tensor_cell(d: &TensorDiff) -> String {
    format!(
        "c{} {} step {} out {} `{}` {:?}",
        d.context, d.function, d.step, d.output, d.name, d.dims
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
        s.push_str("## Where the outputs part (whole-output dumps of the first steps)\n\nTensors are compared in production order; the first that differs names the step and the partition (function) the change first reaches. A relative move is rms(Δ) / rms(baseline).\n\n| Model | Prompt | Mode | Tokens part at | Tensors paired | Differing | First differing tensor | Rel. move | max abs delta | Largest move | Rel. | max row KL |\n|---|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|\n");
        for p in &r.probes {
            let l = &p.localization;
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                p.model,
                p.prompt,
                if p.mtp { "mtp" } else { "plain" },
                p.parted_token.map(|t| t.to_string()).unwrap_or_else(|| "-".into()),
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
                "- {} {}{}: the first difference is at step {} of `{}` (context {}), output {} `{}`: {} of {} elements differ, max |Δ| {:.3e}, rms Δ {:.3e} (relative {:.2e}){}{}{}. The functions reached by that step: {}.\n",
                p.model,
                p.prompt,
                if p.mtp { " (mtp)" } else { "" },
                first.step,
                first.function,
                first.context,
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
                l.first_step
                    .iter()
                    .map(|d| format!("{} out {} ({:.1e})", d.function, d.output, d.relative))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            if l.unpaired > 0 {
                s.push_str(&format!(
                    "  ({} tensors without a counterpart in the other run)\n",
                    l.unpaired
                ));
            }
        }
        s.push('\n');
        let with_rows: Vec<&ProbeLocalization> =
            r.probes.iter().filter(|p| !p.row_kls.is_empty()).collect();
        if !with_rows.is_empty() {
            s.push_str("## KL over the whole vocabulary (the first logits rows)\n\n| Model | Prompt | Mode | Row (function, step) | KL | max abs delta logit | Same argmax |\n|---|---|---|---|---:|---:|---|\n");
            for p in with_rows {
                for k in p.row_kls.iter().take(12) {
                    s.push_str(&format!(
                        "| {} | {} | {} | {} step {} row {} | {:.2e} | {:.3e} | {} |\n",
                        p.model,
                        p.prompt,
                        if p.mtp { "mtp" } else { "plain" },
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
                "- {} {} (device vs host): {}\n",
                t.model,
                t.kind,
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
        s.push_str("## Autopsies (every kernel node of a step graph timed alone)\n\n| Model | Workload | Mode | Function | Kernels | Warm us | Cold us | Flush us |\n|---|---|---|---|---:|---:|---:|---:|\n");
        for a in &capture.autopsies {
            for f in &a.functions {
                s.push_str(&format!(
                    "| {} | {} | {} | {} | {} | {:.0} | {:.0} | {:.0} |\n",
                    a.model,
                    a.workload,
                    if a.mtp { "mtp" } else { "plain" },
                    f.function,
                    f.kernels.len(),
                    f.warm_us,
                    f.cold_us,
                    f.flush_us
                ));
            }
        }
        s.push('\n');
    }
    s
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
