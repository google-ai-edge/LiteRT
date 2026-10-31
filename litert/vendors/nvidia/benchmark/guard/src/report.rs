// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The comparison of two captures and its rendering (markdown and JSON), and
//! the cost-model predictions of a capture.

use std::collections::BTreeSet;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::capture::Capture;
use crate::config::{workload, Config};
use crate::numerics::{compare_runs, NumericsComparison, Verdict};
use crate::perf::{
    compare_points, decode_context, interpolate, CostModel, PerfComparison, PerfPoint,
};

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
    let verdict = numerics
        .iter()
        .map(|n| n.verdict)
        .chain(perf.iter().map(|p| p.verdict))
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
    if c.base_host_mode != c.cand_host_mode {
        s.push_str(&format!("Host modes differ: baseline {}, candidate {}; timings are not comparable across them.\n\n", c.base_host_mode, c.cand_host_mode));
    }
    if !c.numerics.is_empty() {
        s.push_str("## Numerics (greedy decode, logits per step)\n\n");
        s.push_str("| Model | Prompt | Mode | Steps compared | Parted at | max abs delta | mean KL | top-K overlap | Text | Decode tok/s (base, cand) | Verdict |\n|---|---|---|---:|---:|---:|---:|---:|---|---|---|\n");
        for n in &c.numerics {
            s.push_str(&format!(
                "| {} | {} | {} | {} / {} | {} | {:.4} | {:.2e} | {:.3} | {}{} | {}, {} | {} |\n",
                n.model,
                n.prompt,
                if n.mtp { "mtp" } else { "plain" },
                n.steps_compared,
                n.steps_base.min(n.steps_cand),
                n.diverged_at.map(|d| d.to_string()).unwrap_or_else(|| "-".into()),
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
    if !c.perf.is_empty() {
        s.push_str("## Performance (measured)\n\n| Model | Workload | Mode | Metric | Baseline | Candidate | Change | Verdict |\n|---|---|---|---|---:|---:|---:|---|\n");
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
            s.push_str(&format!(
                "| {} | {} | {} | {} | {:.digits$} | {:.digits$} | {} | {} |\n",
                p.model,
                p.workload,
                mode,
                p.metric,
                p.base,
                p.cand,
                pct(p.ratio),
                p.verdict.label()
            ));
        }
        s.push('\n');
    }
    if !c.predictions.is_empty() {
        s.push_str("## Performance (predicted at unmeasured contexts)\n\nCost models: decode step time a + b·context, prefill time per token u + v·prompt, fitted on the measured workloads; the expected candidate value applies the measured change ratio, interpolated in context, to the record where one is given (else to the baseline's model).\n\n| Model | Workload | Metric | Baseline model | Candidate model | Record | Expected candidate | Change |\n|---|---|---|---:|---:|---:|---:|---:|\n");
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
    s
}

pub fn render_capture(capture: &Capture) -> String {
    let m = &capture.manifest;
    let mut s = format!(
        "# Capture {} (build {}, tier {}, host {}, {})\n\nLiteRT {}\nLiteRT-LM {}\n\n",
        m.name, m.build_id, m.tier, m.host_mode, m.updated, m.rt_head, m.lm_head
    );
    if !capture.numerics.is_empty() {
        s.push_str("## Numerics probes\n\n| Model | Prompt | Mode | Exit | Steps | Prefill tokens | Decode tokens | Decode tok/s | Loopiness | Non-finite | Wall s |\n|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|\n");
        for n in &capture.numerics {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {:.2} | {} | {:.0} |\n",
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
                n.wall_s
            ));
        }
        s.push('\n');
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
        s.push_str("## Cost models\n\n| Model | Decode step ms = a + b·context | Prefill s/token = u + v·prompt | Fit points |\n|---|---|---|---:|\n");
        for model in models {
            let cm = CostModel::fit(model, &capture.perf);
            s.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                model,
                cm.decode_ms
                    .map(|(a, b)| format!("{a:.3} + {b:.3e}·c"))
                    .unwrap_or_else(|| "-".into()),
                cm.prefill_s_per_token
                    .map(|(u, v)| format!("{u:.3e} + {v:.3e}·n"))
                    .unwrap_or_else(|| "-".into()),
                cm.decode_fit_points.len()
            ));
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
