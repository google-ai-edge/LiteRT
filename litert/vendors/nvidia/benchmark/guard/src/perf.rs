// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The performance probe's data: the throughputs of bench.sh runs (the CLI's
//! own benchmark report), the cost model that extrapolates them to other
//! context lengths, and the comparison of two captures.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::{workload, PerfThresholds};
use crate::numerics::Verdict;
use crate::parse::{mean, measured, parse_cli_log, parse_time};

/// One bench.sh run, summarized like report.py does.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfPoint {
    pub model: String,
    pub workload: String,
    pub mtp: bool,
    pub text: bool,
    pub label: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub context: u32,
    pub iterations: usize,
    pub warmups: usize,
    pub measured_iterations: usize,
    pub prompt_tokens: Option<u64>,
    pub decode_tps: Option<f64>,
    pub prefill_tps: Option<f64>,
    pub decode_tps_values: Vec<f64>,
    pub prefill_tps_values: Vec<f64>,
    pub acceptance: Option<f64>,
    pub tokens_per_round: Option<f64>,
    pub round_ms: Option<f64>,
    pub draft_ms: Option<f64>,
    pub verify_ms: Option<f64>,
    pub init_total_s: Option<f64>,
    pub max_rss_gib: Option<f64>,
    pub exit_code: i32,
}

#[derive(Debug, Deserialize)]
struct BenchMeta {
    label: String,
    profile: String,
    workload: String,
    mtp: bool,
    prompt_file: Option<String>,
    input_tokens: u32,
    output_tokens: u32,
    context: u32,
    iterations: usize,
    warmups: usize,
    exit_code: i32,
}

/// Reads a bench.sh run directory (meta.json, cli.log, time.txt).
pub fn summarize_run(run_dir: &Path) -> Result<PerfPoint> {
    let meta_text = std::fs::read_to_string(run_dir.join("meta.json"))
        .with_context(|| format!("reading {}", run_dir.join("meta.json").display()))?;
    let meta: BenchMeta = serde_json::from_str(&meta_text).context("parsing meta.json")?;
    let log_text = std::fs::read_to_string(run_dir.join("cli.log")).unwrap_or_default();
    let log = parse_cli_log(&log_text);
    let its = measured(&log.iterations, meta.warmups);
    let decode_tps_values: Vec<f64> =
        its.iter().filter_map(|i| i.decode.first().map(|t| t.tokens_per_second)).collect();
    let prefill_tps_values: Vec<f64> =
        its.iter().filter_map(|i| i.prefill.first().map(|t| t.tokens_per_second)).collect();
    let time_text = std::fs::read_to_string(run_dir.join("time.txt")).unwrap_or_default();
    let (max_rss_gib, _) = parse_time(&time_text);
    Ok(PerfPoint {
        model: meta.profile,
        workload: meta.workload,
        mtp: meta.mtp,
        text: meta.prompt_file.is_some(),
        label: meta.label,
        input_tokens: meta.input_tokens,
        output_tokens: meta.output_tokens,
        context: meta.context,
        iterations: meta.iterations,
        warmups: meta.warmups,
        measured_iterations: its.len(),
        prompt_tokens: its.iter().filter_map(|i| i.prefill.first().map(|t| t.tokens)).next(),
        decode_tps: mean(decode_tps_values.iter().cloned()),
        prefill_tps: mean(prefill_tps_values.iter().cloned()),
        decode_tps_values,
        prefill_tps_values,
        acceptance: mean(its.iter().filter_map(|i| i.mtp.as_ref().map(|m| m.acceptance))),
        tokens_per_round: mean(
            its.iter().filter_map(|i| i.mtp.as_ref().and_then(|m| m.tokens_per_round)),
        ),
        round_ms: mean(its.iter().filter_map(|i| i.mtp.as_ref().and_then(|m| m.round_ms))),
        draft_ms: mean(its.iter().filter_map(|i| i.mtp.as_ref().and_then(|m| m.draft_ms))),
        verify_ms: mean(its.iter().filter_map(|i| i.mtp.as_ref().and_then(|m| m.verify_ms))),
        init_total_s: log.iterations.first().and_then(|i| i.init_total_s),
        max_rss_gib,
        exit_code: meta.exit_code,
    })
}

/// Least-squares line y = a + b x; one point gives a constant.
pub fn fit_line(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    match points.len() {
        0 => None,
        1 => Some((points[0].1, 0.0)),
        n => {
            let n = n as f64;
            let sx: f64 = points.iter().map(|p| p.0).sum();
            let sy: f64 = points.iter().map(|p| p.1).sum();
            let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
            let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
            let denominator = n * sxx - sx * sx;
            if denominator.abs() < 1e-12 {
                return Some((sy / n, 0.0));
            }
            let b = (n * sxy - sx * sy) / denominator;
            let a = (sy - b * sx) / n;
            Some((a, b))
        }
    }
}

/// The context a decode step of a workload sees on average: the prompt plus
/// half of the decoded tokens.
pub fn decode_context(input_tokens: u32, output_tokens: u32) -> f64 {
    input_tokens as f64 + output_tokens as f64 / 2.0
}

/// Least-squares parabola y = c0 + c1 x + c2 x² (three points or more; a
/// line with two, a constant with one).
pub fn fit_parabola(points: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    if points.len() < 3 {
        return fit_line(points).map(|(a, b)| (a, b, 0.0));
    }
    // Normal equations in a scaled x (the contexts span 1e3..1e5).
    let scale = points.iter().map(|p| p.0.abs()).fold(0.0, f64::max).max(1.0);
    let xs: Vec<f64> = points.iter().map(|p| p.0 / scale).collect();
    let ys: Vec<f64> = points.iter().map(|p| p.1).collect();
    let mut m = [[0.0f64; 3]; 3];
    let mut v = [0.0f64; 3];
    for (x, y) in xs.iter().zip(ys.iter()) {
        let powers = [1.0, *x, x * x];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += powers[i] * powers[j];
            }
            v[i] += powers[i] * y;
        }
    }
    // Gaussian elimination.
    for i in 0..3 {
        let pivot = (i..3).max_by(|a, b| m[*a][i].abs().partial_cmp(&m[*b][i].abs()).unwrap())?;
        m.swap(i, pivot);
        v.swap(i, pivot);
        if m[i][i].abs() < 1e-18 {
            return fit_line(points).map(|(a, b)| (a, b, 0.0));
        }
        for r in (i + 1)..3 {
            let f = m[r][i] / m[i][i];
            for c in i..3 {
                m[r][c] -= f * m[i][c];
            }
            v[r] -= f * v[i];
        }
    }
    let mut c = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut s = v[i];
        for j in (i + 1)..3 {
            s -= m[i][j] * c[j];
        }
        c[i] = s / m[i][i];
    }
    Some((c[0], c[1] / scale, c[2] / (scale * scale)))
}

/// The cost model of one model's plain decoding and prefill: decode step
/// time = a + b * context (the attention over the cache grows with it);
/// prefill time of a prompt = c0 + c1 * N + c2 * N² (a fixed cost, the
/// per-token work, and the attention's growth, which only a third measured
/// context can fit: with two it is folded into c1 and long contexts come out
/// optimistic). The measured change ratios (predictions' `cand_expected`)
/// are the robust yardstick; the model is the physics view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    pub model: String,
    pub decode_fit_points: Vec<(f64, f64)>,
    /// (prompt tokens, prefill seconds)
    pub prefill_fit_points: Vec<(f64, f64)>,
    pub decode_ms: Option<(f64, f64)>,
    pub prefill_s: Option<(f64, f64, f64)>,
}

impl CostModel {
    pub fn fit(model: &str, points: &[PerfPoint]) -> CostModel {
        let plain: Vec<&PerfPoint> =
            points.iter().filter(|p| p.model == model && !p.mtp && !p.text).collect();
        let decode_fit_points: Vec<(f64, f64)> = plain
            .iter()
            .filter_map(|p| {
                p.decode_tps
                    .map(|tps| (decode_context(p.input_tokens, p.output_tokens), 1000.0 / tps))
            })
            .collect();
        let prefill_fit_points: Vec<(f64, f64)> = plain
            .iter()
            .filter_map(|p| {
                p.prefill_tps.map(|tps| (p.input_tokens as f64, p.input_tokens as f64 / tps))
            })
            .collect();
        CostModel {
            model: model.to_string(),
            decode_ms: fit_line(&decode_fit_points),
            prefill_s: fit_parabola(&prefill_fit_points),
            decode_fit_points,
            prefill_fit_points,
        }
    }

    pub fn predict_decode_tps(&self, workload_name: &str) -> Option<f64> {
        let w = workload(workload_name).ok()?;
        let (a, b) = self.decode_ms?;
        let ms = a + b * decode_context(w.input, w.output);
        (ms > 0.0).then(|| 1000.0 / ms)
    }

    pub fn predict_prefill_tps(&self, workload_name: &str) -> Option<f64> {
        let w = workload(workload_name).ok()?;
        let (c0, c1, c2) = self.prefill_s?;
        let n = w.input as f64;
        let s = c0 + c1 * n + c2 * n * n;
        (s > 0.0).then(|| n / s)
    }
}

/// Linear interpolation (extrapolation beyond the ends) of y at x.
pub fn interpolate(points: &[(f64, f64)], x: f64) -> Option<f64> {
    let mut sorted: Vec<(f64, f64)> = points.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    match sorted.len() {
        0 => None,
        1 => Some(sorted[0].1),
        _ => {
            let (lo, hi) = if x <= sorted[0].0 {
                (sorted[0], sorted[1])
            } else if x >= sorted[sorted.len() - 1].0 {
                (sorted[sorted.len() - 2], sorted[sorted.len() - 1])
            } else {
                let i = sorted.iter().position(|p| p.0 > x).unwrap_or(1).max(1);
                (sorted[i - 1], sorted[i])
            };
            if (hi.0 - lo.0).abs() < 1e-12 {
                return Some(lo.1);
            }
            Some(lo.1 + (hi.1 - lo.1) * (x - lo.0) / (hi.0 - lo.0))
        }
    }
}

/// One cell of the performance comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfComparison {
    pub model: String,
    pub workload: String,
    pub mtp: bool,
    pub text: bool,
    pub metric: String,
    pub base: f64,
    pub cand: f64,
    pub ratio: f64,
    /// The spread (max - min) / mean of the measured iterations, baseline and
    /// candidate: a change inside the larger spread is noise, not a verdict.
    pub base_spread: Option<f64>,
    pub cand_spread: Option<f64>,
    pub verdict: Verdict,
}

fn spread(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean > 0.0).then(|| (max - min) / mean)
}

pub fn perf_verdict(ratio: f64, thresholds: &PerfThresholds) -> Verdict {
    if ratio < thresholds.fail_ratio {
        Verdict::Fail
    } else if ratio < thresholds.warn_ratio {
        Verdict::Warn
    } else {
        Verdict::Pass
    }
}

pub fn compare_points(
    base: &[PerfPoint],
    cand: &[PerfPoint],
    thresholds: &PerfThresholds,
) -> Vec<PerfComparison> {
    let mut rows = Vec::new();
    for c in cand {
        let Some(b) = base.iter().find(|b| b.label == c.label && b.model == c.model) else {
            continue;
        };
        let metrics: [(&str, Option<f64>, Option<f64>, Option<f64>, Option<f64>); 4] = [
            (
                "decode",
                b.decode_tps,
                c.decode_tps,
                spread(&b.decode_tps_values),
                spread(&c.decode_tps_values),
            ),
            (
                "prefill",
                b.prefill_tps,
                c.prefill_tps,
                spread(&b.prefill_tps_values),
                spread(&c.prefill_tps_values),
            ),
            ("acceptance", b.acceptance, c.acceptance, None, None),
            ("tokens/round", b.tokens_per_round, c.tokens_per_round, None, None),
        ];
        for (metric, bv, cv, bs, cs) in metrics {
            if let (Some(bv), Some(cv)) = (bv, cv) {
                if bv <= 0.0 {
                    continue;
                }
                let ratio = cv / bv;
                // Acceptance and tokens per round depend on the text the
                // sampler produces; they are reported, not judged. A change
                // within the iterations' own spread is noise: at most a warning.
                let mut verdict = if metric == "decode" || metric == "prefill" {
                    perf_verdict(ratio, thresholds)
                } else {
                    Verdict::Pass
                };
                let noise = bs.unwrap_or(0.0).max(cs.unwrap_or(0.0));
                if verdict == Verdict::Fail && (1.0 - ratio) <= noise {
                    verdict = Verdict::Warn;
                }
                rows.push(PerfComparison {
                    model: c.model.clone(),
                    workload: c.workload.clone(),
                    mtp: c.mtp,
                    text: c.text,
                    metric: metric.to_string(),
                    base: bv,
                    cand: cv,
                    ratio,
                    base_spread: bs,
                    cand_spread: cs,
                    verdict,
                });
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(model: &str, workload: &str, decode: f64, prefill: f64) -> PerfPoint {
        let w = crate::config::workload(workload).unwrap();
        PerfPoint {
            model: model.into(),
            workload: workload.into(),
            mtp: false,
            text: false,
            label: format!("{model}_{workload}_plain_latency"),
            input_tokens: w.input,
            output_tokens: w.output,
            context: w.context,
            iterations: 2,
            warmups: 1,
            measured_iterations: 1,
            prompt_tokens: Some(w.input as u64),
            decode_tps: Some(decode),
            prefill_tps: Some(prefill),
            decode_tps_values: vec![decode],
            prefill_tps_values: vec![prefill],
            acceptance: None,
            tokens_per_round: None,
            round_ms: None,
            draft_ms: None,
            verify_ms: None,
            init_total_s: None,
            max_rss_gib: None,
            exit_code: 0,
        }
    }

    #[test]
    fn line_fit() {
        let (a, b) = fit_line(&[(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)]).unwrap();
        assert!((a - 1.0).abs() < 1e-9 && (b - 2.0).abs() < 1e-9);
        assert_eq!(fit_line(&[(5.0, 7.0)]), Some((7.0, 0.0)));
        assert_eq!(interpolate(&[(0.0, 1.0), (10.0, 2.0)], 5.0), Some(1.5));
        assert_eq!(interpolate(&[(0.0, 1.0), (10.0, 2.0)], 20.0), Some(3.0));
    }

    #[test]
    fn cost_model_reproduces_a_linear_world() {
        // Step time 2.5 ms + 0.00005 ms per context token; prefill 30 ms + 25 us per token.
        let step = |ctx: f64| 2.5 + 5e-5 * ctx;
        let pre = |n: f64| 0.030 + 25e-6 * n;
        let points = vec![
            point("e2b", "short", 1000.0 / step(decode_context(1024, 256)), 1024.0 / pre(1024.0)),
            point("e2b", "8k", 1000.0 / step(decode_context(8192, 256)), 8192.0 / pre(8192.0)),
        ];
        let model = CostModel::fit("e2b", &points);
        let predicted = model.predict_decode_tps("30k").unwrap();
        let expected = 1000.0 / step(decode_context(30720, 256));
        assert!((predicted - expected).abs() / expected < 1e-6);
        let predicted = model.predict_prefill_tps("16k").unwrap();
        let expected = 16384.0 / pre(16384.0);
        assert!((predicted - expected).abs() / expected < 1e-6);
        // Three contexts fit the attention's growth too.
        let pre2 = |n: f64| 0.030 + 25e-6 * n + 1e-9 * n * n;
        let points = vec![
            point("e4b", "short", 200.0, 1024.0 / pre2(1024.0)),
            point("e4b", "8k", 190.0, 8192.0 / pre2(8192.0)),
            point("e4b", "16k", 180.0, 16384.0 / pre2(16384.0)),
        ];
        let model = CostModel::fit("e4b", &points);
        let predicted = model.predict_prefill_tps("30k").unwrap();
        let expected = 30720.0 / pre2(30720.0);
        assert!((predicted - expected).abs() / expected < 1e-4, "{predicted} vs {expected}");
    }

    #[test]
    fn comparison_verdicts() {
        let base = vec![point("e2b", "short", 300.0, 40000.0)];
        let cand = vec![point("e2b", "short", 285.0, 41000.0)];
        let rows = compare_points(&base, &cand, &PerfThresholds::default());
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].metric, "decode");
        assert_eq!(rows[0].verdict, Verdict::Warn);
        assert_eq!(rows[1].verdict, Verdict::Pass);
        let cand = vec![point("e2b", "short", 270.0, 41000.0)];
        assert_eq!(
            compare_points(&base, &cand, &PerfThresholds::default())[0].verdict,
            Verdict::Fail
        );
    }
}
