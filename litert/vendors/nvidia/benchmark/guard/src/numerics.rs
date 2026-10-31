// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The numerics probe's data and its comparison: the greedy decode of a fixed
//! prompt leaves, per step, the K largest logits with their indices (the
//! dispatch's dump) and the decoded text. Two runs of the same prompt are
//! comparable step by step while their argmax tokens agree (the inputs of a
//! step are then identical), and the metrics over those steps say how far
//! the arithmetic moved; where the texts part, and whether the candidate
//! loops or produces non-finite values, say whether it moved too far.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::config::NumericsThresholds;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Verdict {
    Pass,
    Warn,
    Fail,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Warn => "WARN",
            Verdict::Fail => "FAIL",
        }
    }

    pub fn exit_code(self) -> i32 {
        match self {
            Verdict::Pass => 0,
            Verdict::Warn => 1,
            Verdict::Fail => 2,
        }
    }
}

/// One row of logits of one step of one function: the K largest values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepLogits {
    pub context: u32,
    pub function: String,
    pub step: u32,
    pub row: usize,
    pub top: Vec<(u32, f64)>,
}

impl StepLogits {
    pub fn argmax(&self) -> Option<u32> {
        self.top.first().map(|(id, _)| *id)
    }
}

/// What one numerics probe run produced (summary.json of its directory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericsRun {
    pub model: String,
    pub prompt: String,
    pub mtp: bool,
    pub build_id: String,
    pub exit_code: i32,
    pub wall_s: f64,
    pub steps: Vec<StepLogits>,
    /// The argmax of every step row, in order (the greedy tokens of a plain run).
    pub tokens: Vec<u32>,
    pub decode_tokens: u64,
    pub decode_tps: Option<f64>,
    pub prefill_tokens: u64,
    pub prefill_tps: Option<f64>,
    pub acceptance: Option<f64>,
    pub text_len: usize,
    pub text_sha256: String,
    pub loopiness: f64,
    pub nonfinite: bool,
}

/// The share of the last 4,000 characters covered by the most repeated line
/// (lines of 20 characters or more): a repetition loop scores above 0.25.
pub fn loopiness(text: &str) -> f64 {
    let chars: Vec<char> = text.chars().collect();
    let start = chars.len().saturating_sub(4000);
    let tail: String = chars[start..].iter().collect();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for line in tail.split('\n') {
        if line.trim().chars().count() >= 20 {
            *counts.entry(line).or_insert(0) += 1;
        }
    }
    let tail_len = tail.chars().count().max(1);
    counts
        .iter()
        .map(|(line, n)| (*n * line.chars().count()) as f64 / tail_len as f64)
        .fold(0.0, f64::max)
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// The metrics of one comparable step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepDelta {
    pub max_abs_delta: f64,
    pub kl: f64,
    pub overlap: f64,
    pub margin_delta: f64,
    pub same_argmax: bool,
}

/// Compares two rows of top-K logits of the same step.
pub fn step_delta(base: &StepLogits, cand: &StepLogits) -> StepDelta {
    let k = base.top.len().min(cand.top.len()).max(1);
    let base_map: BTreeMap<u32, f64> = base.top.iter().cloned().collect();
    let cand_map: BTreeMap<u32, f64> = cand.top.iter().cloned().collect();
    let base_floor = base.top.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
    let cand_floor = cand.top.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
    let mut union: Vec<u32> = base_map.keys().cloned().collect();
    for id in cand_map.keys() {
        if !base_map.contains_key(id) {
            union.push(*id);
        }
    }
    let mut common = 0usize;
    let mut max_abs = 0.0f64;
    for (id, b) in &base_map {
        if let Some(c) = cand_map.get(id) {
            common += 1;
            max_abs = max_abs.max((c - b).abs());
        }
    }
    // Ids outside one list are below that list's floor; the floor stands in
    // for them, a bound that keeps the KL finite and conservative.
    let base_logits: Vec<f64> =
        union.iter().map(|id| *base_map.get(id).unwrap_or(&base_floor)).collect();
    let cand_logits: Vec<f64> =
        union.iter().map(|id| *cand_map.get(id).unwrap_or(&cand_floor)).collect();
    let p = softmax(&base_logits);
    let q = softmax(&cand_logits);
    let kl: f64 = p
        .iter()
        .zip(q.iter())
        .filter(|(pi, _)| **pi > 0.0)
        .map(|(pi, qi)| pi * (pi.ln() - qi.max(1e-300).ln()))
        .sum::<f64>()
        .max(0.0);
    let margin = |top: &[(u32, f64)]| -> f64 {
        match top {
            [(_, a), (_, b), ..] => a - b,
            _ => 0.0,
        }
    };
    StepDelta {
        max_abs_delta: max_abs,
        kl,
        overlap: common as f64 / k as f64,
        margin_delta: (margin(&base.top) - margin(&cand.top)).abs(),
        same_argmax: base.argmax() == cand.argmax(),
    }
}

/// The comparison of two runs of the same prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericsComparison {
    pub model: String,
    pub prompt: String,
    pub mtp: bool,
    pub steps_base: usize,
    pub steps_cand: usize,
    /// Steps compared (identical inputs): up to and including the first
    /// argmax disagreement.
    pub steps_compared: usize,
    /// The index of the first step whose argmax (or structure) differs.
    pub diverged_at: Option<usize>,
    pub max_abs_delta: f64,
    pub mean_kl: f64,
    pub max_kl: f64,
    pub mean_overlap: f64,
    pub mean_margin_delta: f64,
    pub text_identical: bool,
    pub cand_loopiness: f64,
    pub base_loopiness: f64,
    pub cand_nonfinite: bool,
    pub cand_exit_code: i32,
    pub base_decode_tps: Option<f64>,
    pub cand_decode_tps: Option<f64>,
    pub verdict: Verdict,
    pub reasons: Vec<String>,
}

pub fn compare_runs(
    base: &NumericsRun,
    cand: &NumericsRun,
    thresholds: &NumericsThresholds,
) -> NumericsComparison {
    let mut deltas: Vec<StepDelta> = Vec::new();
    let mut diverged_at = None;
    for (i, (b, c)) in base.steps.iter().zip(cand.steps.iter()).enumerate() {
        if b.function != c.function || b.row != c.row {
            diverged_at = Some(i);
            break;
        }
        let delta = step_delta(b, c);
        let same = delta.same_argmax;
        deltas.push(delta);
        if !same {
            diverged_at = Some(i);
            break;
        }
    }
    if diverged_at.is_none() && base.steps.len() != cand.steps.len() {
        diverged_at = Some(base.steps.len().min(cand.steps.len()));
    }
    let n = deltas.len().max(1) as f64;
    let max_abs_delta = deltas.iter().map(|d| d.max_abs_delta).fold(0.0, f64::max);
    let mean_kl = deltas.iter().map(|d| d.kl).sum::<f64>() / n;
    let max_kl = deltas.iter().map(|d| d.kl).fold(0.0, f64::max);
    let mean_overlap =
        if deltas.is_empty() { 0.0 } else { deltas.iter().map(|d| d.overlap).sum::<f64>() / n };
    let mean_margin_delta = deltas.iter().map(|d| d.margin_delta).sum::<f64>() / n;

    let mut verdict = Verdict::Pass;
    let mut reasons = Vec::new();
    let mut raise = |level: Verdict, reason: String| {
        if level > verdict {
            verdict = level;
        }
        reasons.push(format!("{}: {}", level.label(), reason));
    };
    if cand.exit_code != 0 {
        raise(Verdict::Fail, format!("the candidate run exited with {}", cand.exit_code));
    }
    if cand.nonfinite {
        raise(Verdict::Fail, "non-finite logits in the candidate run".to_string());
    }
    if deltas.is_empty() {
        raise(
            Verdict::Fail,
            "no comparable step (no logits dumped, or the first token differs in structure)"
                .to_string(),
        );
    }
    if cand.loopiness >= thresholds.fail_loopiness && base.loopiness < thresholds.fail_loopiness {
        raise(
            Verdict::Fail,
            format!("the candidate's text loops (loopiness {:.2})", cand.loopiness),
        );
    }
    if max_abs_delta >= thresholds.fail_max_abs_delta {
        raise(
            Verdict::Fail,
            format!("max |Δlogit| {max_abs_delta:.3} ≥ {}", thresholds.fail_max_abs_delta),
        );
    } else if max_abs_delta >= thresholds.warn_max_abs_delta {
        raise(
            Verdict::Warn,
            format!("max |Δlogit| {max_abs_delta:.3} ≥ {}", thresholds.warn_max_abs_delta),
        );
    }
    if mean_kl >= thresholds.fail_mean_kl {
        raise(Verdict::Fail, format!("mean KL {mean_kl:.2e} ≥ {:.0e}", thresholds.fail_mean_kl));
    } else if mean_kl >= thresholds.warn_mean_kl {
        raise(Verdict::Warn, format!("mean KL {mean_kl:.2e} ≥ {:.0e}", thresholds.warn_mean_kl));
    }
    if !deltas.is_empty() {
        if mean_overlap <= thresholds.fail_min_overlap {
            raise(
                Verdict::Fail,
                format!("top-K overlap {mean_overlap:.3} ≤ {}", thresholds.fail_min_overlap),
            );
        } else if mean_overlap <= thresholds.warn_min_overlap {
            raise(
                Verdict::Warn,
                format!("top-K overlap {mean_overlap:.3} ≤ {}", thresholds.warn_min_overlap),
            );
        }
    }
    if let Some(at) = diverged_at {
        if (at as u32) < thresholds.fail_diverge_before {
            raise(
                Verdict::Fail,
                format!(
                    "the greedy tokens part at step {at} (before {})",
                    thresholds.fail_diverge_before
                ),
            );
        } else {
            raise(
                Verdict::Warn,
                format!(
                    "the greedy tokens part at step {at} of {}",
                    base.steps.len().min(cand.steps.len())
                ),
            );
        }
    }
    NumericsComparison {
        model: cand.model.clone(),
        prompt: cand.prompt.clone(),
        mtp: cand.mtp,
        steps_base: base.steps.len(),
        steps_cand: cand.steps.len(),
        steps_compared: deltas.len(),
        diverged_at,
        max_abs_delta,
        mean_kl,
        max_kl,
        mean_overlap,
        mean_margin_delta,
        text_identical: base.text_sha256 == cand.text_sha256,
        cand_loopiness: cand.loopiness,
        base_loopiness: base.loopiness,
        cand_nonfinite: cand.nonfinite,
        cand_exit_code: cand.exit_code,
        base_decode_tps: base.decode_tps,
        cand_decode_tps: cand.decode_tps,
        verdict,
        reasons,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(function: &str, step: u32, top: &[(u32, f64)]) -> StepLogits {
        StepLogits { context: 0, function: function.to_string(), step, row: 0, top: top.to_vec() }
    }

    fn run(steps: Vec<StepLogits>, text: &str) -> NumericsRun {
        NumericsRun {
            model: "e2b".into(),
            prompt: "p".into(),
            mtp: false,
            build_id: "x".into(),
            exit_code: 0,
            wall_s: 1.0,
            tokens: steps.iter().filter_map(|s| s.argmax()).collect(),
            steps,
            decode_tokens: 3,
            decode_tps: Some(300.0),
            prefill_tokens: 10,
            prefill_tps: Some(1000.0),
            acceptance: None,
            text_len: text.len(),
            text_sha256: text.to_string(),
            loopiness: loopiness(text),
            nonfinite: false,
        }
    }

    #[test]
    fn identical_runs_pass_with_zero_deltas() {
        let steps = vec![
            step("decode", 1, &[(5, 2.0), (3, 1.0)]),
            step("decode", 2, &[(1, 3.0), (2, 2.5)]),
        ];
        let a = run(steps.clone(), "abc");
        let b = run(steps, "abc");
        let c = compare_runs(&a, &b, &NumericsThresholds::default());
        assert_eq!(c.verdict, Verdict::Pass);
        assert_eq!(c.steps_compared, 2);
        assert_eq!(c.max_abs_delta, 0.0);
        assert!(c.mean_kl.abs() < 1e-12);
        assert_eq!(c.mean_overlap, 1.0);
        assert!(c.text_identical);
    }

    #[test]
    fn small_drift_warns_large_drift_fails() {
        let base = run(vec![step("decode", 1, &[(5, 2.0), (3, 1.0), (7, 0.5)])], "abc");
        let drift = run(vec![step("decode", 1, &[(5, 2.06), (3, 1.0), (7, 0.5)])], "abc");
        let c = compare_runs(&base, &drift, &NumericsThresholds::default());
        assert_eq!(c.verdict, Verdict::Warn, "{:?}", c.reasons);
        let broken = run(vec![step("decode", 1, &[(5, 4.0), (3, 1.0), (7, 0.5)])], "abc");
        let c = compare_runs(&base, &broken, &NumericsThresholds::default());
        assert_eq!(c.verdict, Verdict::Fail);
    }

    #[test]
    fn early_divergence_fails_late_divergence_warns() {
        // Every step a near-tie: the candidate's rounding flips one of them.
        let mut steps = Vec::new();
        for i in 0..20u32 {
            steps.push(step("decode", i, &[(i, 2.0), (i + 100, 1.995)]));
        }
        let base = run(steps.clone(), "abc");
        let mut late = steps.clone();
        late[15].top = vec![(115, 2.0), (15, 1.995)];
        let c = compare_runs(&base, &run(late, "abd"), &NumericsThresholds::default());
        assert_eq!(c.diverged_at, Some(15));
        assert_eq!(c.steps_compared, 16);
        assert_eq!(c.verdict, Verdict::Warn, "{:?}", c.reasons);
        let mut early = steps.clone();
        early[2].top = vec![(102, 2.0), (2, 1.995)];
        let c = compare_runs(&base, &run(early, "abd"), &NumericsThresholds::default());
        assert_eq!(c.verdict, Verdict::Fail);
        assert!(c.reasons.iter().any(|r| r.contains("part at step 2")), "{:?}", c.reasons);
    }

    #[test]
    fn loop_detection() {
        let line = "this line repeats itself over and over again\n";
        let looping = line.repeat(200);
        assert!(loopiness(&looping) > 0.9);
        assert!(loopiness("short\nlines\n") < 0.01);
    }

    #[test]
    fn kl_of_shifted_distribution() {
        // Top lists are in descending order, as the dispatch writes them.
        let a = step("decode", 1, &[(1, 1.0), (2, 0.0)]);
        let b = step("decode", 1, &[(2, 1.0), (1, 0.0)]);
        let d = step_delta(&a, &b);
        assert!(!d.same_argmax);
        assert!(d.kl > 0.3 && d.kl < 1.0, "kl {}", d.kl);
        assert_eq!(d.overlap, 1.0);
        assert_eq!(d.max_abs_delta, 1.0);
    }
}
