// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The dispatch's outputs dump (LITERT_NVIDIA_DISPATCH_DUMP_OUTPUTS): every
//! output of the first steps of every function, in the order they were
//! produced, with an index. Two runs' dumps compared in that order name the
//! first tensor that differs: the step, the context, the function (a
//! TensorRT partition) and the output, with how far it moved. The logits
//! rows among them give the KL over the whole distribution, not a top-K.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// LiteRtElementType values (litert/c/litert_model_types.h).
const ELEMENT_FLOAT32: i32 = 1;
const ELEMENT_INT8: i32 = 9;
const ELEMENT_FLOAT16: i32 = 10;
const ELEMENT_BFLOAT16: i32 = 19;

/// A vocabulary's width: outputs at least this wide are logits.
pub const LOGITS_COLUMNS: u64 = 65536;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexLine {
    pub seq: u64,
    pub context: u32,
    pub function: String,
    pub step: u32,
    pub output: u32,
    #[serde(default)]
    pub name: String,
    pub element_type: i32,
    #[serde(default)]
    pub dims: Vec<i64>,
    #[serde(default)]
    pub stored_half: bool,
    /// An output whose device memory is an input's (an in-place KV cache),
    /// dumped at a context's first step only.
    #[serde(default)]
    pub state: bool,
    /// The dims of the step's first input (the tokens): a prefill chunk's
    /// count, a decode step's one, a verify step's few.
    #[serde(default)]
    pub input_dims: Vec<i64>,
    pub bytes: u64,
    pub file: String,
}

impl IndexLine {
    /// The key two runs' lines are paired by.
    pub fn key(&self) -> (u32, String, u32, u32) {
        (self.context, self.function.clone(), self.step, self.output)
    }

    pub fn columns(&self) -> u64 {
        self.dims.last().copied().unwrap_or(0).max(0) as u64
    }

    /// The step's input tokens (the product of the first input's dims), 0
    /// when the dump did not record them.
    pub fn input_rows(&self) -> i64 {
        if self.input_dims.is_empty() {
            0
        } else {
            self.input_dims.iter().map(|d| (*d).max(0)).product()
        }
    }

    pub fn is_logits(&self) -> bool {
        self.columns() >= LOGITS_COLUMNS
    }

    fn element_bytes(&self) -> Option<u64> {
        match self.element_type {
            ELEMENT_FLOAT32 if !self.stored_half => Some(4),
            ELEMENT_FLOAT32 | ELEMENT_FLOAT16 | ELEMENT_BFLOAT16 => Some(2),
            ELEMENT_INT8 => Some(1),
            _ => None,
        }
    }

    /// Whether the guard reads this element type: the floats, and the int8
    /// of the quantized caches (compared as integers).
    pub fn is_comparable(&self) -> bool {
        self.element_bytes().is_some()
    }
}

/// The index of a dump directory, in production order.
pub fn load_index(dir: &Path) -> Result<Vec<IndexLine>> {
    let path = dir.join("index.jsonl");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return Ok(Vec::new()),
    };
    let mut lines = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: IndexLine = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: not an index line", path.display(), n + 1))?;
        lines.push(parsed);
    }
    lines.sort_by_key(|l| l.seq);
    Ok(lines)
}

fn half_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x3ff) as f64;
    match exponent {
        0 => sign * mantissa * 2f64.powi(-24),
        0x1f => {
            if mantissa == 0.0 {
                sign * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => sign * (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15),
    }
}

fn bfloat16_to_f64(bits: u16) -> f64 {
    f32::from_bits((bits as u32) << 16) as f64
}

/// A tensor of the dump as f64 values (float16 and the float32 tensors the
/// engine keeps as half widened exactly; int8 as its integers).
pub fn read_tensor(dir: &Path, line: &IndexLine) -> Result<Vec<f64>> {
    let path = dir.join(&line.file);
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let values = match (line.element_type, line.stored_half) {
        (ELEMENT_FLOAT32, false) => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        (ELEMENT_BFLOAT16, _) => bytes
            .chunks_exact(2)
            .map(|c| bfloat16_to_f64(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        (ELEMENT_FLOAT32, true) | (ELEMENT_FLOAT16, _) => {
            bytes.chunks_exact(2).map(|c| half_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect()
        }
        (ELEMENT_INT8, _) => bytes.iter().map(|b| *b as i8 as f64).collect(),
        _ => anyhow::bail!(
            "{}: element type {} is not one the guard reads",
            line.file,
            line.element_type
        ),
    };
    Ok(values)
}

/// How far one output moved between two runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDiff {
    pub seq: u64,
    pub context: u32,
    pub function: String,
    pub step: u32,
    pub output: u32,
    pub name: String,
    #[serde(default)]
    pub state: bool,
    /// The step's input tokens (0 when unknown).
    #[serde(default)]
    pub input_rows: i64,
    pub dims: Vec<i64>,
    pub elements: usize,
    pub differing: usize,
    pub max_abs_delta: f64,
    pub rms_delta: f64,
    pub base_rms: f64,
    /// rms(delta) / rms(baseline): the relative size of the move.
    pub relative: f64,
    pub first_index: Option<usize>,
    pub nonfinite: bool,
    /// For a logits output: the largest values of the first differing row in
    /// both runs, by index, descending.
    pub base_top: Vec<(u32, f64)>,
    pub cand_top: Vec<(u32, f64)>,
}

fn top_of_row(row: &[f64], k: usize) -> Vec<(u32, f64)> {
    let mut indexed: Vec<(u32, f64)> =
        row.iter().enumerate().map(|(i, v)| (i as u32, *v)).collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
    });
    indexed.truncate(k);
    indexed
}

pub fn tensor_diff(line: &IndexLine, base: &[f64], cand: &[f64]) -> TensorDiff {
    let n = base.len().min(cand.len());
    let mut differing = 0usize;
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut base_sq = 0.0f64;
    let mut first_index = None;
    let mut nonfinite = false;
    for i in 0..n {
        let b = base[i];
        let c = cand[i];
        nonfinite |= !c.is_finite();
        base_sq += b * b;
        if b.to_bits() != c.to_bits() {
            differing += 1;
            if first_index.is_none() {
                first_index = Some(i);
            }
            let d = (c - b).abs();
            if d.is_finite() {
                max_abs = max_abs.max(d);
                sum_sq += d * d;
            } else {
                max_abs = f64::INFINITY;
            }
        }
    }
    if base.len() != cand.len() {
        differing += base.len().max(cand.len()) - n;
        if first_index.is_none() {
            first_index = Some(n);
        }
    }
    let rms = (sum_sq / n.max(1) as f64).sqrt();
    let base_rms = (base_sq / n.max(1) as f64).sqrt();
    let (base_top, cand_top) = if line.is_logits() && differing > 0 {
        let columns = line.columns() as usize;
        let row = first_index.unwrap_or(0) / columns.max(1);
        let start = row * columns;
        let end = (start + columns).min(n);
        (top_of_row(&base[start..end], 8), top_of_row(&cand[start..end], 8))
    } else {
        (Vec::new(), Vec::new())
    };
    TensorDiff {
        seq: line.seq,
        context: line.context,
        function: line.function.clone(),
        step: line.step,
        output: line.output,
        name: line.name.clone(),
        state: line.state,
        input_rows: line.input_rows(),
        dims: line.dims.clone(),
        elements: n,
        differing,
        max_abs_delta: max_abs,
        rms_delta: rms,
        base_rms,
        relative: if base_rms > 0.0 { rms / base_rms } else { rms },
        first_index,
        nonfinite,
        base_top,
        cand_top,
    }
}

/// What the dumps of two runs say about where they part.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Localization {
    /// Output tensors present in both runs.
    pub pairs: usize,
    /// Output tensors of either run without a counterpart in the other.
    pub unpaired: usize,
    /// Tensors that differ, over all pairs.
    pub differing: usize,
    /// The first tensor, in production order, that differs.
    pub first: Option<TensorDiff>,
    /// Every differing output of the first differing step (its context and
    /// step): how much of that function's output the difference covers.
    pub first_step: Vec<TensorDiff>,
    /// The outputs paired at that step, differing or not.
    #[serde(default)]
    pub first_step_outputs: usize,
    /// The largest relative move over all pairs.
    pub largest: Option<TensorDiff>,
}

/// Compares the two dumps in the baseline's production order.
pub fn localize(base_dir: &Path, cand_dir: &Path) -> Result<Localization> {
    let base = load_index(base_dir)?;
    let cand = load_index(cand_dir)?;
    let cand_by_key: BTreeMap<(u32, String, u32, u32), &IndexLine> =
        cand.iter().map(|l| (l.key(), l)).collect();
    let base_keys: std::collections::BTreeSet<_> = base.iter().map(|l| l.key()).collect();
    let mut out = Localization::default();
    out.unpaired = base.iter().filter(|l| !cand_by_key.contains_key(&l.key())).count()
        + cand.iter().filter(|l| !base_keys.contains(&l.key())).count();
    let mut first_step: Option<(u32, u32)> = None;
    for line in &base {
        let Some(other) = cand_by_key.get(&line.key()) else { continue };
        if !line.is_comparable() || !other.is_comparable() {
            continue;
        }
        out.pairs += 1;
        let base_bytes = std::fs::read(base_dir.join(&line.file))
            .with_context(|| format!("reading {}", line.file))?;
        let cand_bytes = std::fs::read(cand_dir.join(&other.file))
            .with_context(|| format!("reading {}", other.file))?;
        if base_bytes == cand_bytes {
            continue;
        }
        out.differing += 1;
        let diff = tensor_diff(line, &read_tensor(base_dir, line)?, &read_tensor(cand_dir, other)?);
        if out.first.is_none() {
            first_step = Some((line.context, line.step));
            out.first = Some(diff.clone());
        }
        if first_step == Some((line.context, line.step)) {
            out.first_step.push(diff.clone());
        }
        if out.largest.as_ref().map(|l| diff.relative > l.relative).unwrap_or(true) {
            out.largest = Some(diff);
        }
    }
    if let Some((context, step)) = first_step {
        out.first_step_outputs = base
            .iter()
            .filter(|l| l.context == context && l.step == step && l.is_comparable())
            .filter(|l| cand_by_key.get(&l.key()).map(|o| o.is_comparable()).unwrap_or(false))
            .count();
    }
    Ok(out)
}

/// The KL divergence of one logits row between two runs, over the whole
/// vocabulary, and the largest logit move.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowKl {
    pub context: u32,
    pub function: String,
    pub step: u32,
    pub row: usize,
    pub kl: f64,
    pub max_abs_delta: f64,
    pub same_argmax: bool,
}

fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|l| (l - max).exp()).sum();
    let log_sum = max + sum.ln();
    logits.iter().map(|l| l - log_sum).collect()
}

pub fn row_kl(base: &[f64], cand: &[f64]) -> f64 {
    let p = log_softmax(base);
    let q = log_softmax(cand);
    p.iter()
        .zip(q.iter())
        .map(|(lp, lq)| {
            let pi = lp.exp();
            if pi > 0.0 {
                pi * (lp - lq)
            } else {
                0.0
            }
        })
        .sum::<f64>()
        .max(0.0)
}

fn argmax(row: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, v) in row.iter().enumerate() {
        if best.map(|(_, b)| *v > b).unwrap_or(true) {
            best = Some((i, *v));
        }
    }
    best.map(|(i, _)| i)
}

/// The full-distribution KL of every logits row the two dumps share, in
/// production order.
pub fn full_row_kls(base_dir: &Path, cand_dir: &Path) -> Result<Vec<RowKl>> {
    let base = load_index(base_dir)?;
    let cand = load_index(cand_dir)?;
    let cand_by_key: BTreeMap<(u32, String, u32, u32), &IndexLine> =
        cand.iter().map(|l| (l.key(), l)).collect();
    let mut out = Vec::new();
    for line in base.iter().filter(|l| l.is_logits() && l.is_comparable()) {
        let Some(other) = cand_by_key.get(&line.key()) else { continue };
        let b = read_tensor(base_dir, line)?;
        let c = read_tensor(cand_dir, other)?;
        let columns = line.columns() as usize;
        let rows = b.len().min(c.len()) / columns.max(1);
        for row in 0..rows {
            let bs = &b[row * columns..(row + 1) * columns];
            let cs = &c[row * columns..(row + 1) * columns];
            out.push(RowKl {
                context: line.context,
                function: line.function.clone(),
                step: line.step,
                row,
                kl: row_kl(bs, cs),
                max_abs_delta: bs
                    .iter()
                    .zip(cs.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max),
                same_argmax: argmax(bs) == argmax(cs),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_dump(dir: &Path, entries: &[(u32, &str, u32, u32, i32, &[i64], &[f32])]) {
        let mut index = String::new();
        for (seq, (context, function, step, output, element_type, dims, values)) in
            entries.iter().enumerate()
        {
            let file = format!("c{context}/{function}/step{step}/out_{output}.raw");
            let path = dir.join(&file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut bytes = Vec::new();
            for v in *values {
                if *element_type == ELEMENT_FLOAT16 {
                    bytes.extend_from_slice(&half_bits(*v).to_le_bytes());
                } else if *element_type == ELEMENT_INT8 {
                    bytes.push(*v as i8 as u8);
                } else {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            std::fs::write(&path, &bytes).unwrap();
            // An int8 tensor stands for a quantized cache: state.
            let state = *element_type == ELEMENT_INT8;
            index.push_str(&format!(
                "{{\"seq\":{seq},\"context\":{context},\"function\":\"{function}\",\"step\":{step},\"output\":{output},\"name\":\"t{output}\",\"element_type\":{element_type},\"dims\":{},\"stored_half\":false,\"state\":{state},\"bytes\":{},\"file\":\"{file}\"}}\n",
                serde_json::to_string(dims).unwrap(),
                bytes.len()
            ));
        }
        std::fs::write(dir.join("index.jsonl"), index).unwrap();
    }

    fn half_bits(v: f32) -> u16 {
        // Round-to-nearest conversion for the magnitudes the tests use.
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x7fffff;
        if v == 0.0 {
            return sign;
        }
        assert!((1..31).contains(&exp), "test value out of half range: {v}");
        let rounded = (mant + 0x1000) >> 13;
        sign | ((exp as u16) << 10) + rounded as u16
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("nvguard-dumps-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn int8_state_tensors_compare_as_integers_and_come_first() {
        let a = temp_dir("i");
        let b = temp_dir("j");
        let base: &[(u32, &str, u32, u32, i32, &[i64], &[f32])] = &[
            (
                2,
                "prefill",
                1,
                3,
                ELEMENT_INT8,
                &[1, 1, 4, 2],
                &[1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -128.0],
            ),
            (0, "decode", 1, 0, ELEMENT_FLOAT32, &[1, 1, 4], &[0.5, 1.5, -2.0, 3.0]),
        ];
        let cand: &[(u32, &str, u32, u32, i32, &[i64], &[f32])] = &[
            (
                2,
                "prefill",
                1,
                3,
                ELEMENT_INT8,
                &[1, 1, 4, 2],
                &[1.0, -2.0, 3.0, 4.0, 5.0, 9.0, 7.0, -128.0],
            ),
            (0, "decode", 1, 0, ELEMENT_FLOAT32, &[1, 1, 4], &[0.5, 1.5, -2.0, 3.25]),
        ];
        write_dump(&a, base);
        write_dump(&b, cand);
        let index = load_index(&a).unwrap();
        assert!(index[0].state && index[0].is_comparable());
        assert_eq!(index[0].input_rows(), 0, "the test index carries no input dims");
        let with_dims: IndexLine = serde_json::from_str(r#"{"seq":0,"context":2,"function":"f","step":1,"output":0,"element_type":9,"dims":[1,1,4,2],"state":true,"input_dims":[1,1024],"bytes":8,"file":"x"}"#).unwrap();
        assert_eq!(with_dims.input_rows(), 1024);
        assert_eq!(
            read_tensor(&a, &index[0]).unwrap(),
            vec![1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -128.0]
        );
        let l = localize(&a, &b).unwrap();
        assert_eq!(l.pairs, 2);
        assert_eq!(l.differing, 2);
        let first = l.first.unwrap();
        assert!(first.state);
        assert_eq!(
            (first.function.as_str(), first.step, first.output, first.first_index),
            ("prefill", 1, 3, Some(5))
        );
        assert_eq!(first.max_abs_delta, 3.0);
        assert_eq!(l.first_step.len(), 1, "the decode step is another context's step");
        assert_eq!(l.first_step_outputs, 1);
    }

    #[test]
    fn identical_dumps_localize_nothing() {
        let a = temp_dir("a");
        let b = temp_dir("b");
        let entries: &[(u32, &str, u32, u32, i32, &[i64], &[f32])] = &[
            (0, "prefill", 1, 0, ELEMENT_FLOAT32, &[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (0, "decode", 2, 0, ELEMENT_FLOAT16, &[1, 1, 4], &[0.5, 1.5, -2.0, 3.0]),
        ];
        write_dump(&a, entries);
        write_dump(&b, entries);
        let l = localize(&a, &b).unwrap();
        assert_eq!(l.pairs, 2);
        assert_eq!(l.differing, 0);
        assert!(l.first.is_none());
        assert_eq!(l.unpaired, 0);
    }

    #[test]
    fn the_first_differing_tensor_is_named_in_production_order() {
        let a = temp_dir("c");
        let b = temp_dir("d");
        write_dump(
            &a,
            &[
                (0, "prefill", 1, 0, ELEMENT_FLOAT32, &[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                (0, "prefill", 1, 1, ELEMENT_FLOAT32, &[1, 3], &[0.0, 1.0, 2.0]),
                (1, "decode", 1, 0, ELEMENT_FLOAT32, &[1, 1, 4], &[0.5, 1.5, -2.0, 3.0]),
                (1, "decode", 2, 0, ELEMENT_FLOAT32, &[1, 1, 4], &[0.5, 1.5, -2.0, 3.0]),
            ],
        );
        write_dump(
            &b,
            &[
                (0, "prefill", 1, 0, ELEMENT_FLOAT32, &[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                (0, "prefill", 1, 1, ELEMENT_FLOAT32, &[1, 3], &[0.0, 1.0, 2.5]),
                (1, "decode", 1, 0, ELEMENT_FLOAT32, &[1, 1, 4], &[0.5, 1.5, -2.0, 3.25]),
                (1, "decode", 2, 0, ELEMENT_FLOAT32, &[1, 1, 4], &[0.5, 1.5, -2.0, 3.0]),
            ],
        );
        let l = localize(&a, &b).unwrap();
        assert_eq!(l.pairs, 4);
        assert_eq!(l.differing, 2);
        let first = l.first.unwrap();
        assert_eq!(
            (first.context, first.function.as_str(), first.step, first.output),
            (0, "prefill", 1, 1)
        );
        assert_eq!(first.differing, 1);
        assert_eq!(first.first_index, Some(2));
        assert!((first.max_abs_delta - 0.5).abs() < 1e-12);
        assert_eq!(l.first_step.len(), 1, "only the prefill output differs at that step");
        let largest = l.largest.unwrap();
        assert_eq!(largest.function, "prefill");
    }

    #[test]
    fn full_row_kl_over_the_vocabulary() {
        let a = temp_dir("e");
        let b = temp_dir("f");
        let columns = LOGITS_COLUMNS as usize;
        let mut base: Vec<f32> = (0..columns).map(|i| (i % 7) as f32 * 0.5).collect();
        base[100] = 12.0;
        base[101] = 0.0;
        let mut cand = base.clone();
        cand[100] = 11.0;
        cand[101] = 10.0;
        let dims: Vec<i64> = vec![1, 1, columns as i64];
        write_dump(&a, &[(0, "decode", 1, 0, ELEMENT_FLOAT32, &dims, &base)]);
        write_dump(&b, &[(0, "decode", 1, 0, ELEMENT_FLOAT32, &dims, &cand)]);
        let kls = full_row_kls(&a, &b).unwrap();
        assert_eq!(kls.len(), 1);
        assert!(kls[0].kl > 0.1, "kl {}", kls[0].kl);
        assert!(kls[0].same_argmax);
        assert_eq!(kls[0].max_abs_delta, 10.0);
        let same = full_row_kls(&a, &a).unwrap();
        assert_eq!(same[0].kl, 0.0);
        let l = localize(&a, &b).unwrap();
        let first = l.first.unwrap();
        assert_eq!(first.base_top.first().map(|(i, _)| *i), Some(100));
        assert_eq!(first.cand_top.len(), 8);
    }

    #[test]
    fn half_values_widen_exactly() {
        assert_eq!(half_to_f64(0x3c00), 1.0);
        assert_eq!(half_to_f64(0xc000), -2.0);
        assert_eq!(half_to_f64(0x0001), 2f64.powi(-24));
        assert!(half_to_f64(0x7c00).is_infinite());
        assert!(half_to_f64(0x7e00).is_nan());
        assert_eq!(bfloat16_to_f64(0x3f80), 1.0);
    }
}
