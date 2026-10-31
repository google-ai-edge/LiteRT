// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The guard's configuration: models, probe prompts, performance workloads and
//! the thresholds of the verdicts. `nvguard.toml` next to the crate holds the
//! defaults; `--config FILE` replaces it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

pub const DEFAULT_CONFIG: &str = include_str!("../nvguard.toml");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub prompts: Vec<PromptSpec>,
    #[serde(default)]
    pub numerics: NumericsConfig,
    #[serde(default)]
    pub perf: PerfConfig,
    #[serde(default)]
    pub probes: ProbesConfig,
    #[serde(default)]
    pub acceptance: AcceptanceThresholds,
    #[serde(default)]
    pub disk: DiskConfig,
}

/// The space a capture needs free: its numerics probes dump whole outputs
/// (numerics.full_steps), GBs per model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskConfig {
    /// Below this many GiB free on the guard root's filesystem, a capture
    /// stops before its next probe (it resumes later).
    pub min_free_gib: f64,
}

impl Default for DiskConfig {
    fn default() -> Self {
        Self { min_free_gib: 20.0 }
    }
}

/// The MTP acceptance of the greedy MTP numerics probes, pooled per model
/// (accepted over drafted tokens across the prompts): a drop judged on the
/// pool and on how many prompts fell, since one prompt's acceptance moves
/// with its text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceThresholds {
    /// Fewer probes paired than this: reported, not judged.
    pub min_probes: usize,
    /// A pooled drop (absolute) at or above this, with more prompts falling
    /// than rising, warns.
    pub warn_drop: f64,
    /// At or above this, with at least two thirds of the prompts falling,
    /// fails.
    pub fail_drop: f64,
}

impl Default for AcceptanceThresholds {
    fn default() -> Self {
        Self { min_probes: 3, warn_drop: 0.02, fail_drop: 0.05 }
    }
}

/// The probes beside the greedy decodes and the latency runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbesConfig {
    #[serde(default)]
    pub sampled: SampledProbeConfig,
    #[serde(default)]
    pub constraint: ConstraintProbeConfig,
    #[serde(default)]
    pub autopsy: AutopsyProbeConfig,
    #[serde(default)]
    pub conversations: ConversationsProbeConfig,
    #[serde(default)]
    pub memory: MemoryProbeConfig,
}

/// A sampled decode (top-k / top-p, a fixed seed) of a numerics prompt, run
/// with the MTP verifier sampling on the device and on the host.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampledProbeConfig {
    pub prompt: String,
    pub max_output_tokens: u32,
    pub context: u32,
    pub top_k: u32,
    pub top_p: f64,
    pub temperature: f64,
    pub seed: u32,
}

impl Default for SampledProbeConfig {
    fn default() -> Self {
        Self {
            prompt: "code_stdin".into(),
            max_output_tokens: 200,
            context: 2048,
            top_k: 64,
            top_p: 0.95,
            temperature: 1.0,
            seed: 1,
        }
    }
}

/// A constrained decode (a regex, a repetition penalty) in the CLI's session
/// mode, the only one that takes a constraint, sampled like the above.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintProbeConfig {
    pub prompt_text: String,
    /// Space-free: with Gemma 4's tokenizer a regex with spaces fails on
    /// every backend ("token ▁\" doesn't satisfy the grammar").
    pub regex: String,
    pub repetition_penalty: f64,
    pub max_output_tokens: u32,
    pub context: u32,
    pub top_k: u32,
    pub top_p: f64,
    pub temperature: f64,
    pub seed: u32,
}

impl Default for ConstraintProbeConfig {
    fn default() -> Self {
        Self {
            prompt_text: "Describe a fictional person as a JSON object with the fields name, age and city. Output only the JSON object.".into(),
            regex: r#"\{"name":"[A-Z][a-z]+","age":[1-9][0-9]?,"city":"[A-Z][a-z]+"\}"#.into(),
            repetition_penalty: 1.3,
            max_output_tokens: 80,
            context: 2048,
            top_k: 64,
            top_p: 0.95,
            temperature: 1.0,
            seed: 1,
        }
    }
}

/// The per-kernel autopsy of a step graph: one bench.sh run of a workload in
/// the dispatch's own-graph mode with the autopsy and DOT writers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutopsyProbeConfig {
    pub workload: String,
    pub mtp: bool,
}

impl Default for AutopsyProbeConfig {
    fn default() -> Self {
        Self { workload: "short".into(), mtp: false }
    }
}

/// The memory probe: bench.sh's memory metric of a workload, plain and with
/// MTP, and the growth that warns and fails (memory.rs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProbeConfig {
    pub workload: String,
    pub warn_gpu_mib: f64,
    pub fail_gpu_mib: f64,
    pub warn_rss_gib: f64,
    pub fail_rss_gib: f64,
}

impl Default for MemoryProbeConfig {
    fn default() -> Self {
        Self {
            workload: "short".into(),
            warn_gpu_mib: 64.0,
            fail_gpu_mib: 256.0,
            warn_rss_gib: 0.1,
            fail_rss_gib: 0.25,
        }
    }
}

/// The conversation probe: a numerics prompt decoded greedily in
/// `conversations` conversations of one process (the CLI's benchmark
/// iterations), whose texts must all be the first one's.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationsProbeConfig {
    pub prompt: String,
    pub max_output_tokens: u32,
    pub conversations: u32,
    pub mtp: bool,
}

impl Default for ConversationsProbeConfig {
    fn default() -> Self {
        Self { prompt: "doc_8k".into(), max_output_tokens: 256, conversations: 2, mtp: false }
    }
}

/// One numerics probe: a prompt file, decoded greedily for a fixed number of
/// tokens inside a fixed context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSpec {
    pub name: String,
    /// Relative to the guard directory (so `../prompts/prompt_8k.txt` is the
    /// benchmark's own long prompt).
    pub file: String,
    pub max_output_tokens: u32,
    pub context: u32,
    /// quick | standard | long: the tiers are cumulative.
    #[serde(default = "default_tier")]
    pub tier: String,
    /// Models the prompt applies to; empty means every model.
    #[serde(default)]
    pub models: Vec<String>,
}

fn default_tier() -> String {
    "quick".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericsConfig {
    /// How many logits per row the dispatch records (LITERT_NVIDIA_DISPATCH_DUMP_IO=top<K>).
    #[serde(default = "default_top_k")]
    pub top_k: u32,
    /// How many steps of every function the dispatch dumps whole
    /// (LITERT_NVIDIA_DISPATCH_DUMP_OUTPUTS): the full logits rows for the
    /// whole-distribution KL, the partitions' outputs for the localization.
    #[serde(default = "default_full_steps")]
    pub full_steps: u32,
    /// Outputs above this size are not dumped (caches never are).
    #[serde(default = "default_outputs_max_mb")]
    pub outputs_max_mb: u32,
    #[serde(default)]
    pub thresholds: NumericsThresholds,
}

fn default_top_k() -> u32 {
    32
}
fn default_full_steps() -> u32 {
    8
}
fn default_outputs_max_mb() -> u32 {
    16
}

impl Default for NumericsConfig {
    fn default() -> Self {
        Self {
            top_k: default_top_k(),
            full_steps: default_full_steps(),
            outputs_max_mb: default_outputs_max_mb(),
            thresholds: NumericsThresholds::default(),
        }
    }
}

/// Verdict thresholds of the numerics comparison. A metric at or above a
/// `fail_` value fails, at or above a `warn_` value warns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericsThresholds {
    /// Largest |candidate - baseline| logit among the top-K of either run, over
    /// the comparable steps. BF16 logits move by 0.125 per ulp at their usual
    /// magnitude and a different summation order moves them one to three ulp
    /// per step, more as the sequence amplifies it, so this warns late and
    /// fails only on gross change.
    pub warn_max_abs_delta: f64,
    pub fail_max_abs_delta: f64,
    /// Mean KL(baseline || candidate) over the union of both top-K sets,
    /// over the first steps (identical inputs, nothing amplified yet): the
    /// regression signal. A reordered reduction gives about 1e-5; a broken
    /// kernel orders of magnitude more.
    #[serde(default = "default_first_steps")]
    pub first_steps: usize,
    #[serde(default = "default_warn_first_kl")]
    pub warn_first_kl: f64,
    #[serde(default = "default_fail_first_kl")]
    pub fail_first_kl: f64,
    /// Mean per-step KL over every comparable step (near-ties spike it).
    pub warn_mean_kl: f64,
    pub fail_mean_kl: f64,
    /// Mean |top-K(baseline) ∩ top-K(candidate)| / K; a value at or below warns/fails.
    pub warn_min_overlap: f64,
    pub fail_min_overlap: f64,
    /// A greedy text that parts from the baseline's before this many decode
    /// steps fails (a near-tie a sentence later is normal).
    pub fail_diverge_before: u32,
    /// Repetition in the last 4,000 characters (the most repeated line's share).
    pub fail_loopiness: f64,
}

fn default_first_steps() -> usize {
    4
}
fn default_warn_first_kl() -> f64 {
    1e-3
}
fn default_fail_first_kl() -> f64 {
    1e-2
}

impl Default for NumericsThresholds {
    fn default() -> Self {
        Self {
            warn_max_abs_delta: 1.0,
            fail_max_abs_delta: 8.0,
            first_steps: default_first_steps(),
            warn_first_kl: default_warn_first_kl(),
            fail_first_kl: default_fail_first_kl(),
            warn_mean_kl: 1e-3,
            fail_mean_kl: 1e-1,
            warn_min_overlap: 0.95,
            fail_min_overlap: 0.8,
            fail_diverge_before: 8,
            fail_loopiness: 0.25,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfConfig {
    /// bench.sh workloads measured by a capture, in order.
    #[serde(default = "default_workloads")]
    pub workloads: Vec<String>,
    /// Iterations per workload (bench.sh keeps its own warm-up count per workload).
    #[serde(default = "default_iterations")]
    pub iterations: BTreeMap<String, u32>,
    /// Workloads the cost model predicts from the measured ones.
    #[serde(default = "default_predict")]
    pub predict: Vec<String>,
    #[serde(default)]
    pub thresholds: PerfThresholds,
}

fn default_workloads() -> Vec<String> {
    vec!["short".to_string(), "8k".to_string(), "16k".to_string()]
}

fn default_iterations() -> BTreeMap<String, u32> {
    BTreeMap::from([("short".to_string(), 5), ("8k".to_string(), 3), ("16k".to_string(), 3)])
}

fn default_predict() -> Vec<String> {
    vec!["30k".to_string(), "128k".to_string()]
}

impl Default for PerfConfig {
    fn default() -> Self {
        Self {
            workloads: default_workloads(),
            iterations: default_iterations(),
            predict: default_predict(),
            thresholds: PerfThresholds::default(),
        }
    }
}

/// Verdict thresholds of the performance comparison, on the ratio
/// candidate / baseline of a throughput (decode steps/s, prefill tok/s).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfThresholds {
    pub warn_ratio: f64,
    pub fail_ratio: f64,
}

impl Default for PerfThresholds {
    fn default() -> Self {
        Self { warn_ratio: 0.97, fail_ratio: 0.93 }
    }
}

/// The bench.sh workload presets (prefill tokens, decode tokens, context,
/// default iterations, warm-ups).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workload {
    pub name: &'static str,
    pub input: u32,
    pub output: u32,
    pub context: u32,
    pub iterations: u32,
    pub warmups: u32,
}

pub const WORKLOADS: &[Workload] = &[
    Workload { name: "short", input: 1024, output: 256, context: 2048, iterations: 8, warmups: 2 },
    Workload { name: "8k", input: 8192, output: 256, context: 9216, iterations: 4, warmups: 1 },
    Workload { name: "16k", input: 16384, output: 256, context: 17408, iterations: 4, warmups: 1 },
    Workload { name: "30k", input: 30720, output: 256, context: 31744, iterations: 4, warmups: 1 },
    Workload { name: "32k", input: 32768, output: 256, context: 33792, iterations: 4, warmups: 1 },
    Workload {
        name: "128k",
        input: 128001,
        output: 128,
        context: 130048,
        iterations: 4,
        warmups: 1,
    },
    Workload {
        name: "256k",
        input: 256001,
        output: 128,
        context: 262144,
        iterations: 4,
        warmups: 1,
    },
];

pub fn workload(name: &str) -> Result<Workload> {
    WORKLOADS
        .iter()
        .copied()
        .find(|w| w.name == name)
        .ok_or_else(|| anyhow!("unknown workload {name} (short|8k|16k|30k|32k|128k|256k)"))
}

/// The benchmark's own real-text prompt of a workload (matrix.sh's mapping).
pub fn workload_prompt(name: &str) -> Option<&'static str> {
    match name {
        "short" => Some("prompt_1024.txt"),
        "8k" => Some("prompt_8k.txt"),
        "16k" => Some("prompt_16k.txt"),
        "30k" => Some("prompt_30k.txt"),
        "128k" => Some("prompt_128k.txt"),
        _ => None,
    }
}

pub const TIERS: &[&str] = &["quick", "standard", "long"];

pub fn tier_rank(tier: &str) -> Result<usize> {
    TIERS
        .iter()
        .position(|t| *t == tier)
        .ok_or_else(|| anyhow!("unknown tier {tier} (quick|standard|long)"))
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Config> {
        let text = match path {
            Some(p) => {
                std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?
            }
            None => DEFAULT_CONFIG.to_string(),
        };
        let config: Config = toml::from_str(&text).context("parsing the guard configuration")?;
        for prompt in &config.prompts {
            tier_rank(&prompt.tier)?;
        }
        for name in &config.perf.workloads {
            workload(name)?;
        }
        for name in &config.perf.predict {
            workload(name)?;
        }
        workload(&config.probes.autopsy.workload)?;
        workload(&config.probes.memory.workload)?;
        config.prompt(&config.probes.sampled.prompt)?;
        config.prompt(&config.probes.conversations.prompt)?;
        if config.probes.conversations.conversations < 2 {
            anyhow::bail!("the conversation probe needs at least 2 conversations");
        }
        regex::Regex::new(&format!("^(?:{})$", config.probes.constraint.regex))
            .context("the constraint probe's regex")?;
        Ok(config)
    }

    /// A prompt by name.
    pub fn prompt(&self, name: &str) -> Result<&PromptSpec> {
        self.prompts
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| anyhow!("no prompt {name} in the configuration"))
    }

    /// The prompts of a tier (and the tiers below it) for a model.
    pub fn prompts_for(&self, model: &str, tier: &str) -> Result<Vec<&PromptSpec>> {
        let rank = tier_rank(tier)?;
        Ok(self
            .prompts
            .iter()
            .filter(|p| tier_rank(&p.tier).map(|r| r <= rank).unwrap_or(false))
            .filter(|p| p.models.is_empty() || p.models.iter().any(|m| m == model))
            .collect())
    }

    pub fn prompt_path(&self, guard_dir: &Path, prompt: &PromptSpec) -> PathBuf {
        guard_dir.join(&prompt.file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_parses() {
        let config = Config::load(None).unwrap();
        assert!(config.models.contains(&"e2b".to_string()));
        assert!(!config.prompts.is_empty());
        let quick = config.prompts_for("e2b", "quick").unwrap();
        let long = config.prompts_for("e2b", "long").unwrap();
        assert!(long.len() >= quick.len());
        assert_eq!(workload("short").unwrap().context, 2048);
        assert!(workload("4k").is_err());
    }
}
