// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! Parsers of what a run leaves behind: the CLI's BenchmarkInfo blocks (the
//! same lines report.py reads), the dispatch's top-K logits JSON lines and
//! GNU time's accounting.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

fn re(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("valid regex"))
}

/// absl::Duration text ("1.2s", "27.3ms", "1m2.5s") in seconds.
pub fn duration_seconds(text: &str) -> f64 {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = re(&RE, r"([0-9.]+)(ms|us|µs|ns|h|m|s)");
    let mut total = 0.0;
    for cap in re.captures_iter(text) {
        let value: f64 = cap[1].parse().unwrap_or(0.0);
        let unit = match &cap[2] {
            "h" => 3600.0,
            "m" => 60.0,
            "s" => 1.0,
            "ms" => 1e-3,
            "us" | "µs" => 1e-6,
            "ns" => 1e-9,
            _ => 0.0,
        };
        total += value * unit;
    }
    total
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Turn {
    pub tokens: u64,
    pub seconds: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MtpStats {
    pub acceptance: f64,
    pub accepted: u64,
    pub drafted: u64,
    pub rounds: Option<u64>,
    pub tokens_per_round: Option<f64>,
    pub emitted: Option<u64>,
    pub round_ms: Option<f64>,
    pub draft_ms: Option<f64>,
    pub verify_ms: Option<f64>,
    pub plain_steps: Option<u64>,
    pub plain_step_ms: Option<f64>,
}

/// One BenchmarkInfo block: one iteration of the CLI's benchmark.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Iteration {
    pub prefill: Vec<Turn>,
    pub decode: Vec<Turn>,
    pub init_total_s: Option<f64>,
    pub init_executor_s: Option<f64>,
    pub time_to_first_token_s: Option<f64>,
    pub mtp: Option<MtpStats>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CliLog {
    pub iterations: Vec<Iteration>,
    pub captured_output: Option<String>,
    pub cli_peak_rss_gib: Option<f64>,
    pub aot_hits: usize,
    pub aot_misses: usize,
    pub compiled_partitions: usize,
    pub sampler_lines: Vec<String>,
    /// "Applying plugin with SDK version: TensorRT-RTX via NvInfer headers X":
    /// the SDK the compiler plugin was built against.
    pub sdk_version: Option<String>,
    /// The compile used the shared weight store (plans without their weights).
    pub weight_store: Option<bool>,
    /// "NVIDIA MTP GPU ... sampling enabled" and "... candidate selection
    /// enabled" lines: the device samplers in use (the greedy argmax and the
    /// top-k candidate selection).
    pub device_samplers: usize,
    /// The "candidate selection enabled" lines alone: the verifier's device
    /// path at k > 1.
    pub device_candidates: usize,
}

fn turns(block: &str, kind: &str) -> Vec<Turn> {
    let pattern = format!(
        r"{kind} Turn \d+: Processed (\d+) tokens in (\S+) duration\.\s*{kind} Speed: ([0-9.]+) tokens/sec"
    );
    let re = Regex::new(&pattern).expect("valid regex");
    re.captures_iter(block)
        .map(|cap| Turn {
            tokens: cap[1].parse().unwrap_or(0),
            seconds: duration_seconds(&cap[2]),
            tokens_per_second: cap[3].parse().unwrap_or(0.0),
        })
        .collect()
}

fn capture_f64(re: &Regex, text: &str, group: usize) -> Option<f64> {
    re.captures(text).and_then(|c| c.get(group)).and_then(|m| m.as_str().parse().ok())
}

fn capture_u64(re: &Regex, text: &str, group: usize) -> Option<u64> {
    re.captures(text).and_then(|c| c.get(group)).and_then(|m| m.as_str().parse().ok())
}

/// What the CLI printed to its log.
pub fn parse_cli_log(text: &str) -> CliLog {
    static INIT: OnceLock<Regex> = OnceLock::new();
    static TTFT: OnceLock<Regex> = OnceLock::new();
    static ACCEPT: OnceLock<Regex> = OnceLock::new();
    static ROUNDS: OnceLock<Regex> = OnceLock::new();
    static PER_ROUND: OnceLock<Regex> = OnceLock::new();
    static ROUND_MS: OnceLock<Regex> = OnceLock::new();
    static PLAIN: OnceLock<Regex> = OnceLock::new();
    static PEAK: OnceLock<Regex> = OnceLock::new();
    static CAPTURED: OnceLock<Regex> = OnceLock::new();
    static PARTITIONS: OnceLock<Regex> = OnceLock::new();
    let init = re(&INIT, r"(?m)^\s+- ([^:\n]+): ([0-9.]+) ms$");
    let ttft = re(&TTFT, r"Time to first token: ([0-9.]+) s");
    let accept = re(&ACCEPT, r"Acceptance rate:\s*([0-9.]+)% \((\d+) of (\d+) drafted");
    let rounds = re(&ROUNDS, r"Rounds:\s*(\d+) \(([0-9.]+)% of decode steps\)");
    let per_round = re(&PER_ROUND, r"Tokens per round:\s*([0-9.]+) \((\d+) emitted\)");
    let round_ms = re(
        &ROUND_MS,
        r"Time per round:\s*([0-9.]+) ms \(draft ([0-9.]+) ms, verify ([0-9.]+) ms\)",
    );
    let plain = re(&PLAIN, r"Plain decodes:\s*(\d+) \(incl\. re-priming\), ([0-9.]+) ms/step");
    let peak = re(&PEAK, r"Peak system ram usage: ([0-9.]+) ?MB");
    let captured = re(&CAPTURED, r"Captured model output:\s*(.*)");
    let partitions = re(&PARTITIONS, r"compiling partition \d+/\d+");

    let mut log = CliLog::default();
    let blocks: Vec<&str> = text.split("BenchmarkInfo:\n").collect();
    for block in blocks.iter().skip(1) {
        let mut item = Iteration {
            prefill: turns(block, "Prefill"),
            decode: turns(block, "Decode"),
            ..Default::default()
        };
        for cap in init.captures_iter(block) {
            let seconds = cap[2].parse::<f64>().unwrap_or(0.0) / 1e3;
            match &cap[1] {
                "Init Total" => item.init_total_s = Some(seconds),
                "Init Executor" => item.init_executor_s = Some(seconds),
                _ => {}
            }
        }
        item.time_to_first_token_s = capture_f64(ttft, block, 1);
        if let Some(cap) = accept.captures(block) {
            let mut mtp = MtpStats {
                acceptance: cap[1].parse::<f64>().unwrap_or(0.0) / 100.0,
                accepted: cap[2].parse().unwrap_or(0),
                drafted: cap[3].parse().unwrap_or(0),
                ..Default::default()
            };
            mtp.rounds = capture_u64(rounds, block, 1);
            mtp.tokens_per_round = capture_f64(per_round, block, 1);
            mtp.emitted = capture_u64(per_round, block, 2);
            mtp.round_ms = capture_f64(round_ms, block, 1);
            mtp.draft_ms = capture_f64(round_ms, block, 2);
            mtp.verify_ms = capture_f64(round_ms, block, 3);
            mtp.plain_steps = capture_u64(plain, block, 1);
            mtp.plain_step_ms = capture_f64(plain, block, 2);
            item.mtp = Some(mtp);
        }
        log.iterations.push(item);
    }
    log.cli_peak_rss_gib = peak
        .captures_iter(text)
        .filter_map(|c| c[1].parse::<f64>().ok())
        .fold(None, |acc: Option<f64>, v| Some(acc.map_or(v, |a| a.max(v))))
        .map(|mb| mb * (1u64 << 20) as f64 / (1u64 << 30) as f64);
    log.captured_output = captured.captures(text).map(|c| c[1].trim().to_string());
    log.aot_hits = text.matches("AOT cache hit:").count();
    log.aot_misses = text.matches("AOT cache miss:").count();
    log.compiled_partitions = partitions.find_iter(text).count();
    log.sampler_lines = text
        .lines()
        .filter(|l| l.contains("sampler") || l.contains("Sampler"))
        .filter(|l| l.contains("NVIDIA") || l.contains("GPU") || l.contains("CUDA"))
        .map(|l| l.splitn(5, ' ').last().unwrap_or(l).to_string())
        .collect();
    static SDK: OnceLock<Regex> = OnceLock::new();
    let sdk = re(&SDK, r"Applying plugin with SDK version: ([^\n]*?) cache-schema");
    log.sdk_version = sdk.captures(text).map(|c| c[1].trim().to_string());
    if log.compiled_partitions > 0
        || text.contains("compiled self-contained partition")
        || text.contains("compiled weight-store partition")
    {
        log.weight_store = Some(text.contains("weight store enabled"));
    }
    log.device_candidates = text.matches("candidate selection enabled for").count();
    log.device_samplers = text.matches("sampling enabled for").count() + log.device_candidates;
    log
}

/// Mean over the measured iterations (after the warm-ups, or the last one when
/// a run has no more iterations than warm-ups), the rule of report.py.
pub fn measured<'a>(iterations: &'a [Iteration], warmups: usize) -> &'a [Iteration] {
    if iterations.len() > warmups {
        &iterations[warmups..]
    } else if iterations.is_empty() {
        iterations
    } else {
        &iterations[iterations.len() - 1..]
    }
}

pub fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 {
        None
    } else {
        Some(sum / n as f64)
    }
}

/// One JSON line of LITERT_NVIDIA_DISPATCH_DUMP_IO=top<K>:<file>.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogitsLine {
    pub context: u32,
    pub function: String,
    pub step: u32,
    pub output: String,
    pub rows: usize,
    pub columns: usize,
    /// Per row, the (index, value) pairs in descending value order.
    pub top: Vec<Vec<(u32, f64)>>,
}

pub fn parse_logits_jsonl(text: &str) -> Result<Vec<LogitsLine>> {
    let mut lines = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: LogitsLine =
            serde_json::from_str(line).with_context(|| format!("logits line {}", number + 1))?;
        lines.push(parsed);
    }
    Ok(lines)
}

/// GNU time's `-v` accounting: the peak RSS in GiB and the wall time.
pub fn parse_time(text: &str) -> (Option<f64>, Option<f64>) {
    static RSS: OnceLock<Regex> = OnceLock::new();
    static WALL: OnceLock<Regex> = OnceLock::new();
    let rss = re(&RSS, r"Maximum resident set size \(kbytes\): (\d+)");
    let wall = re(&WALL, r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)");
    let rss_gib = capture_f64(rss, text, 1).map(|kb| kb * 1024.0 / (1u64 << 30) as f64);
    let wall_s = wall.captures(text).map(|c| {
        c[1].split(':').filter_map(|p| p.parse::<f64>().ok()).fold(0.0, |acc, v| acc * 60.0 + v)
    });
    (rss_gib, wall_s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_sampler_lines_count_the_greedy_and_the_candidate_paths() {
        let log = parse_cli_log("I x NVIDIA MTP GPU greedy sampling enabled for 4 rows\nI x NVIDIA MTP GPU candidate selection enabled for 4 rows of 262144 (k 64, p 0.95, temperature 1)\n");
        assert_eq!(log.device_candidates, 1);
        assert_eq!(log.device_samplers, 2);
        assert_eq!(parse_cli_log("nothing").device_samplers, 0);
    }

    const LOG: &str = "I0000 litert_lm_lib.cc:431] BenchmarkInfo:
  Init Phases (7):
    - Init Conversation: 0.06 ms
    - Init Executor: 17877.14 ms
    - Init Total: 17886.37 ms
--------------------------------------------------
  Time to first token: 1.37 s
--------------------------------------------------
  Prefill Turns (Total 1 turns):
    Prefill Turn 1: Processed 8025 tokens in 1.36304557s duration.
      Prefill Speed: 5887.55 tokens/sec.
--------------------------------------------------
  Decode Turns (Total 1 turns):
    Decode Turn 1: Processed 256 tokens in 1.109332937s duration.
      Decode Speed: 230.77 tokens/sec.
--------------------------------------------------
  Speculative Decoding (MTP):
    Rounds:            82 (98.8% of decode steps)
    Acceptance rate:   52.7% (173 of 328 drafted tokens)
    Tokens per round:  3.11 (255 emitted)
    Time per round:    13.27 ms (draft 2.91 ms, verify 10.23 ms)
    Plain decodes:     1 (incl. re-priming), 16.67 ms/step
--------------------------------------------------
I0000 litert_lm_lib.cc:181] Captured model output: Paris is the capital.
I0000 litert_lm_lib.cc:431] BenchmarkInfo:
  Prefill Turns (Total 1 turns):
    Prefill Turn 1: Processed 1024 tokens in 41.561308ms duration.
      Prefill Speed: 24638.21 tokens/sec.
  Decode Turns (Total 1 turns):
    Decode Turn 1: Processed 256 tokens in 778.450015ms duration.
      Decode Speed: 328.86 tokens/sec.
";

    #[test]
    fn durations() {
        assert!((duration_seconds("1.36304557s") - 1.36304557).abs() < 1e-9);
        assert!((duration_seconds("41.561308ms") - 0.041561308).abs() < 1e-12);
        assert!((duration_seconds("1m2.5s") - 62.5).abs() < 1e-9);
    }

    #[test]
    fn benchmark_blocks() {
        let log = parse_cli_log(LOG);
        assert_eq!(log.iterations.len(), 2);
        let first = &log.iterations[0];
        assert_eq!(first.prefill[0].tokens, 8025);
        assert!((first.decode[0].tokens_per_second - 230.77).abs() < 1e-9);
        assert_eq!(first.init_total_s, Some(17.88637));
        assert_eq!(first.time_to_first_token_s, Some(1.37));
        let mtp = first.mtp.as_ref().unwrap();
        assert_eq!(mtp.accepted, 173);
        assert_eq!(mtp.rounds, Some(82));
        assert_eq!(mtp.tokens_per_round, Some(3.11));
        assert_eq!(mtp.verify_ms, Some(10.23));
        assert_eq!(mtp.plain_step_ms, Some(16.67));
        assert!(log.iterations[1].mtp.is_none());
        assert_eq!(log.captured_output.as_deref(), Some("Paris is the capital."));
        assert_eq!(measured(&log.iterations, 1).len(), 1);
        assert_eq!(measured(&log.iterations, 5).len(), 1);
        assert_eq!(measured(&log.iterations, 0).len(), 2);
    }

    #[test]
    fn logits_lines() {
        let text = r#"{"context":1,"function":"decode","step":3,"output":"logits","rows":1,"columns":262144,"top":[[[12,4.5],[7,4.25]]]}
{"context":1,"function":"decode","step":4,"output":"logits","rows":1,"columns":262144,"top":[[[7,5.0],[12,4.0]]]}
"#;
        let lines = parse_logits_jsonl(text).unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].top[0][0], (12, 4.5));
        assert_eq!(lines[1].step, 4);
    }

    #[test]
    fn time_file() {
        let (rss, wall) = parse_time("\tMaximum resident set size (kbytes): 1048576\n\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02.50\n");
        assert_eq!(rss, Some(1.0));
        assert_eq!(wall, Some(62.5));
    }
}
