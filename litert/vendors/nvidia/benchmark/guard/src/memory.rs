// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The memory probe: bench.sh's memory metric of one workload, plain and
//! MTP. The device memory is the dispatch's memory profile's largest
//! device-wide use less its first one (taken once the process's CUDA context
//! exists: the context and the other processes' memory are in both), the host
//! memory the profile's peak RSS: allocations, not a sampled peak, so a
//! same-build pair agrees to a few MiB where nvidia-smi's samples move by a
//! hundred. A run that compiled its engines in the process
//! (an AOT cache miss) holds the compiler's memory: the probe runs it again
//! from the warm cache.

use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryRun {
    pub model: String,
    pub mtp: bool,
    pub workload: String,
    pub build_id: String,
    pub exit_code: i32,
    /// The most device memory the run allocated past its first memory-profile
    /// line, MiB (None: no memory profile).
    pub gpu_mib: Option<f64>,
    /// The process's peak RSS, GiB.
    pub rss_gib: Option<f64>,
    /// The run compiled its engines (its peaks are the compiler's).
    pub compiled: bool,
    pub label: String,
}

/// A bench.sh memory run directory (meta.json, cli.log) as a MemoryRun.
pub fn summarize_memory_run(
    run_dir: &Path,
    model: &str,
    mtp: bool,
    workload: &str,
) -> Result<MemoryRun> {
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("meta.json"))
            .with_context(|| format!("reading {}/meta.json", run_dir.display()))?,
    )?;
    let log = std::fs::read_to_string(run_dir.join("cli.log")).unwrap_or_default();
    let (first, used, rss) = profile_peaks(&log);
    Ok(MemoryRun {
        model: model.to_string(),
        mtp,
        workload: workload.to_string(),
        build_id: meta.get("build_id").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
        exit_code: meta.get("exit_code").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
        gpu_mib: used.zip(first).map(|(most, first)| (most as f64 - first as f64) / 1048576.0),
        rss_gib: rss.map(|bytes| bytes as f64 / 1073741824.0),
        compiled: log.contains("AOT cache miss") || log.contains("phase=aot_cache_miss"),
        label: run_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
    })
}

/// The memory profile's first and largest device-wide use and its peak RSS,
/// in bytes.
fn profile_peaks(log: &str) -> (Option<u64>, Option<u64>, Option<u64>) {
    let used = Regex::new(r"cuda_device_used_bytes=(\d+)").expect("valid regex");
    let rss = Regex::new(r"cpu_peak_rss_bytes=(\d+)").expect("valid regex");
    let values = |re: &Regex| {
        re.captures_iter(log).filter_map(|c| c[1].parse::<u64>().ok()).collect::<Vec<_>>()
    };
    let used = values(&used);
    (used.first().copied(), used.iter().max().copied(), values(&rss).into_iter().max())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_memory_run_is_its_profiles_largest_use_less_its_first() {
        let dir = std::env::temp_dir().join(format!("nvguard_memory_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("meta.json"),
            r#"{"build_id": "b", "exit_code": 0, "gpu_baseline_mib": 481}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("cli.log"),
            "I x NVIDIA memory profile component=dispatch phase=a cpu_peak_rss_bytes=1073741824 cuda_device_used_bytes=1048576000\n\
             I x NVIDIA memory profile component=dispatch phase=b cpu_peak_rss_bytes=2147483648 cuda_device_used_bytes=2097152000\n",
        )
        .unwrap();
        let run = summarize_memory_run(&dir, "e2b", true, "short").unwrap();
        assert_eq!(run.gpu_mib, Some(2000.0 - 1000.0));
        assert_eq!(run.rss_gib, Some(2.0));
        assert!(!run.compiled);
        std::fs::write(dir.join("cli.log"), "I x NVIDIA memory profile component=compiler phase=aot_cache_miss cpu_peak_rss_bytes=1\n").unwrap();
        let run = summarize_memory_run(&dir, "e2b", true, "short").unwrap();
        assert!(run.compiled);
        assert_eq!(run.gpu_mib, None);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
