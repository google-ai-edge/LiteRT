// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The benchmark environment: the paths of env.sh (the trees, the state
//! directory, the built binaries, the model files), the build id and the
//! TensorRT-RTX environment of a model cache pool. Everything comes from
//! env.sh itself, run through bash, so the guard and the scripts agree.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context, Result};

#[derive(Debug, Clone)]
pub struct BenchEnv {
    /// litert/vendors/nvidia/benchmark of the LiteRT tree.
    pub bench_dir: PathBuf,
    pub rt: PathBuf,
    pub lm: PathBuf,
    pub root: PathBuf,
    pub runtime_libs: PathBuf,
    pub models_dir: PathBuf,
    pub build_id: String,
    pub smi: PathBuf,
    /// Extra environment of every probe (capture --env NAME=VALUE): a runtime
    /// knob's A/B as a capture of its own. Its probes run in a pool of their
    /// own (pool_suffix), so a compile-time knob compiles its own engines.
    pub extra_env: Vec<(String, String)>,
    pub pool_suffix: String,
}

fn bash(script: &str, args: &[&str]) -> Result<String> {
    let output = Command::new("bash")
        .arg("-c")
        .arg(script)
        .arg("nvguard")
        .args(args)
        .output()
        .context("running bash")?;
    if !output.status.success() {
        bail!(
            "bash failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

impl BenchEnv {
    pub fn resolve(bench_dir: &Path) -> Result<BenchEnv> {
        let env_sh = bench_dir.join("env.sh");
        if !env_sh.is_file() {
            bail!("not a benchmark directory (no env.sh): {}", bench_dir.display());
        }
        let script = r#"set -e; source "$1/env.sh"; nb_resolve_paths
printf 'NB_RT=%s\nNB_LM=%s\nNB_ROOT=%s\nNB_RUNTIME_LIBS=%s\nNB_MODELS=%s\nBUILD_ID=%s\n' \
  "$NB_RT" "$NB_LM" "$NB_ROOT" "$NB_RUNTIME_LIBS" "$NB_MODELS" "$(nb_build_id)""#;
        let text = bash(script, &[&bench_dir.to_string_lossy()])?;
        let mut values = BTreeMap::new();
        for line in text.lines() {
            if let Some((key, value)) = line.split_once('=') {
                values.insert(key.to_string(), value.to_string());
            }
        }
        let get = |key: &str| -> Result<String> {
            values.get(key).cloned().ok_or_else(|| anyhow!("env.sh gave no {key}"))
        };
        let smi = match Command::new("nvidia-smi").arg("--version").output() {
            Ok(o) if o.status.success() => PathBuf::from("nvidia-smi"),
            _ => PathBuf::from("/usr/lib/wsl/lib/nvidia-smi"),
        };
        Ok(BenchEnv {
            bench_dir: bench_dir.to_path_buf(),
            rt: PathBuf::from(get("NB_RT")?),
            lm: PathBuf::from(get("NB_LM")?),
            root: PathBuf::from(get("NB_ROOT")?),
            runtime_libs: PathBuf::from(get("NB_RUNTIME_LIBS")?),
            models_dir: PathBuf::from(get("NB_MODELS")?),
            build_id: get("BUILD_ID")?,
            smi,
            extra_env: Vec::new(),
            pool_suffix: String::new(),
        })
    }

    pub fn cli(&self) -> PathBuf {
        self.runtime_libs.join("litert_lm_advanced_main")
    }

    pub fn built(&self) -> bool {
        self.cli().is_file() && self.runtime_libs.join("libLiteRtDispatch_Nvidia.so").is_file()
    }

    /// The litertlm file of a profile name (e2b, e4b, 12b, 26b) or a file path.
    pub fn model_file(&self, model: &str) -> Result<PathBuf> {
        let script = r#"set -e; source "$1/env.sh"; nb_resolve_paths; nb_model_file "$2""#;
        let text = bash(script, &[&self.bench_dir.to_string_lossy(), model])?;
        let path = PathBuf::from(text.trim());
        if !path.is_file() {
            bail!("model file missing: {}", path.display());
        }
        Ok(path)
    }

    /// The model cache pool of a profile for this build (and this
    /// environment's extra environment, when it has one).
    pub fn pool(&self, model: &str) -> PathBuf {
        self.root
            .join("cache")
            .join("models")
            .join(model)
            .join(format!("{}{}", self.build_id, self.pool_suffix))
    }

    /// bench.sh's arguments for this environment: its pool when the extra
    /// environment asks for one of its own, and the extra environment.
    pub fn bench_args(&self, model: &str) -> Vec<String> {
        let mut args = Vec::new();
        if !self.pool_suffix.is_empty() {
            args.push("--pool".to_string());
            args.push(self.pool(model).display().to_string());
        }
        for (key, value) in &self.extra_env {
            args.push("--env".to_string());
            args.push(format!("{key}={value}"));
        }
        args
    }

    /// The build the binaries in runtime-libs came from (build.sh's BUILD_ID
    /// stamp), when there is one.
    pub fn binaries_build_id(&self) -> Option<String> {
        std::fs::read_to_string(self.runtime_libs.join("BUILD_ID"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }

    /// clock_check.py's report over a window of `seconds` (the kernel tick,
    /// the monotonic clock's rate against the raw clock) and whether it finds
    /// the clock fit for timings.
    pub fn host_clock(&self, seconds: u32) -> Result<(serde_json::Value, bool)> {
        let script = self.bench_dir.join("clock_check.py");
        if !script.is_file() {
            bail!("no {}", script.display());
        }
        let output = Command::new("python3")
            .arg("-I")
            .arg(&script)
            .arg(seconds.to_string())
            .output()
            .context("running clock_check.py")?;
        let report: serde_json::Value =
            serde_json::from_slice(&output.stdout).with_context(|| {
                format!(
                    "clock_check.py printed no report: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                )
            })?;
        let fit =
            output.status.success() && report.get("fit").and_then(|v| v.as_bool()).unwrap_or(false);
        Ok((report, fit))
    }

    pub fn pool_has_engines(&self, pool: &Path) -> bool {
        std::fs::read_dir(pool.join("aot"))
            .map(|entries| {
                entries.flatten().any(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    name.starts_with("tensorrt_aot") && name.ends_with(".bin")
                })
            })
            .unwrap_or(false)
    }

    /// The environment of a CLI run in a pool: nb_runtime_env's exports on top
    /// of the current environment.
    pub fn runtime_env(&self, pool: &Path, model_file: &Path) -> Result<Vec<(String, String)>> {
        let script =
            r#"set -e; source "$1/env.sh"; nb_resolve_paths; nb_runtime_env "$2" "$3"; env -0"#;
        let output = Command::new("bash")
            .arg("-c")
            .arg(script)
            .arg("nvguard")
            .arg(&self.bench_dir)
            .arg(pool)
            .arg(model_file)
            .output()
            .context("running bash for the pool environment")?;
        if !output.status.success() {
            bail!("nb_runtime_env failed: {}", String::from_utf8_lossy(&output.stderr).trim());
        }
        let mut env: Vec<(String, String)> = Vec::new();
        for entry in output.stdout.split(|b| *b == 0) {
            if entry.is_empty() {
                continue;
            }
            let entry = String::from_utf8_lossy(entry);
            if let Some((key, value)) = entry.split_once('=') {
                if key == "_" || key == "SHLVL" || key == "PWD" || key == "OLDPWD" {
                    continue;
                }
                env.push((key.to_string(), value.to_string()));
            }
        }
        for (key, value) in &self.extra_env {
            env.retain(|(k, _)| k != key);
            env.push((key.clone(), value.clone()));
        }
        Ok(env)
    }

    /// Whether another CUDA compute process holds the GPU (bench.sh refuses to
    /// measure beside one).
    /// The space free on the filesystem that holds `path`, GiB (df -Pk).
    pub fn free_gib(path: &Path) -> Result<f64> {
        let output = Command::new("df").arg("-Pk").arg(path).output().context("running df")?;
        let text = String::from_utf8_lossy(&output.stdout);
        let available = text
            .lines()
            .nth(1)
            .and_then(|line| line.split_whitespace().nth(3))
            .and_then(|v| v.parse::<f64>().ok());
        match available {
            Some(kib) if output.status.success() => Ok(kib / 1048576.0),
            _ => bail!(
                "df -Pk {} gave no available space: {}",
                path.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
        }
    }

    pub fn gpu_busy(&self) -> Result<bool> {
        let output = Command::new(&self.smi)
            .args(["--query-compute-apps=pid", "--format=csv,noheader"])
            .output()
            .with_context(|| format!("running {}", self.smi.display()))?;
        if !output.status.success() {
            bail!("nvidia-smi failed: {}", String::from_utf8_lossy(&output.stderr).trim());
        }
        Ok(String::from_utf8_lossy(&output.stdout).lines().any(|l| !l.trim().is_empty()))
    }

    pub fn git_head(&self, tree: &Path) -> String {
        Command::new("git")
            .args(["-C", &tree.to_string_lossy(), "log", "-1", "--format=%h %s"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    }
}
