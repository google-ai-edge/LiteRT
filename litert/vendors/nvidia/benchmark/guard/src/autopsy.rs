// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! A step's per-kernel profile (the dispatch's autopsy of its own CUDA graph:
//! every kernel node timed alone, warm and cold) joined with the graph's DOT,
//! which names the kernels; and the diff of two such profiles, by kernel, so
//! a performance change is attributed to the kernels that moved.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Kernel {
    pub id: u32,
    /// The kernel's name as the graph names it (mangled, or a TensorRT fusion).
    pub name: String,
    /// A readable name: demangled and cut to its identifier.
    pub short: String,
    pub grid: [u32; 3],
    pub block: u32,
    pub cold_us: f64,
    pub warm_us: f64,
    /// The node's time in the real step (the timeline: the time from the
    /// previous node's end to this one's, its launch gap included), when the
    /// run wrote one.
    #[serde(default)]
    pub graph_us: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionAutopsy {
    pub function: String,
    pub flush_us: f64,
    pub kernels: Vec<Kernel>,
    pub warm_us: f64,
    pub cold_us: f64,
    /// The step's whole time in the timeline replay, and the kernel nodes'
    /// part of it; None without a timeline.
    #[serde(default)]
    pub graph_total_us: Option<f64>,
    #[serde(default)]
    pub graph_kernel_us: Option<f64>,
}

/// The autopsy of one run: every function's step graph, kernel by kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Autopsy {
    pub model: String,
    pub workload: String,
    pub mtp: bool,
    pub functions: Vec<FunctionAutopsy>,
}

/// The kernel names of a graph's DOT (cudaGraphDebugDotPrint), by node id.
pub fn parse_dot_names(text: &str) -> BTreeMap<u32, String> {
    // | {ID | 12 (topoId: 870) | _ZN...Kernel...\<\<\<1024,256,0\>\>\>}
    let re = Regex::new(r"\{ID \| (\d+) \([^)]*\) \| ([^\\}]+?)(?:\\<\\<\\<[^}]*)?\}").unwrap();
    let mut names = BTreeMap::new();
    for caps in re.captures_iter(text) {
        if let Ok(id) = caps[1].parse::<u32>() {
            names.insert(id, caps[2].trim().to_string());
        }
    }
    names
}

/// Demangles with c++filt where it is, else keeps the names.
fn demangle_all(names: &[String]) -> Vec<String> {
    let mangled: Vec<&String> = names.iter().filter(|n| n.starts_with("_Z")).collect();
    if mangled.is_empty() {
        return names.to_vec();
    }
    let demangled: Option<Vec<String>> = (|| {
        let mut child = Command::new("c++filt")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        {
            let mut stdin = child.stdin.take()?;
            for name in &mangled {
                writeln!(stdin, "{name}").ok()?;
            }
        }
        let output = child.wait_with_output().ok()?;
        if !output.status.success() {
            return None;
        }
        let lines: Vec<String> =
            String::from_utf8_lossy(&output.stdout).lines().map(|l| l.to_string()).collect();
        (lines.len() == mangled.len()).then_some(lines)
    })();
    let mut out = Vec::with_capacity(names.len());
    let mut next = 0;
    for name in names {
        if name.starts_with("_Z") {
            out.push(demangled.as_ref().map(|d| d[next].clone()).unwrap_or_else(|| name.clone()));
            next += 1;
        } else {
            out.push(name.clone());
        }
    }
    out
}

/// The identifier of a demangled kernel: the function's own name, without
/// its namespaces and template arguments; a TensorRT fusion keeps its head.
pub fn short_name(demangled: &str) -> String {
    let text = demangled.replace("(anonymous namespace)", "_anonymous_");
    let text = text.trim_start_matches("void ").trim();
    // Drop template arguments at any depth and stop at the parameter list
    // (the first parenthesis outside template arguments).
    let mut depth = 0;
    let mut stripped = String::new();
    for ch in text.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            '(' if depth == 0 => break,
            _ if depth == 0 => stripped.push(ch),
            _ => {}
        }
    }
    let stripped = stripped.trim();
    // A return type, when the demangler prints one, precedes a space.
    let stripped = stripped.rsplit(' ').next().unwrap_or(stripped);
    let last = stripped.rsplit("::").next().unwrap_or(stripped).trim().to_string();
    let last = if last.is_empty() { stripped.trim().to_string() } else { last };
    if last.starts_with("__myl_") {
        let cut: String = last.chars().take(48).collect();
        return cut;
    }
    last
}

/// Parses one function's timeline file (`# total_us T nodes N`, then `node
/// type end_us dur_us` lines): the total and each node's duration by index.
pub fn parse_timeline(text: &str) -> (Option<f64>, BTreeMap<u32, f64>) {
    let mut total = None;
    let mut durations = BTreeMap::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# total_us") {
            total = rest.split_whitespace().next().and_then(|v| v.parse().ok());
            continue;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            continue;
        }
        if let (Ok(id), Ok(dur)) = (fields[0].parse::<u32>(), fields[3].parse::<f64>()) {
            durations.insert(id, dur);
        }
    }
    (total, durations)
}

/// Parses one function's autopsy file (`# node grid_x grid_y grid_z block_x
/// cold_us warm_us` lines) with the DOT's names, and the timeline's node
/// times when given.
pub fn parse_autopsy_with_timeline(
    function: &str,
    autopsy_text: &str,
    dot_text: Option<&str>,
    timeline_text: Option<&str>,
) -> FunctionAutopsy {
    let mut autopsy = parse_autopsy(function, autopsy_text, dot_text);
    if let Some(text) = timeline_text {
        let (total, durations) = parse_timeline(text);
        for kernel in &mut autopsy.kernels {
            kernel.graph_us = durations.get(&kernel.id).copied();
        }
        autopsy.graph_total_us = total;
        autopsy.graph_kernel_us = Some(autopsy.kernels.iter().filter_map(|k| k.graph_us).sum());
    }
    autopsy
}

/// Parses one function's autopsy file (`# node grid_x grid_y grid_z block_x
/// cold_us warm_us` lines) with the DOT's names.
pub fn parse_autopsy(
    function: &str,
    autopsy_text: &str,
    dot_text: Option<&str>,
) -> FunctionAutopsy {
    let names = dot_text.map(parse_dot_names).unwrap_or_default();
    let mut flush_us = 0.0;
    let mut raw: Vec<(u32, [u32; 3], u32, f64, f64)> = Vec::new();
    for line in autopsy_text.lines() {
        if let Some(rest) = line.strip_prefix("# flush_us") {
            flush_us = rest.trim().parse().unwrap_or(0.0);
            continue;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 7 {
            continue;
        }
        let num = |i: usize| fields[i].parse::<f64>().unwrap_or(0.0);
        raw.push((
            fields[0].parse().unwrap_or(0),
            [num(1) as u32, num(2) as u32, num(3) as u32],
            num(4) as u32,
            num(5),
            num(6),
        ));
    }
    let mangled: Vec<String> = raw
        .iter()
        .map(|(id, ..)| names.get(id).cloned().unwrap_or_else(|| format!("node {id}")))
        .collect();
    let demangled = demangle_all(&mangled);
    let kernels: Vec<Kernel> = raw
        .iter()
        .zip(mangled.iter())
        .zip(demangled.iter())
        .map(|(((id, grid, block, cold, warm), name), dem)| Kernel {
            id: *id,
            name: name.clone(),
            short: short_name(dem),
            graph_us: None,
            grid: *grid,
            block: *block,
            cold_us: *cold,
            warm_us: *warm,
        })
        .collect();
    FunctionAutopsy {
        function: function.to_string(),
        flush_us,
        warm_us: kernels.iter().map(|k| k.warm_us.max(0.0)).sum(),
        cold_us: kernels.iter().map(|k| k.cold_us.max(0.0)).sum(),
        kernels,
        graph_total_us: None,
        graph_kernel_us: None,
    }
}

/// Reads an autopsy run directory: `autopsy.txt.<function>` files and the
/// DOT of each function's first captured step under `dot/`.
pub fn load_autopsy_dir(dir: &Path, model: &str, workload: &str, mtp: bool) -> Result<Autopsy> {
    let mut functions = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(function) = name.strip_prefix("autopsy.txt.") else { continue };
        let text = std::fs::read_to_string(entry.path())?;
        let timeline = std::fs::read_to_string(dir.join(format!("timeline.txt.{function}"))).ok();
        let dot = std::fs::read_dir(dir.join("dot"))
            .ok()
            .and_then(|d| {
                d.flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .map(|f| f.to_string_lossy().starts_with(&format!("{function}_")))
                            .unwrap_or(false)
                    })
                    .min()
            })
            .and_then(|p| std::fs::read_to_string(p).ok());
        functions.push(parse_autopsy_with_timeline(
            function,
            &text,
            dot.as_deref(),
            timeline.as_deref(),
        ));
    }
    Ok(Autopsy { model: model.to_string(), workload: workload.to_string(), mtp, functions })
}

/// One kernel (by function and short name) in both autopsies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelDelta {
    pub function: String,
    pub short: String,
    pub base_count: usize,
    pub cand_count: usize,
    pub base_us: f64,
    pub cand_us: f64,
    pub delta_us: f64,
    /// The in-graph (timeline) sums, when both captures have them.
    #[serde(default)]
    pub base_graph_us: Option<f64>,
    #[serde(default)]
    pub cand_graph_us: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDelta {
    pub function: String,
    pub base_kernels: usize,
    pub cand_kernels: usize,
    pub base_us: f64,
    pub cand_us: f64,
    pub delta_us: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutopsyDiff {
    pub model: String,
    pub workload: String,
    pub mtp: bool,
    pub functions: Vec<FunctionDelta>,
    /// Kernels by |delta|, descending.
    pub kernels: Vec<KernelDelta>,
}

/// Warm times summed by (function, kernel name) on both sides.
pub fn diff_autopsies(base: &Autopsy, cand: &Autopsy) -> AutopsyDiff {
    let mut sums: BTreeMap<(String, String), (usize, usize, f64, f64, Option<f64>, Option<f64>)> =
        BTreeMap::new();
    let mut functions: BTreeMap<String, (usize, usize, f64, f64)> = BTreeMap::new();
    let add_graph = |slot: &mut Option<f64>, value: Option<f64>| {
        if let Some(v) = value {
            *slot = Some(slot.unwrap_or(0.0) + v);
        }
    };
    for f in &base.functions {
        let entry = functions.entry(f.function.clone()).or_default();
        entry.0 = f.kernels.len();
        entry.2 = f.warm_us;
        for k in &f.kernels {
            let s = sums.entry((f.function.clone(), k.short.clone())).or_default();
            s.0 += 1;
            s.2 += k.warm_us.max(0.0);
            add_graph(&mut s.4, k.graph_us);
        }
    }
    for f in &cand.functions {
        let entry = functions.entry(f.function.clone()).or_default();
        entry.1 = f.kernels.len();
        entry.3 = f.warm_us;
        for k in &f.kernels {
            let s = sums.entry((f.function.clone(), k.short.clone())).or_default();
            s.1 += 1;
            s.3 += k.warm_us.max(0.0);
            add_graph(&mut s.5, k.graph_us);
        }
    }
    let mut kernels: Vec<KernelDelta> = sums
        .into_iter()
        .map(|((function, short), (bc, cc, bu, cu, bg, cg))| KernelDelta {
            function,
            short,
            base_count: bc,
            cand_count: cc,
            base_us: bu,
            cand_us: cu,
            delta_us: cu - bu,
            base_graph_us: bg,
            cand_graph_us: cg,
        })
        .collect();
    kernels.sort_by(|a, b| {
        b.delta_us.abs().partial_cmp(&a.delta_us.abs()).unwrap_or(std::cmp::Ordering::Equal)
    });
    AutopsyDiff {
        model: cand.model.clone(),
        workload: cand.workload.clone(),
        mtp: cand.mtp,
        functions: functions
            .into_iter()
            .map(|(function, (bk, ck, bu, cu))| FunctionDelta {
                function,
                base_kernels: bk,
                cand_kernels: ck,
                base_us: bu,
                cand_us: cu,
                delta_us: cu - bu,
            })
            .collect(),
        kernels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOT: &str = r#"digraph dot {
"graph_1_node_0"[style="bold" shape="record" label="{KERNEL
| {ID | 0 (topoId: 881) | __myl_CastCastMulMeanAddSqrtDivMulCast_0x79a974d26df997acf47414e94793d5c9\<\<\<1024,128,7744\>\>\>}
| {cooperative | 0}
}"];
"graph_1_node_1"[style="bold" shape="record" label="{KERNEL
| {ID | 1 (topoId: 870) | _ZN48_GLOBAL__N__60fd1c23_15_subbyte_gemm_cu_7e170b1415ScaleRowsKernelEPK13__nv_bfloat16PKiiPKfffPfP6__half\<\<\<1024,256,0\>\>\>}
}"];
}"#;

    const AUTOPSY: &str = "# flush_us 657.4\n# node grid_x grid_y grid_z block_x cold_us warm_us\n0 1024 1 1 128 36.24 11.69\n1 1024 1 1 256 20.00 8.50\n";

    #[test]
    fn a_timeline_gives_every_kernel_its_time_in_the_step() {
        let base = parse_autopsy_with_timeline("decode", AUTOPSY, Some(DOT), Some("# total_us 30.5 nodes 3\n# node type end_us dur_us\n0 K 12.0 12.0\n2 S 13.0 1.0\n1 K 30.5 17.5\n"));
        assert_eq!(base.graph_total_us, Some(30.5));
        assert_eq!(base.kernels[0].graph_us, Some(12.0));
        assert_eq!(base.kernels[1].graph_us, Some(17.5));
        assert_eq!(base.graph_kernel_us, Some(29.5));
        let plain = parse_autopsy("decode", AUTOPSY, Some(DOT));
        assert_eq!(plain.graph_total_us, None);
        assert_eq!(plain.kernels[0].graph_us, None);
    }

    #[test]
    fn dot_names_by_id_and_short_names() {
        let names = parse_dot_names(DOT);
        assert_eq!(names.len(), 2);
        assert!(names[&0].starts_with("__myl_CastCast"));
        assert!(names[&1].starts_with("_ZN48_GLOBAL"));
        assert_eq!(
            short_name(
                "void (anonymous namespace)::ScaleRowsKernel<__half>(__nv_bfloat16 const*, int)"
            ),
            "ScaleRowsKernel"
        );
        assert_eq!(short_name("ns::Outer<int>::Inner<float>::Run(int)"), "Run");
        assert_eq!(short_name("void gemm::Kernel<void (*)(int), 4>(float const*)"), "Kernel");
        assert_eq!(short_name("plain_kernel(int)"), "plain_kernel");
        // c++filt's rendering of the dispatch's own kernels.
        assert_eq!(
            short_name("void litert::nvidia::(anonymous namespace)::Int8CacheWriteKernel<__nv_bfloat16, false>(signed char*, __nv_bfloat16 const*, int const*, int, int, int, int, float, int)"),
            "Int8CacheWriteKernel"
        );
        assert_eq!(
            short_name("void (anonymous namespace)::Bf16SubbytePerChannelGemvKernel<4, 1, 128>(__nv_bfloat16 const*, unsigned char const*, __nv_bfloat16 const*, __nv_bfloat16*, int, int)"),
            "Bf16SubbytePerChannelGemvKernel"
        );
        assert_eq!(
            short_name("cudnn_generated_fort_native_sm120_matMul_knob_25_32x32x32_0_0"),
            "cudnn_generated_fort_native_sm120_matMul_knob_25_32x32x32_0_0"
        );
        assert_eq!(short_name("__myl_CastCastMul_0x79a974d26df997acf47414e94793d5c9").len(), 48);
    }

    #[test]
    fn autopsy_joins_times_with_names_and_diffs_by_kernel() {
        let base = parse_autopsy("decode", AUTOPSY, Some(DOT));
        assert_eq!(base.kernels.len(), 2);
        assert_eq!(base.flush_us, 657.4);
        assert_eq!(base.kernels[1].grid, [1024, 1, 1]);
        assert_eq!(base.kernels[1].block, 256);
        assert!((base.warm_us - 20.19).abs() < 1e-9);
        // The second kernel's name resolves through c++filt when present.
        assert!(
            base.kernels[1].short == "ScaleRowsKernel"
                || base.kernels[1].short.starts_with("_ZN48")
        );
        let slower = "# flush_us 657.4\n0 1024 1 1 128 36.24 11.69\n1 1024 1 1 256 20.00 12.50\n";
        let cand = parse_autopsy("decode", slower, Some(DOT));
        let a = Autopsy {
            model: "12b".into(),
            workload: "short".into(),
            mtp: false,
            functions: vec![base],
        };
        let b = Autopsy {
            model: "12b".into(),
            workload: "short".into(),
            mtp: false,
            functions: vec![cand],
        };
        let diff = diff_autopsies(&a, &b);
        assert_eq!(diff.functions.len(), 1);
        assert!((diff.functions[0].delta_us - 4.0).abs() < 1e-9);
        assert!((diff.kernels[0].delta_us - 4.0).abs() < 1e-9, "{:?}", diff.kernels);
        assert_eq!(diff.kernels[0].base_count, 1);
        assert_eq!(diff.kernels[1].delta_us, 0.0);
    }
}
