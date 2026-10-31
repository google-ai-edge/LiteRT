// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The text probes: a sampled decode (top-k / top-p with a fixed seed) and a
//! constrained one (a regex and a repetition penalty, in the CLI's session
//! mode), each run with the MTP verifier sampling on the device and on the
//! host. Their texts must agree, and the constrained one must match its
//! regex: what the greedy probes cannot see.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRun {
    pub model: String,
    /// sampled | constraint
    pub kind: String,
    /// device | host
    pub variant: String,
    pub build_id: String,
    pub exit_code: i32,
    pub wall_s: f64,
    pub text: String,
    pub text_len: usize,
    /// Whether the text matched the probe's regex (constraint probes).
    pub regex_ok: Option<bool>,
    pub device_samplers: usize,
    pub parser_errors: usize,
}

/// Where two texts part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextComparison {
    pub model: String,
    pub kind: String,
    pub identical: bool,
    /// The character offset where they part (the shorter's length when one
    /// is a prefix of the other).
    pub parted_char: Option<usize>,
    /// Whole words in common before they part.
    pub common_words: usize,
    pub device_exit: i32,
    pub host_exit: i32,
    pub device_regex_ok: Option<bool>,
    pub host_regex_ok: Option<bool>,
    pub device_samplers: usize,
    pub device_text_len: usize,
    pub host_text_len: usize,
}

pub fn parted_at(a: &str, b: &str) -> Option<usize> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    if n == a.len() && n == b.len() {
        None
    } else {
        Some(n)
    }
}

pub fn common_words(a: &str, b: &str) -> usize {
    let n = parted_at(a, b).unwrap_or(a.chars().count());
    let prefix: String = a.chars().take(n).collect();
    prefix.split_whitespace().count()
}

pub fn compare_texts(device: &TextRun, host: &TextRun) -> TextComparison {
    TextComparison {
        model: device.model.clone(),
        kind: device.kind.clone(),
        identical: device.text == host.text,
        parted_char: parted_at(&device.text, &host.text),
        common_words: common_words(&device.text, &host.text),
        device_exit: device.exit_code,
        host_exit: host.exit_code,
        device_regex_ok: device.regex_ok,
        host_regex_ok: host.regex_ok,
        device_samplers: device.device_samplers,
        device_text_len: device.text_len,
        host_text_len: host.text_len,
    }
}

/// The session mode's answer: the CLI prints "output: <text>".
pub fn session_output(stdout: &str) -> String {
    match stdout.split_once("output: ") {
        Some((_, rest)) => rest.trim().to_string(),
        None => stdout.trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parting_and_common_words() {
        assert_eq!(parted_at("abc", "abc"), None);
        assert_eq!(parted_at("abc", "abd"), Some(2));
        assert_eq!(parted_at("abc", "abcd"), Some(3));
        assert_eq!(common_words("the cat sat on", "the cat ran"), 2);
        assert_eq!(common_words("same text", "same text"), 2);
        assert_eq!(session_output("junk\noutput: {\"a\":1}\n"), "{\"a\":1}");
        assert_eq!(session_output("no marker"), "no marker");
    }
}
