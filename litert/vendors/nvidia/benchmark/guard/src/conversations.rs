// Copyright 2026 The ODML Authors. Licensed under the Apache License, Version 2.0.
//! The conversation probe: one prompt decoded greedily in two conversations
//! of one process. A later conversation must give the first one's text: one
//! that parts depends on what an earlier conversation left behind (a cache's
//! stale rows, a workspace, a buffer read before it is written), which the
//! probes of one conversation each cannot see. The INT8 GEMM's outlier dims
//! chosen over a prefill chunk's padding rows, whose attention averages the
//! cache, are such a dependence (LITERT_NVIDIA_GEMM_INT8_REAL_ROWS).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRun {
    pub model: String,
    pub prompt: String,
    pub mtp: bool,
    pub build_id: String,
    pub exit_code: i32,
    pub wall_s: f64,
    /// Each conversation's text, in order.
    pub texts: Vec<String>,
}

impl ConversationRun {
    /// The first conversation (counted from 1) whose text is not the first
    /// one's, and the character where it parts from it; None when every text
    /// is the first one's.
    pub fn parting(&self) -> Option<(usize, usize)> {
        let first = self.texts.first()?;
        self.texts
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, text)| *text != first)
            .map(|(i, text)| (i + 1, parted_char(first, text)))
    }
}

/// The characters two texts have in common before they part.
pub fn parted_char(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(texts: &[&str]) -> ConversationRun {
        ConversationRun {
            model: "e4b".into(),
            prompt: "doc_8k".into(),
            mtp: false,
            build_id: "b".into(),
            exit_code: 0,
            wall_s: 1.0,
            texts: texts.iter().map(|t| t.to_string()).collect(),
        }
    }

    #[test]
    fn a_later_conversation_that_parts_is_named_with_its_character() {
        assert_eq!(run(&["abc", "abc", "abc"]).parting(), None);
        assert_eq!(run(&["abcdef", "abcxyz"]).parting(), Some((2, 3)));
        assert_eq!(run(&["abc", "abc", "ab"]).parting(), Some((3, 2)));
        assert_eq!(run(&[]).parting(), None);
    }
}
