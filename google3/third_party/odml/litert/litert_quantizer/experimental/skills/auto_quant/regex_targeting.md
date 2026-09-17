---
name: aeq-regex-targeting
description: >-
  Guides the agent to construct clean, block-level,
  hardware-friendly regular expressions for selective quantization, preventing
  scattered bit-depth changes and giant, unreadable joined regex strings.
---

# AEQ Regular Expression Targeting & Block Isolation

When performing selective quantization, you must configure precision overrides
(e.g., `no_quantize` float fallbacks or INT4 squashing) using regular
expressions matching tensor paths.

Choosing the right granularity of regular expressions is critical for both
recipe readability and model inference performance.

## 1. The Cost of Fine-Grained (Scattered) Regexes

When an agent identifies specific tensors (e.g., specific activations, bias
additions, or single operations) that collapse in quality, it is tempting to
generate a regex that targets *only* those specific leaf nodes. This is an
anti-pattern for two reasons:

1.  **Quantization/Dequantization Overhead**: In TFLite and edge hardware
    (NPU/CPU), switching back and forth between float and quantized integer
    domains introduces dequantize/quantize operators. If you quantize layer $N$,
    keep $N+1$ in float, and quantize $N+2$, you force two boundary crossings.
    This destroys runtime efficiency and latency.
2.  **Recipe Bloat**: Joining dozens of fully-qualified leaf paths with `|` (OR)
    creates unreadable, giant regex strings (often hundreds of characters long).
    These can easily hit memory/parsing issues and make the JSON recipe
    completely incomprehensible.

### The Correct Approach: Block-Level Isolation

Identify the **common structural parent module (block)** containing the
sensitive layers, and override the entire block. If a huge chunk of the block is
running in a uniform precision, dequantization/quantization transitions only
occur at the boundary of the block, minimizing execution overhead and keeping
the regex simple & readable.

--------------------------------------------------------------------------------

## 2. Core Heuristic for Block Extraction

To extract the core structural block name programmatically from a list of
sensitive tensor paths:

1.  **Analyze Path Segments**: Splitting a tensor path on `/` and `;` reveals
    the structural hierarchy of the PyTorch/TFLite model.
2.  **Find the Common Ancestor**: Group the sensitive tensors by their
    sub-module paths. Look for ancestors that represent meaningful layers or
    stages.
3.  **Threshold-based Promotion**: If one or more key components of a structural
    block (like a specific U-net stage, residual block, or attention head)
    collapse in precision, promote the fallback config to the entire block.
4.  **Filter Out Universal Model Wrappers**: When programmatically extracting
    structural parent blocks by splitting tensor path strings, dynamically
    evaluate the frequency of each path segment across the entire model graph.
    If a starting path segment or namespace prefix appears across more than 50%
    of all tensors in the network, treat it as a universal wrapper or global
    container. Discard that universal prefix and extract the first distinct
    internal sub-module segment instead to guarantee your generated regular
    expressions target granular architectural blocks rather than acting as
    unintended global wildcards (`.*`).

    **Example: Dynamically Stripping Universal Wrappers**

    ```python
    def get_universal_prefix(all_tensor_names):
        """Finds a common path prefix (separated by '/') shared across all tensors."""
        split_paths = [t.replace(';', '/').split('/') for t in all_tensor_names]
        if not split_paths: return []

        common_prefix = []
        min_len = min(len(p) for p in split_paths)

        for i in range(min_len):
            first_segment = split_paths[0][i]
            # If the segment is identical across all tensors, it's a structural wrapper
            if all(p[i] == first_segment for p in split_paths):
                common_prefix.append(first_segment)
            else:
                break

        return common_prefix

    # Usage:
    # 1. Identify the universal prefix length: len(get_universal_prefix(all_tensors))
    # 2. When creating a regex for a specific tensor, skip those universal segments
    #    and target the subsequent meaningful sub-module block afterward.
    ```

--------------------------------------------------------------------------------

## 3. Hazard: Fused-Name Cross-Capture (Empirically Verified)

Converters such as `ai-edge-torch` FUSE tensor names from multiple module
scopes into one op name, joined with `;`. AEQ matches a rule's regex anywhere
in that fused string. Two consequences:

1.  **Sibling-name shadowing**: a bare block name like `encoder_stage1` also
    matches `encoder_stage1d`. Always terminate block regexes with a
    boundary, e.g. `re.escape(block) + r"([/;]|$)"`.
2.  **Cross-capture through fusion**: even WITH the boundary, a rule for
    block `X` can capture ops of a different block `Y` whose fused name
    references `X` (observed in a U-Net-style search: an encoder-stage INT4
    rule also captured the corresponding decoder stage's convs, because the
    decoder ops' fused names embedded encoder-stage scopes). This
    over-matching only ever WIDENS a rule's coverage, so a correctly gated
    search remains valid — but your committed per-block accounting becomes
    unreliable.

Required mitigations in the search loop:

*   Gate every commit on the measured stopping metric of the actual exported
    model, never on the assumption that a rule touched only its named block.
*   Detect **no-op commits**: if a newly committed rule changes neither the
    exported byte size nor the stopping metric versus the previous committed
    step, the block was already covered by an earlier rule. Log it as a
    no-op and exclude it from per-block claims in the report.
*   Treat the exported recipe `.json` files as the single source of truth
    for what each model contains, not the loop's committed-block list.

