# Generative LLM Evaluation

> [!NOTE] This is one member of the **pluggable task evaluation family**
> (`image_segmentation_eval.md`, `llm_eval.md`, `classification_eval.md`).
> Only read and apply this file if the target model is a generative language
> model (or any autoregressive token-producing model).

When compressing LLMs with the AI Edge Quantizer, agents must evaluate the
final quantized model against the float reference using token-level metrics,
because autoregressive decoding cascades small logit errors into entirely
different output sequences.

## 1. Know the Model Topology First

LiteRT LLMs differ structurally from single-signature vision models:

*   **Multiple signatures**: Converted LLMs typically expose `prefill` (process
    the prompt) and `decode` (generate one token) signatures. Enumerate them
    via `validation_results.available_signature_keys()` and evaluate BOTH —
    quantization noise can affect them differently because they exercise
    different sequence lengths. `qt.validate()` accepts a per-signature
    `test_data` dict; you MUST supply real tokenized prompts for token-id
    inputs. Do NOT rely on the API's random-normal default test data: token-id
    inputs are integers indexing an embedding table, and random floats produce
    meaningless activations.
*   **KV cache tensors**: Cache input/output tensors appear as signature I/O.
    Exclude KV cache round-trip tensors from sensitivity rankings (they are
    state, not computation) but monitor their metrics as a diagnostic —
    degraded cache values compound across decode steps.
*   **Conversion boundary**: Models converted through the ai-edge-torch
    Generative API may embed their own quantization annotations. If the source
    model already carries a generative-API quantization config, confirm with
    the user whether AEQ recipe exploration should override it before
    proceeding.

## 2. Search Metrics vs Report Metrics

Follow the golden rule from `snr_best_practices.md`: cheap tensor metrics for
the search loop, task metrics only for the final report.

*   **Search loop (every iteration)**: `KL_DIVERGENCE` on the output logits
    tensor as the stopping metric; `COSINE_SIMILARITY` on intermediate block
    outputs for sensitivity ranking (see `error_metric_selection.md`).
*   **Final report (Top 3 profiles only)**: the token-level metrics below.

## 3. Standard Token-Level Report Metrics

*   **Top-1 Token Agreement Rate**: Fraction of positions where the quantized
    model's argmax token matches the float model's argmax token, measured
    teacher-forced (feed both models the SAME ground-truth token sequence and
    compare their next-token predictions independently at each position).
    This is the single most interpretable metric; report it prominently.
    Healthy INT8 models typically stay above 99%; investigate anything below
    95%.
*   **Mean Token-Level KL Divergence**: KL between the float and quantized
    softmax distributions averaged over positions. Captures probability drift
    invisible to argmax agreement.
*   **Perplexity Delta**: If a small ground-truth text corpus is available,
    report `ppl(quantized) - ppl(float)` on it. Do NOT use this in the search
    loop — it is far too slow.
*   **Sample Generations (Qualitative)**: Greedy-decode 3–5 fixed prompts on
    both models and save the side-by-side outputs to
    `results_fig/{base_name}_generations.txt`. Divergence position (how many
    tokens match before the first difference) is a useful summary statistic.

## 4. Boilerplate: Teacher-Forced Token Agreement

```python
import numpy as np

def token_agreement(float_logits, quant_logits):
    """Computes top-1 agreement and mean KL over teacher-forced positions.

    Args:
      float_logits: [num_positions, vocab] float reference logits.
      quant_logits: [num_positions, vocab] quantized model logits.
    """
    agree = np.mean(
        np.argmax(float_logits, axis=-1) == np.argmax(quant_logits, axis=-1)
    )

    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    p, q = softmax(float_logits), softmax(quant_logits)
    kl = np.mean(np.sum(p * np.log((p + 1e-10) / (q + 1e-10)), axis=-1))
    return float(agree), float(kl)
```

Save the resulting statistics to
`results_fig/{base_name}_token_agreement.json` per `file_naming.md`.

## 5. Cost Control

LLM evaluation is expensive; keep it inside the Tier L budgets from
`model_scale_tiers.md`:

*   Use SHORT fixed prompts (e.g. 16–64 tokens) and short generation lengths
    (e.g. 32 tokens) during evaluation.
*   Evaluate token-level metrics only on the final Top 3 profiles, never
    inside the greedy loop.
*   Use a fixed random seed and identical prompts across all configurations so
    numbers are comparable.
