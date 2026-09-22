# Output Report for AEQ Recipe Exploration

This guide dictates the structural formatting of the final delivery report when
performing Pareto-frontier optimizations via the AI Edge Quantizer.

The output report must clearly contrast the baseline model against the
generated, optimized quantization recipes. The agent should summarize its
iterative decisions for regex layer skipping or algorithm changes, and present
the final empirical results (extracted directly from `os.path.getsize(tflite)`
and the deployment-faithful stopping-metric measurements defined in
`experiment_runner.md`).

## Markdown Quality Requirements

The generated markdown report must strictly adhere to `markdownlint` rules. Pay
special attention to:

-   Proper heading increments (no skipping heading levels).
-   Consistent spacing before and after headings, lists, and code blocks.
-   Correct list indentation and formatting.
-   Removing trailing spaces and trailing blank lines.
-   Wrapping text correctly to limit line length (e.g. 80 columns), excepting
    long URLs, table rows, and code blocks where applicable.

Failing to adhere to these rules will result in linting failures during
automated reviews.

## Output Footprint Structure

You must save all generated assets in the following strict locations:

-   **Quantized Models**: Save `.tflite` variations and their associated `.json`
    recipes in the `model/quantized/` and `model/quantized/recipes/` folders.
-   **Reports**: Save the markdown output report in the `reports/` folder.
-   **Result Figures**: Save any generated visualizations or figures in the
    `results_fig/` folder. **You must generate and save a graph of the Pareto
    curve showing all the models you tried (e.g. Size vs the primary stopping
    metric).** Additionally, you must run inference on a sample input and save
    a modality-appropriate qualitative artifact demonstrating fidelity of the
    final recommended models:
    -   Segmentation / detection: baseline vs. quantized output masks and
        difference heatmaps (see `image_segmentation_eval.md`).
    -   Generative LLMs: side-by-side sample generations and token agreement
        statistics (see `llm_eval.md`).
    -   Classification: Top-1/Top-5 agreement tables (see
        `classification_eval.md`).
    -   Other modalities: at least one qualitative sample appropriate to the
        task (e.g. spectrogram comparison, rendered depth map).

    Provide the python snippet or artifact directly in the report.

## Output Overview Example

The report must include a technical summary of the bounds achieved and a strict
Markdown table comparing the final configurations:

```text
**Optimization Target Result:** Successfully managed to fit the model inside the user's required size envelope without surpassing the acceptable stopping-metric threshold. (All names and numbers in this example are illustrative placeholders — derive yours from your own measurements.)

### Step-by-step Decisions on Recipe Update
* **Iteration 1**: Explored `dynamic_wi8_afp32()` as the baseline. The resulting model was 42.1 MB but suffered severe degradation in the primary metric (e.g., SNR dropped to 12.4 dB) in specific final layers. I inspected the worst-offending tensors and noticed `.*target_layer_regex.*` was particularly severely degraded, likely because it is a highly sensitive output projection.
* **Iteration 2**: I applied an explicit `no_quantize` override for the `.*target_layer_regex.*` scope. The model size increased to 45.4 MB, but the metric for those layers recovered significantly, hitting our target bounds.
* **Iteration 3 (Mixed-Precision)**: On layers that weren't skipped, I iteratively applied mixed quantization. I tested lowering robust layers (highest relative stability) to INT4 by targeting `.*robust_regex.*` with `op_config={'weight_tensor_config': {'num_bits': 4}}`, discovering optimal points between the target and size models.

### Pareto Options Comparison

You must list **EVERY SINGLE recipe** you evaluated during your exploration in this table so the user can see the full Pareto distribution. Be sure to explicitly recommend and highlight the **Top 3** optimal points (Quality, Balanced, and Compact profiles) clearly among the other intermediate runs.

| Config | Quality (Primary Metric) | Quality (Secondary Metric) | Size (MB) | Compression Ratio | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 (Baseline)** | Inf | 0.0 | 167.3 | 1.0x | Uncompressed payload |
| **dynamic_int8_baseline** | 22.1 | 0.15 | 42.1 | 4.0x | Base: `dynamic_wi8_afp32()`. Blind pure dynamic int8 quantization. |
| **dynamic_int8_selective_ver1 (Quality)** | 29.5 | 0.0009 | 45.4 | 3.7x | Base: `dynamic_wi8_afp32()`. Override: `regex='.*target_layer_regex.*'` -> `algorithm_key='no_quantize'`. Skipped final output to recover critical bounds |
| **dynamic_mixed_4_8_selective_ver2 (Balanced)** | 27.2 | 0.005 | 34.0 | 4.9x | Base: `dynamic_wi8_afp32()`. Override 1: Skip target layers. Override 2: pushed robust layers (highest stability) to `num_bits: 4` while retaining `num_bits: 8` for the rest. |
| **dynamic_int4_selective_ver3 (Compact)** | 18.2 | 0.03 | 26.2 | 6.4x | Base: `dynamic_wi8_afp32()`. Override 1: Skip target layers. Override 2: `regex='.*'` -> `num_bits: 4`. Aggressively pushed all remaining non-skipped layers to 4-bit |

### Strategy Insights & Recipe Details
For each configuration explored above, you **must detail the exact recipe updates applied**:
- **dynamic_int8_selective_iter1 (Quality)**: We started with `dynamic_wi8_afp32()`. The final projection block (`.*troubled_block_regex.*`) suffered massive structural noise due to wide feature channels mapping to a strict 8-bit dynamic runtime. We applied `algorithm_key='no_quantize'` via regex to this layer specifically, recovering significant signal precision globally.
- **dynamic_mixed_4_8_selective_iter2 (Balanced)**: After rescuing the skipped layers, we evaluated the remaining int8 layers. We identified that intermediate linear layers (`.*robust_block_regex.*`) maintained excellent relative stability in the upper percentiles during the baseline sweep. We aggressively lowered these specific layers to 4-bit via `op_config`, saving 11 MB without degrading the target layers, cleanly establishing a Pareto optimal mid-point.
- **dynamic_int4_selective_iter3 (Compact)**: To aggressively compress, we used `dynamic_wi8_afp32()` as the base, retained the bypass on `.*troubled_block_regex.*`, and applied an `op_config` override pushing `num_bits: 4` to all other layers.

```

When detailing the "Strategy Insights", you must clearly specify:

1.  **The Base Recipe**: Which default recipe was used as the foundation (e.g.
    `dynamic_wi8_afp32`, `weight_only_wi8_afp32`, `static_wi8_ai16`).
2.  **Skipped Layers**: Exactly which layers were selectively skipped (using
    `algorithm_key='no_quantize'`).
3.  **Quantization Algorithm Overrides**: Any specific advanced mathematical
    algorithms you injected via `algorithm_key` (e.g., explicitly shifting a
    layer to `hadamard` or `gptq`).
4.  **Specific Config Overrides**: Any specific tensor configurations you
    overrode via `op_config` parameter dictionaries (e.g., passing
    `op_config={'weight_tensor_config': {'num_bits': 16}}` to explicitly change
    the bit depth, or changing granularity/symmetric toggles).
5.  **The Rationale**: Exactly **why** you made these decisions, using the error
    metrics pulled directly from your Python evaluation script.

