<!-- disableFinding(LINE_OVER_80) -->

# Guide to Hadamard & Kronecker Rotation Schemes and Recipe Parameters

This document explains how the orthogonal rotation schemes in
`hadamard_rotation.py` work, how to configure them in quantization recipes, and
how to compare rotation configurations quickly using offline weight quantization
error.

--------------------------------------------------------------------------------

## 1. Mental Model: How Orthogonal Rotation Works

Low-bit symmetric uniform quantization (`int4` / `int2`) suffers when a weight
row or activation channel has large outlier spikes: a single outlier forces a
wide per-channel quantization step size (`scale`), compressing all normal values
into only a few quantization bins.

Multiplying by an orthogonal rotation matrix $$R\in\mathbb{R}^{D\times{D}}$$
($$RR^T=I_D$$) mixes features across the reduction dimension $$D$$, turning
sharp channel spikes into a near-Gaussian distribution with a much smaller
peak-to-RMS ratio:

-   **Linear layers (`FULLY_CONNECTED`, $$Y=XW^T$$)**:
    -   **Offline (during quantization)**: Weight rows are rotated
        $$W_{\text{rot}}=WR$$ and symmetrically quantized (`int4`, `zero_point =
        0`).
    -   **Online (at inference time)**: Input activations are rotated
        $$X_{\text{rot}}=XR$$ before the matrix multiply, so
        $$(XR)(WR)^T=XRR^TW^T=XW^T$$.
-   **Token embeddings (`EMBEDDING_LOOKUP`, $$Y=E[\text{ids}]$$)**:
    -   **Offline (during quantization)**: Embedding rows are rotated
        $$E_{\text{rot}}=ER$$ and symmetrically quantized (`int4` or `int2`,
        `zero_point = 0`).
    -   **Online (at inference time)**: Looked-up embedding rows are un-rotated
        $$Y=E_{\text{rot}}[\text{ids}]R^T$$ immediately after the lookup.

### 1-Stage Block-Hadamard vs. 2-Stage Kronecker Rotation

Sylvester Hadamard matrices $$H_d$$ only exist when $$d$$ is a power of 2
($$d=2^k$$). When a layer dimension $$D$$ is not a power of 2—such as
$$D=2560=10\times{256}$$ or $$D=10240=40\times{256}$$—we factor $$D=m\cdot{d}$$
(where $$d$$ is a power of 2 that evenly divides $$D$$, and $$m=D/d$$) and
choose between two structures:

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Structure                      | Mathematical Form        | Runtime Ops Inserted per Layer               | Outlier Mixing Scope                                                                   |
| :----------------------------- | :----------------------- | :------------------------------------------- | :------------------------------------------------------------------------------------- |
| **1-Stage Block-Hadamard**     | $$R = I_m \otimes H_d$$  | **1** `FULLY_CONNECTED` (`[d, d]`)           | Mixes only **within** each $$d$$-element block ($$m$$ independent blocks).             |
| **2-Stage Kronecker Rotation** | $$R = Q_m \otimes H_d$$  | **2** `FULLY_CONNECTED` (`[d, d]`, `[m, m]`) | Mixes **within** $$d$$-blocks via $$H_d$$ AND **across** all $$m$$ blocks via $$Q_m$$. |
<!-- mdformat on -->

--------------------------------------------------------------------------------

## 2. Quantization Recipe Parameters (`algorithm_key` & `algorithm_params`)

### 2.1 Top-Level `algorithm_key` Options

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| `algorithm_key`                  | What It Does                                                                                               | Hardware / Delegate Support                                      |
| :------------------------------- | :--------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------- |
| `"min_max_uniform_quantize"`     | Naive per-channel symmetric min/max quantization (`scale = max(\|w\|) / q_max`). No clipping, no rotation. | Standard TFLite (`CPU`, `XNNPACK`, `GPU`).                       |
| `"OCTAV"`                        | Iterative L2-optimal clipping threshold (10 Newton-like steps) without rotation.                           | Standard TFLite (`CPU`, `XNNPACK`, `GPU`).                       |
| `"HADAMARD_ROTATION"`            | Rotates weights offline and inserts custom op `aeq.hadamard_rotation` online.                              | **CPU custom kernel only** (not supported on `XNNPACK` / `GPU`). |
| `"DECOMPOSED_HADAMARD_ROTATION"` | Rotates weights offline and inserts standard `RESHAPE -> FULLY_CONNECTED -> RESHAPE` ops online.           | Standard TFLite (`CPU`, `XNNPACK`, `GPU`).                       |
<!-- mdformat on -->

### 2.2 `algorithm_params` Options (for `"DECOMPOSED_HADAMARD_ROTATION"`)

```json
"algorithm_params": {
  "q_init_mode": "none",          // "none" | "auto"
  "sizing_mode": "min_compute",   // "min_compute" | "max_power_of_two"
  "max_hadamard_size": 256,       // power-of-2 cap on d, or null
  "scale_refinement": "mse_grid", // "none" | "mse_grid"
  "randomize": false,             // true | false
  "q_seed": 0
}
```

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| Parameter           | Type / Default           | Allowed Values                                 | What It Controls                                                                                                                               |
| :------------------ | :----------------------- | :--------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------- |
| `q_init_mode`       | `str` (`"none"`)         | `"none"`, `"auto"`                             | `"none"` uses 1-stage Block-Hadamard ($$I_m \otimes H_d$$). `"auto"` enables 2-stage Kronecker rotation ($$Q_m \otimes H_d$$).                 |
| `sizing_mode`       | `str` (`"min_compute"`)  | `"min_compute"`, `"max_power_of_two"`          | How $$D = m \cdot d$$ is factored into block count $$m$$ and Hadamard block size $$d$$ when `q_init_mode="auto"`.                              |
| `max_hadamard_size` | `int` or `None` (`None`) | Power of 2 (`64`, `128`, `256`, ...) or `None` | Upper bound $$d_{\max}$$ on the Hadamard block size $$d$$.                                                                                     |
| `scale_refinement`  | `str` (`"none"`)         | `"none"`, `"mse_grid"`                         | Post-OCTAV per-channel scale refinement to minimize discrete rounding MSE $$\lVert w - s \cdot \text{round}(w/s) \rVert_2^2$$.                 |
| `randomize`         | `bool` (`False`)         | `true`, `false`                                | Multiplies $$H_d$$ by a deterministic random $$\pm 1$$ sign diagonal ($$\text{diag}(s) H_d$$) to break coherent alignment with weights.        |
| `q_seed`            | `int` (`0`)              | Integer seed                                   | Deterministic RNG seed used when `randomize=true` or `q_init_mode="auto"`.                                                                     |
<!-- mdformat on -->

#### How `q_init_mode` Constructs the Rotation

-   **`"none"` (1-Stage Block-Hadamard, $$R=I_m\otimes{H_d}$$)**:
    -   Inserts a single `[d, d]` `FULLY_CONNECTED` op per rotated tensor, where
        $$d$$ is the largest power-of-2 divisor of $$D$$ (up to
        `max_hadamard_size`).
-   **`"auto"` (2-Stage Kronecker, $$R=Q_m\otimes{H_d}$$)**:
    -   Inserts two `FULLY_CONNECTED` ops (`[d, d]` for $$H_d$$ and `[m, m]` for
        $$Q_m$$), selecting the cross-block orthogonal matrix $$Q_m\in{O(m)}$$
        automatically based on $$m=D/d$$:
        1.  **$$m$$ is a power of 2**: Uses an exact Sylvester Hadamard matrix
            $$H_m$$.
        2.  **$$m\le{48}$$ (not a power of 2)**: Uses the Cayley transform
            $$Q_m=(I-A)(I+A)^{-1}$$ of a skew-symmetric Rademacher matrix
            $$A_{ij}=\pm\frac{1}{\sqrt{m}+1}$$, which produces near-uniform
            off-diagonal entries without small-dimension SVD tail spikes.
        3.  **$$m>{48}$$ (not a power of 2)**: Uses the orthogonal polar factor
            $$Q_m=UV^T$$ from the SVD of a random Rademacher matrix
            $$S\in\{\pm{1}\}^{m\times{m}}$$, which has zero diagonal bias and
            $$O(\sqrt{\ln(m)/m})$$ peak magnitude as $$m$$ grows.

#### How `sizing_mode` Selects $$(m,d)$$ When `q_init_mode="auto"`

Let $$d_{\max}$$ be the largest power of 2 that evenly divides $$D$$, capped at
`max_hadamard_size` (if set).

-   When `q_init_mode="none"`, `sizing_mode` is not used: $$d$$ is always
    $$d_{\max}$$.
-   When `q_init_mode="auto"`, `sizing_mode` chooses $$d$$ from the candidate
    powers of 2 $$\{2,4,8,\dots,d_{\max}\}$$.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| `sizing_mode`        | Which Power-of-2 $$d \in \{2, 4, \dots, d_{\max}\}$$ Is Chosen                                                                                                                               | Example on $$D = 2560$$ (`max_hadamard_size=256`) | Runtime Compute ($$m + d$$ MACs/elem) |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------ | :------------------------------------ |
| `"min_compute"`      | Searches all powers of 2 $$d \in \{2, 4, \dots, d_{\max}\}$$ and picks the $$d$$ that minimizes runtime MACs $$(D/d) + d$$ (so $$d \approx m \approx \sqrt{D}$$; ties broken toward $$d$$).  | $$d = 64, m = 40$$                                | $$40 + 64 = 104$$ MACs/elem           |
| `"max_power_of_two"` | Always picks $$d = d_{\max}$$ (the largest power-of-2 divisor of $$D$$ up to `max_hadamard_size`), minimizing $$m = D / d_{\max}$$.                                                          | $$d = 256, m = 10$$                               | $$10 + 256 = 266$$ MACs/elem          |
<!-- mdformat on -->

#### How `scale_refinement` Works

OCTAV estimates per-channel clipping scales under a continuous uniform-noise
approximation, which is accurate at `int8` but suboptimal at `int4` (15 bins)
and `int2` (3 bins) where discrete rounding boundaries matter.

<!-- mdformat off(reason: standard GFM table rendering in Critique and Code Search) -->
| `scale_refinement` | What It Does                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| :----------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"none"`           | Uses the raw OCTAV scale directly.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `"mse_grid"`       | 1. **Grid search**: Tests 25 candidate multipliers $$\alpha \in [0.80, 1.20]$$ around each channel's OCTAV scale ($$s = \alpha \cdot s_{\text{octav}}$$) and keeps the $$\alpha$$ with lowest rounding MSE $$\lVert w - s \cdot \text{round}(w/s) \rVert_2^2$$.<br>2. **Alternating least-squares refinement (2 steps)**: Holding the rounded integers $$q = \text{clip}(\text{round}(w/s), q_{\min}, q_{\max})$$ fixed, updates the scale to $$s^* = \frac{w \cdot q}{q \cdot q}$$ and re-rounds. |
<!-- mdformat on -->

### 2.3 Two-Pass Quantization Pattern (Model Weights vs. Inserted Rotation Matrices)

When `"DECOMPOSED_HADAMARD_ROTATION"` is applied to a `FULLY_CONNECTED` or
`EMBEDDING_LOOKUP` op, `litert_quantizer` rotates and quantizes that op's weight
table offline, and inserts 1 or 2 new `FULLY_CONNECTED` ops into the graph to
perform the online rotation ($$H_d$$ and $$Q_m$$). Those newly inserted rotation
ops are created with **`float32` constant weights**.

To quantize both the model weights and the newly inserted rotation matrices
cleanly in two passes:

1.  **Pass 1 Recipe (All Model Weights: Transformer Linear Layers +
    Embeddings)**:
    -   Applies `"DECOMPOSED_HADAMARD_ROTATION"` (or `"OCTAV"`) to all
        transformer `FULLY_CONNECTED` layers (e.g., `int4` weights, `int8`
        dynamic activations) **and** to `EMBEDDING_LOOKUP` ops (`int8`, `int4`,
        or `int2`).
    -   Output of Pass 1: All transformer and embedding weights are rotated and
        quantized; all newly inserted rotation matrices (`hadamard_rotation` and
        `kronecker_q_rotation`, on both linear layers and embeddings) have
        `float32` weights.
2.  **Pass 2 Recipe (Inserted Rotation Matrices Only)**:
    -   Targets only the inserted rotation ops via regex
        `.*(?:hadamard_rotation|kronecker_q_rotation).*` with
        `"min_max_uniform_quantize"` and dynamic `int8` activations, setting the
        **rotation matrix weight precision** to:
        -   **8-bit rotation weights** (`num_bits: 8` on rotation matrix
            weights, `int8` dynamic activations), or
        -   **4-bit rotation weights** (`num_bits: 4` on rotation matrix
            weights, `int8` dynamic activations).

--------------------------------------------------------------------------------

## 3. Exploring and Comparing Schemes Quickly via Offline Weight RMSE

Full end-to-end model quantization, validation, and downstream benchmarking
takes **~15–20 minutes per model variant**. Instead of running full model
quantization for every candidate configuration, rotation schemes can be explored
and ranked in **three progressively narrower stages**:

### Stage 1: Stratified Offline Weight Relative RMSE (~2 Minutes for 10+ Schemes)

1.  **Sample 1 representative weight tensor per shape class** directly from the
    float `.tflite` model (e.g., from Layer 0 and a middle transformer layer,
    plus the token embedding table):
    -   `q_proj`: `[4096, 2560]` ($$D=2560$$)
    -   `o_proj`: `[2560, 2048]` ($$D=2048$$, power of 2)
    -   `gate_proj`: `[10240, 2560]` ($$D=2560$$)
    -   `down_proj`: `[2560, 10240]` ($$D=10240$$)
    -   `embedder`: `[262144, 2560]` ($$D=2560$$)
2.  **Rotate, quantize, and dequantize each sample tensor in memory**:
    -   Rotate the weight matrix along its reduction axis:
        $$W_{\text{rot}}=WR$$.
    -   Compute per-channel symmetric quantization scales via OCTAV (+ optional
        `scale_refinement`) and quantize to `int4` or `int2` integers $$Q$$.
    -   Because `hadamard_rotation.py` enforces **symmetric quantization**
        (`symmetric=True`, `zero_point = 0` for all rotated weight and embedding
        tensors), dequantization is simply
        $$\widehat{W}_{\text{rot}}=\text{scale}\cdot{Q}$$.
    -   *(In code, `hadamard_rotation.get_tensor_quant_params()`—the standard
        `litert_quantizer` per-tensor entry point implemented by every
        quantization algorithm module—performs this exact rotation + OCTAV +
        refinement step and returns `scale` and `quantized_data`.)*
3.  **Compute Relative Frobenius RMSE**:

    -   **When rotation matrices are kept in `float32` ($$\widehat{R}=R$$)**:

        $$\text{RelRMSE}=\frac{\|WR-\widehat{W}_{\text{rot}}\|_F}{\|WR\|_F}$$

        *(Note: Because $$R$$ is orthogonal, $$\|WR\|_F=\|W\|_F$$ and
        $$\|WR-\widehat{W}_{\text{rot}}\|_F=\|W-\widehat{W}_{\text{rot}}R^T\|_F$$,
        so measuring error in the rotated domain or un-rotated domain gives the
        exact same value.)*
    -   **When rotation matrices are quantized to 8-bit or 4-bit weights
        ($$\widehat{R}=\text{dequant}(\text{quant}(R))$$)**:

        $$\text{RelRMSE}_{\text{effective}}=\frac{\|W-\widehat{W}_{\text{rot}}\widehat{R}^T\|_F}{\|W\|_F}$$

        This unified metric reduces to
        $$\frac{\|WR-\widehat{W}_{\text{rot}}\|_F}{\|WR\|_F}$$ when
        $$\widehat{R}=R$$, while also capturing any error introduced by
        quantizing $$H_d$$ and $$Q_m$$ to `int8` or `int4`.

#### Example: Stage 1 Offline Relative RMSE on Gemma 4 4.5B

-   **4-bit (`int4`) Transformer Linear Weights**:
    -   Unrotated `min_max`: **`0.1976`**
    -   Unrotated `OCTAV`: **`0.1334`** ($$-32.5\%$$ vs. `min_max`)
    -   1-Stage Block-Hadamard (`q_init_mode="none"`, $$d=256$$): **`0.1134`**
        ($$-42.6\%$$)
    -   2-Stage Kronecker (`q_init_mode="auto"`, `sizing_mode="min_compute"`,
        $$m=40,d=64$$ on $$D=2560$$): **`0.1110`** ($$-43.8\%$$)
    -   1-Stage Block-Hadamard + `scale_refinement="mse_grid"`: **`0.1109`**
        ($$-43.9\%$$)
-   **2-bit (`int2`) Token Embedding Table (`[262144, 2560]`)**:
    -   Unrotated `min_max`: **`0.9151`**
    -   Unrotated `OCTAV`: **`0.5143`** ($$-43.8\%$$ vs. `min_max`)
    -   1-Stage Block-Hadamard (`q_init_mode="none"`, $$d=256$$): **`0.5077`**
        ($$-44.5\%$$)
    -   2-Stage Kronecker (`q_init_mode="auto"`, `sizing_mode="min_compute"`,
        $$m=40,d=64$$ on $$D=2560$$): **`0.5071`** ($$-44.6\%$$)
    -   1-Stage Block-Hadamard + `scale_refinement="mse_grid"`: **`0.3915`**
        ($$-57.2\%$$ vs. `min_max`, $$-22.9\%$$ vs. `OCTAV`—showing that
        `"mse_grid"` is the single largest lever at 2 bits)

### Stage 2: Forward-Pass Subgraph & Logits Validation (~5 Minutes per Survivor)

Offline weight RMSE only sees static weights $$W$$; it does not see **input
activation outliers** in $$X$$ or how errors accumulate across transformer
layers. For the top 3–5 survivors from Stage 1:

-   **Fast Embedding Subgraph Check (~8 seconds)**: A `.litertlm` bundle
    packages the token embedding table as a separate TFLite flatbuffer subgraph
    (`tf_lite_embedder`, exported as `embedder.tflite`) alongside the main
    transformer flatbuffer (`model.tflite`). Because `embedder.tflite` contains
    only the `EMBEDDING_LOOKUP` op (plus its online un-rotation op after
    quantization), `qt.validate()` can run `embedder.tflite` by itself in ~8
    seconds to measure exact output embedding MSE and Cosine Similarity before
    touching the multi-GB transformer backbone.
-   **Single-Pass Prompt Logits Check (`qt.validate()`, ~5 minutes)**: Feeds a
    fixed set of calibration prompt token IDs through both the float model and
    the quantized model in a single forward pass (without autoregressive
    token-by-token generation) and compares their output logit vectors using
    **SNR**, **Cosine Similarity**, and **KL Divergence**.

### Stage 3: Autoregressive Generation Benchmark (Open ARTS)

In real usage, the model generates tokens autoregressively (feeding each
predicted token back in as the next input), which can amplify small biases over
hundreds of steps. Running a small sample (e.g., 20 examples) of `open_mmlu` and
`open_mgsm` on the final Pareto-optimal `.litertlm` bundles verifies end-to-end
task accuracy.
