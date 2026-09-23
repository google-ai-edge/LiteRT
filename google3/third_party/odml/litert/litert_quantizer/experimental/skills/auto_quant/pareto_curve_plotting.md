# Pareto Curve Visualization Best Practices

When rendering the final Pareto frontier graph in an AEQ Exploration report,
agents must consistently format their `matplotlib` code to produce readable,
analytically useful charts. Given the massive scale differences between baseline
models and failed empirical combinations, naive plotting quickly results in
illegible graphs.

Please adhere to the following best practices when generating the
`pareto_curve.png` artifact:

## 1. Metric Selection & Axes
The user prioritizes the functional End-to-End quality over internal tensor
fidelity — but during the search you only possess the cheap **stopping
metric** for every explored point; the end-to-end task metric (IoU, token
agreement, Top-1) is computed only for the final recommended models.

* **Primary Graph**: The X-axis must always be **Model Size (MB)**. The
Y-axis must be the **stopping metric** (e.g. output MSE / KL), which exists
for ALL explored points. Every point must come from the SAME sample count —
do not mix quick-step and final re-validation measurements of the same model
as separate points; plot one point per exported model and note re-validation
values in the report instead.
* **Optional Task-Metric Panel**: If end-to-end task metrics were computed
for the finalists, add a SECOND panel (or figure) plotting size vs the task
metric for just those models. Do not pretend the task metric exists for
points where it was never measured.
* **Dual Metrics**: If you choose to plot an internal
tensor metric (like SNR) alongside the functional metric, you MUST use dual
Y-axes (`ax.twinx()`). You must explicitly differentiate them using distinct
markers and colors and provide a clear legend. Be warned: dual axes graphs can
easily become visually cluttered.

## 2. Managing Scale and Outliers
During your empirical search loop, you will inevitably generate "broken"
configurations (e.g., a blind INT8 config producing massive noise like 10,000+
MSE). Plotting these linearly will compress the legitimate, cluster-optimized
models on the frontier into a single overlapping dot at the bottom of the graph.

* **Logarithmic Scaling**: You MUST use a logarithmic scale for your Y-axis if
the validation error spans multiple orders of magnitude. Because the FP32
baseline has exactly `0.0` error, use `symlog` (Symmetrical Log) and plot the
FP32 point at TRUE `0.0` — never fabricate a small epsilon for it, which
misrepresents the baseline and distorts the frontier.
* **Choosing `linthresh`**: the linear region of `symlog` visually compresses
everything inside it. Set `linthresh` BELOW your smallest nonzero error (e.g.
half of it), so the differences between your best models stay log-resolved.
A hardcoded `linthresh=0.01` will flatten a frontier whose best points sit at
`0.003` vs `0.0008` into indistinguishable dots.
    ```python
    nonzero = [e for _, e, _ in data if e > 0]
    ax.set_yscale('symlog', linthresh=min(nonzero) / 2)
    ```

## 3. Drawing the Actual Frontier Line
A collection of scatter points is not technically a Pareto curve until the
non-dominated frontier line is explicitly drawn!
You must mathematically identify the non-dominated points (configs where no
other config exists that is both smaller *and* has higher accuracy), sort them
by size, and draw a distinct line connecting them.

* **Exclude rejected configs from the frontier line.** The smallest model is
always technically non-dominated, so a broken extreme (e.g. a blind
INT4-everything config with collapsed quality) will drag the frontier line
through garbage. Exclude points that violate the user's quality/size bound
from the LINE, but still scatter them with a visually distinct marker (e.g.
`marker='x'`) labeled as rejected.
* **Draw the user's constraint.** Render the size bound as a vertical line
(`ax.axvline`) or the error bound as a horizontal line (`ax.axhline`), so the
reader immediately sees which points are admissible.

```python
# Identifying the Pareto Frontier (Assuming lower error is better)
# `data` is a list of tuples: (size_mb, error, name)
admissible = [d for d in data if passes_user_bound(d)]
admissible.sort(key=lambda x: x[0])  # Sort by Size ascending
pareto_points = []
best_error = float('inf')

for size, error, name in admissible:
    if error < best_error:
        pareto_points.append((size, error))
        best_error = error

# Draw the step line connecting the optimal threshold
pareto_sizes, pareto_errors = zip(*pareto_points)
ax.plot(pareto_sizes, pareto_errors, color='red', linestyle='--', label='Pareto Frontier')
ax.axvline(size_bound_mb, color='gray', linestyle=':', label='Size bound')
```

## 4. Preventing Annotation Clutter
If you generate 10+ intermediate loop versions (e.g., `ver1` through `ver5`),
they will likely cluster at nearly identical coordinates, causing `plt.annotate`
text labels to instantly over-write each other into a black, illegible blob.

* **Annotation Selection**: **DO NOT** annotate every single dot. Only annotate
the Baseline, the extreme limits, and your explicitly recommended Top 3 profiles
(e.g., "Quality", "Balanced", "Compact").
* **Offsetting**: Visually offset the text slightly using `xytext=(5, 5)`
combined with `textcoords='offset points'`.

## 5. Optional Third Axis: Latency

File size is only a proxy for what users usually care about (on-device speed
and memory). If latency measurements were collected per
[`latency_benchmarking.md`](latency_benchmarking.md), render a SECOND figure
(`pareto_latency.png`) plotting **Latency (ms) on the X-axis** vs the primary
task metric on the Y-axis, using the same annotation rules. Do not cram
latency into the size figure as a third encoded dimension (e.g. marker size) —
it becomes unreadable.

