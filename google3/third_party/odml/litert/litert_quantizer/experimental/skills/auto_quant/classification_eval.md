# Classification Evaluation

> [!NOTE] This is one member of the **pluggable task evaluation family**
> (`image_segmentation_eval.md`, `llm_eval.md`, `classification_eval.md`).
> Only read and apply this file if the target model performs classification
> (image, text, or audio).

When compressing classification models, the operationally meaningful question
is not "how much did the logits move" but "did the predicted class change".
Evaluate the final quantized models against the float reference using
prediction-agreement metrics.

## 1. Search Metrics vs Report Metrics

*   **Search loop (every iteration)**: `KL_DIVERGENCE` on the output logits as
    the stopping metric (see `error_metric_selection.md`). It is cheap and
    strictly more sensitive than argmax agreement.
*   **Final report (Top 3 profiles only)**: the agreement metrics below.

## 2. Standard Report Metrics

*   **Top-1 Agreement**: Fraction of inputs where quantized argmax equals
    float argmax. This measures *consistency with the float model*, which
    requires no ground-truth labels and is the primary deliverable metric.
*   **Top-5 Agreement**: Fraction of inputs where the float top-1 class
    appears in the quantized top-5. Softens penalty for near-tie flips.
*   **Accuracy Delta (optional)**: If a labeled evaluation set is available,
    report `accuracy(quantized) - accuracy(float)` directly. Prefer this over
    agreement when labels exist, but never require the user to supply labels.
*   **Confidence Drift**: Mean absolute change in the top-1 softmax
    probability. High drift with high agreement warns of calibration loss
    (relevant if downstream code thresholds on confidence).

## 3. Boilerplate: Agreement Metrics

```python
import numpy as np

def classification_agreement(float_logits, quant_logits):
    """Agreement metrics between float and quantized classifiers.

    Args:
      float_logits: [num_samples, num_classes] float reference logits.
      quant_logits: [num_samples, num_classes] quantized model logits.
    """
    f_top1 = np.argmax(float_logits, axis=-1)
    q_top1 = np.argmax(quant_logits, axis=-1)
    top1_agree = np.mean(f_top1 == q_top1)

    q_top5 = np.argsort(quant_logits, axis=-1)[:, -5:]
    top5_agree = np.mean([f in row for f, row in zip(f_top1, q_top5)])

    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    f_conf = softmax(float_logits)[np.arange(len(f_top1)), f_top1]
    q_conf = softmax(quant_logits)[np.arange(len(f_top1)), f_top1]
    conf_drift = np.mean(np.abs(f_conf - q_conf))

    return {
        "top1_agreement": float(top1_agree),
        "top5_agreement": float(top5_agree),
        "confidence_drift": float(conf_drift),
    }
```

Save the resulting statistics to
`results_fig/{base_name}_topk_agreement.json` per `file_naming.md`.

## 4. Sample Size Guidance

Agreement estimates stabilize quickly: 100–500 evaluation samples are
sufficient for the final report. Use a fixed random seed and the identical
sample set across all configurations so numbers are directly comparable. Do
NOT reuse the calibration dataset (see `calibration_data.md`) for final
evaluation — that constitutes leakage.
