# Image Segmentation Evaluation

> [!NOTE] This is one member of the **pluggable task evaluation family**
> (`image_segmentation_eval.md`, `llm_eval.md`, `classification_eval.md`).
> Only read and apply this file if the target model performs image
> segmentation or detection. For other modalities, use the matching sibling
> skill instead.

When utilizing the AI Edge Quantizer (AEQ) to compress models executing image
segmentation, agents must rigorously evaluate the final quantized model against
the float reference model using End-to-End semantic segmentation metrics.

## 1. Why Standard Metrics Fail for Segmentation

Unlike classification (which reduces features to categorical logits),
segmentation networks predict dense spatial layouts. Quantization corruption
does not merely drop confidence; it destroys the physical output, resulting in
structural "ghosting", shredded boundary edges, and foreground inversion. Thus,
simple scalar Mean Squared Error (MSE) on the output tensor is insufficient for
reporting final operational accuracy.

## 2. Standard Evaluation Metrics

When benchmarking and reporting the final quality of a quantized segmentation
model, implement the following standard evaluation metrics on the post-processed
spatial masks. Do not use these metrics as iterative logic for searching layers
(use SNR/MSE for that).

*   **Mean Intersection over Union (mIoU) / Jaccard Index**: The primary and
    most universally accepted metric for semantic segmentation frameworks
    (PASCAL VOC, Cityscapes). It measures the overlap area between the predicted
    mask and ground truth mask divided by the union area.
*   **Dice Similarity Coefficient (DSC) / F1 Score**: Often preferred in tasks
    exhibiting extreme class imbalance (e.g., medical imaging like MICCAI or
    tiny object isolation). It effectively computes twice the intersection
    divided by the sum of the predicted and ground truth areas.
*   **Pixel Accuracy**: The percentage of pixels correctly classified. While
    useful as a baseline diagnostic, it can be dangerously misleading in
    datasets with high class imbalance (e.g., large background regions
    dominating the image).

## 3. Implementing Task Metrics in AEQ

In AEQ, you can define custom task metrics that process the output masks.

### Example Boilerplate: Computing IoU and Dice

```python
import numpy as np

def calculate_mask_metrics(float_mask, quant_mask, threshold=0.5):
    """Calculates Intersection over Union and Dice Coefficient for a binary mask."""
    # Convert arbitrary output logits to strict binary thresholds
    # Note: Ensure you account for batch and channel dimensions in real implementation
    gt_bool = (float_mask > threshold).astype(bool)
    pred_bool = (quant_mask > threshold).astype(bool)

    intersection = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()

    # Calculate IoU
    iou = 1.0 if union == 0 and intersection == 0 else (0.0 if union == 0 else intersection / union)

    # Calculate Dice
    dice = 2.0 * intersection / (gt_bool.sum() + pred_bool.sum())

    return iou, dice
```

## 4. Visual Evaluation and Heatmaps

Agents must render visual differences into exported graphs rather than just
logging a decimal number. **Critical Setup**: You must use `plt.savefig()` and
avoid `plt.show()` when validating in headless mode.

### Example Plotting Pipeline (Difference Heatmap)

```python
import matplotlib.pyplot as plt
import numpy as np

def save_fidelity_heatmap(float_mask, quantized_mask, save_path):
    # Slice to a specific 2D channel
    diff_mask = np.abs(float_mask - quantized_mask)[0, :, :, 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Base Truth
    axes[0].imshow(float_mask[0, :, :, 0], cmap='gray')
    axes[0].set_title('FP32 Baseline Mask')
    axes[0].axis('off')

    # Quantized Mask
    axes[1].imshow(quantized_mask[0, :, :, 0], cmap='gray')
    axes[1].set_title('Quantized Output')
    axes[1].axis('off')

    # Difference Heatmap
    heatmap = axes[2].imshow(diff_mask, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)

    # Save visualization for Output Report
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
```

## 5. Reporting Validation Results

Because raw tensor MSE and SNR do not intuitively map to visual quality for end
users, agents must calculate these end-to-end task metrics (mIoU, Dice) for the
final optimized model and explicitly export them in the final markdown or CSV
report.

Always present a clear summary table contrasting the baseline Float32 Mask
metrics with the Quantized Mask metrics alongside the generated difference
heatmaps.
