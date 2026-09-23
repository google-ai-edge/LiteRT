*LiteRT Quantizer*

LiteRT Quantizer is a **post training quantization (PTQ) tool** for
[LiteRT](https://github.com/google-ai-edge/LiteRT) (formerly known as TFLite)
and [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), with the following
key features:

* Selective Quantization:
    * Enables users to quantize specific operations at layer level (e.g., only
      quantizing FullyConnected Ops in FeedForward layers and leaving all other
      Ops as float).
* Mixed Precision Quantization:
    * Allows users to specify different precision levels (activation and weight)
      for operators at layer level (e.g., INT4 weights FullyConnected in
      FeedForward layers but INT8 weights in Attention layers).
* Advanced Quantization Functions:
    * Provides functionalities like block-based/sub-channel quantization, along
      with weight-only and dynamic range quantization, ensuring compatibility
      with the DarwiNN toolchain.
* Full Integer Quantization:
    * Offers full integer quantization, including INT16/INT8 activations with
      INT8/INT4 weights, a requirement for many Android OEM toolchains and
      hardware like Qualcomm and Samsung NPUs.

The LiteRT Quantizer is one of many different tools for quantization across
various teams and use-cases. Other quantization tooling with a path to
on-device deployment include [TensorFlow Quantizer](http://go/tf-quantizer) and
[TfLite quantizer][TfLite]. Feature parity of LiteRT Quantizer to TFLite
Quantizer is WIP.

See http://go/litert-quantizer for more details.

## Repository Structure & Architecture

To assist developers and AI Agents in navigating the codebase, the repository is
structured into distinct modular components:

* `quantizer.py`: Top-level user and agent orchestration API (`Quantizer`,
  `QuantizationResult`).
* `recipe.py` & `recipe_manager.py`: Pre-defined quantization recipes (e.g.,
  `dynamic_wi8_afp32`, `weight_only_wi8_afp32`) and recipe construction/matching
  logic.
* `qtyping.py`: Core type definitions, Enum configurations
  (`TFLOperationName`, `QuantGranularity`), and flatbuffer schema wrappers.
* `calibrator.py`: Provides the TFLite interpreter wrappers
  (`CalibrationInterpreter`) and the core calibration engine (`Calibrator`)
  responsible for invoking the model on sample data and collecting
  Quantization Statistics Values (QSVs) across tensors.
* `algorithms/`: Underlying low-level quantization implementations (GPTQ,
  Hadamard Rotation, MinMax).

## The Quantization Lifecycle / Pipeline

The typical end-to-end execution pipeline for quantizing a model follows this
chronological sequence:
```
Load Model -> Load/Build Recipe -> Calibrate (if needed) ->
Quantize (Materialize) -> Validate (if needed) -> Export
```

## Quick Start & MNIST Benchmarking Suite

To help developers explore quantization trade-offs firsthand, the repository
includes an extensive runnable benchmarking suite and tutorials under
[`examples/mnist/`](examples/mnist/):

* **[`quantize_toy_model.py`](examples/mnist/quantize_toy_model.py)**: A
  practical end-to-end tutorial demonstrating how to load a float model,
  calibrate with sample data, quantize, save artifacts to disk, and run
  inference on real images.

[TfLite]: https://www.tensorflow.org/lite/performance/post_training_quantization
