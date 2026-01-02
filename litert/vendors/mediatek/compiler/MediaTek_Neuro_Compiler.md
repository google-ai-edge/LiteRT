# LiteRT to MediaTek Neuron Compiler Support

This document outlines the supported data types, quantization schemes, and
operations for the LiteRT to MediaTek Neuron compiler, based on an analysis of
the compiler plugin source code.

## Supported Data Types

The compiler maps `litert::ElementType` to a corresponding `NeuronTensorType`.
The final Neuron type depends on the element type and its quantization scheme.

| LiteRT Element Type | Mapped Neuron Tensor Type | Note |
| :--- | :--- | :--- |
| `Float32` | `NEURON_TENSOR_FLOAT32` | |
| `Float16` | `NEURON_TENSOR_FLOAT16` | |
| `Int32` | `NEURON_TENSOR_INT32` | |
| `Int16` | `NEURON_TENSOR_QUANT16_SYMM` | Only per-tensor quantization is supported. |
| `UInt8` | `NEURON_TENSOR_QUANT8_ASYMM` | |
| `Int8` | `NEURON_TENSOR_QUANT8_SYMM` or `NEURON_TENSOR_QUANT8_SYMM_PER_CHANNEL` | The final type depends on the quantization scheme. |
| `Int4` | `NEURON_EXT_TENSOR_QUANT4_SYMM` or `NEURON_EXT_TENSOR_QUANT4_SYMM_PER_CHANNEL` | Constant weights are unpacked into `Int8` tensors during compilation. |
| `Bool` | `NEURON_TENSOR_BOOL8` | |
| `Int64` | `NEURON_TENSOR_INT32` | Only supported for constant tensors (weights). Values are cast from `Int64` to `Int32`. |

## Supported Quantization Types

| LiteRT Quantization Type | Supported | Note |
| :--- | :--- | :--- |
| `kLiteRtQuantizationPerTensor` | **Yes** | |
| `kLiteRtQuantizationPerChannel` | **Yes** | |
| `kLiteRtQuantizationBlockWise` | **No** | Explicitly marked as unsupported. |
| `kLiteRtQuantizationNone` | **Yes** | |

<!-- LINT.IfChange(supported_ops) -->
## Supported Operations

The following `LiteRtOpCode` operations are supported. For an operation to be
delegated, it must also pass a set of constraints checked by
the `VerifyCommonOp` function.

| LiteRT Op Code | Note |
| :--- | :--- |
| `kLiteRtOpCodeTflAbs` | Legalized to `NEURON_ABS`. |
| `kLiteRtOpCodeTflAdd` | Legalized to `NEURON_ADD`. Supports fused activation. |
| `kLiteRtOpCodeTflAveragePool2d` | Legalized to `NEURON_AVERAGE_POOL_2D`. Supports padding, stride, filter size, and fused activation. |
| `kLiteRtOpCodeTflBatchMatmul` | Legalized to `NEURON_BATCH_MATMUL`. Supports `adj_x` and `adj_y` attributes. |
| `kLiteRtOpCodeTflCast` | Legalized to `NEURON_CAST`. |
| `kLiteRtOpCodeTflConcatenation` | Legalized to `NEURON_CONCATENATION`. Supports `axis` attribute. |
| `kLiteRtOpCodeTflConv2d` | Legalized to `NEURON_CONV_2D`. Supports padding, stride, fused activation, data format, and dilation. |
| `kLiteRtOpCodeTflDequantize` | Legalized to `NEURON_DEQUANTIZE`. |
| `kLiteRtOpCodeTflDepthwiseConv2d`| Legalized to `NEURON_DEPTHWISE_CONV_2D`. Supports padding, stride, depth multiplier, fused activation, data format, and dilation. |
| `kLiteRtOpCodeTflDiv` | Legalized to `NEURON_DIV`. Supports fused activation. |
| `kLiteRtOpCodeTflFullyConnected` | Legalized to `NEURON_FULLY_CONNECTED`. Supports fused activation. Adds a zero bias if one is not provided. |
| `kLiteRtOpCodeTflGelu` | Legalized to `NEURON_GELU_V2` with `approximate` set to true. |
| `kLiteRtOpCodeTflGreater` | Legalized to `NEURON_GREATER`. |
| `kLiteRtOpCodeTflHardSwish` | Legalized to `NEURON_HARD_SWISH`. |
| `kLiteRtOpCodeTflLogistic` | Legalized to `NEURON_LOGISTIC`. |
| `kLiteRtOpCodeTflMaximum` | Legalized to `NEURON_MAXIMUM`. |
| `kLiteRtOpCodeTflMaxPool2d` | Legalized to `NEURON_MAX_POOL_2D`. Supports padding, stride, filter size, and fused activation. |
| `kLiteRtOpCodeTflMean` | Legalized to `NEURON_MEAN`. Supports `keep_dims` attribute. |
| `kLiteRtOpCodeTflMinimum` | Legalized to `NEURON_MINIMUM`. |
| `kLiteRtOpCodeTflMul` | Legalized to `NEURON_MUL`. Supports fused activation. |
| `kLiteRtOpCodeTflPad` | Legalized to `NEURON_PAD`. |
| `kLiteRtOpCodeTflPadv2` | Legalized to `NEURON_PAD_V2`. |
| `kLiteRtOpCodeTflPrelu` | Legalized to `NEURON_PRELU`. |
| `kLiteRtOpCodeTflQuantize` | Legalized to `NEURON_QUANTIZE`. |
| `kLiteRtOpCodeTflReduceMax` | Legalized to `NEURON_REDUCE_MAX`. Supports `keep_dims` attribute. |
| `kLiteRtOpCodeTflRelu` | Legalized to `NEURON_RELU`. |
| `kLiteRtOpCodeTflReshape` | Legalized to `NEURON_RESHAPE`. |
| `kLiteRtOpCodeTflResizeBilinear` | Legalized to `NEURON_RESIZE_BILINEAR`. Supports `align_corners` and `half_pixel_centers`. |
| `kLiteRtOpCodeTflResizeNearestNeighbor`| Legalized to `NEURON_RESIZE_NEAREST_NEIGHBOR`. Supports `align_corners` and `half_pixel_centers`. |
| `kLiteRtOpCodeTflRsqrt` | Legalized to `NEURON_RSQRT`. |
| `kLiteRtOpCodeTflSlice` | Legalized to `NEURON_SLICE`. |
| `kLiteRtOpCodeTflSoftmax` | Legalized to `NEURON_SOFTMAX`. Supports `beta` attribute. |
| `kLiteRtOpCodeTflSplit` | Legalized to `NEURON_SPLIT`. Supports `num_splits` attribute. |
| `kLiteRtOpCodeTflSqrt` | Legalized to `NEURON_SQRT`. |
| `kLiteRtOpCodeTflSquaredDifference`| Legalized as an OEM extension: `MTKEXT_SQUARED_DIFFERENCE`. |
| `kLiteRtOpCodeTflStridedSlice` | Legalized to `NEURON_STRIDED_SLICE`. Supports `begin_mask`, `end_mask`, and `shrink_axis_mask`. |
| `kLiteRtOpCodeTflSub` | Legalized to `NEURON_SUB`. Supports fused activation. |
| `kLiteRtOpCodeTflSum` | Legalized to `NEURON_REDUCE_SUM`. Supports `keep_dims` attribute. |
| `kLiteRtOpCodeTflTanh` | Legalized to `NEURON_TANH`. |
| `kLiteRtOpCodeTflTranspose` | Legalized to `NEURON_TRANSPOSE`. |
| `kLiteRtOpCodeTflTransposeConv` | Legalized to `NEURON_TRANSPOSE_CONV`. Supports padding and stride. Adds a zero bias if one is not provided. |
| `kLiteRtOpCodeShloComposite` | Supports `odml.rms_norm` (as OEM extension `MTKEXT_RMS_NORMALIZATION`) and `odml.l2_norm` (as `NEURON_L2_NORMALIZATION`). |
<!-- LINT.ThenChange(./compiler_plugin.cc:supported_ops) -->
