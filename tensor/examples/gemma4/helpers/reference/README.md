# Gemma4 numerical reference fixtures

These NumPy modules compute small, deterministic mathematical examples for the
[C++ helper tests](../README.md) and [graph tests](../../gemma4_graph_test.cc).
Their imports use the `tensor.examples` Python package path.

| File | Responsibility |
| --- | --- |
| [rmsnorm.py](rmsnorm.py) | Last-axis RMS normalization with optional scale and epsilon. |
| [rope.py](rope.py) | Split-half rotation, `x * cos + [-x2, x1] * sin`. |
| [feed_forward_network.py](feed_forward_network.py) | Gated feed-forward block using the tanh approximation to GELU. |
| [attention.py](attention.py) | Q/K/V projection, normalization, RoPE, grouped-query attention, masks, and optional KV caches. |
| [transformer.py](transformer.py) | Attention and feed-forward residual blocks, optional per-layer inputs, and the layer scalar. |
| [gemma4_graph.py](gemma4_graph.py) | Synthetic model weights, stacked layers, cache sharing, final projection, and logit softcapping. |
| [utils.py](utils.py) | Deterministic example angles and formatting NumPy arrays as C++ float literals. |

The files compose in the same order as the C++ helpers: normalization, rotation,
and feed-forward primitives feed attention/transformer blocks, which feed the
model graph. The numerical modules' `main()` functions print fixture values;
the graph module also generates the four-token reference used to compare full
prefill with a shorter
prefill followed by cached decode. Reduced E4B cases explicitly preserve the
sharing flags used by their corresponding C++ tests.

When changing a fixture, match its weights, shapes, masks, positions, and sharing
configuration to the C++ case before updating expected numbers. These scripts
do not simulate XNNPACK's integer activation rounding or run pretrained Hugging
Face models. Static quantization has its own
[integer reference test](../mobile_fully_connected_test.cc).

NumPy is required to evaluate these scripts; it is not a dependency of the C++
runtime. See the [Gemma4 example](../../README.md) for the supported development
and test workflow.
