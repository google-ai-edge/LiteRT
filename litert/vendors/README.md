<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# LiteRT Hardware Vendor Integrations (`litert/vendors/`)

This directory holds LiteRT's ABI-stable C plugin headers, reference examples,
and in-tree SoC backends.

> [!IMPORTANT]
> **Building a new NPU backend?**
> You do not need to check code into `litert/vendors/` or modify LiteRT. See
> [Build Your Own NPU Backend](../BUILD_YOUR_OWN_BACKEND.md) to build and ship
> standalone `libLiteRtDispatch_<Vendor>.so` and
> `libLiteRtCompilerPlugin_<Vendor>.so` libraries out-of-tree.

## Directory Layout

| Subdirectory | Purpose |
| :--- | :--- |
| [`c/`](./c/) | ABI-stable C11 headers (`litert_dispatch_api.h`, `litert_compiler_plugin_api.h`) with no `libLiteRt.so` link dependency |
| [`cc/`](./cc/) | Optional C++ helpers for in-tree vendor plugins |
| [`examples/`](./examples/) | Runnable `example_plugin.cc` + `example_dispatch.cc`, CMake build, and ABI version-skew tests (`compatibility_test/`) |
| [`qualcomm/`](./qualcomm/README.md) | Qualcomm AI Engine Direct (QAIRT / QNN) compiler plugin and dispatch runtime |
| [`mediatek/`](./mediatek/) | MediaTek NeuroPilot compiler plugin and dispatch runtime |
| [`google_tensor/`](./google_tensor/) | Google Tensor compiler plugin and dispatch runtime |
| [`samsung/`](./samsung/) | Samsung Exynos AI LiteCore compiler plugin and dispatch runtime |
| [`intel_openvino/`](./intel_openvino/) | Intel OpenVINO compiler plugin and dispatch runtime |

## Related Documentation

*   [Build Your Own NPU Backend](../BUILD_YOUR_OWN_BACKEND.md)
*   [Dispatch API Guide](../DISPATCH_API.md)
*   [Compiler Plugin Guide](../COMPILER_PLUGIN.md)
*   [JIT Compilation in LiteRT](../JIT_COMPILATION.md)
*   [Transformation Pattern Matching Guide](../PATTERN_MATCHING.md)
*   [Accelerator Test Suite (`ats`)](../TESTING.md)
