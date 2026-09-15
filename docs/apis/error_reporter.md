# LiteRT Error Reporter API User Guide

This document explains how to configure and consume the LiteRT Error Reporter
API so that you can capture detailed diagnostics from model compilation and
runtime execution.

## Core Concepts

The Error Reporter API revolves around two concepts:

*   **`LiteRtErrorReporterMode`**: Runtime option that selects how errors are
    surfaced. Available modes are:
    *   `kLiteRtErrorReporterModeNone` – disables error forwarding.
    *   `kLiteRtErrorReporterModeStderr` – prints errors via stderr.
    *   `kLiteRtErrorReporterModeBuffer` – accumulates formatted messages in a
        `BufferErrorReporter` so they can be retrieved programmatically.
*   **Buffer utilities**: Helper functions that allow you to inject custom error
    messages (`ReportError`) and read or clear the buffer (`GetErrorMessages`,
    `ClearErrors`) when buffer mode is active.

## Configuring the Error Reporter

Set the error reporter mode during compilation by attaching a
`RuntimeOptions` opaque option to the compilation options.

### Code Snippet: Configure Buffer Mode

```cpp
#include "third_party/odml/litert/litert/cc/litert_compiled_model.h"
#include "third_party/odml/litert/litert/cc/litert_environment.h"
#include "third_party/odml/litert/litert/cc/litert_model.h"
#include "third_party/odml/litert/litert/cc/options/litert_options.h"
#include "third_party/odml/litert/litert/cc/options/litert_runtime_options.h"

LITERT_ASSIGN_OR_ABORT(litert::Options compilation_options,
                       litert::Options::Create());
compilation_options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);

LITERT_ASSIGN_OR_ABORT(auto runtime_options, litert::RuntimeOptions::Create());
runtime_options.SetErrorReporterMode(kLiteRtErrorReporterModeBuffer);
compilation_options.AddOpaqueOptions(std::move(runtime_options));

LITERT_ASSERT_OK_AND_ASSIGN(
    litert::CompiledModel compiled_model,
    litert::CompiledModel::Create(env, model, compilation_options));
```

If you prefer printing to stderr, replace `kLiteRtErrorReporterModeBuffer` with
`kLiteRtErrorReporterModeStderr`.

## Reporting and Inspecting Errors

Once the compiled model is created with buffer mode, you can log application or
runtime errors directly into the reporter and retrieve them later.

### Code Snippet: Logging and Reading Errors

```cpp
// Inject an application-specific error message.
compiled_model.ReportError("Input %s failed validation: expected %d elements",
                           signature.InputNames()[0].c_str(), expected_size);

// Fetch all buffered messages as a single string.
LITERT_ASSERT_OK_AND_ASSIGN(auto messages, compiled_model.GetErrorMessages());
if (!messages.empty()) {
  std::cout << "LiteRT errors:\n" << messages << std::endl;
  LITERT_ASSERT_OK(compiled_model.ClearErrors());
}
```

## Tips

*   **Use buffer mode when debugging**: Buffer mode lets you collect internal
    runtime failures together with application-specific errors before surfacing
    them to users or telemetry.
*   **Clear the buffer after handling messages**: Call `ClearErrors()`
    once you have processed the payload so subsequent runs start from a clean
    slate.
