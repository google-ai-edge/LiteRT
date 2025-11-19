# LiteRT Profiler API User Guide

This document provides a guide to using the LiteRT Profiler API for performance
analysis and debugging of your models.

## Introduction

The LiteRT Profiler is a powerful tool that allows you to gain insights into the
execution of your models. It provides detailed timing information for various
stages of the inference process, helping you identify performance bottlenecks
and debug issues.

## Core Concepts

The Profiler API revolves around two main components:

*   **`Profiler`**: The main object used to control profiling. It is obtained
from a `CompiledModel` instance.
*   **`ProfiledEventData`**: Represents a single timed event captured by the
profiler. It contains detailed information about the event.

## Enabling the Profiler

To use the profiler, you must enable it during the model compilation phase. This
is done by setting the appropriate runtime options.

### Code Snippet: Enabling Profiling

```cpp
#include "third_party/odml/litert/litert/cc/litert_options.h"
#include "third_party/odml/litert/litert/cc/options/litert_runtime_options.h"

// Create compilation options.
LITERT_ASSIGN_OR_ABORT(litert::Options compilation_options,
                       litert::Options::Create());
compilation_options.SetHardwareAccelerators(kLiteRtHwAcceleratorCpu);

// Create and configure runtime options to enable profiling.
LITERT_ASSIGN_OR_ABORT(auto runtime_options, litert::RuntimeOptions::Create());
runtime_options.SetEnableProfiling(/*enabled=*/true);

// Add the runtime options to the compilation options.
compilation_options.AddOpaqueOptions(std::move(runtime_options));

// Create the CompiledModel with the specified options.
LITERT_ASSERT_OK_AND_ASSIGN(
    litert::CompiledModel compiled_model,
    litert::CompiledModel::Create(env, model, compilation_options));
```

## Using the Profiler

Once you have a `CompiledModel` with profiling enabled, you can obtain and use
the `Profiler` object.

### Workflow

1.  **Get the Profiler**: Obtain the `Profiler` instance from your `CompiledModel`.
2.  **Start Profiling**: Begin capturing events.
3.  **Run Inference**: Execute your model.
4.  **Get Events**: Retrieve the captured profiling events.
5.  **Reset (Optional)**: Clear the captured events to prepare for a new
profiling session.

### Code Snippet: Profiling an Inference Run

```cpp
#include "third_party/odml/litert/litert/cc/litert_profiler.h"

// 1. Get the Profiler from the CompiledModel.
LITERT_ASSERT_OK_AND_ASSIGN(auto profiler, compiled_model.GetProfiler());
ASSERT_TRUE(profiler);

// 2. Start Profiling.
ASSERT_TRUE(profiler.StartProfiling());

// Prepare input and output buffers...
// ...

// 3. Run Inference.
compiled_model.Run(input_buffers, output_buffers);

// 4. Get Events.
LITERT_ASSERT_OK_AND_ASSIGN(auto events, profiler.GetEvents());

// Process and print the events.
for (const auto& event : events) {
  // See "Understanding the Output" section for details on the event structure.
  std::cout << "Event Tag: " << event.tag
            << ", Start (us): " << event.start_timestamp_us
            << ", Elapsed (us): " << event.elapsed_time_us << std::endl;
}

// 5. Reset the profiler for the next run.
ASSERT_TRUE(profiler.Reset());
LITERT_ASSERT_OK_AND_ASSIGN(events, profiler.GetEvents());
// The events vector should now be empty.
EXPECT_EQ(events.size(), 0);
```

## Understanding the Output

The `GetEvents()` method returns a vector of `ProfiledEventData` objects. Each
`ProfiledEventData` has the following structure (from
`third_party/odml/litert/litert/c/litert_profiler_event.h`):

```cpp
struct ProfiledEventData {
  const char* tag;
  tflite::Profiler::EventType event_type;
  uint64_t start_timestamp_us;
  uint64_t elapsed_time_us;
  tflite::profiling::memory::MemoryUsage begin_mem_usage;
  tflite::profiling::memory::MemoryUsage end_mem_usage;
  uint64_t event_metadata1;
  uint64_t event_metadata2;
  ProfiledEventSource event_source;
};
```

### Sample Profiler Output

When you print the events, the output will look similar to this, showing the
tag, start time, and elapsed time for each event.

```
Event Tag: LiteRT::Run[buffer registration], Start (us): 1107775913177, Elapsed (us): 640
Event Tag: AllocateTensors, Start (us): 1107775913852, Elapsed (us): 293
Event Tag: Invoke, Start (us): 1107775914174, Elapsed (us): 50
Event Tag: Add (ND), Start (us): 0, Elapsed (us): 12
Event Tag: Invoke, Start (us): 0, Elapsed (us): 4294967296
Event Tag: LiteRT::Run[Buffer sync], Start (us): 1107775914236, Elapsed (us): 9
```

This output shows the time spent in different stages of the inference, such as
buffer registration, tensor allocation, and the actual invocation.

## Use Cases

The LiteRT Profiler is useful in a variety of scenarios:

*   **Performance Bottleneck Analysis**: By examining the `elapsed_time_us` of
different events, you can identify which operations in your model are taking the
most time.
*   **Delegate Verification**: The event tags can help you confirm that your
model is running on the intended delegate (e.g., you would expect to see
GPU-related events when using the GPU delegate).
*   **Comparing Hardware Performance**: You can run the same model on different
hardware and use the profiler to compare the performance characteristics.
*   **Regression Testing**: Incorporate profiling into your testing pipeline to
catch performance regressions.

By integrating the LiteRT Profiler into your development workflow, you can
ensure your models are running efficiently and correctly.