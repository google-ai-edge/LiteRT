# LiteRT Runtime

This directory contains the core runtime components of the LiteRT framework.
These components are responsible for managing the execution of models, including
memory allocation, the dispatch of operations to hardware accelerators,
and the synchronization of events.

## Core Concepts

The LiteRT runtime is built around the following core concepts:

*   **`Accelerator`**: An `Accelerator` is a hardware accelerator that can be
    used to speed up the execution of a model. The LiteRT runtime provides a
    registry for accelerators, which allows developers to register their own
    custom accelerators.
*   **`TensorBuffer`**: A `TensorBuffer` is a block of memory that can be used to store the tensor data. It can be backed by host memory, hardware buffers, or other types of memory.
*   **`Event`**: An `Event` is used to synchronize the execution of operations on different hardware accelerators.
*   **`CompiledModel`**: A `CompiledModel` is a model that has been optimized for a specific hardware platform. It is created from a `Model` and a set of compilation options.

## Files

The `litert/runtime` directory contains the following files:

*   **`accelerator.h`**: Defines the `LiteRtAccelerator` struct, which represents a hardware accelerator.
*   **`accelerator_registry.h`**: This file defines the `AcceleratorRegistry` class, which is used to register and manage hardware accelerators.
*   **`ahwb_buffer.h`**: This file defines the `AhwbBuffer` struct, which represents an Android Hardware Buffer.
*   **`compiled_model.h`**: This file defines the `LiteRtCompiledModelT` class, which is the internal implementation of the `litert::CompiledModel` class.
*   **`custom_op_dispatcher.h`**: This file defines the `CustomOpDispatcher` class, which is used to dispatch the execution of custom operators.
*   **`dmabuf_buffer.h`**: This file defines the `DmaBufBuffer` struct, which represents a DMA-BUF buffer.
*   **`event.h`**: This file defines the `LiteRtEventT` struct, which is the internal implementation of the `litert::Event` class.
*   **`external_litert_buffer_context.h`**: This file defines the `ExternalLiteRtBufferContext` class, which is used to manage the lifetime of external tensor buffers.
*   **`fastrpc_buffer.h`**: This file defines the `FastRpcBuffer` struct, which represents a FastRPC buffer.
*   **`gl_buffer.h`**: This file defines the `GlBuffer` class, which represents an OpenGL buffer.
*   **`gl_texture.h`**: This file defines the `GlTexture` class, which represents an OpenGL texture.
*   **`gpu_environment.h`**: This file defines the `GpuEnvironment` class, which is used to manage the GPU environment.
*   **`ion_buffer.h`**: This file defines the `IonBuffer` struct, which represents an ION buffer.
*   **`litert_cpu_options.h`**: This file defines the `LiteRtCpuOptions` struct, which is used to configure the CPU backend.
*   **`litert_google_tensor.h`**: This file defines the `LiteRtGoogleTensorOptions` struct, which is used to configure the Google Tensor backend.
*   **`litert_runtime_options.h`**: This file defines the `LiteRtRuntimeOptions` struct, which is used to configure the LiteRT runtime.
*   **`metrics.h`**: This file defines the `LiteRtMetricsT` struct, which is used to collect metrics.
*   **`open_cl_memory.h`**: This file defines the `OpenClMemory` class, which represents an OpenCL memory object.
*   **`open_cl_sync.h`**: This file defines functions for synchronizing with OpenCL.
*   **`profiler.h`**: This file defines the `LiteRtProfilerT` class, which is the internal implementation of the `litert::Profiler` class.
*   **`tensor_buffer.h`**: This file defines the `LiteRtTensorBufferT` class, which is the internal implementation of the `litert::TensorBuffer` class.
*   **`tensor_buffer_conversion.h`**: This file defines functions for converting between different tensor buffer types.
*   **`tensor_buffer_requirements.h`**: This file defines the `LiteRtTensorBufferRequirementsT` class, which is the internal implementation of the `litert::TensorBufferRequirements` class.
*   **`tfl_utils.h`**: This file defines utility functions for working with TensorFlow Lite.

## Subdirectories

The `litert/runtime` directory contains the following subdirectories:

*   **`accelerators`**: Contains the implementation of the hardware accelerators that are supported by LiteRT.
*   **`compiler`**: This directory contains the LiteRT compiler, which is used to optimize models for specific hardware platforms.
*   **`dispatch`**: This directory contains the `Dispatch` class, which is used to dispatch the execution of a model to a different backend.

## Relationship to the C and C++ APIs

The `litert/runtime` directory contains the implementation of the LiteRT
runtime. The C and C++ APIs are built on top of the `litert/runtime` components.
The C API provides a low-level interface to the LiteRT runtime, while the C++
API provides a higher-level, more convenient interface.

The `litert/runtime` components are not part of the public API and are subject
to change without notice. Therefore, you should not use them directly in your
applications. Instead, you should use the C or C++ APIs.

