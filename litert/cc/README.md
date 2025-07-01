# LiteRT C++ API

This directory contains the C++ API for the LiteRT runtime. The C++ API is a higher-level interface that is built on top of the C API. It provides a more convenient and type-safe way to interact with the LiteRT runtime.

## Core Concepts

The LiteRT C++ API is built around a few core concepts:

*   **`litert::Environment`**: The `litert::Environment` class is the top-level object in the C++ API. It is responsible for managing the lifecycle of all other objects in the C++ API.
*   **`litert::Model`**: The `litert::Model` class represents a LiteRT model. It contains the model's graph, weights, and metadata.
*   **`litert::CompiledModel`**: The `litert::CompiledModel` class represents a model that has been optimized for a specific hardware platform. It is created from a `litert::Model` and a set of compilation options.
*   **`litert::TensorBuffer`**: The `litert::TensorBuffer` class is a block of memory that can be used to store the data of a tensor. It can be backed by host memory, hardware buffers, or other types of memory.
*   **`litert::Accelerator`**: The `litert::Accelerator` class represents a hardware accelerator that can be used to speed up the execution of a model.
*   **`litert::Expected<T>`**: The `litert::Expected<T>` class is a type that can hold either a value of type `T` or an error. It is used to return values from functions that can fail.

## API Reference

The C++ API is defined in the following header files:

*   [`litert_any.h`](litert_any.h): This file defines the `litert::Any` class, which is a type-safe container for a single value of any type.
*   [`litert_buffer_ref.h`](litert_buffer_ref.h): This file defines the `litert::BufferRef` class, which is a non-owning reference to a buffer.
*   [`litert_c_types_printing.h`](litert_c_types_printing.h): This file defines functions for printing the C types defined in the `litert/c` directory.
*   [`litert_compiled_model.h`](litert_compiled_model.h): This file defines the `litert::CompiledModel` class, which represents a model that has been optimized for a specific hardware platform.
*   [`litert_consts.h`](litert_consts.h): This file defines constants that are used throughout the LiteRT C++ API.
*   [`litert_custom_op_kernel.h`](litert_custom_op_kernel.h): This file defines the `litert::CustomOpKernel` class, which can be used to implement custom operators.
*   [`litert_detail.h`](litert_detail.h): This file contains implementation details of the LiteRT C++ API.
*   [`litert_dispatch_delegate.h`](litert_dispatch_delegate.h): This file defines the `litert::DispatchDelegate` class, which is a delegate that can be used to dispatch the execution of a model to a different backend.
*   [`litert_element_type.h`](litert_element_type.h): This file defines the `litert::ElementType` enum, which represents the data type of a tensor element.
*   [`litert_environment.h`](litert_environment.h): This file defines the `litert::Environment` class, which is the top-level object in the C++ API.
*   [`litert_environment_options.h`](litert_environment_options.h): This file defines the `litert::EnvironmentOptions` class, which is used to configure the `litert::Environment`.
*   [`litert_event.h`](litert_event.h): This file defines the `litert::Event` class, which can be used to profile the execution of a model.
*   [`litert_expected.h`](litert_expected.h): This file defines the `litert::Expected<T>` class, which is a type that can hold either a value of type `T` or an error.
*   [`litert_handle.h`](litert_handle.h): This file defines the `litert::Handle` class, which is a smart pointer that is used to manage the lifetime of C API objects.
*   [`litert_layout.h`](litert_layout.h): This file defines the `litert::Layout` class, which describes the physical layout of a tensor in memory.
*   [`litert_logging.h`](litert_logging.h): This file defines functions for logging.
*   [`litert_macros.h`](litert_macros.h): This file defines macros that are used throughout the LiteRT C++ API.
*   [`litert_model.h`](litert_model.h): This file defines the `litert::Model` class, which represents a LiteRT model.
*   [`litert_model_predicates.h`](litert_model_predicates.h): This file defines predicates that can be used to match patterns in the graph.
*   [`litert_numerics.h`](litert_numerics.h): This file defines numeric types and functions.
*   [`litert_op_options.h`](litert_op_options.h): This file defines the `litert::OpOptions` class, which is used to configure operator-specific options.
*   [`litert_opaque_options.h`](litert_opaque_options.h): This file defines the `litert::OpaqueOptions` class, which is used to pass custom options to the LiteRT runtime.
*   [`litert_options.h`](litert_options.h): This file defines the `litert::Options` class, which is used to configure general options.
*   [`litert_platform_support.h`](litert_platform_support.h): This file defines functions for checking for platform support.
*   [`litert_profiler.h`](litert_profiler.h): This file defines the `litert::Profiler` class, which can be used to collect performance data about the execution of a model.
*   [`litert_rng.h`](litert_rng.h): This file defines functions for random number generation.
*   [`litert_shared_library.h`](litert_shared_library.h): This file defines the `litert::SharedLibrary` class, which can be used to dynamically load shared libraries.
*   [`litert_source_location.h`](litert_source_location.h): This file defines the `litert::SourceLocation` class, which is used to represent a location in the source code.
*   [`litert_tensor_buffer.h`](litert_tensor_buffer.h): This file defines the `litert::TensorBuffer` class, which is a block of memory that can be used to store the data of a tensor.
*   [`litert_tensor_buffer_requirements.h`](litert_tensor_buffer_requirements.h): This file defines the `litert::TensorBufferRequirements` class, which describes the properties of a tensor buffer that are required by a model.
*   [`litert_tensor_buffer_utils.h`](litert_tensor_buffer_utils.h): This file defines utility functions for working with tensor buffers.
*   [`litert_tflite_error_status_builder.h`](litert_tflite_error_status_builder.h): This file defines the `litert::TfLiteErrorStatusBuilder` class, which is used to build `TfLiteStatus` objects from `litert::ErrorStatusBuilder` objects.
