# LiteRT Core

This directory contains the core components of the LiteRT runtime. These
components are not part of the public API and are subject to change without
notice.

## Files

The `litert/core` directory contains the following files:

*   **`build_stamp.h`**: This file defines the build stamp that is embedded in
    the LiteRT runtime. The build stamp contains information about the build,
    such as the date, time, and the user who built the runtime.
*   **`dispatch_op_schema.h`**: This file defines the schema for the dispatch
    op. The dispatch op is a custom operator that is used to dispatch the
    execution of a model to a different backend.
*   **`dynamic_loading.h`**: This file defines functions for dynamically loading shared libraries.
*   **`environment.h`**: This file defines the `LiteRtEnvironment` class, which
    is the top-level object in the LiteRT runtime. It is responsible for
    managing the lifecycle of all other objects in the runtime.
*   **`environment_options.h`**: This file defines the
    `LiteRtEnvironmentOptions` class, which is used to configure the
    `LiteRtEnvironment`.
*   **`filesystem.h`**: This file defines functions for interacting with the filesystem.
*   **`insert_order_map.h`**: This file defines the `InsertOrderMap` class,
    which is a map that preserves the insertion order of its elements.
*   **`options.h`**: This file defines the `LiteRtOptions` struct, which is
    used to configure the LiteRT runtime.
*   **`version.h`**: This file defines the version of the LiteRT runtime.

## Subdirectories

The `litert/core` directory contains the following subdirectories:

*   **`model`**: This directory contains the `Model` class, which represents a LiteRT model.
*   **`util`**: This directory contains utility functions that are used by the LiteRT runtime.

## Relationship to the C and C++ APIs

The `litert/core` directory contains the implementation of the LiteRT runtime.
The C and C++ APIs are built on top of the `litert/core` components. The C API
provides a low-level interface to the LiteRT runtime, while the C++ API provides
a higher-level, more convenient interface.

The `litert/core` components are not part of the public API and are subject to
change without notice. Therefore, you should not use them directly in your
applications. Instead, you should use the C or C++ APIs.

