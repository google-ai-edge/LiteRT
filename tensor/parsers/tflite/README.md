# LiteRT TFLite Flatbuffer Parser

This directory contains the `FlatbufferToTensorGraphParser`, which reconstructs
a
LiteRT in-memory tensor graph from a TFLite flatbuffer model.

Two key extension mechanisms are used during parsing to customize the
reconstructed graph:

1.  **Custom Operation Parser Registry**: Maps TFLite custom operators to
    concrete LiteRT typed operations.
2.  **Mixin Registrar**: Attaches backend-specific execution logic (mixins)
    to the parsed backend-neutral operations.

---

## 1. Custom Operation Parser Registry

By default, when the parser encounters a `BuiltinOperator_CUSTOM` in the
flatbuffer, it parses it into a generic `litert::tensor::graph::CustomOperation`
node, storing the raw custom options as a byte vector.

If you have a concrete LiteRT operation that represents this custom op (e.g., a
`RotaryEmbeddingOperation` representing a `"CUSTOM_ROPE"` custom operator), you
can register a custom parser to reconstruct the concrete type instead.

### Registration

Register a parser callback associated with the custom code:

```cpp
#include "tensor/parsers/tflite/tflite_flatbuffer_parser.h"

FlatbufferToTensorGraphParser::RegisterCustomOpParser(
    "CUSTOM_ROPE",
    [](const tflite::Operator* tfl_op)
        -> absl::StatusOr<std::shared_ptr<graph::Operation>> {
      auto op = std::make_shared<graph::RotaryEmbeddingOperation>();
      
      // Parse custom options from the flatbuffer operator
      if (tfl_op->custom_options()) {
        std::vector<uint8_t> options(tfl_op->custom_options()->begin(),
                                     tfl_op->custom_options()->end());
        // Unpack options into your operation attributes...
        if (options.size() >= sizeof(float)) {
          std::memcpy(&op->min_timescale, options.data(), sizeof(float));
        }
      }
      return op;
    });
```

When `Parse` is called, the parser will lookup `"CUSTOM_ROPE"` in the registry
and invoke your callback to instantiate the correct operation type.

---

## 2. Mixin Registrar

Reconstructed graphs are backend-neutral by default. They contain core
operations (like `AddOperation`, `RotaryEmbeddingOperation`) but no
information on how to execute them on a specific backend.

LiteRT uses **Mixins** to associate backend execution logic (e.g.,
`MlDriftOperation` for ML Drift, `TfLiteOperation` for TFLite) with core
operations. To make a parsed graph executable, you must attach these mixins to
the nodes using a `MixinRegistrar` during parsing.

### How it Works

1.  **Define the Registrar**: Implement a class inheriting from
`graph::MixinRegistrar` that registers mixins for the operations you want to
support.

    ```cpp
    #include "tensor/internal/mixin.h"
    
    // Define the list of operations supported by your backend target
    using MyBackendOps = std::tuple<
        graph::AddOperation,
        graph::RotaryEmbeddingOperation
    >;

    class MyBackendMixinRegistrar : public graph::MixinRegistrar {
     public:
      void Register(std::shared_ptr<graph::Operation> op) override {
        // Automatically attaches the MyBackendMixinTag mixin to the op
        // if it matches any type in MyBackendOps.
        graph::RegisterMixin<MyBackendMixinTag, MyBackendOps>(op);
      }
    };
    ```

2.  **Pass to Parser**: Pass an instance of your registrar to the `Parse`
    method:

    ```cpp
    MyBackendMixinRegistrar registrar;
    auto parsed_outputs = FlatbufferToTensorGraphParser::Parse(
        fb_buffer_span,
        &registrar
    );
    ```

As the parser reconstructs each operation, it calls `registrar.Register(op)`.
If the operation is in your supported list, the corresponding
`OpMixin<Op, MyBackendMixinTag>` specialization is instantiated and attached
to the operation node.

The resulting graph is now fully equipped with the backend mixins and can be
passed to the backend code generator (e.g., `AppendGpuModelBuilder` for ML
Drift).
