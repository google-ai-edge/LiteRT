NPU just-in-time compilation caching
====================================

LiteRT supports NPU just-in-time (JIT) compilation of `.tflite` models. JIT
compilation can be especially useful in situations where compiling the model
ahead of time is not feasible.

JIT compilation however can come with some latency and memory overhead to
translate the user-provided model into NPU bytecode instructions on-demand. To
minimize the performance impact NPU compilation artifacts can be cached.

When caching is enabled LiteRT will only trigger the re-compilation of the model
when required, e.g.:

- The vendor's NPU compiler plugin version changed;
- The Android build fingerprint changed;
- The use-provided model changed;
- The compilation options changed.

Example
-------

In order to enable NPU compilation caching, specify the `CompilerCacheDir`
environment tag in the environment options. The value must be set to an
existing writable path of the application.

```C++
   const std::array environment_options = {
        litert::Environment::Option{
            /*.tag=*/litert::Environment::OptionTag::CompilerPluginLibraryDir,
            /*.value=*/kCompilerPluginLibSearchPath,
        },
        litert::Environment::Option{
            litert::Environment::OptionTag::DispatchLibraryDir,
            kDispatchLibraryDir,
        },
        // 'kCompilerCacheDir' will be used to store NPU-compiled model
        // artifacts.
        litert::Environment::Option{
            litert::Environment::OptionTag::CompilerCacheDir,
            kCompilerCacheDir,
        },
    };

    // Create an environment.
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto environment, litert::Environment::Create(environment_options));

    // Load a model.
    auto model_path = litert::testing::GetTestFilePath(kModelFileName);
    LITERT_ASSERT_OK_AND_ASSIGN(auto model,
                                litert::Model::CreateFromFile(model_path));

    // Create a compiled model, which only triggers NPU compilation if
    // required.
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto compiled_model, litert::CompiledModel::Create(
                                 environment, model, kLiteRtHwAcceleratorNpu));
```

Example latency and memory savings
----------------------------------

The time and memory required for NPU compilation can vary based on several
factors, like the underlying NPU chip, the complexity of the input model etc.

The following table compares the runtime initialization time and memory
consumption when NPU compilation is required vs when compilation can be skipped
due to caching. On one sample device we obtain the following:

| TFLite model | model init with NPU compilation | model init with cached compilation | init memory footprint with NPU compilation | init memory with cached compilation |
| :------------------------------ | :---------------------------------: | :-----------------------------------------: | :----------------------------------: | :---: |
| torchvision_resnet152.tflite | 7465.22 ms | 198.34 ms | 1525.24 MB | 355.07 MB |
| torchvision_lraspp_mobilenet_v3_large.tflite | 1592.54 ms | 166.47 ms | 254.90 MB | 33.78 MB |

On another device we obtain the following:

| TFLite model | model init with NPU compilation | model init with cached compilation | init memory footprint with NPU compilation | init memory with cached compilation |
| :------------------------------ | :---------------------------------: | :-----------------------------------------: | :----------------------------------: | :---: |
| torchvision_resnet152.tflite | 2766.44 ms | 379.86 ms | 653.54 MB | 501.21 MB |
| torchvision_lraspp_mobilenet_v3_large.tflite | 784.14 ms | 231.76 ms | 113.14 MB | 67.49 MB |
