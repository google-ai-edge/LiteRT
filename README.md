# LiteRT Next

LiteRT Next is a new set of APIs that improves upon LiteRT, particularly in
terms of hardware acceleration and performance for on-device ML and AI
applications. The APIs are an alpha release and available in Kotlin and C++.

The LiteRT Next CompiledModel API builds on the TensorFlow Lite Interpreter
API, and simplifies the model loading and execution process for on-device
machine learning. The new APIs provide a new streamlined way to use hardware
acceleration, removing the need to deal with model FlatBuffers, I/O buffer
interoperability, and delegates. The LiteRT Next APIs are not compatible with
the LiteRT APIs.

## Early Access Program for NPU Acceleration

> Get exclusive access to unified NPU acceleration in LiteRT Next. Join our Early Access Program to be among the first to integrate seamless NPU support from major chipset providers into your applications.
>
> [**Sign up for the EAP today!**](https://forms.gle/CoH4jpLwxiEYvDvF6)

## Key features

LiteRT Next contains the following key benefits and features:

-   **New LiteRT API:** Streamline development with automated accelerator
    selection, true async execution, and efficient I/O buffer handling.

-   **Best-in-class GPU Performance:** Use state-of-the-art GPU acceleration for
    on-device ML. The new buffer interoperability enables zero-copy and
    minimizes latency across various GPU buffer types.

-   **Superior Generative AI inference:** Enable the simplest integration with
    the best performance for GenAI models.

-   **Unified NPU Acceleration:** Offer seamless access to NPUs from major
    chipset providers with a consistent developer experience.

## Key improvements

LiteRT Next (CompiledModel API) contains the following key improvements on
LiteRT (TFLite Interpreter API). For a comprehensive guide to setting up your
application with LiteRT Next, see the Get Started guide.

-   **Accelerator usage:** Running models on GPU with LiteRT requires explicit
    delegate creation, function calls, and graph modifications. With LiteRT
    Next, just specify the accelerator.
-   **Native hardware buffer interoperability:** LiteRT does not provide the
    option of buffers, and forces all data through CPU memory. With LiteRT Next,
    you can pass in Android Hardware Buffers (AHWB), OpenCL buffers, OpenGL
    buffers, or other specialized buffers.

-   **Async execution:** LiteRT Next comes with a redesigned async API,
    providing a true async mechanism based on sync fences. This enables faster
    overall execution times through the use of diverse hardware – like CPUs,
    GPUs, CPUs, and NPUs – for different tasks.

-   **Model loading:** LiteRT Next does not require a separate builder step when
    loading a model.

For more details, check our
[official documentation](https://ai.google.dev/edge/litert/next/overview).

## Build From Source

1.  Start a docker daemon

2.  Run [build_with_docker.sh](./build/build_with_docker.sh) under
    [build/](./build)

3.  For more information about how to use docker interactive shell/ building
    different targets. Please refer to [build/README.md](./build/README.md)
