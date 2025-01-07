# LiteRT
GitHub repository for Google's open-source high-performance runtime for
on-device AI which has been renamed from TensorFlow Lite to LiteRT.

More details of the LiteRT announcement are in this [blog
post](https://developers.googleblog.com/en/tensorflow-lite-is-now-litert/).

The official documentation can be found at https://ai.google.dev/edge/litert

In its current state, the LiteRT repository is not intended for open source
development because it is pulling in existing TensorFlow code via a git
submodule. We intend to evolve this repo to a point where developers can
directly build and contribute here, at which time we will make a separate
announcement.


## PyPi Installation Requirements

 * Python versions:  3.9, 3.10, 3.11
 * Operating system: Linux, MacOS


## FAQs

1.  How do I contribute code?

    For now, please contribute code to the
    [existing TensorFlow Lite repository](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md).

2.  What is happening to the .tflite file extension and file format?

    No changes are being made to the .tflite file extension or format.
    Conversion tools will continue to output .tflite flatbuffer files, and
    .tflite files will be readable by LiteRT.

3.  How do I convert models to .tflite format?

    For [Tensorflow](https://ai.google.dev/edge/litert/models/convert),
    [Keras](https://ai.google.dev/edge/litert/models/convert) and
    [Jax](https://ai.google.dev/edge/litert/models/jax_to_tflite) you can
    continue to use the same flows. For PyTorch support check out
    [ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch).

4.  Will there be any changes to classes and methods?

    No. Aside from package names, you won't have to change any code you've
    written for now.

5.  Is TensorFlow Lite still being actively developed?

    Yes, but under the name LiteRT. Active development will continue on the
    runtime (now called LiteRT), as well as the conversion and optimization
    tools. To ensure you're using the most up-to-date version of the runtime,
    please use LiteRT.
