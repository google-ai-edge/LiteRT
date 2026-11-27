Kotlin interface for the new LiteRT API.

`TensorBuffer.writeFrom(ByteBuffer)` and `TensorBuffer.readInto(ByteBuffer)` copy
packed tensor bytes directly between a caller-owned direct buffer and the tensor,
without allocating an intermediate Java array. They copy one packed tensor
payload, starting at the buffer's position, and advance that position after a
successful copy. The limit stays unchanged, and insufficient remaining space
raises `LiteRtException` without advancing the position.

The input must already contain the tensor's expected packed representation.
These methods do not convert pixels, data types, or byte order, or
quantize/dequantize values. Use native byte order for multi-byte values.
`writeFrom` accepts read-only direct buffers; `readInto` requires a writable
direct buffer. Existing array methods remain available.

```kotlin
inputTensor.writeFrom(preprocessedInput)
model.run(inputTensors, outputTensors)
outputTensor.readInto(output)
output.flip()
```


APK callers must adopt the new methods to benefit.
