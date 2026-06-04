// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import LiteRtC

/// A compiled LiteRT model ready for execution.
public final class CompiledModel {
  internal let cCompiledModel: LiteRtCompiledModel
  public let environment: Environment
  public let model: Model
  public let options: Options?

  /// Creates a compiled model from an environment and a model, with optional configuration options.
  ///
  /// - Parameters:
  ///   - environment: The runtime environment to use for execution.
  ///   - model: The model to compile.
  ///   - options: Optional configuration options for the compiled model.
  /// - Throws: `LiteRtError` if the compilation fails.
  public init(environment: Environment, model: Model, options: Options? = nil) throws {
    var compiledModel: LiteRtCompiledModel?
    let status: LiteRtStatus

    if let options {
      status = withExtendedLifetime((environment, model, options)) {
        LiteRtCreateCompiledModel(
          environment.cEnvironment,
          model.cModel,
          options.cOptions,
          &compiledModel
        )
      }
    } else {
      status = withExtendedLifetime((environment, model)) {
        LiteRtCreateCompiledModel(
          environment.cEnvironment,
          model.cModel,
          nil,
          &compiledModel
        )
      }
    }

    try checkStatus(status)
    guard let compiledModel else {
      throw LiteRtError.runtimeFailure
    }
    self.cCompiledModel = compiledModel
    self.environment = environment
    self.model = model
    self.options = options
  }

  deinit {
    LiteRtDestroyCompiledModel(cCompiledModel)
  }

  /// Runs model inference with specified inputs and outputs and optional per-invocation options.
  ///
  /// - Parameters:
  ///   - inputs: The input tensor buffers.
  ///   - outputs: The output tensor buffers where results will be written.
  ///   - options: Optional per-invocation execution options.
  ///   - signatureIndex: The index of the signature to run. Defaults to 0.
  /// - Throws: `LiteRtError` if the inference fails.
  public func run(
    inputs: [TensorBuffer],
    outputs: [TensorBuffer],
    options: Options? = nil,
    signatureIndex: Int = 0
  ) throws {
    var cInputs = inputs.map { $0.cTensorBuffer as LiteRtTensorBuffer? }
    var cOutputs = outputs.map { $0.cTensorBuffer as LiteRtTensorBuffer? }
    let status: LiteRtStatus
    if let options {
      status = withExtendedLifetime(options) {
        LiteRtRunCompiledModelWithOptions(
          cCompiledModel,
          size_t(signatureIndex),
          inputs.count,
          &cInputs,
          outputs.count,
          &cOutputs,
          options.cOptions
        )
      }
    } else {
      status = LiteRtRunCompiledModel(
        cCompiledModel,
        size_t(signatureIndex),
        inputs.count,
        &cInputs,
        outputs.count,
        &cOutputs
      )
    }
    try checkStatus(status)
  }

  /// Runs model inference with optional per-invocation options and returns newly allocated output buffers safely.
  ///
  /// - Parameters:
  ///   - inputs: The input tensor buffers.
  ///   - options: Optional per-invocation execution options.
  ///   - signatureIndex: The index of the signature to run. Defaults to 0.
  /// - Returns: An array of newly allocated output tensor buffers containing the inference results.
  /// - Throws: `LiteRtError` if the inference fails.
  public func run(
    inputs: [TensorBuffer],
    options: Options? = nil,
    signatureIndex: Int = 0
  ) throws -> [TensorBuffer] {
    let outputs = try createOutputBuffers(signatureIndex: signatureIndex)
    try run(inputs: inputs, outputs: outputs, options: options, signatureIndex: signatureIndex)
    return outputs
  }

  /// Dispatches model inference asynchronously if possible with specified inputs and outputs.
  ///
  /// This is a synchronous, non-blocking hardware dispatch on the CPU that initiates
  /// execution on hardware accelerators (e.g., GPU or NPU) and attaches synchronization events
  /// (fences) to the output tensor buffers. Downstream operations or buffer reads (e.g.,
  /// `TensorBuffer.read()`) will automatically wait for these events to signal.
  ///
  /// - Parameters:
  ///   - inputs: The input tensor buffers.
  ///   - outputs: The output tensor buffers where results will be written.
  ///   - options: Optional per-invocation execution options.
  ///   - signatureIndex: The index of the signature to run. Defaults to 0.
  /// - Returns: `true` if the model was successfully dispatched for asynchronous execution.
  ///   `false` if asynchronous execution was not possible (e.g. due to hardware limitations
  ///   or unsupported ops), in which case the model was executed synchronously.
  /// - Throws: `LiteRtError` if the dispatch or synchronous execution fails.
  public func dispatch(
    inputs: [TensorBuffer],
    outputs: [TensorBuffer],
    options: Options? = nil,
    signatureIndex: Int = 0
  ) throws -> Bool {
    var cInputs = inputs.map { $0.cTensorBuffer as LiteRtTensorBuffer? }
    var cOutputs = outputs.map { $0.cTensorBuffer as LiteRtTensorBuffer? }
    var isAsync = false
    let status: LiteRtStatus
    if let options {
      status = withExtendedLifetime(options) {
        LiteRtRunCompiledModelAsyncWithOptions(
          cCompiledModel,
          size_t(signatureIndex),
          inputs.count,
          &cInputs,
          outputs.count,
          &cOutputs,
          &isAsync,
          options.cOptions
        )
      }
    } else {
      status = LiteRtRunCompiledModelAsync(
        cCompiledModel,
        size_t(signatureIndex),
        inputs.count,
        &cInputs,
        outputs.count,
        &cOutputs,
        &isAsync
      )
    }
    try checkStatus(status)
    return isAsync
  }

  /// Dispatches inference asynchronously if possible and returns newly allocated output buffers safely.
  ///
  /// This is a synchronous, non-blocking hardware dispatch on the CPU that initiates
  /// execution on hardware accelerators (e.g., GPU or NPU) and attaches synchronization events
  /// (fences) to the output tensor buffers. Downstream operations or buffer reads (e.g.,
  /// `TensorBuffer.read()`) will automatically wait for these events to signal.
  ///
  /// - Parameters:
  ///   - inputs: The input tensor buffers.
  ///   - options: Optional per-invocation execution options.
  ///   - signatureIndex: The index of the signature to run. Defaults to 0.
  /// - Returns: A tuple containing the newly allocated output buffers (`buffers`) and a boolean
  ///   (`isAsync`) indicating `true` if the model was successfully dispatched for asynchronous
  ///   execution, or `false` if it was executed synchronously.
  /// - Throws: `LiteRtError` if the dispatch or synchronous execution fails.
  public func dispatch(
    inputs: [TensorBuffer],
    options: Options? = nil,
    signatureIndex: Int = 0
  ) throws -> (buffers: [TensorBuffer], isAsync: Bool) {
    let outputs = try createOutputBuffers(signatureIndex: signatureIndex)
    let isAsync = try dispatch(
      inputs: inputs, outputs: outputs, options: options, signatureIndex: signatureIndex)
    return (outputs, isAsync)
  }

  /// Automatically creates input buffers based on compiling requirements.
  ///
  /// - Parameter signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: An array of newly created input tensor buffers.
  /// - Throws: `LiteRtError` if buffer creation fails.
  public func createInputBuffers(signatureIndex: Int = 0) throws -> [TensorBuffer] {
    let count = try inputCount(signatureIndex: signatureIndex)
    var inputs: [TensorBuffer] = []
    for i in 0..<count {
      let requirements = try getInputBufferRequirements(
        inputIndex: i, signatureIndex: signatureIndex)
      let tensorType = try getInputTensorType(inputIndex: i, signatureIndex: signatureIndex)
      let bufferType = requirements.supportedTypes.first ?? .hostMemory
      let buf = try TensorBuffer(
        environment: environment,
        bufferType: bufferType,
        tensorType: tensorType,
        size: requirements.bufferSize
      )
      inputs.append(buf)
    }
    return inputs
  }

  /// Automatically creates output buffers based on compiling requirements.
  ///
  /// - Parameter signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: An array of newly created output tensor buffers.
  /// - Throws: `LiteRtError` if buffer creation fails.
  public func createOutputBuffers(signatureIndex: Int = 0) throws -> [TensorBuffer] {
    let count = try outputCount(signatureIndex: signatureIndex)
    var outputs: [TensorBuffer] = []
    for i in 0..<count {
      let requirements = try getOutputBufferRequirements(
        outputIndex: i, signatureIndex: signatureIndex)
      let tensorType = try getOutputTensorType(outputIndex: i, signatureIndex: signatureIndex)
      let bufferType = requirements.supportedTypes.first ?? .hostMemory
      let buf = try TensorBuffer(
        environment: environment,
        bufferType: bufferType,
        tensorType: tensorType,
        size: requirements.bufferSize
      )
      outputs.append(buf)
    }
    return outputs
  }

  /// Queries the input count for a signature.
  ///
  /// - Parameter signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: The number of inputs for the signature.
  /// - Throws: `LiteRtError` if the query fails.
  public func inputCount(signatureIndex: Int = 0) throws -> Int {
    var signature: LiteRtSignature?
    var status = LiteRtGetModelSignature(model.cModel, size_t(signatureIndex), &signature)
    try checkStatus(status)
    guard let signature else { throw LiteRtError.runtimeFailure }

    var count: size_t = 0
    status = LiteRtGetNumSignatureInputs(signature, &count)
    try checkStatus(status)
    return Int(count)
  }

  /// Queries the output count for a signature.
  ///
  /// - Parameter signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: The number of outputs for the signature.
  /// - Throws: `LiteRtError` if the query fails.
  public func outputCount(signatureIndex: Int = 0) throws -> Int {
    var signature: LiteRtSignature?
    var status = LiteRtGetModelSignature(model.cModel, size_t(signatureIndex), &signature)
    try checkStatus(status)
    guard let signature else { throw LiteRtError.runtimeFailure }

    var count: size_t = 0
    status = LiteRtGetNumSignatureOutputs(signature, &count)
    try checkStatus(status)
    return Int(count)
  }

  /// Queries the input name for a signature.
  ///
  /// - Parameters:
  ///   - inputIndex: The index of the input.
  ///   - signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: The name of the input.
  /// - Throws: `LiteRtError` if the query fails.
  public func inputName(
    inputIndex: Int,
    signatureIndex: Int = 0
  ) throws -> String {
    var signature: LiteRtSignature?
    var status = LiteRtGetModelSignature(model.cModel, size_t(signatureIndex), &signature)
    try checkStatus(status)
    guard let signature else { throw LiteRtError.runtimeFailure }

    var cName: UnsafePointer<CChar>?
    status = LiteRtGetSignatureInputName(signature, size_t(inputIndex), &cName)
    try checkStatus(status)
    guard let cName else { throw LiteRtError.runtimeFailure }
    return String(cString: cName)
  }

  /// Queries the output name for a signature.
  ///
  /// - Parameters:
  ///   - outputIndex: The index of the output.
  ///   - signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: The name of the output.
  /// - Throws: `LiteRtError` if the query fails.
  public func outputName(
    outputIndex: Int,
    signatureIndex: Int = 0
  ) throws -> String {
    var signature: LiteRtSignature?
    var status = LiteRtGetModelSignature(model.cModel, size_t(signatureIndex), &signature)
    try checkStatus(status)
    guard let signature else { throw LiteRtError.runtimeFailure }

    var cName: UnsafePointer<CChar>?
    status = LiteRtGetSignatureOutputName(signature, size_t(outputIndex), &cName)
    try checkStatus(status)
    guard let cName else { throw LiteRtError.runtimeFailure }
    return String(cString: cName)
  }

  /// Queries buffer requirements for a specified input.
  ///
  /// - Parameters:
  ///   - inputIndex: The index of the input.
  ///   - signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: The buffer requirements for the input.
  /// - Throws: `LiteRtError` if the query fails.
  public func getInputBufferRequirements(
    inputIndex: Int,
    signatureIndex: Int = 0
  ) throws -> TensorBufferRequirements {
    var requirements: LiteRtTensorBufferRequirements?
    let status = LiteRtGetCompiledModelInputBufferRequirements(
      cCompiledModel,
      size_t(signatureIndex),
      size_t(inputIndex),
      &requirements
    )
    try checkStatus(status)
    guard let requirements else { throw LiteRtError.runtimeFailure }
    return TensorBufferRequirements(cRequirements: requirements, owned: false)
  }

  /// Queries buffer requirements for a specified output.
  ///
  /// - Parameters:
  ///   - outputIndex: The index of the output.
  ///   - signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: The buffer requirements for the output.
  /// - Throws: `LiteRtError` if the query fails.
  public func getOutputBufferRequirements(
    outputIndex: Int,
    signatureIndex: Int = 0
  ) throws -> TensorBufferRequirements {
    var requirements: LiteRtTensorBufferRequirements?
    let status = LiteRtGetCompiledModelOutputBufferRequirements(
      cCompiledModel,
      size_t(signatureIndex),
      size_t(outputIndex),
      &requirements
    )
    try checkStatus(status)
    guard let requirements else { throw LiteRtError.runtimeFailure }
    return TensorBufferRequirements(cRequirements: requirements, owned: false)
  }

  /// Queries the tensor type for a specified input.
  ///
  /// - Parameters:
  ///   - inputIndex: The index of the input.
  ///   - signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: The tensor type of the input.
  /// - Throws: `LiteRtError` if the query fails.
  public func getInputTensorType(
    inputIndex: Int,
    signatureIndex: Int = 0
  ) throws -> TensorType {
    var signature: LiteRtSignature?
    var status = LiteRtGetModelSignature(model.cModel, size_t(signatureIndex), &signature)
    try checkStatus(status)
    guard let signature else { throw LiteRtError.runtimeFailure }

    var tensor: LiteRtTensor?
    status = LiteRtGetSignatureInputTensorByIndex(signature, size_t(inputIndex), &tensor)
    try checkStatus(status)
    guard let tensor else { throw LiteRtError.runtimeFailure }

    var cRankedType = LiteRtRankedTensorType()
    status = LiteRtGetRankedTensorType(tensor, &cRankedType)
    try checkStatus(status)

    return TensorType(cRankedType: cRankedType)
  }

  /// Queries the tensor type for a specified output.
  ///
  /// - Parameters:
  ///   - outputIndex: The index of the output.
  ///   - signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: The tensor type of the output.
  /// - Throws: `LiteRtError` if the query fails.
  public func getOutputTensorType(
    outputIndex: Int,
    signatureIndex: Int = 0
  ) throws -> TensorType {
    var signature: LiteRtSignature?
    var status = LiteRtGetModelSignature(model.cModel, size_t(signatureIndex), &signature)
    try checkStatus(status)
    guard let signature else { throw LiteRtError.runtimeFailure }

    var tensor: LiteRtTensor?
    status = LiteRtGetSignatureOutputTensorByIndex(signature, size_t(outputIndex), &tensor)
    try checkStatus(status)
    guard let tensor else { throw LiteRtError.runtimeFailure }

    var cRankedType = LiteRtRankedTensorType()
    status = LiteRtGetRankedTensorType(tensor, &cRankedType)
    try checkStatus(status)

    return TensorType(cRankedType: cRankedType)
  }

  /// Queries the current runtime input tensor layout (reflecting dynamic resizing).
  ///
  /// - Parameters:
  ///   - inputIndex: The index of the input.
  ///   - signatureIndex: The index of the signature to query. Defaults to 0.
  /// - Returns: The layout of the input tensor.
  /// - Throws: `LiteRtError` if the query fails.
  public func getInputTensorLayout(
    inputIndex: Int,
    signatureIndex: Int = 0
  ) throws -> Layout {
    var cLayout = LiteRtLayout()
    let status = LiteRtGetCompiledModelInputTensorLayout(
      cCompiledModel,
      size_t(signatureIndex),
      size_t(inputIndex),
      &cLayout
    )
    try checkStatus(status)

    // Decode first byte containing rank and hasStrides (C Bitfields)
    let firstByte = withUnsafeBytes(of: cLayout) { bytes in
      bytes[0]
    }
    let rank = Int(firstByte & 0x7F)
    let hasStrides = (firstByte & 0x80) != 0

    var dimensions: [Int] = []
    withUnsafeBytes(of: cLayout.dimensions) { rawDims in
      let dimsPointer = rawDims.bindMemory(to: Int32.self)
      for i in 0..<min(rank, 8) {
        dimensions.append(Int(dimsPointer[i]))
      }
    }

    var strides: [Int] = []
    if hasStrides {
      withUnsafeBytes(of: cLayout.strides) { rawStrides in
        let stridesPointer = rawStrides.bindMemory(to: UInt32.self)
        for i in 0..<min(rank, 8) {
          strides.append(Int(stridesPointer[i]))
        }
      }
    }

    return Layout(dimensions: dimensions, strides: strides)
  }

  /// Queries current runtime output tensor layouts (reflecting dynamic resizing).
  ///
  /// - Parameters:
  ///   - signatureIndex: The index of the signature to query. Defaults to 0.
  ///   - updateAllocation: Whether to update the underlying allocation. Defaults to `true`.
  /// - Returns: An array of layouts for the output tensors.
  /// - Throws: `LiteRtError` if the query fails.
  public func getOutputTensorLayouts(
    signatureIndex: Int = 0,
    updateAllocation: Bool = true
  ) throws -> [Layout] {
    let count = try outputCount(signatureIndex: signatureIndex)
    guard count > 0 else { return [] }

    var cLayouts = [LiteRtLayout](repeating: LiteRtLayout(), count: count)
    let status = cLayouts.withUnsafeMutableBufferPointer { layoutsBuffer -> LiteRtStatus in
      guard let baseAddr = layoutsBuffer.baseAddress else {
        return kLiteRtStatusErrorInvalidArgument
      }
      return LiteRtGetCompiledModelOutputTensorLayouts(
        cCompiledModel,
        size_t(signatureIndex),
        count,
        baseAddr,
        updateAllocation
      )
    }
    try checkStatus(status)

    var layouts: [Layout] = []
    for cLayout in cLayouts {
      let firstByte = withUnsafeBytes(of: cLayout) { bytes in bytes[0] }
      let rank = Int(firstByte & 0x7F)
      let hasStrides = (firstByte & 0x80) != 0

      var dimensions: [Int] = []
      withUnsafeBytes(of: cLayout.dimensions) { rawDims in
        let dimsPointer = rawDims.bindMemory(to: Int32.self)
        for i in 0..<min(rank, 8) {
          dimensions.append(Int(dimsPointer[i]))
        }
      }

      var strides: [Int] = []
      if hasStrides {
        withUnsafeBytes(of: cLayout.strides) { rawStrides in
          let stridesPointer = rawStrides.bindMemory(to: UInt32.self)
          for i in 0..<min(rank, 8) {
            strides.append(Int(stridesPointer[i]))
          }
        }
      }
      layouts.append(Layout(dimensions: dimensions, strides: strides))
    }
    return layouts
  }

  /// Checks if the compiled model is fully accelerated by hardware.
  ///
  /// - Returns: `true` if fully accelerated, `false` otherwise.
  /// - Throws: `LiteRtError` if the check fails.
  public func isFullyAccelerated() throws -> Bool {
    var fullyAccelerated = false
    let status = LiteRtCompiledModelIsFullyAccelerated(cCompiledModel, &fullyAccelerated)
    try checkStatus(status)
    return fullyAccelerated
  }

  /// Resizes a specified input tensor.
  ///
  /// - Parameters:
  ///   - inputIndex: The index of the input tensor to resize.
  ///   - dimensions: The new dimensions for the input tensor.
  ///   - signatureIndex: The index of the signature. Defaults to 0.
  /// - Throws: `LiteRtError` if the resize operation fails.
  public func resizeInputTensor(
    inputIndex: Int,
    dimensions: [Int],
    signatureIndex: Int = 0
  ) throws {
    let cDims = dimensions.map { Int32($0) }
    let status = cDims.withUnsafeBufferPointer { dimsBuffer -> LiteRtStatus in
      guard let baseAddr = dimsBuffer.baseAddress else {
        return kLiteRtStatusErrorInvalidArgument
      }
      return LiteRtCompiledModelResizeInputTensor(
        cCompiledModel,
        size_t(signatureIndex),
        size_t(inputIndex),
        baseAddr,
        dimensions.count
      )
    }
    try checkStatus(status)
  }

  /// Resizes a specified input tensor (non-strict mode).
  ///
  /// - Parameters:
  ///   - inputIndex: The index of the input tensor to resize.
  ///   - dimensions: The new dimensions for the input tensor.
  ///   - signatureIndex: The index of the signature. Defaults to 0.
  /// - Throws: `LiteRtError` if the resize operation fails.
  public func resizeInputTensorNonStrict(
    inputIndex: Int,
    dimensions: [Int],
    signatureIndex: Int = 0
  ) throws {
    let cDims = dimensions.map { Int32($0) }
    let status = cDims.withUnsafeBufferPointer { dimsBuffer -> LiteRtStatus in
      guard let baseAddr = dimsBuffer.baseAddress else {
        return kLiteRtStatusErrorInvalidArgument
      }
      return LiteRtCompiledModelResizeInputTensorNonStrict(
        cCompiledModel,
        size_t(signatureIndex),
        size_t(inputIndex),
        baseAddr,
        dimensions.count
      )
    }
    try checkStatus(status)
  }

  /// Clears all errors on the compiled model (buffer error reporter mode).
  ///
  /// - Throws: `LiteRtError` if clearing errors fails.
  public func clearErrors() throws {
    let status = LiteRtCompiledModelClearErrors(cCompiledModel)
    try checkStatus(status)
  }

  /// Gets all error messages as a single string (buffer error reporter mode).
  ///
  /// - Returns: A string containing all error messages.
  /// - Throws: `LiteRtError` if retrieving error messages fails.
  public func getErrorMessages() throws -> String {
    var cMessages: UnsafeMutablePointer<CChar>?
    let status = LiteRtCompiledModelGetErrorMessages(cCompiledModel, &cMessages)
    try checkStatus(status)
    guard let cMessages else {
      throw LiteRtError.runtimeFailure
    }
    defer {
      free(cMessages)
    }
    return String(cString: cMessages)
  }

  /// Sets a callback function to check if model execution should be cancelled.
  ///
  /// - Parameters:
  ///   - userData: Optional user data to pass to the callback function.
  ///   - checkCancelledFunc: A closure that returns `true` if execution should be cancelled.
  /// - Throws: `LiteRtError` if setting the cancellation function fails.
  public func setCancellationFunction(
    userData: UnsafeMutableRawPointer? = nil,
    checkCancelledFunc: @escaping @convention(c) (UnsafeMutableRawPointer?) -> Bool
  ) throws {
    let status = LiteRtSetCompiledModelCancellationFunction(
      cCompiledModel,
      userData,
      checkCancelledFunc
    )
    try checkStatus(status)
  }
}
