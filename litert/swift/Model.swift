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

/// A compiled LiteRT model.
///
/// **Basic Usage Example:**
/// ```swift
/// let env = try Environment()
/// let model = try CompiledModel(file: "model.tflite",
///     options: try CompiledModel.Options(), environment: env)
///
/// // 1. Create default managed buffers
/// let inBuffers = try model.createInputBuffers()
/// let outBuffers = try model.createOutputBuffers()
///
/// // 2. Copy data into the input buffer
/// let inputPixels: [Float32] = getMyPixels()
/// try inBuffers[0].write(inputPixels)
///
/// // 3. Run the model
/// try model.run(signatureIndex: 0, inputBuffers: inBuffers, outputBuffers: outBuffers)
///
/// // 4. Read the results (allocates a new array)
/// let results: [Float32] = try outBuffers[0].read()
/// ```
//
/// **Usage Example (Metal Zero-Copy):**
/// ```swift
/// let env = try Environment()
/// let options = try CompiledModel.Options()
/// // Note: Configure GPU/Metal options as needed
/// let model = try CompiledModel(file: "model.tflite", options: options, environment: env)
///
/// // 1. Determine the expected tensor type
/// let inputType = try model.createInputBuffers()[0].tensorType
///
/// // 2. Create a zero-copy TensorBuffer directly from an active MTLBuffer
/// // let myMTLBuffer = device.makeBuffer(length: size, options: .storageModeShared)
/// let tensorBuffer = try TensorBuffer(tensorType: inputType, environment: env, metalBuffer: myMTLBuffer)
///
/// // 3. Run the model using the Metal zero-copy buffer
/// try model.run(signatureIndex: 0, inputBuffers: [tensorBuffer], outputBuffers: outBuffers)
/// ```
public class CompiledModel {
  let model: LiteRtModel
  let handle: LiteRtCompiledModel  // CompiledModel handle
  let environment: Environment

  /// Options for the model.
  public class Options {
    var options: LiteRtOptions

    public init() throws {
      var opts: LiteRtOptions?
      let status = LiteRtCreateOptions(&opts)
      guard status == kLiteRtStatusOk, let opts = opts else {
        throw LiteRtError.memoryAllocationFailure("Failed to create options")
      }
      self.options = opts
    }

    deinit {
      LiteRtDestroyOptions(options)
    }

    public func setAccelerators(_ accelerators: HwAccelerators) throws {
      let status = LiteRtSetOptionsHardwareAccelerators(options, accelerators.rawValue)
      guard status == kLiteRtStatusOk else {
        throw status.toError(message: "Failed to set accelerators")
          ?? .unknown("Failed to set accelerators")
      }
    }

    public func getAccelerators() throws -> HwAccelerators {
      var accelerators: LiteRtHwAcceleratorSet = 0
      let status = LiteRtGetOptionsHardwareAccelerators(options, &accelerators)
      guard status == kLiteRtStatusOk else {
        throw status.toError(message: "Failed to get accelerators")
          ?? .unknown("Failed to get accelerators")
      }
      return HwAccelerators(rawValue: accelerators)
    }

    public class CpuOptions {
      var handle: LiteRtOpaqueOptions

      public init() throws {
        var opts: LiteRtOpaqueOptions?
        let status = LiteRtCreateCpuOptions(&opts)
        guard status == kLiteRtStatusOk, let opts = opts else {
          throw LiteRtError.memoryAllocationFailure("Failed to create CPU options")
        }
        self.handle = opts
      }

      public func setNumThreads(_ numThreads: Int) throws {
        let status = LiteRtSetCpuOptionsNumThread(handle, Int32(numThreads))
        guard status == kLiteRtStatusOk else {
          throw status.toError(message: "Failed to set num threads")
            ?? .unknown("Failed to set num threads")
        }
      }

      public func getNumThreads() throws -> Int {
        var numThreads: Int32 = 0
        let status = LiteRtGetCpuOptionsNumThread(handle, &numThreads)
        guard status == kLiteRtStatusOk else {
          throw status.toError(message: "Failed to get num threads")
            ?? .unknown("Failed to get num threads")
        }
        return Int(numThreads)
      }

      public func setXnnPackFlags(_ flags: Int) throws {
        let status = LiteRtSetCpuOptionsXNNPackFlags(handle, UInt32(flags))
        guard status == kLiteRtStatusOk else {
          throw status.toError(message: "Failed to set XNNPack flags")
            ?? .unknown("Failed to set XNNPack flags")
        }
      }

      public func setXnnPackWeightCachePath(_ path: String) throws {
        let status = LiteRtSetCpuOptionsXnnPackWeightCachePath(handle, path)
        guard status == kLiteRtStatusOk else {
          throw status.toError(message: "Failed to set XNNPack weight cache path")
            ?? .unknown("Failed to set XNNPack weight cache path")
        }
      }
    }

    public func setCpuOptions(_ cpuOptions: CpuOptions) throws {
      let status = LiteRtAddOpaqueOptions(options, cpuOptions.handle)
      guard status == kLiteRtStatusOk else {
        throw status.toError(message: "Failed to add CPU options")
          ?? .unknown("Failed to add CPU options")
      }
    }

    public class GpuOptions {
      var handle: LiteRtOpaqueOptions

      public init() throws {
        var opts: LiteRtOpaqueOptions?
        let status = LiteRtCreateGpuOptions(&opts)
        guard status == kLiteRtStatusOk, let opts = opts else {
          throw LiteRtError.memoryAllocationFailure("Failed to create GPU options")
        }
        self.handle = opts
      }

      public enum Priority: Int32 {
        case auto = 0  // kLiteRtGpuPriorityDefault
        case low = 1  // kLiteRtGpuPriorityLow
        case normal = 2  // kLiteRtGpuPriorityNormal
        case high = 3  // kLiteRtGpuPriorityHigh
      }

      public enum Precision: Int32 {
        case standard = 0  // kLiteRtDelegatePrecisionDefault
        case fp16 = 1  // kLiteRtDelegatePrecisionFp16
        case fp32 = 2  // kLiteRtDelegatePrecisionFp32
      }

      public enum Backend: Int32 {
        case metal = 0  // kLiteRtGpuBackendAutomatic (Default/Metal)
        case webGpu = 2  // kLiteRtGpuBackendWebGpu
      }

      // On MacOS, only Metal is supported, so no backend options are available.

      public func setPriority(_ priority: Priority) throws {
        let status = LiteRtSetGpuOptionsGpuPriority(
          handle, LiteRtGpuPriority(UInt32(priority.rawValue)))
        guard status == kLiteRtStatusOk else {
          throw status.toError(message: "Failed to set GPU priority")
            ?? .unknown("Failed to set GPU priority")
        }
      }

      public func setBackend(_ backend: Backend) throws {
        let status = LiteRtSetGpuOptionsGpuBackend(
          handle, LiteRtGpuBackend(UInt32(backend.rawValue)))
        guard status == kLiteRtStatusOk else {
          throw status.toError(message: "Failed to set GPU backend")
            ?? .unknown("Failed to set GPU backend")
        }
      }

      public func setSerializationDir(_ dir: String) throws {
        let status = LiteRtSetGpuAcceleratorCompilationOptionsSerializationDir(handle, dir)
        guard status == kLiteRtStatusOk else {
          throw status.toError(message: "Failed to set serialization dir")
            ?? .unknown("Failed to set serialization dir")
        }
      }

      public func setPrecision(_ precision: Precision) throws {
        let status = LiteRtSetGpuAcceleratorCompilationOptionsPrecision(
          handle, LiteRtDelegatePrecision(UInt32(precision.rawValue)))
        guard status == kLiteRtStatusOk else {
          throw status.toError(message: "Failed to set precision")
            ?? .unknown("Failed to set precision")
        }
      }

      // Metal specific
      // setMetalCommandQueue removed as it is handled via EnvironmentOptions

      public func setMetalDisableWait(_ disableWait: Bool) throws {
        let waitType = disableWait ? kLiteRtGpuWaitTypeDoNotWait : kLiteRtGpuWaitTypeDefault
        let status = LiteRtSetGpuAcceleratorRuntimeOptionsWaitType(handle, waitType)
        guard status == kLiteRtStatusOk else {
          throw status.toError(message: "Failed to set Metal disable wait")
            ?? .unknown("Failed to set Metal disable wait")
        }
      }
    }
    public func setGpuOptions(_ gpuOptions: GpuOptions) throws {
      let status = LiteRtAddOpaqueOptions(options, gpuOptions.handle)
      guard status == kLiteRtStatusOk else {
        throw status.toError(message: "Failed to add GPU options")
          ?? .unknown("Failed to add GPU options")
      }
    }
  }

  /// Creates a model from a file.
  public init(file: String, options: Options? = nil, environment: Environment) throws {
    self.environment = environment

    // 1. Load Model
    var model: LiteRtModel?
    var status = LiteRtCreateModelFromFile(file, &model)
    guard status == kLiteRtStatusOk, let model = model else {
      throw LiteRtError.fileIO("Failed to load model from file: \(file)")
    }
    self.model = model

    // 2. Compile
    let opts = try options ?? Options()
    var compiledModel: LiteRtCompiledModel?
    status = LiteRtCreateCompiledModel(environment.handle, model, opts.options, &compiledModel)
    guard status == kLiteRtStatusOk, let compiledModel = compiledModel else {
      throw status.toError(message: "Failed to create compiled model")
        ?? .unknown("Unknown error creating compiled model")
    }
    self.handle = compiledModel
  }

  deinit {
    LiteRtDestroyCompiledModel(handle)
    LiteRtDestroyModel(model)
  }

  /// Runs the model.
  public func run(signatureIndex: Int, inputBuffers: [TensorBuffer], outputBuffers: [TensorBuffer])
    throws
  {
    var inputHandles: [LiteRtTensorBuffer?] = inputBuffers.map { $0.handle }
    var outputHandles: [LiteRtTensorBuffer?] = outputBuffers.map { $0.handle }

    let status = LiteRtRunCompiledModel(
      handle,
      LiteRtParamIndex(signatureIndex),
      inputHandles.count,
      &inputHandles,
      outputHandles.count,
      &outputHandles)

    guard status == kLiteRtStatusOk else {
      throw status.toError(message: "Failed to run model")
        ?? .unknown("Unknown error running model")
    }
  }

  private func getSignature(key: String?) throws -> LiteRtSignature {
    if let targetKey = key {
      var numSignatures: LiteRtParamIndex = 0
      let status = LiteRtGetNumModelSignatures(model, &numSignatures)
      guard status == kLiteRtStatusOk else {
        throw status.toError(message: "Failed to get number of model signatures")
          ?? .unknown("Failed to get number of model signatures")
      }

      for i in 0..<numSignatures {
        var signature: LiteRtSignature?
        let getSigStatus = LiteRtGetModelSignature(model, i, &signature)
        guard getSigStatus == kLiteRtStatusOk, let signature = signature else {
          throw getSigStatus.toError(message: "Failed to get signature at index \(i)")
            ?? .internalError("Failed to get signature at index \(i)")
        }

        var signatureKey: UnsafePointer<CChar>?
        let getKeyStatus = LiteRtGetSignatureKey(signature, &signatureKey)
        guard getKeyStatus == kLiteRtStatusOk, let signatureKey = signatureKey else {
          throw getKeyStatus.toError(message: "Failed to get signature key at index \(i)")
            ?? .internalError("Failed to get signature key at index \(i)")
        }

        if String(cString: signatureKey) == targetKey {
          return signature
        }
      }
      throw LiteRtError.invalidArgument("Signature not found: \(targetKey)")
    } else {
      var numSignatures: LiteRtParamIndex = 0
      let status = LiteRtGetNumModelSignatures(model, &numSignatures)
      guard status == kLiteRtStatusOk else {
        throw status.toError(message: "Failed to get number of model signatures")
          ?? .unknown("Failed to get number of model signatures")
      }
      guard numSignatures > 0 else {
        throw LiteRtError.internalError("Model has no signatures")
      }

      var signature: LiteRtSignature?
      let sigStatus = LiteRtGetModelSignature(model, 0, &signature)
      guard sigStatus == kLiteRtStatusOk, let signature = signature else {
        throw sigStatus.toError(message: "Failed to get default signature")
          ?? .internalError("Failed to get default signature")
      }
      return signature
    }
  }

  private func getTensorType(tensor: LiteRtTensor) throws -> TensorType {
    var type = LiteRtRankedTensorType()
    let status = LiteRtGetRankedTensorType(tensor, &type)
    guard status == kLiteRtStatusOk else {
      throw status.toError(message: "Failed to get tensor type")
        ?? .unknown("Failed to get tensor type")
    }
    return TensorType(cType: type)
  }

  public func createInputBuffers(signatureKey: String? = nil) throws -> [TensorBuffer] {
    let sig = try getSignature(key: signatureKey)
    var numInputs: LiteRtParamIndex = 0
    let status = LiteRtGetNumSignatureInputs(sig, &numInputs)
    guard status == kLiteRtStatusOk else {
      throw status.toError(message: "Failed to get number of inputs")
        ?? .unknown("Failed to get number of inputs")
    }

    var buffers: [TensorBuffer] = []
    buffers.reserveCapacity(Int(numInputs))
    for i in 0..<numInputs {
      var tensor: LiteRtTensor?
      let status = LiteRtGetSignatureInputTensorByIndex(sig, i, &tensor)
      guard status == kLiteRtStatusOk, let tensor = tensor else {
        throw LiteRtError.internalError("Failed to get input tensor at index \(i)")
      }
      let type = try getTensorType(tensor: tensor)
      buffers.append(try TensorBuffer(tensorType: type, environment: environment))
    }
    return buffers
  }

  public func createOutputBuffers(signatureKey: String? = nil) throws -> [TensorBuffer] {
    let sig = try getSignature(key: signatureKey)
    var numOutputs: LiteRtParamIndex = 0
    let status = LiteRtGetNumSignatureOutputs(sig, &numOutputs)
    guard status == kLiteRtStatusOk else {
      throw status.toError(message: "Failed to get number of outputs")
        ?? .unknown("Failed to get number of outputs")
    }

    var buffers: [TensorBuffer] = []
    buffers.reserveCapacity(Int(numOutputs))
    for i in 0..<numOutputs {
      var tensor: LiteRtTensor?
      let status = LiteRtGetSignatureOutputTensorByIndex(sig, i, &tensor)
      guard status == kLiteRtStatusOk, let tensor = tensor else {
        throw LiteRtError.internalError("Failed to get output tensor at index \(i)")
      }
      let type = try getTensorType(tensor: tensor)
      buffers.append(try TensorBuffer(tensorType: type, environment: environment))
    }
    return buffers
  }

  public func createInputBuffer(signature: String? = nil, inputName: String) throws -> TensorBuffer
  {
    let sig = try getSignature(key: signature)
    var tensor: LiteRtTensor?
    let status = LiteRtGetSignatureInputTensor(sig, inputName, &tensor)
    guard status == kLiteRtStatusOk, let tensor = tensor else {
      throw LiteRtError.invalidArgument("Input not found: \(inputName)")
    }

    let type = try getTensorType(tensor: tensor)
    return try TensorBuffer(tensorType: type, environment: environment)
  }

  public func createOutputBuffer(signature: String? = nil, outputName: String) throws
    -> TensorBuffer
  {
    let sig = try getSignature(key: signature)
    var tensor: LiteRtTensor?
    let status = LiteRtGetSignatureOutputTensor(sig, outputName, &tensor)
    guard status == kLiteRtStatusOk, let tensor = tensor else {
      throw LiteRtError.invalidArgument("Output not found: \(outputName)")
    }

    let type = try getTensorType(tensor: tensor)
    return try TensorBuffer(tensorType: type, environment: environment)
  }

  public func getInputRequirements(signatureIndex: Int, inputIndex: Int) throws
    -> TensorBuffer.Requirements
  {
    var reqs: LiteRtTensorBufferRequirements?
    let status = LiteRtGetCompiledModelInputBufferRequirements(
      handle, LiteRtParamIndex(signatureIndex), LiteRtParamIndex(inputIndex), &reqs)
    guard status == kLiteRtStatusOk, let reqs = reqs else {
      throw status.toError(message: "Failed to get input requirements")
        ?? .unknown("Unknown error")
    }
    return TensorBuffer.Requirements(handle: reqs, owned: false)
  }

  public func getOutputRequirements(signatureIndex: Int, outputIndex: Int) throws
    -> TensorBuffer.Requirements
  {
    var reqs: LiteRtTensorBufferRequirements?
    let status = LiteRtGetCompiledModelOutputBufferRequirements(
      handle, LiteRtParamIndex(signatureIndex), LiteRtParamIndex(outputIndex), &reqs)
    guard status == kLiteRtStatusOk, let reqs = reqs else {
      throw status.toError(message: "Failed to get output requirements")
        ?? .unknown("Unknown error")
    }
    return TensorBuffer.Requirements(handle: reqs, owned: false)
  }
}
