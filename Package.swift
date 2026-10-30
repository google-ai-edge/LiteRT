// swift-tools-version: 5.9
// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import PackageDescription

let package = Package(
  name: "LiteRT",
  platforms: [
    .iOS(.v15),
    .macOS(.v12),
  ],
  products: [
    .library(
      name: "LiteRT",
      targets: ["LiteRT"]
    ),
    .library(
      name: "LiteRT_static",
      type: .static,
      targets: ["LiteRT"]
    ),
    .library(
      name: "LiteRT_dynamic",
      type: .dynamic,
      targets: ["LiteRT"]
    ),
    .library(
      name: "LiteRtMetalAccelerator",
      targets: ["LiteRtMetalAccelerator"]
    ),
    .library(
      name: "CLiteRT_static",
      targets: ["CLiteRT_static"]
    ),
    .library(
      name: "TensorFlowLite",
      targets: ["TensorFlowLite"]
    ),
  ],
  targets: [
    // The Prebuilt Binary Target
    .binaryTarget(
      name: "CLiteRT",
      url: "https://github.com/google-ai-edge/LiteRT/releases/download/v2.3.0/CLiteRT.xcframework.zip",
      checksum: "b36a1a4f1b3eeb04546b4cc2986902cd704439b37c34597ab0f804aed08f5a00"
    ),
    .binaryTarget(
      name: "CLiteRT_mac",
      url: "https://github.com/google-ai-edge/LiteRT/releases/download/v2.3.0/CLiteRT_mac.xcframework.zip",
      checksum: "071b9f4bebb483709ac3d7faf98123da5453a2ebf60ce8f819aa1e30b9535a98"
    ),
    // Static build of the C API for C and Objective-C consumers (iOS only).
    .binaryTarget(
      name: "CLiteRT_static",
      url: "https://github.com/google-ai-edge/LiteRT/releases/download/v2.3.0/CLiteRT_static.xcframework.zip",
      checksum: "3742400af2c2b8e4a7f8dd87ccf1470bccf7f9866d80da56eefa76f899a7fb46"
    ),
    // Optional GPU Accelerator Plugin Target
    .binaryTarget(
      name: "LiteRtMetalAccelerator",
      url: "https://github.com/google-ai-edge/LiteRT/releases/download/v2.3.0/LiteRtMetalAccelerator.xcframework.zip",
      checksum: "e564e6678959098a35c1dbcffc60041afac25a845abcee998a53f958e342aeec"
    ),
    // The Swift Wrapper Target
    .target(
      name: "LiteRT",
      dependencies: [
        .target(name: "CLiteRT", condition: .when(platforms: [.iOS])),
        .target(name: "CLiteRT_mac", condition: .when(platforms: [.macOS])),
      ],
      path: "litert/swift/Sources/LiteRT"
    ),
    // The Test Target
    .testTarget(
      name: "LiteRTTests",
      dependencies: ["LiteRT"],
      path: "litert/swift/Tests/LiteRT"
    ),
    // The Prebuilt TensorFlow Lite C Binary Target
    .binaryTarget(
      name: "TensorFlowLiteC",
      url: "https://github.com/google-ai-edge/LiteRT/releases/download/v2.3.0/TensorFlowLiteC.xcframework.zip",
      checksum: "f77274f9249af1e7dd1de88f57661944f80d177ff2c386b28aa3d466b7173f50"
    ),
    // The TensorFlow Lite Swift Wrapper Target
    .target(
      name: "TensorFlowLite",
      dependencies: [
        .target(name: "TensorFlowLiteC", condition: .when(platforms: [.iOS])),
      ],
      path: "litert/swift/Sources/TensorFlowLite",
      // The Core ML and Metal delegate C libraries are not shipped in the
      // package, so the delegates that import them are left out.
      exclude: [
        "CoreMLDelegate.swift",
        "MetalDelegate.swift",
      ]
    ),
    // The TensorFlow Lite Test Target
    .testTarget(
      name: "TensorFlowLiteTests",
      dependencies: ["TensorFlowLite"],
      path: "litert/swift/Tests/TensorFlowLite",
      exclude: [
        "BUILD",
        // Tests `MetalDelegate`, which the `TensorFlowLite` target leaves out.
        "MetalDelegateTests.swift",
      ]
    ),
  ]
)
