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

// Development manifest of the LiteRT Swift package.
//
// This manifest is for developing the Swift package from this repository. It
// is not meant to be added to apps: its binary targets resolve the prebuilt
// xcframework archives from the `prebuilt/` directory, which is not checked in,
// so resolving the package fails until the archives are built or downloaded
// there. See litert/swift/README.md for how to populate `prebuilt/`.
//
// Apps depend on the package from https://github.com/google-ai-edge/LiteRT-Swift
// instead, which mirrors the Swift sources of litert/swift and downloads the
// archives published with each LiteRT release.

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
      path: "prebuilt/CLiteRT.xcframework.zip"
    ),
    .binaryTarget(
      name: "CLiteRT_mac",
      path: "prebuilt/CLiteRT_mac.xcframework.zip"
    ),
    // Static build of the C API for C and Objective-C consumers (iOS only).
    .binaryTarget(
      name: "CLiteRT_static",
      path: "prebuilt/CLiteRT_static.xcframework.zip"
    ),
    // Optional GPU Accelerator Plugin Target
    .binaryTarget(
      name: "LiteRtMetalAccelerator",
      path: "prebuilt/LiteRtMetalAccelerator.xcframework.zip"
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
    // The Prebuilt TensorFlow Lite C Binary Target
    .binaryTarget(
      name: "TensorFlowLiteC",
      path: "prebuilt/TensorFlowLiteC.xcframework.zip"
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
    // The tests run with Bazel (see litert/swift/README.md), as they need Bazel
    // runfiles for the test models and accelerator libraries, so this manifest
    // declares no test targets.
  ]
)
