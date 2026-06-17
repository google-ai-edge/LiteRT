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
  ],
  products: [
    .library(
      name: "LiteRT",
      targets: ["LiteRT"]
    )
  ],
  targets: [
    // The Prebuilt Binary Target
    .binaryTarget(
      name: "CLiteRT",
      path: "prebuilt/CLiteRT.xcframework.zip"
    ),
    // The Swift Wrapper Target
    .target(
      name: "LiteRT",
      dependencies: [
        .target(name: "CLiteRT", condition: .when(platforms: [.iOS])),
      ],
      path: "litert/swift/Sources",
      exclude: [
        "BUILD",
        "Info.plist",
      ],
      linkerSettings: [
        .unsafeFlags(["-Xlinker", "-all_load"])
      ]
    ),
    // The Test Target
    .testTarget(
      name: "LiteRTTests",
      dependencies: ["LiteRT"],
      path: "litert/swift/Tests"
    ),
  ]
)
