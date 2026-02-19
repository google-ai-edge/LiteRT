// Copyright 2024 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

@main
struct BinaryHeaderMapTool {
    struct Arguments {
        var output: URL
        var moduleName: String?
        var headers: [String]

        init(arguments: [String]) {
            var arguments = arguments
            var output: URL?

            // If the first argument starts with @, treat it as a response file.
            // Since this is an internal tool, it's safe to just check the first argument.
            if let arg = arguments.first, arg.hasPrefix("@") {
                // Remove @ prefix to get file path
                let filePath = String(arg.dropFirst())

                guard FileManager.default.fileExists(atPath: filePath) else {
                    fatalError("The response file doesn't exist: \(filePath)")
                }

                guard let fileContents = try? String(contentsOfFile: filePath, encoding: .utf8) else {
                    fatalError("The response file cannot be read: \(filePath)")
                }

                arguments = fileContents.components(separatedBy: .newlines)
            }

            if let outputIndex = arguments.firstIndex(of: "--output") {
                guard outputIndex + 1 < arguments.count else {
                    fatalError("Missing output path after --output")
                }
                output = URL(fileURLWithPath: arguments[outputIndex + 1])
                arguments.removeSubrange(outputIndex...outputIndex + 1)
            }

            if let moduleNameIndex = arguments.firstIndex(of: "--module_name") {
                guard moduleNameIndex + 1 < arguments.count else {
                    fatalError("Missing module_name argument after --module_name")
                }
                self.moduleName = arguments[moduleNameIndex + 1]
                arguments.removeSubrange(moduleNameIndex...moduleNameIndex + 1)
            }

            guard let output = output else {
                fatalError("Missing output path")
            }

            self.output = output
            self.headers = arguments
        }
    }

    static func main() {
        let arguments = Arguments(arguments: Array(CommandLine.arguments.dropFirst()))
        var entries: [BinaryHeaderMap.Entry] = []

        // Parse entries into a list of BinaryHeaderMap.Entry
        // Following the format:
        // <key=[module_name]/basename> <prefix=dir path> <suffix=filename>
        for header in arguments.headers {
            // Get the basename
            let basename = header.components(separatedBy: "/").last!
            let `prefix` = header.replacingOccurrences(of: basename, with: "")
            let suffix = basename
            // Add an entry with just the basename for the key
            entries.append(
                BinaryHeaderMap.Entry(
                    key: basename,
                    prefix: `prefix`,
                    suffix: suffix
                )
            )
            // If a module name was provided, add an entry with the module name and basename for the key
            if let moduleName = arguments.moduleName {
                entries.append(
                    BinaryHeaderMap.Entry(
                        key: "\(moduleName)/\(basename)",
                        prefix: `prefix`,
                        suffix: suffix
                    )
                )
            }
        }

        do {
            let data = try BinaryHeaderMapEncoder.encode(BinaryHeaderMap(entries: entries))
            try data.write(to: arguments.output)
        } catch {
            fatalError("Failed to encode header map: \(error)")
        }
    }
}
