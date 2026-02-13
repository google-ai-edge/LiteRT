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

/// Interface for creating binary header maps.
public struct BinaryHeaderMap {
    public struct Entry {
        public let key: String
        public let prefix: String
        public let suffix: String

        public init(key: String, prefix: String, suffix: String) {
            self.key = key
            self.prefix = prefix
            self.suffix = suffix
        }
    }

    public typealias EntryIndex = [Data: Entry]
    public let entries: EntryIndex

    public init(entries: [Entry]) {
        self.entries = makeIndexedEntries(from: entries)
    }

    public subscript(key: String) -> BinaryHeaderMap.Entry? {
        guard let lowercasedBytes = try? key.clangLowercasedBytes() else {
            return nil
        }

        return entries[lowercasedBytes]
    }
}

// MARK: - Hashable

extension BinaryHeaderMap.Entry: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(key)
        hasher.combine(prefix)
        hasher.combine(suffix)
    }

    public static func == (lhs: BinaryHeaderMap.Entry, rhs: BinaryHeaderMap.Entry) -> Bool {
        return lhs.key == rhs.key &&
            lhs.prefix == rhs.prefix &&
            lhs.suffix == rhs.suffix
    }
}

// MARK: - Internal/Private

extension String {
    func clangLowercasedBytes() throws -> Data {
        guard let charBytes = data(using: BinaryHeaderMapEncoder.encoding) else {
            fatalError("Unable to encode string")
        }

        let lowercasedBytes = charBytes.map { $0.asciiLowercased }
        return Data(lowercasedBytes)
    }
}

private extension UInt8 {
    var asciiLowercased: UInt8 {
        let isUppercase = (65 <= self && self <= 90)
        return self + (isUppercase ? 32 : 0)
    }
}

private func makeIndexedEntries(from entries: [BinaryHeaderMap.Entry]) -> BinaryHeaderMap.EntryIndex {
    return Dictionary(
        sanitize(headerEntries: entries).map { (try! $0.key.clangLowercasedBytes(), $0) },
        uniquingKeysWith: { $1 }
    )
}
