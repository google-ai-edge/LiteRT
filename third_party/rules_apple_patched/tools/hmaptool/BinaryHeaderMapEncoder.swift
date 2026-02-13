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

public enum BinaryHeaderMapEncoder {
    static let encoding: String.Encoding = .utf8

    public static func encode(_ headerMap: BinaryHeaderMap) throws -> Data {
        let entries: [BinaryHeaderMap.Entry] = headerMap.entries.map { $0.value }.sorted { $0.key < $1.key }
        return try makeHeaderMapBinaryData(withEntries: entries)
    }
}

private func makeHeaderMapBinaryData(withEntries unsafeEntries: [BinaryHeaderMap.Entry]) throws -> Data {
    let safeEntries = sanitize(headerEntries: unsafeEntries)
    let allStrings = Set(safeEntries.flatMap { [$0.key, $0.prefix, $0.suffix] }).sorted()
    let stringSection = try makeStringSection(allStrings: allStrings)
    let bucketSection = try makeBucketSection(
        forEntries: safeEntries,
        stringSection: stringSection
    )

    let maxValueLength = safeEntries
        .max(by: { lhs, rhs in lhs.valueLength < rhs.valueLength })
        .map { $0.valueLength } ?? 0

    let headerSize = DataHeader.packedSize
    let stringSectionOffset = headerSize + bucketSection.data.count

    let header = DataHeader(
        magic: .hmap,
        version: .version1,
        reserved: .none,
        stringSectionOffset: UInt32(stringSectionOffset),
        stringCount: UInt32(stringSection.stringCount),
        bucketCount: UInt32(bucketSection.bucketCount),
        maxValueLength: UInt32(maxValueLength)
    )

    let encoder = ByteBufferEncoder()
    encoder.encode(header.magic.rawValue)
    encoder.encode(header.version.rawValue)
    encoder.encode(header.reserved.rawValue)
    encoder.encode(header.stringSectionOffset)
    encoder.encode(header.stringCount)
    encoder.encode(header.bucketCount)
    encoder.encode(header.maxValueLength)
    encoder.append(bucketSection.data)
    encoder.append(stringSection.data)

    return encoder.bytes
}

private protocol Packable {
    static var packedSize: Int { get }
}

private struct DataHeader: Packable {
    typealias MagicType = Magic
    typealias VersionType = Version
    typealias ReservedType = Reserved
    typealias StringSectionOffsetType = UInt32
    typealias StringCountType = UInt32
    typealias BucketCountType = UInt32
    typealias MaxValueLengthType = UInt32

    let magic: MagicType
    let version: VersionType
    let reserved: ReservedType
    let stringSectionOffset: StringSectionOffsetType
    let stringCount: StringCountType
    let bucketCount: BucketCountType // Must be power of 2
    let maxValueLength: MaxValueLengthType

    static var packedSize: Int {
        MemoryLayout<Magic.RawValue>.size +
            MemoryLayout<Version.RawValue>.size +
            MemoryLayout<Reserved.RawValue>.size +
            MemoryLayout<UInt32>.size +
            MemoryLayout<UInt32>.size +
            MemoryLayout<UInt32>.size +
            MemoryLayout<UInt32>.size
    }

    static func headerPlusBucketsSize(bucketCount: DataHeader.BucketCountType) -> Int {
        let bucketsSectionSize = Int(bucketCount) * Bucket.packedSize
        return packedSize + bucketsSectionSize
    }
}

private enum Magic: UInt32 {
    case hmap = 0x68_6D_61_70 // 'hmap'
}

private enum Version: UInt16 {
    case version1 = 1
}

private enum Reserved: UInt16 {
    case none = 0
}

private struct Bucket: Packable {
    typealias OffsetType = StringSectionOffset

    let key: OffsetType
    let prefix: OffsetType
    let suffix: OffsetType

    static var packedSize: Int {
        OffsetType.packedSize * 3
    }
}

private struct StringSectionOffset: Packable {
    typealias OffsetType = UInt32

    /** Indicates an invalid offset */
    static let reserved = OffsetType(0)

    let offset: OffsetType

    init?(offset: OffsetType) {
        if offset == StringSectionOffset.reserved {
            // The first byte is reserved and means 'empty'.
            return nil
        }

        self.offset = offset
    }

    static var packedSize: Int {
        return MemoryLayout<OffsetType>.size
    }
}

private struct BucketSection {
    let data: Data
    let bucketCount: Int
}

private struct StringSection {
    let data: Data
    let stringCount: Int
    let offsets: [String: StringSectionOffset]
}

private func makeStringSection(allStrings: [String]) throws -> StringSection {

    var buffer = Data()
    var offsets: [String: StringSectionOffset] = [:]

    if !allStrings.isEmpty {
        buffer.append(UInt8(StringSectionOffset.reserved))
    }

    for string in allStrings {
        guard let stringBytes = string.data(using: BinaryHeaderMapEncoder.encoding) else {
            fatalError("Unable to get data in proper encoding")
        }

        let bufferCount = UInt32(buffer.count)
        let offset = StringSectionOffset(offset: bufferCount)!

        offsets[string] = offset
        buffer.append(stringBytes)
        buffer.append(0x0) // NUL character
    }

    return StringSection(
        data: buffer,
        stringCount: allStrings.count,
        offsets: offsets
    )
}

private func isBucketSlotEmpty(
    bytePointer: UnsafeMutablePointer<UInt8>,
    index: UInt32
) -> Bool {

    let slotOffset = Int(index) * Bucket.packedSize
    let slotPointer = bytePointer.advanced(by: slotOffset)
    return slotPointer.withMemoryRebound(
        to: StringSectionOffset.OffsetType.self,
        capacity: 1
    ) {
        return $0.pointee == StringSectionOffset.reserved
    }
}

private func writeTo<T>(bytePointer: UnsafeMutablePointer<UInt8>, value: T) {
    var mutableValue = value
    withUnsafeBytes(of: &mutableValue) { (pointer) in
        let valueBytes = Array(pointer[0..<pointer.count])
        for (index, byte) in valueBytes.enumerated() {
            bytePointer.advanced(by: index).pointee = byte
        }
    }
}

private func makeBucketSection(
    forEntries entries: [BinaryHeaderMap.Entry],
    stringSection: StringSection
) throws -> BucketSection {

    let bucketCount = numberOfBuckets(forEntryCount: entries.count)
    var bytes = Data(count: bucketCount * Bucket.packedSize)
    bytes.withUnsafeMutableBytes { (rawBytes: UnsafeMutableRawBufferPointer) in

        guard let rawBasePointer = rawBytes.baseAddress else {
            fatalError("Empty data buffer")
        }

        let bytePointer = rawBasePointer.bindMemory(to: UInt8.self, capacity: rawBytes.count)

        for entry in entries {
            guard let keyHash = entry.key.headerMapHash else {
                fatalError("Unable to hash key")
            }

            guard
                let keyOffset = stringSection.offsets[entry.key],
                let prefixOffset = stringSection.offsets[entry.prefix],
                let suffixOffset = stringSection.offsets[entry.suffix]
            else {
                fatalError("Unable to find string offset")
            }

            let maybeEmptySlotIndex: UInt32? = probeHashTable(
                bucketCount: UInt32(bucketCount),
                start: keyHash
            ) { (probeIndex) in
                if isBucketSlotEmpty(bytePointer: bytePointer, index: probeIndex) {
                    return .stop(value: probeIndex)
                }
                return .keepProbing
            }

            guard let emptySlotIndex = maybeEmptySlotIndex else {
                fatalError("Unable to find empty slot in hash table")
            }

            let slotPointer = bytePointer.advanced(by: Int(emptySlotIndex) * Bucket.packedSize)
            let fieldSize = MemoryLayout<Bucket.OffsetType>.size

            writeTo(bytePointer: slotPointer, value: keyOffset.offset)
            writeTo(bytePointer: slotPointer.advanced(by: fieldSize), value: prefixOffset.offset)
            writeTo(bytePointer: slotPointer.advanced(by: 2 * fieldSize), value: suffixOffset.offset)
        }
    }

    return BucketSection(
        data: bytes,
        bucketCount: bucketCount
    )
}

private func numberOfBuckets(forEntryCount entryCount: Int) -> Int {
    let minimumSlots = Int(ceil(Double(entryCount) * (1.0 / 0.7)))
    var bucketCount = 1
    while bucketCount < minimumSlots {
        bucketCount <<= 1
    }
    return bucketCount
}

func sanitize(headerEntries entries: [BinaryHeaderMap.Entry]) -> [BinaryHeaderMap.Entry] {
    var allKeys = Set<String>()
    return entries.compactMap { (entry) in
        guard !allKeys.contains(entry.key) else { return nil }
        allKeys.insert(entry.key)
        return entry
    }
}

private extension BinaryHeaderMap.Entry {
    var valueLength: Int {
        prefix.lengthOfBytes(using: BinaryHeaderMapEncoder.encoding) +
            suffix.lengthOfBytes(using: BinaryHeaderMapEncoder.encoding)
    }
}

private extension String {
    var headerMapHash: UInt32? {
        guard
            let characterBytes = try? clangLowercasedBytes()
        else { return nil }

        return characterBytes.withUnsafeBytes { (charBytes: UnsafeRawBufferPointer) -> UInt32 in
            var result = UInt32(0)
            for i in 0 ..< charBytes.count {
                result += UInt32(charBytes[i]) * 13
            }
            return result
        }
    }
}

private enum ProbeAction<T> {
    case keepProbing
    case stop(value: T?)
}

private func probeHashTable<T>(
    bucketCount: DataHeader.BucketCountType,
    start: UInt32,
    _ body: (UInt32) -> ProbeAction<T>
) -> T? {
    var probeAttempts = DataHeader.BucketCountType(0)
    var nextUnboundedBucketIndex = start
    while probeAttempts < bucketCount {
        let nextBucketIndex = nextUnboundedBucketIndex & (bucketCount - 1)
        switch body(nextBucketIndex) {
        case .keepProbing:
            break
        case .stop(let value):
            return value
        }
        nextUnboundedBucketIndex += 1
        probeAttempts += 1
    }

    return nil
}

private final class ByteBufferEncoder {
    private(set) var bytes = Data()

    func encode<T>(_ value: T) {
        var mutableValue = value
        withUnsafeBytes(of: &mutableValue) { (pointer) in
            let valueBytes = Array(pointer[0..<pointer.count])
            bytes.append(contentsOf: valueBytes)
        }
    }

    func append(_ data: Data) {
        bytes.append(data)
    }
}
