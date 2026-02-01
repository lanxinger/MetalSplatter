import Foundation

public protocol UnsafeRawPointerConvertible {
    // MARK: Reading from UnsafeRawPointer

    // Assumes that data size - offset >= byteWidth
    init(_ data: UnsafeRawPointer, from offset: Int, bigEndian: Bool)
    // Assumes that data size >= byteWidth
    init(_ data: UnsafeRawPointer, bigEndian: Bool)

    // Assumes that data size - offset >= count * byteWidth
    static func array(_ data: UnsafeRawPointer, from offset: Int, count: Int, bigEndian: Bool) -> [Self]
    // Assumes that data size >= count * byteWidth
    static func array(_ data: UnsafeRawPointer, count: Int, bigEndian: Bool) -> [Self]

    // MARK: Writing to UnsafeMutableRawPointer

    // Assumes that data size - offset >= byteWidth
    // Returns number of bytes stored
    @discardableResult
    func store(to data: UnsafeMutableRawPointer, at offset: Int, bigEndian: Bool) -> Int
    // Assumes that data size >= byteWidth
    // Returns number of bytes stored
    @discardableResult
    func store(to data: UnsafeMutableRawPointer, bigEndian: Bool) -> Int

    // Assumes that data size - offset >= count * byteWidth
    // Returns number of bytes stored
    @discardableResult
    static func store(_ values: [Self], to data: UnsafeMutableRawPointer, at offset: Int, bigEndian: Bool) -> Int
    // Assumes that data size >= count * byteWidth
    // Returns number of bytes stored
    @discardableResult
    static func store(_ values: [Self], to data: UnsafeMutableRawPointer, bigEndian: Bool) -> Int
}

fileprivate enum UnsafeRawPointerConvertibleConstants {
    fileprivate static let isBigEndian = 42 == 42.bigEndian
}

public extension BinaryInteger where Self: UnsafeRawPointerConvertible, Self: EndianConvertible {
    // MARK: Reading from UnsafeRawPointer

    init(_ data: UnsafeRawPointer, from offset: Int, bigEndian: Bool) {
        let value = (data + offset).loadUnaligned(as: Self.self)
        self = (bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian) ? value : value.byteSwapped
    }

    init(_ data: UnsafeRawPointer, bigEndian: Bool) {
        let value = data.loadUnaligned(as: Self.self)
        self = (bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian) ? value : value.byteSwapped
    }

    static func array(_ data: UnsafeRawPointer, from offset: Int, count: Int, bigEndian: Bool) -> [Self] {
        let size = MemoryLayout<Self>.size
        var values: [Self] = Array(repeating: .zero, count: count)
        if bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian {
            for i in 0..<count {
                values[i] = (data + offset + size*i).loadUnaligned(as: Self.self)
            }
        } else {
            for i in 0..<count {
                values[i] = (data + offset + size*i).loadUnaligned(as: Self.self).byteSwapped
            }
        }
        return values
    }

    static func array(_ data: UnsafeRawPointer, count: Int, bigEndian: Bool) -> [Self] {
        array(data, from: 0, count: count, bigEndian: bigEndian)
    }

    // MARK: Writing to UnsafeMutableRawPointer

    @discardableResult
    func store(to data: UnsafeMutableRawPointer, at offset: Int, bigEndian: Bool) -> Int {
        let value = (bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian) ? self : byteSwapped
        data.storeBytes(of: value, toByteOffset: offset, as: Self.self)
        return Self.byteWidth
    }

    @discardableResult
    func store(to data: UnsafeMutableRawPointer, bigEndian: Bool) -> Int {
        let value = (bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian) ? self : byteSwapped
        data.storeBytes(of: value, as: Self.self)
        return Self.byteWidth
    }

    @discardableResult
    static func store(_ values: [Self], to data: UnsafeMutableRawPointer, at offset: Int, bigEndian: Bool) -> Int {
        guard !values.isEmpty else { return 0 }
        if bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian {
            values.withUnsafeBytes {
                guard let baseAddress = $0.baseAddress else { return }
                (data + offset).copyMemory(from: baseAddress, byteCount: values.count * byteWidth)
            }
        } else {
            for (index, value) in values.enumerated() {
                let byteSwapped = value.byteSwapped
                data.storeBytes(of: byteSwapped, toByteOffset: offset + index * byteWidth, as: Self.self)
            }
        }
        return values.count * byteWidth
    }

    @discardableResult
    static func store(_ values: [Self], to data: UnsafeMutableRawPointer, bigEndian: Bool) -> Int {
        guard !values.isEmpty else { return 0 }
        if bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian {
            values.withUnsafeBytes {
                guard let baseAddress = $0.baseAddress else { return }
                data.copyMemory(from: baseAddress, byteCount: values.count * byteWidth)
            }
        } else {
            for (index, value) in values.enumerated() {
                let byteSwapped = value.byteSwapped
                data.storeBytes(of: byteSwapped, toByteOffset: index * byteWidth, as: Self.self)
            }
        }
        return values.count * byteWidth
    }
}

public extension BinaryFloatingPoint where Self: UnsafeRawPointerConvertible, Self: BitPatternConvertible, Self.BitPattern: EndianConvertible {
    // MARK: Reading from UnsafeRawPointer

    init(_ data: UnsafeRawPointer, from offset: Int, bigEndian: Bool) {
        self = if bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian {
            (data + offset).loadUnaligned(as: Self.self)
        } else {
            Self(bitPattern: (data + offset).loadUnaligned(as: BitPattern.self).byteSwapped)
        }
    }

    init(_ data: UnsafeRawPointer, bigEndian: Bool) {
        self = if bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian {
            data.loadUnaligned(as: Self.self)
        } else {
            Self(bitPattern: data.loadUnaligned(as: BitPattern.self).byteSwapped)
        }
    }

    static func array(_ data: UnsafeRawPointer, from offset: Int, count: Int, bigEndian: Bool) -> [Self] {
        let size = MemoryLayout<Self>.size
        var values: [Self] = Array(repeating: .zero, count: count)
        if bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian {
            for i in 0..<count {
                values[i] = (data + offset + size*i).loadUnaligned(as: Self.self)
            }
        } else {
            for i in 0..<count {
                values[i] = Self(bitPattern: (data + offset + size*i).loadUnaligned(as: BitPattern.self).byteSwapped)
            }
        }
        return values
    }

    static func array(_ data: UnsafeRawPointer, count: Int, bigEndian: Bool) -> [Self] {
        array(data, from: 0, count: count, bigEndian: bigEndian)
    }

    // MARK: Writing to UnsafeMutableRawPointer

    @discardableResult
    func store(to data: UnsafeMutableRawPointer, at offset: Int, bigEndian: Bool) -> Int {
        let value = (bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian) ? bitPattern : bitPattern.byteSwapped
        data.storeBytes(of: value, toByteOffset: offset, as: BitPattern.self)
        return Self.byteWidth
    }

    @discardableResult
    func store(to data: UnsafeMutableRawPointer, bigEndian: Bool) -> Int {
        let value = (bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian) ? bitPattern : bitPattern.byteSwapped
        data.storeBytes(of: value, as: BitPattern.self)
        return Self.byteWidth
    }

    @discardableResult
    static func store(_ values: [Self], to data: UnsafeMutableRawPointer, at offset: Int, bigEndian: Bool) -> Int {
        guard !values.isEmpty else { return 0 }
        if bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian {
            values.withUnsafeBytes {
                guard let baseAddress = $0.baseAddress else { return }
                (data + offset).copyMemory(from: baseAddress, byteCount: values.count * byteWidth)
            }
        } else {
            for (index, value) in values.enumerated() {
                let byteSwapped = value.bitPattern.byteSwapped
                data.storeBytes(of: byteSwapped, toByteOffset: offset + index * byteWidth, as: BitPattern.self)
            }
        }
        return values.count * byteWidth
    }

    @discardableResult
    static func store(_ values: [Self], to data: UnsafeMutableRawPointer, bigEndian: Bool) -> Int {
        guard !values.isEmpty else { return 0 }
        if bigEndian == UnsafeRawPointerConvertibleConstants.isBigEndian {
            values.withUnsafeBytes {
                guard let baseAddress = $0.baseAddress else { return }
                data.copyMemory(from: baseAddress, byteCount: values.count * byteWidth)
            }
        } else {
            for (index, value) in values.enumerated() {
                let byteSwapped = value.bitPattern.byteSwapped
                data.storeBytes(of: byteSwapped, toByteOffset: index * byteWidth, as: BitPattern.self)
            }
        }
        return values.count * byteWidth
    }
}

extension Int8: UnsafeRawPointerConvertible {}
extension UInt8: UnsafeRawPointerConvertible {}
extension Int16: UnsafeRawPointerConvertible {}
extension UInt16: UnsafeRawPointerConvertible {}
extension Int32: UnsafeRawPointerConvertible {}
extension UInt32: UnsafeRawPointerConvertible {}
extension Int64: UnsafeRawPointerConvertible {}
extension UInt64: UnsafeRawPointerConvertible {}
extension Float: UnsafeRawPointerConvertible {}
extension Double: UnsafeRawPointerConvertible {}

// MARK: - Bounds-Checked Array Reading

/// Internal parse error for malformed binary data
internal enum BinaryParseError: Error {
    case invalidData
}

/// Bounds-checked array reading from raw pointer.
/// - Parameters:
///   - type: The element type to read
///   - data: Base pointer to read from
///   - offset: Position within buffer to start reading
///   - count: Number of elements to read
///   - availableBytes: Total valid bytes from buffer start (NOT remaining-from-offset)
///   - bigEndian: Whether to interpret bytes as big-endian
/// - Throws: `BinaryParseError.invalidData` if bounds would be exceeded or overflow occurs
/// - Returns: Array of decoded values
internal func checkedReadArray<T: UnsafeRawPointerConvertible>(
    _ type: T.Type,
    from data: UnsafeRawPointer,
    offset: Int,
    count: Int,
    availableBytes: Int,
    bigEndian: Bool
) throws -> [T] {
    // Guard against negative inputs
    guard count >= 0, offset >= 0, availableBytes >= 0 else {
        throw BinaryParseError.invalidData
    }
    let size = MemoryLayout<T>.size
    // Check for overflow: count * size
    let (totalBytes, overflow1) = count.multipliedReportingOverflow(by: size)
    guard !overflow1 else { throw BinaryParseError.invalidData }
    // Check: offset + totalBytes <= availableBytes
    let (endOffset, overflow2) = offset.addingReportingOverflow(totalBytes)
    guard !overflow2, endOffset <= availableBytes else {
        throw BinaryParseError.invalidData
    }
    // Safe to use unchecked version
    return T.array(data, from: offset, count: count, bigEndian: bigEndian)
}
