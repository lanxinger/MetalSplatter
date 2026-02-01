import Foundation

extension PLYElement {
    public enum ASCIIDecodeError: LocalizedError {
        case bodyInvalidStringForPropertyType(PLYHeader.Element, Int, PLYHeader.Property)
        case bodyMissingPropertyValuesInElement(PLYHeader.Element, Int, PLYHeader.Property)
        case bodyUnexpectedValuesInElement(PLYHeader.Element, Int)

        public var errorDescription: String? {
            switch self {
            case .bodyInvalidStringForPropertyType(let headerElement, let elementIndex, let headerProperty):
                "Invalid type string for property \(headerProperty.name) in element \(headerElement.name), index \(elementIndex)"
            case .bodyMissingPropertyValuesInElement(let headerElement, let elementIndex, let headerProperty):
                "Missing values for property \(headerProperty.name) in element \(headerElement.name), index \(elementIndex)"
            case .bodyUnexpectedValuesInElement(let headerElement, let elementIndex):
                "Unexpected values in element \(headerElement.name), index \(elementIndex)"
            }
        }
    }

    // Parse the given element type from the single line from the body of an ASCII PLY file.
    // Considers only bytes from offset..<(offset+size)
    // May modify the body; after this returns, the body contents are undefined.
    mutating func decodeASCII(type elementHeader: PLYHeader.Element,
                              fromMutable body: UnsafeMutablePointer<UInt8>,
                              at offset: Int,
                              bodySize: Int,
                              elementIndex: Int) throws {
        if properties.count != elementHeader.properties.count {
            properties = Array(repeating: .uint8(0), count: elementHeader.properties.count)
        }

        var stringParser = UnsafeStringParser(data: body, offset: offset, size: bodySize)

        for (i, propertyHeader) in elementHeader.properties.enumerated() {
            switch propertyHeader.type {
            case .primitive(let primitiveType):
                // Use direct parsing to avoid String allocation
                guard let value = Property.tryDecodeASCIIPrimitive(type: primitiveType, from: &stringParser) else {
                    throw ASCIIDecodeError.bodyInvalidStringForPropertyType(elementHeader, elementIndex, propertyHeader)
                }
                properties[i] = value
            case .list(countType: let countType, valueType: let valueType):
                // Use direct parsing for list count to avoid String allocation
                guard let count = Property.tryDecodeASCIIPrimitive(type: countType, from: &stringParser)?.uint64Value else {
                    throw ASCIIDecodeError.bodyInvalidStringForPropertyType(elementHeader, elementIndex, propertyHeader)
                }
                // Validate count is reasonable (match binary 100M limit) and fits in Int
                guard count <= 100_000_000, count <= UInt64(Int.max) else {
                    throw ASCIIDecodeError.bodyInvalidStringForPropertyType(elementHeader, elementIndex, propertyHeader)
                }

                properties[i] = try PLYElement.Property.tryDecodeASCIIList(valueType: valueType,
                                                                           count: Int(count),
                                                                           from: &stringParser,
                                                                           elementHeader: elementHeader,
                                                                           elementIndex: elementIndex,
                                                                           propertyHeader: propertyHeader)
            }
        }

        guard stringParser.nextStringSeparatedByWhitespace() == nil else {
            throw ASCIIDecodeError.bodyUnexpectedValuesInElement(elementHeader, elementIndex)
        }
    }
}

fileprivate extension PLYElement.Property {
    // Direct parsing from UnsafeStringParser (zero allocation for common types)
    static func tryDecodeASCIIPrimitive(type: PLYHeader.PrimitivePropertyType,
                                        from parser: inout UnsafeStringParser) -> PLYElement.Property? {
        switch type {
        case .int8:
            guard let value = parser.nextInt64(), value >= Int8.min && value <= Int8.max else { return nil }
            return .int8(Int8(value))
        case .uint8:
            guard let value = parser.nextUInt64(), value <= UInt8.max else { return nil }
            return .uint8(UInt8(value))
        case .int16:
            guard let value = parser.nextInt64(), value >= Int16.min && value <= Int16.max else { return nil }
            return .int16(Int16(value))
        case .uint16:
            guard let value = parser.nextUInt64(), value <= UInt16.max else { return nil }
            return .uint16(UInt16(value))
        case .int32:
            guard let value = parser.nextInt64(), value >= Int32.min && value <= Int32.max else { return nil }
            return .int32(Int32(value))
        case .uint32:
            guard let value = parser.nextUInt64(), value <= UInt32.max else { return nil }
            return .uint32(UInt32(value))
        case .float32:
            guard let value = parser.nextFloat32() else { return nil }
            return .float32(value)
        case .float64:
            guard let value = parser.nextFloat64() else { return nil }
            return .float64(value)
        }
    }

    // String-based parsing (legacy, for compatibility)
    static func tryDecodeASCIIPrimitive(type: PLYHeader.PrimitivePropertyType,
                                        from string: String) -> PLYElement.Property? {
        switch type {
        case .int8   : if let value = Int8(  string) { .int8(   value) } else { nil }
        case .uint8  : if let value = UInt8( string) { .uint8(  value) } else { nil }
        case .int16  : if let value = Int16( string) { .int16(  value) } else { nil }
        case .uint16 : if let value = UInt16(string) { .uint16( value) } else { nil }
        case .int32  : if let value = Int32( string) { .int32(  value) } else { nil }
        case .uint32 : if let value = UInt32(string) { .uint32( value) } else { nil }
        case .float32: if let value = Float( string) { .float32(value) } else { nil }
        case .float64: if let value = Double(string) { .float64(value) } else { nil }
        }
    }

    static func tryDecodeASCIIList(valueType: PLYHeader.PrimitivePropertyType,
                                   from strings: [String]) -> PLYElement.Property? {
        switch valueType {
        case .int8:
            let values = strings.compactMap { Int8($0) }
            return values.count == strings.count ? .listInt8(values) : nil
        case .uint8:
            let values = strings.compactMap { UInt8($0) }
            return values.count == strings.count ? .listUInt8(values) : nil
        case .int16:
            let values = strings.compactMap { Int16($0) }
            return values.count == strings.count ? .listInt16(values) : nil
        case .uint16:
            let values = strings.compactMap { UInt16($0) }
            return values.count == strings.count ? .listUInt16(values) : nil
        case .int32:
            let values = strings.compactMap { Int32($0) }
            return values.count == strings.count ? .listInt32(values) : nil
        case .uint32:
            let values = strings.compactMap { UInt32($0) }
            return values.count == strings.count ? .listUInt32(values) : nil
        case .float32:
            let values = strings.compactMap { Float($0) }
            return values.count == strings.count ? .listFloat32(values) : nil
        case .float64:
            let values = strings.compactMap { Double($0) }
            return values.count == strings.count ? .listFloat64(values) : nil
        }
    }

    static func tryDecodeASCIIList(valueType: PLYHeader.PrimitivePropertyType,
                                   count: Int,
                                   from stringParser: inout UnsafeStringParser,
                                   elementHeader: PLYHeader.Element,
                                   elementIndex: Int,
                                   propertyHeader: PLYHeader.Property) throws -> PLYElement.Property {
        do {
            switch valueType {
            case .int8:
                return .listInt8(try (0..<count).map { _ in try stringParser.assumeNextElementSeparatedByWhitespace() })
            case .uint8:
                return .listUInt8(try (0..<count).map { _ in try stringParser.assumeNextElementSeparatedByWhitespace() })
            case .int16:
                return .listInt16(try (0..<count).map { _ in try stringParser.assumeNextElementSeparatedByWhitespace() })
            case .uint16:
                return .listUInt16(try (0..<count).map { _ in try stringParser.assumeNextElementSeparatedByWhitespace() })
            case .int32:
                return .listInt32(try (0..<count).map { _ in try stringParser.assumeNextElementSeparatedByWhitespace() })
            case .uint32:
                return .listUInt32(try (0..<count).map { _ in try stringParser.assumeNextElementSeparatedByWhitespace() })
            case .float32:
                return .listFloat32(try (0..<count).map { _ in try stringParser.assumeNextElementSeparatedByWhitespace() })
            case .float64:
                return .listFloat64(try (0..<count).map { _ in try stringParser.assumeNextElementSeparatedByWhitespace() })
            }
        } catch UnsafeStringParser.Error.invalidFormat {
            throw PLYElement.ASCIIDecodeError.bodyInvalidStringForPropertyType(elementHeader, elementIndex, propertyHeader)
        } catch UnsafeStringParser.Error.unexpectedEndOfData {
            throw PLYElement.ASCIIDecodeError.bodyMissingPropertyValuesInElement(elementHeader, elementIndex, propertyHeader)
        }
    }
}

extension PLYElement: CustomStringConvertible {
    public var description: String {
        properties.map(\.description).joined(separator: " ")
    }
}

extension PLYElement.Property: CustomStringConvertible {
    public var description: String {
        if let listCount, listCount == 0 {
            return "0"
        }
        return switch self {
        case .int8(       let value ): "\(value)"
        case .uint8(      let value ): "\(value)"
        case .int16(      let value ): "\(value)"
        case .uint16(     let value ): "\(value)"
        case .int32(      let value ): "\(value)"
        case .uint32(     let value ): "\(value)"
        case .float32(    let value ): "\(value)"
        case .float64(    let value ): "\(value)"
        case .listInt8(   let values): "\(values.count) \(values.map(\.description).joined(separator: " "))"
        case .listUInt8(  let values): "\(values.count) \(values.map(\.description).joined(separator: " "))"
        case .listInt16(  let values): "\(values.count) \(values.map(\.description).joined(separator: " "))"
        case .listUInt16( let values): "\(values.count) \(values.map(\.description).joined(separator: " "))"
        case .listInt32(  let values): "\(values.count) \(values.map(\.description).joined(separator: " "))"
        case .listUInt32( let values): "\(values.count) \(values.map(\.description).joined(separator: " "))"
        case .listFloat32(let values): "\(values.count) \(values.map(\.description).joined(separator: " "))"
        case .listFloat64(let values): "\(values.count) \(values.map(\.description).joined(separator: " "))"
        }
    }
}

fileprivate struct UnsafeStringParser {
    enum Error: Swift.Error {
        case invalidFormat(String)
        case unexpectedEndOfData
    }

    var data: UnsafeMutablePointer<UInt8>
    var offset: Int
    var size: Int
    var currentPosition = 0

    /// Find the next token boundaries without allocating.
    ///
    /// **Safety**: All pointer accesses are guarded by `start < size` or `end < size` checks,
    /// ensuring we never read beyond the buffer. The `offset` base is validated
    /// by PLYReader's body slice management.
    private mutating func findNextTokenBounds() -> (start: Int, end: Int)? {
        var start = currentPosition
        var end = start

        // Skip leading whitespace
        while start < size {
            let byte = (data + offset + start).pointee
            if byte != PLYReader.Constants.space && byte != 0 {
                break
            }
            start += 1
        }

        guard start < size else { return nil }
        end = start

        // Find end of token
        while end < size {
            let byte = (data + offset + end).pointee
            if byte == PLYReader.Constants.space || byte == 0 {
                break
            }
            end += 1
        }

        return start < end ? (start, end) : nil
    }

    // Temporarily null-terminate and execute closure
    // Handles edge case where end == size (last token without trailing delimiter)
    private func withNullTerminatedToken<T>(start: Int, end: Int, _ body: (UnsafePointer<CChar>) -> T) -> T {
        // When end < size, we can safely do in-place null termination
        if end < size {
            let savedByte = (data + offset + end).pointee
            (data + offset + end).pointee = 0
            defer { (data + offset + end).pointee = savedByte }

            return (data + offset + start).withMemoryRebound(to: CChar.self, capacity: end - start + 1) { charPtr in
                body(charPtr)
            }
        } else {
            // end == size: we're at the buffer boundary, must copy to temporary buffer
            let tokenLength = end - start
            return withUnsafeTemporaryAllocation(of: CChar.self, capacity: tokenLength + 1) { tempBuffer in
                // Copy token bytes
                for i in 0..<tokenLength {
                    tempBuffer[i] = CChar(bitPattern: (data + offset + start + i).pointee)
                }
                tempBuffer[tokenLength] = 0  // Null terminate
                return body(tempBuffer.baseAddress!)
            }
        }
    }

    // MARK: - Direct numeric parsing (zero allocation)

    mutating func nextFloat32() -> Float? {
        guard let (start, end) = findNextTokenBounds() else { return nil }

        let result = withNullTerminatedToken(start: start, end: end) { charPtr -> Float? in
            var endPtr: UnsafeMutablePointer<CChar>?
            let value = strtof(charPtr, &endPtr)
            // Verify entire token was consumed (endPtr should point to null terminator)
            guard let endPtr, endPtr.pointee == 0 else { return nil }
            return value
        }

        currentPosition = end + 1
        return result
    }

    mutating func nextFloat64() -> Double? {
        guard let (start, end) = findNextTokenBounds() else { return nil }

        let result = withNullTerminatedToken(start: start, end: end) { charPtr -> Double? in
            var endPtr: UnsafeMutablePointer<CChar>?
            let value = strtod(charPtr, &endPtr)
            guard let endPtr, endPtr.pointee == 0 else { return nil }
            return value
        }

        currentPosition = end + 1
        return result
    }

    mutating func nextInt64() -> Int64? {
        guard let (start, end) = findNextTokenBounds() else { return nil }

        let result = withNullTerminatedToken(start: start, end: end) { charPtr -> Int64? in
            var endPtr: UnsafeMutablePointer<CChar>?
            let value = strtoll(charPtr, &endPtr, 10)
            guard let endPtr, endPtr.pointee == 0 else { return nil }
            return value
        }

        currentPosition = end + 1
        return result
    }

    mutating func nextUInt64() -> UInt64? {
        guard let (start, end) = findNextTokenBounds() else { return nil }

        let result = withNullTerminatedToken(start: start, end: end) { charPtr -> UInt64? in
            var endPtr: UnsafeMutablePointer<CChar>?
            let value = strtoull(charPtr, &endPtr, 10)
            guard let endPtr, endPtr.pointee == 0 else { return nil }
            return value
        }

        currentPosition = end + 1
        return result
    }

    // MARK: - String-based parsing (fallback, allocates)

    mutating func nextStringSeparatedByWhitespace() -> String? {
        guard let (start, end) = findNextTokenBounds() else { return nil }

        let result = withNullTerminatedToken(start: start, end: end) { charPtr in
            String(cString: charPtr)
        }

        currentPosition = end + 1
        return result
    }

    mutating func nextElementSeparatedByWhitespace<T: LosslessStringConvertible>() throws -> T? {
        guard let s = nextStringSeparatedByWhitespace() else { return nil }
        guard let result = T(s) else {
            throw Error.invalidFormat(s)
        }
        return result
    }

    mutating func assumeNextElementSeparatedByWhitespace<T: LosslessStringConvertible>() throws -> T {
        guard let result: T = try nextElementSeparatedByWhitespace() else {
            throw Error.unexpectedEndOfData
        }
        return result
    }
}

fileprivate func min<T>(_ x: T?, _ y: T?) -> T? where T : Comparable {
    guard let x else { return y }
    guard let y else { return x }
    return min(x, y)
}
