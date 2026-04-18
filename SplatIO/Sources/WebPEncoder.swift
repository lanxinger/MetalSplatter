import Foundation
import libwebp

enum WebPEncoder {
    enum Error: Swift.Error {
        case invalidPixelCount(expected: Int, actual: Int)
        case encodingFailed
    }

    static func encodeLosslessRGBA(_ pixels: Data, width: Int, height: Int) throws -> Data {
        let expectedByteCount = width * height * 4
        guard pixels.count == expectedByteCount else {
            throw Error.invalidPixelCount(expected: expectedByteCount, actual: pixels.count)
        }

        var outputPointer: UnsafeMutablePointer<UInt8>?
        let outputSize = pixels.withUnsafeBytes { rawBuffer -> Int in
            guard let baseAddress = rawBuffer.bindMemory(to: UInt8.self).baseAddress else {
                return 0
            }

            return Int(WebPEncodeLosslessRGBA(
                baseAddress,
                Int32(width),
                Int32(height),
                Int32(width * 4),
                &outputPointer
            ))
        }

        guard outputSize > 0, let outputPointer else {
            throw Error.encodingFailed
        }
        defer {
            WebPFree(outputPointer)
        }

        return Data(bytes: outputPointer, count: outputSize)
    }
}
