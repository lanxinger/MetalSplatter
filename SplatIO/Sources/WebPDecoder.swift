import Foundation
import CoreGraphics
import ImageIO
import libwebp

public struct WebPDecoder {
    public enum WebPError: Error {
        case decodingFailed
        case unsupportedFormat
        case invalidImageData
        case imageTooLarge
    }

    /// Maximum allowed decoded image bytes (platform-based)
    #if os(macOS)
    private static let maxDecodedBytes = 512 * 1024 * 1024  // 512 MB
    #else
    private static let maxDecodedBytes = 256 * 1024 * 1024  // 256 MB (iOS/visionOS)
    #endif
    
    /// Decoded WebP image data
    public struct DecodedImage: Sendable {
        public let pixels: Data
        public let width: Int
        public let height: Int
        public let bytesPerPixel: Int
        
        public init(pixels: Data, width: Int, height: Int, bytesPerPixel: Int) {
            self.pixels = pixels
            self.width = width
            self.height = height
            self.bytesPerPixel = bytesPerPixel
        }
    }
    
    /// Decodes WebP data to straight (non-premultiplied) RGBA8 pixels with libwebp.
    ///
    /// SOG/SOGS textures store data, not colour: indices in RGB and an unrelated value
    /// (opacity, a quaternion tag) in A. Core Graphics can only draw into premultiplied
    /// contexts and may colour-manage the image, both of which corrupt those values.
    public static func decode(_ webpData: Data) throws -> DecodedImage {
        var decodedWidth: Int32 = 0
        var decodedHeight: Int32 = 0
        let hasInfo = webpData.withUnsafeBytes { input -> Bool in
            guard let base = input.bindMemory(to: UInt8.self).baseAddress else { return false }
            return WebPGetInfo(base, input.count, &decodedWidth, &decodedHeight) != 0
        }
        guard hasInfo else {
            throw WebPError.decodingFailed
        }

        let width = Int(decodedWidth)
        let height = Int(decodedHeight)
        let bytesPerPixel = 4
        let totalBytes = try validatedByteCount(width: width, height: height, bytesPerPixel: bytesPerPixel)

        var pixels = Data(count: totalBytes)
        let decodingSucceeded = pixels.withUnsafeMutableBytes { output -> Bool in
            webpData.withUnsafeBytes { input -> Bool in
                guard let inputBase = input.bindMemory(to: UInt8.self).baseAddress,
                      let outputBase = output.bindMemory(to: UInt8.self).baseAddress else {
                    return false
                }
                return WebPDecodeRGBAInto(inputBase, input.count,
                                          outputBase, output.count,
                                          Int32(width * bytesPerPixel)) != nil
            }
        }
        guard decodingSucceeded else {
            throw WebPError.decodingFailed
        }

        return DecodedImage(
            pixels: pixels,
            width: width,
            height: height,
            bytesPerPixel: bytesPerPixel
        )
    }

    private static func validatedByteCount(width: Int, height: Int, bytesPerPixel: Int) throws -> Int {
        guard width > 0 && height > 0 else {
            throw WebPError.invalidImageData
        }
        guard width <= 16384 && height <= 16384 else {
            throw WebPError.invalidImageData  // Reject unreasonably large images
        }
        let totalBytes = width * height * bytesPerPixel
        guard totalBytes <= Self.maxDecodedBytes else {
            throw WebPError.imageTooLarge
        }
        return totalBytes
    }
    
    /// Alternative decoder using ImageIO (fallback method). Lossy for data textures: pixels
    /// pass through a premultiplied context, so RGB precision drops as alpha decreases.
    public static func decodeWithImageIO(_ webpData: Data) throws -> DecodedImage {
        guard let source = CGImageSourceCreateWithData(webpData as CFData, nil) else {
            throw WebPError.decodingFailed
        }

        // Read image properties BEFORE creating CGImage to avoid allocation
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int else {
            throw WebPError.decodingFailed
        }

        // Validate dimensions BEFORE decoding
        guard width > 0 && height > 0 else {
            throw WebPError.invalidImageData
        }
        guard width <= 16384 && height <= 16384 else {
            throw WebPError.invalidImageData  // Reject unreasonably large images
        }

        // Check total allocation size against platform limit
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow
        guard totalBytes <= Self.maxDecodedBytes else {
            throw WebPError.imageTooLarge
        }

        // NOW safe to create CGImage
        guard let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw WebPError.decodingFailed
        }

        var pixels = Data(count: totalBytes)
        var decodingSucceeded = false

        pixels.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }

            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            ) else { return }

            context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

            // Core Graphics only draws premultiplied; DecodedImage holds straight alpha.
            let bytes = ptr.bindMemory(to: UInt8.self)
            for offset in stride(from: 0, to: totalBytes, by: bytesPerPixel) {
                let alpha = bytes[offset + 3]
                guard alpha > 0, alpha < 255 else { continue }
                let scale = 255.0 / Float(alpha)
                for channel in 0..<3 {
                    bytes[offset + channel] = (Float(bytes[offset + channel]) * scale).quantized(clampedTo: 0...255)
                }
            }
            decodingSucceeded = true
        }

        guard decodingSucceeded else {
            throw WebPError.decodingFailed
        }

        return DecodedImage(
            pixels: pixels,
            width: width,
            height: height,
            bytesPerPixel: bytesPerPixel
        )
    }
    
    /// Get pixel value at specific coordinates
    public static func getPixel(from image: DecodedImage, x: Int, y: Int) -> SIMD4<UInt8> {
        guard x >= 0 && x < image.width && y >= 0 && y < image.height else {
            return SIMD4<UInt8>(0, 0, 0, 0)
        }

        // Use checked arithmetic to prevent integer overflow
        let rowOffset = y.multipliedReportingOverflow(by: image.width)
        guard !rowOffset.overflow else { return SIMD4<UInt8>(0, 0, 0, 0) }

        let pixelOffset = rowOffset.partialValue.addingReportingOverflow(x)
        guard !pixelOffset.overflow else { return SIMD4<UInt8>(0, 0, 0, 0) }

        let index = pixelOffset.partialValue.multipliedReportingOverflow(by: image.bytesPerPixel)
        guard !index.overflow else { return SIMD4<UInt8>(0, 0, 0, 0) }

        // Bounds check: ensure we have 4 bytes available at index (RGBA)
        guard index.partialValue >= 0, index.partialValue + 3 < image.pixels.count else {
            return SIMD4<UInt8>(0, 0, 0, 0)
        }

        let base = image.pixels.startIndex + index.partialValue
        return SIMD4<UInt8>(image.pixels[base],
                            image.pixels[base + 1],
                            image.pixels[base + 2],
                            image.pixels[base + 3])
    }
    
    /// Get normalized float pixel value at specific coordinates
    public static func getPixelFloat(from image: DecodedImage, x: Int, y: Int) -> SIMD4<Float> {
        let pixel = getPixel(from: image, x: x, y: y)
        return SIMD4<Float>(
            Float(pixel.x) / 255.0,
            Float(pixel.y) / 255.0,
            Float(pixel.z) / 255.0,
            Float(pixel.w) / 255.0
        )
    }
    
    /// Get raw UInt8 pixel value at specific coordinates (alias for getPixel for clarity)
    public static func getPixelUInt8(from image: DecodedImage, x: Int, y: Int) -> SIMD4<UInt8> {
        return getPixel(from: image, x: x, y: y)
    }
} 
