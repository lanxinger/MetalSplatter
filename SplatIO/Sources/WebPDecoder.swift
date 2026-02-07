import Foundation
import CoreImage
import CoreGraphics
import ImageIO

#if canImport(UIKit)
import UIKit
#endif

#if canImport(AppKit)
import AppKit
#endif

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
    
    /// Decode WebP data to RGBA pixel data using Core Image
    /// This requires iOS 14+/macOS 11+ for WebP support
    public static func decode(_ webpData: Data) throws -> DecodedImage {
        guard let ciImage = CIImage(data: webpData) else {
            throw WebPError.decodingFailed
        }

        let context = CIContext()
        let width = Int(ciImage.extent.width)
        let height = Int(ciImage.extent.height)

        // Validate image dimensions to prevent excessive memory allocation
        guard width > 0 && height > 0 else {
            throw WebPError.invalidImageData
        }
        guard width <= 16384 && height <= 16384 else {
            throw WebPError.invalidImageData  // Reject unreasonably large images
        }

        // Create a bitmap context for RGBA8 pixel data
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow

        // Check total allocation size against platform limit
        guard totalBytes <= Self.maxDecodedBytes else {
            throw WebPError.imageTooLarge
        }

        var pixels = Data(count: totalBytes)
        var decodingSucceeded = false

        pixels.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return }

            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

            guard let cgContext = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo.rawValue
            ) else { return }

            // Render the CIImage to a CGImage first, then draw to the context
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
            cgContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
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
    
    /// Alternative decoder using ImageIO (fallback method)
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

        let r = image.pixels[index.partialValue]
        let g = image.pixels[index.partialValue + 1]
        let b = image.pixels[index.partialValue + 2]
        let a = image.pixels[index.partialValue + 3]

        // Un-premultiply alpha to get original color values
        // Since CGContext uses premultiplied alpha, we need to divide by alpha to get original colors
        if a > 0 {
            let alpha = Float(a) / 255.0
            let unpremultipliedR = UInt8(min(255.0, Float(r) / alpha))
            let unpremultipliedG = UInt8(min(255.0, Float(g) / alpha))
            let unpremultipliedB = UInt8(min(255.0, Float(b) / alpha))
            return SIMD4<UInt8>(unpremultipliedR, unpremultipliedG, unpremultipliedB, a)
        } else {
            return SIMD4<UInt8>(r, g, b, a)
        }
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
