import Foundation
import CoreImage
import CoreGraphics

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
    }
    
    /// Decoded WebP image data
    public struct DecodedImage {
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
        
        // Create a bitmap context for RGBA8 pixel data
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow
        
        var pixels = Data(count: totalBytes)
        
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
        guard let source = CGImageSourceCreateWithData(webpData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw WebPError.decodingFailed
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let totalBytes = height * bytesPerRow
        
        var pixels = Data(count: totalBytes)
        
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
        
        let index = (y * image.width + x) * image.bytesPerPixel
        let r = image.pixels[index]
        let g = image.pixels[index + 1]
        let b = image.pixels[index + 2]
        let a = image.pixels[index + 3]
        
        return SIMD4<UInt8>(r, g, b, a)
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
} 