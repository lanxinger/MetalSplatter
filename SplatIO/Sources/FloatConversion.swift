import Foundation
import simd

#if canImport(Metal)
import Metal
#endif

/**
 * Utility class for high-performance float16 conversions
 * Uses software implementation for compatibility
 */
enum FloatConversion {
    
    /**
     * Converts a single half-precision float (UInt16) to a 32-bit float
     * Uses software implementation for cross-platform compatibility
     */
    static func float16ToFloat32(_ half: UInt16) -> Float {
        // Software implementation for cross-platform compatibility
        let sign = (half & 0x8000) != 0
        let exponent = Int((half & 0x7C00) >> 10)
        let mantissa = Int(half & 0x03FF)
        
        let signMul: Float = sign ? -1.0 : 1.0
        
        if exponent == 0 {
            // Zero or denormalized
            if mantissa == 0 {
                return 0.0 * signMul
            }
            
            // Denormalized
            return signMul * pow(2.0, -14.0) * (Float(mantissa) / 1024.0)
        }
        
        if exponent == 31 {
            // Infinity or NaN
            return mantissa != 0 ? Float.nan : Float.infinity * signMul
        }
        
        // Normalized
        return signMul * pow(2.0, Float(exponent - 15)) * (1.0 + Float(mantissa) / 1024.0)
    }
    
    /**
     * Batch convert an array of half-precision floats to 32-bit floats
     */
    static func convertFloat16ArrayToFloat32(_ float16Data: [UInt8]) -> [Float] {
        // Handle empty or odd-sized arrays
        if float16Data.count < 2 {
            return []
        }
        
        let halfCount = float16Data.count / 2
        var result = [Float](repeating: 0, count: halfCount)
        
        // Process in chunks for better performance
        let chunkSize = 64 // Process values in chunks for cache locality
        
        for chunkStart in stride(from: 0, to: halfCount, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, halfCount)
            
            // Process each element in the chunk
            for i in chunkStart..<chunkEnd {
                let byteOffset = i * 2
                let halfValue = UInt16(float16Data[byteOffset]) | 
                               (UInt16(float16Data[byteOffset + 1]) << 8)
                result[i] = float16ToFloat32(halfValue)
            }
        }
        
        return result
    }
    
    /**
     * Convert a batch of 3-component float16 position vectors to float32
     * Optimized for SPZ file format handling
     */
    static func convertFloat16PositionsToFloat32(_ float16Data: [UInt8], count: Int) -> [SIMD3<Float>] {
        // Check for valid input
        if float16Data.count < 6 || count < 1 {
            return []
        }
        
        // Limit count to available data
        let maxPositionCount = float16Data.count / 6 // Each position is 3 components × 2 bytes
        let safeCount = min(count, maxPositionCount)
        
        var result = [SIMD3<Float>](repeating: SIMD3<Float>(0, 0, 0), count: safeCount)
        
        // Process in chunks for better memory locality
        let chunkSize = 64 // Process positions in chunks
        
        for chunkStart in stride(from: 0, to: safeCount, by: chunkSize) {
            let chunkEnd = min(chunkStart + chunkSize, safeCount)
            
            // Process each position in the chunk
            for i in chunkStart..<chunkEnd {
                let byteOffset = i * 6 // 6 bytes per position (3 components × 2 bytes)
                
                // Extract x,y,z components
                var position = SIMD3<Float>(0, 0, 0)
                for j in 0..<3 {
                    let componentOffset = byteOffset + j * 2
                    if componentOffset + 1 < float16Data.count {
                        let halfValue = UInt16(float16Data[componentOffset]) | 
                                       (UInt16(float16Data[componentOffset + 1]) << 8)
                        position[j] = float16ToFloat32(halfValue)
                    }
                }
                
                result[i] = position
            }
        }
        
        return result
    }
}
