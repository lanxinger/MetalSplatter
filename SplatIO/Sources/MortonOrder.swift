import Foundation
import simd

/// Morton code (Z-order curve) utilities for spatial ordering of Gaussian splats.
///
/// Morton ordering clusters spatially nearby 3D points together in memory,
/// improving GPU cache coherency during rendering.
///
/// Reference: https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
public enum MortonOrder {

    /// Computes the axis-aligned bounding box for an array of splat points.
    /// - Parameter points: Array of splat scene points
    /// - Returns: Tuple of (minimum bounds, maximum bounds)
    public static func computeBounds(_ points: [SplatScenePoint]) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        guard !points.isEmpty else {
            return (SIMD3<Float>.zero, SIMD3<Float>.zero)
        }

        var minBounds = SIMD3<Float>(repeating: .infinity)
        var maxBounds = SIMD3<Float>(repeating: -.infinity)

        for point in points {
            minBounds = simd_min(minBounds, point.position)
            maxBounds = simd_max(maxBounds, point.position)
        }

        return (minBounds, maxBounds)
    }

    /// Expands a 10-bit integer into 30 bits by inserting 2 zeros between each bit.
    /// This is the "magic number" method for Morton code encoding.
    ///
    /// Input:  ---- ---- ---- ---- ---- --98 7654 3210
    /// Output: --9- -8-- 7--6 --5- -4-- 3--2 --1- -0--
    @inline(__always)
    private static func expandBits(_ v: UInt32) -> UInt32 {
        var x = v & 0x3FF  // Mask to 10 bits
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8))  & 0x0300F00F
        x = (x | (x << 4))  & 0x030C30C3
        x = (x | (x << 2))  & 0x09249249
        return x
    }

    /// Encodes a 3D position (each component 10-bit) into a 30-bit Morton code.
    /// - Parameters:
    ///   - x: X component (0-1023)
    ///   - y: Y component (0-1023)
    ///   - z: Z component (0-1023)
    /// - Returns: 30-bit Morton code with interleaved bits: ...zyx zyx zyx
    @inline(__always)
    public static func encode(_ x: UInt32, _ y: UInt32, _ z: UInt32) -> UInt32 {
        expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2)
    }

    /// Computes Morton codes for an array of splat points.
    /// - Parameters:
    ///   - points: Array of splat scene points
    ///   - bounds: Optional pre-computed bounds. If nil, bounds will be computed.
    /// - Returns: Array of Morton codes, one per point
    public static func computeMortonCodes(
        _ points: [SplatScenePoint],
        bounds: (min: SIMD3<Float>, max: SIMD3<Float>)? = nil
    ) -> [UInt32] {
        guard !points.isEmpty else { return [] }

        let (minBounds, maxBounds) = bounds ?? computeBounds(points)
        let size = maxBounds - minBounds

        // Prevent division by zero for degenerate cases (all points on a plane/line)
        let invSize = SIMD3<Float>(
            size.x > 0 ? 1.0 / size.x : 0,
            size.y > 0 ? 1.0 / size.y : 0,
            size.z > 0 ? 1.0 / size.z : 0
        )

        var codes = [UInt32](repeating: 0, count: points.count)

        for i in 0..<points.count {
            let pos = points[i].position
            let normalized = (pos - minBounds) * invSize

            // Quantize to 10-bit integers (0-1023)
            let qx = UInt32(min(max(normalized.x * 1023, 0), 1023))
            let qy = UInt32(min(max(normalized.y * 1023, 0), 1023))
            let qz = UInt32(min(max(normalized.z * 1023, 0), 1023))

            codes[i] = encode(qx, qy, qz)
        }

        return codes
    }

    /// Computes the reordering indices that would sort points by Morton code.
    /// - Parameters:
    ///   - points: Array of splat scene points
    ///   - bounds: Optional pre-computed bounds
    /// - Returns: Array of indices representing the Morton-ordered permutation
    public static func computeReorderingIndices(
        _ points: [SplatScenePoint],
        bounds: (min: SIMD3<Float>, max: SIMD3<Float>)? = nil
    ) -> [Int] {
        guard !points.isEmpty else { return [] }

        let codes = computeMortonCodes(points, bounds: bounds)

        // Create index array and sort by Morton code
        var indices = Array(0..<points.count)
        indices.sort { codes[$0] < codes[$1] }

        return indices
    }

    /// Reorders an array of splat points according to Morton code ordering.
    /// This improves GPU cache coherency by clustering spatially nearby points in memory.
    /// - Parameter points: Array of splat scene points to reorder
    /// - Returns: New array with points reordered by Morton code
    public static func reorder(_ points: [SplatScenePoint]) -> [SplatScenePoint] {
        guard points.count > 1 else { return points }

        let indices = computeReorderingIndices(points)
        return indices.map { points[$0] }
    }

    /// Reorders splat points in-place using the provided reordering indices.
    /// - Parameters:
    ///   - points: Array of splat scene points to reorder (modified in place)
    ///   - indices: Reordering indices from `computeReorderingIndices`
    public static func reorder(_ points: inout [SplatScenePoint], using indices: [Int]) {
        guard points.count > 1 && indices.count == points.count else { return }

        // Use a copy to avoid aliasing issues during reordering
        let original = points
        for i in 0..<indices.count {
            points[i] = original[indices[i]]
        }
    }
}

// MARK: - Parallel Morton Code Computation

extension MortonOrder {

    /// Computes Morton codes in parallel using multiple threads.
    /// Recommended for large datasets (>100K points).
    /// - Parameters:
    ///   - points: Array of splat scene points
    ///   - bounds: Optional pre-computed bounds
    /// - Returns: Array of Morton codes, one per point
    public static func computeMortonCodesParallel(
        _ points: [SplatScenePoint],
        bounds: (min: SIMD3<Float>, max: SIMD3<Float>)? = nil
    ) -> [UInt32] {
        guard !points.isEmpty else { return [] }

        let (minBounds, maxBounds) = bounds ?? computeBounds(points)
        let size = maxBounds - minBounds

        let invSize = SIMD3<Float>(
            size.x > 0 ? 1.0 / size.x : 0,
            size.y > 0 ? 1.0 / size.y : 0,
            size.z > 0 ? 1.0 / size.z : 0
        )

        var codes = [UInt32](repeating: 0, count: points.count)

        // Use concurrent dispatch for parallel computation
        DispatchQueue.concurrentPerform(iterations: points.count) { i in
            let pos = points[i].position
            let normalized = (pos - minBounds) * invSize

            let qx = UInt32(min(max(normalized.x * 1023, 0), 1023))
            let qy = UInt32(min(max(normalized.y * 1023, 0), 1023))
            let qz = UInt32(min(max(normalized.z * 1023, 0), 1023))

            codes[i] = encode(qx, qy, qz)
        }

        return codes
    }

    /// Reorders splat points by Morton code using parallel computation.
    /// Recommended for large datasets (>100K points).
    /// - Parameter points: Array of splat scene points to reorder
    /// - Returns: New array with points reordered by Morton code
    public static func reorderParallel(_ points: [SplatScenePoint]) -> [SplatScenePoint] {
        guard points.count > 1 else { return points }

        let codes = computeMortonCodesParallel(points)

        // Create index array and sort by Morton code
        var indices = Array(0..<points.count)
        indices.sort { codes[$0] < codes[$1] }

        // Parallel reordering
        var reordered = [SplatScenePoint](repeating: points[0], count: points.count)
        DispatchQueue.concurrentPerform(iterations: points.count) { i in
            reordered[i] = points[indices[i]]
        }

        return reordered
    }
}

// MARK: - Statistics

extension MortonOrder {

    /// Statistics about Morton code distribution.
    public struct Statistics {
        /// Total number of points
        public let pointCount: Int
        /// Number of unique Morton codes
        public let uniqueCodes: Int
        /// Bounding box minimum
        public let boundsMin: SIMD3<Float>
        /// Bounding box maximum
        public let boundsMax: SIMD3<Float>
        /// Bounding box diagonal length
        public let diagonalLength: Float
        /// Ratio of unique codes to total points (higher = better spatial distribution)
        public var uniqueRatio: Float {
            pointCount > 0 ? Float(uniqueCodes) / Float(pointCount) : 0
        }
    }

    /// Computes statistics about the Morton code distribution for a set of points.
    /// - Parameter points: Array of splat scene points
    /// - Returns: Statistics about the Morton code distribution
    public static func computeStatistics(_ points: [SplatScenePoint]) -> Statistics {
        let bounds = computeBounds(points)
        let codes = computeMortonCodes(points, bounds: bounds)
        let uniqueCodes = Set(codes).count
        let diagonal = simd_length(bounds.max - bounds.min)

        return Statistics(
            pointCount: points.count,
            uniqueCodes: uniqueCodes,
            boundsMin: bounds.min,
            boundsMax: bounds.max,
            diagonalLength: diagonal
        )
    }
}

// MARK: - Recursive Bucket Refinement

extension MortonOrder {

    /// Default threshold for recursive bucket refinement.
    /// Buckets larger than this will be recursively subdivided.
    public static let defaultBucketThreshold = 256

    /// Reorders splat points by Morton code with recursive bucket refinement.
    ///
    /// This method improves upon basic Morton ordering by recursively re-sorting
    /// buckets of points that hash to the same Morton code. When many points fall
    /// into the same bucket (due to 10-bit quantization limits), this provides
    /// finer-grained spatial ordering within those dense regions.
    ///
    /// - Parameters:
    ///   - points: Array of splat scene points to reorder
    ///   - bucketThreshold: Buckets larger than this will be recursively refined (default: 256)
    /// - Returns: New array with points reordered by Morton code with recursive refinement
    public static func reorderRecursive(
        _ points: [SplatScenePoint],
        bucketThreshold: Int = defaultBucketThreshold
    ) -> [SplatScenePoint] {
        guard points.count > 1 else { return points }

        // Check for degenerate bounds (all points at same position or zero-size bounds)
        // This prevents infinite recursion when points can't be further subdivided
        let bounds = computeBounds(points)
        let size = bounds.max - bounds.min
        if size.x == 0 && size.y == 0 && size.z == 0 {
            return points // All points at same position, no further refinement possible
        }

        // Compute Morton codes and sort
        let codes = computeMortonCodes(points, bounds: bounds)

        // Check if all codes are identical (no refinement possible)
        let firstCode = codes[0]
        if codes.allSatisfy({ $0 == firstCode }) {
            return points
        }

        var indices = Array(0..<points.count)
        indices.sort { codes[$0] < codes[$1] }

        // Create initial sorted result
        var result = indices.map { points[$0] }
        let sortedCodes = indices.map { codes[$0] }

        // Find contiguous runs with the same Morton code and recursively refine large buckets
        var start = 0
        while start < sortedCodes.count {
            // Find the end of this bucket (contiguous run with same code)
            var end = start + 1
            while end < sortedCodes.count && sortedCodes[end] == sortedCodes[start] {
                end += 1
            }

            let bucketSize = end - start
            if bucketSize > bucketThreshold {
                // Extract bucket, recursively refine with fresh bounds, and replace
                let bucket = Array(result[start..<end])
                let refined = reorderRecursive(bucket, bucketThreshold: bucketThreshold)
                for i in 0..<bucketSize {
                    result[start + i] = refined[i]
                }
            }

            start = end
        }

        return result
    }

    /// Reorders splat points by Morton code with recursive bucket refinement using parallel computation.
    ///
    /// Recommended for large datasets (>100K points). Combines parallel Morton code computation
    /// with recursive bucket refinement for optimal performance and spatial locality.
    ///
    /// - Parameters:
    ///   - points: Array of splat scene points to reorder
    ///   - bucketThreshold: Buckets larger than this will be recursively refined (default: 256)
    /// - Returns: New array with points reordered by Morton code with recursive refinement
    public static func reorderRecursiveParallel(
        _ points: [SplatScenePoint],
        bucketThreshold: Int = defaultBucketThreshold
    ) -> [SplatScenePoint] {
        guard points.count > 1 else { return points }

        // Use parallel Morton code computation for initial pass
        let codes = computeMortonCodesParallel(points)
        var indices = Array(0..<points.count)
        indices.sort { codes[$0] < codes[$1] }

        // Parallel reordering for initial result
        var result = [SplatScenePoint](repeating: points[0], count: points.count)
        DispatchQueue.concurrentPerform(iterations: points.count) { i in
            result[i] = points[indices[i]]
        }
        let sortedCodes = indices.map { codes[$0] }

        // Find buckets that need refinement
        var bucketsToRefine: [(start: Int, end: Int)] = []
        var start = 0
        while start < sortedCodes.count {
            var end = start + 1
            while end < sortedCodes.count && sortedCodes[end] == sortedCodes[start] {
                end += 1
            }
            if end - start > bucketThreshold {
                bucketsToRefine.append((start, end))
            }
            start = end
        }

        // Refine large buckets (could parallelize this too for very large datasets)
        for (bucketStart, bucketEnd) in bucketsToRefine {
            let bucket = Array(result[bucketStart..<bucketEnd])
            // Use non-parallel for recursive calls since buckets are smaller
            let refined = reorderRecursive(bucket, bucketThreshold: bucketThreshold)
            for i in 0..<refined.count {
                result[bucketStart + i] = refined[i]
            }
        }

        return result
    }
}
