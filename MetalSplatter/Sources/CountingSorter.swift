import Metal
import simd
import os

/// O(n) Counting Sort implementation for Gaussian Splat sorting
/// Replaces O(n log n) MPS argSort with faster histogram-based sorting
///
/// The algorithm works in three passes:
/// 1. Histogram: Count splats per depth bin
/// 2. Prefix Sum: Convert counts to starting indices
/// 3. Scatter: Place splat indices in sorted order
///
/// This is significantly faster than radix sort for large splat counts
/// because it's O(n) vs O(n log n), and depth quantization to 16 bits
/// is sufficient for visual correctness.
///
/// Camera-relative binning (optional):
/// - Allocates more precision to near-camera bins where visual quality matters most
/// - Weight tiers: camera=40x, adjacent=20x, nearby=8x, medium=3x, far=1x
/// - Inspired by PlayCanvas gsplat-sort-bin-weights approach
internal class CountingSorter {

    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "MetalSplatter",
                                    category: "CountingSorter")

    // Number of histogram bins - 16 bits provides good precision
    // while keeping buffers small
    static let defaultBinCount: Int = 65536  // 2^16

    // Minimum bin count for small scenes (saves memory)
    static let minBinCount: Int = 4096

    // Camera-relative binning constants
    private static let numDistanceBins: Int = 32

    // Weight tiers for camera-relative precision (from PlayCanvas)
    // Distance from camera bin -> weight multiplier
    private static let weightTiers: [(maxDistance: Int, weight: Float)] = [
        (0, 40.0),              // Camera bin (40x precision)
        (2, 20.0),              // Adjacent bins
        (5, 8.0),               // Nearby bins
        (10, 3.0),              // Medium distance
        (Int.max, 1.0)          // Far bins
    ]

    // Parameters structure matching Metal shader
    struct CountingSortParams {
        var minDepth: Float
        var maxDepth: Float
        var invRange: Float       // COUNTING_SORT_BINS / (maxDepth - minDepth)
        var splatCount: UInt32
        var binCount: UInt32

        init(minDepth: Float, maxDepth: Float, splatCount: Int, binCount: Int) {
            self.minDepth = minDepth
            self.maxDepth = maxDepth
            let range = max(maxDepth - minDepth, 0.001)  // Avoid division by zero
            self.invRange = Float(binCount) / range
            self.splatCount = UInt32(splatCount)
            self.binCount = UInt32(binCount)
        }
    }

    // Camera-relative bin parameters matching Metal shader
    struct CameraRelativeBinParams {
        var binBase: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                      UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                      UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                      UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) // 33 elements
        var binDivider: (UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                         UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                         UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                         UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32) // 33 elements
        var cameraBin: UInt32
        var invRange: Float
        var minDepth: Float
        var totalBuckets: UInt32

        init() {
            binBase = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            binDivider = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            cameraBin = 0
            invRange = 1.0
            minDepth = 0.0
            totalBuckets = 65536
        }
    }

    private let device: MTLDevice

    // Pipeline states
    private let histogramPipeline: MTLComputePipelineState
    private let histogramWeightedPipeline: MTLComputePipelineState
    private let prefixSumPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let resetHistogramPipeline: MTLComputePipelineState
    private let initBinOffsetsPipeline: MTLComputePipelineState

    // Blocked prefix sum pipelines (for large bin counts exceeding threadgroup memory)
    private let blockPrefixSumPipeline: MTLComputePipelineState
    private let blockSumsPrefixSumPipeline: MTLComputePipelineState
    private let addBlockPrefixPipeline: MTLComputePipelineState

    // Maximum block size based on device threadgroup memory
    private let maxPrefixSumBlockSize: Int

    // Reusable buffers (allocated once, reused across frames)
    private var histogramBuffer: MTLBuffer?
    private var prefixSumBuffer: MTLBuffer?
    private var binOffsetsBuffer: MTLBuffer?
    private var cachedBinsBuffer: MTLBuffer?  // Caches bin indices between histogram and scatter passes
    private var blockSumsBuffer: MTLBuffer?   // Block totals for blocked prefix sum
    private var currentBinCount: Int = 0
    private var currentSplatCapacity: Int = 0

    // Cached depth bounds (optional optimization)
    private var cachedMinDepth: Float?
    private var cachedMaxDepth: Float?

    internal init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device

        // Load compute functions
        guard let histogramFunction = library.makeFunction(name: "countingSortHistogram") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "countingSortHistogram")
        }
        guard let histogramWeightedFunction = library.makeFunction(name: "countingSortHistogramWeighted") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "countingSortHistogramWeighted")
        }
        guard let prefixSumFunction = library.makeFunction(name: "countingSortPrefixSum") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "countingSortPrefixSum")
        }
        guard let scatterFunction = library.makeFunction(name: "countingSortScatter") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "countingSortScatter")
        }
        guard let resetFunction = library.makeFunction(name: "countingSortResetHistogram") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "countingSortResetHistogram")
        }
        guard let initOffsetsFunction = library.makeFunction(name: "countingSortInitBinOffsets") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "countingSortInitBinOffsets")
        }

        // Blocked prefix sum functions
        guard let blockPrefixSumFunction = library.makeFunction(name: "countingSortBlockPrefixSum") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "countingSortBlockPrefixSum")
        }
        guard let blockSumsPrefixSumFunction = library.makeFunction(name: "countingSortBlockSumsPrefixSum") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "countingSortBlockSumsPrefixSum")
        }
        guard let addBlockPrefixFunction = library.makeFunction(name: "countingSortAddBlockPrefix") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "countingSortAddBlockPrefix")
        }

        // Create pipeline states
        histogramPipeline = try device.makeComputePipelineState(function: histogramFunction)
        histogramWeightedPipeline = try device.makeComputePipelineState(function: histogramWeightedFunction)
        prefixSumPipeline = try device.makeComputePipelineState(function: prefixSumFunction)
        scatterPipeline = try device.makeComputePipelineState(function: scatterFunction)
        resetHistogramPipeline = try device.makeComputePipelineState(function: resetFunction)
        initBinOffsetsPipeline = try device.makeComputePipelineState(function: initOffsetsFunction)

        // Blocked prefix sum pipelines
        blockPrefixSumPipeline = try device.makeComputePipelineState(function: blockPrefixSumFunction)
        blockSumsPrefixSumPipeline = try device.makeComputePipelineState(function: blockSumsPrefixSumFunction)
        addBlockPrefixPipeline = try device.makeComputePipelineState(function: addBlockPrefixFunction)

        // Calculate max block size based on device threadgroup memory
        // Block size must be power of 2 for Blelloch algorithm
        // Each element is UInt32 (4 bytes)
        let maxThreadgroupMemory = device.maxThreadgroupMemoryLength
        let maxElements = maxThreadgroupMemory / MemoryLayout<UInt32>.stride
        // Find largest power of 2 that fits
        var blockSize = 1
        while blockSize * 2 <= maxElements {
            blockSize *= 2
        }
        maxPrefixSumBlockSize = blockSize
    }

    /// Computes camera-relative bin parameters for weighted sorting
    /// This allocates more precision to bins near the camera where visual quality matters most
    private func computeCameraRelativeBinParams(
        minDepth: Float,
        maxDepth: Float,
        cameraPosition: SIMD3<Float>,
        sortByDistance: Bool,
        binCount: Int
    ) -> CameraRelativeBinParams {
        var params = CameraRelativeBinParams()
        let range = max(maxDepth - minDepth, 0.001)

        // Determine which distance bin contains the camera
        let cameraBin: Int
        if sortByDistance {
            // For radial sort, camera (dist=0) maps to bin 0
            cameraBin = 0
        } else {
            // For linear sort, camera is at depth 0 relative to itself
            let cameraOffsetFromRangeStart = 0 - minDepth
            let cameraBinFloat = (cameraOffsetFromRangeStart / range) * Float(Self.numDistanceBins)
            cameraBin = max(0, min(Self.numDistanceBins - 1, Int(cameraBinFloat)))
        }

        // Calculate weight by distance from camera bin
        var weights = [Float](repeating: 1.0, count: Self.numDistanceBins)
        for i in 0..<Self.numDistanceBins {
            let distFromCamera = abs(i - cameraBin)
            for tier in Self.weightTiers {
                if distFromCamera <= tier.maxDistance {
                    weights[i] = tier.weight
                    break
                }
            }
        }

        // Normalize weights and compute bin bases/dividers
        let totalWeight = weights.reduce(0, +)
        var accumulated: UInt32 = 0

        // Use withUnsafeMutablePointer to set tuple elements
        withUnsafeMutablePointer(to: &params.binBase) { basePtr in
            let base = UnsafeMutableRawPointer(basePtr).assumingMemoryBound(to: UInt32.self)
            withUnsafeMutablePointer(to: &params.binDivider) { dividerPtr in
                let divider = UnsafeMutableRawPointer(dividerPtr).assumingMemoryBound(to: UInt32.self)

                for i in 0..<Self.numDistanceBins {
                    let buckets = max(1, UInt32((weights[i] / totalWeight) * Float(binCount)))
                    divider[i] = buckets
                    base[i] = accumulated
                    accumulated += buckets
                }

                // Adjust last bin to fit exactly
                if accumulated > UInt32(binCount) {
                    let excess = accumulated - UInt32(binCount)
                    let lastIdx = Self.numDistanceBins - 1
                    divider[lastIdx] = max(1, divider[lastIdx] - excess)
                }

                // Safety entry
                base[Self.numDistanceBins] = base[Self.numDistanceBins - 1] + divider[Self.numDistanceBins - 1]
                divider[Self.numDistanceBins] = 0
            }
        }

        params.cameraBin = UInt32(cameraBin)
        params.invRange = Float(Self.numDistanceBins) / range
        params.minDepth = minDepth
        params.totalBuckets = UInt32(binCount)

        return params
    }

    /// Ensures buffers are allocated for the given bin count and splat capacity
    private func ensureBuffers(binCount: Int, splatCount: Int) throws {
        // Reallocate bin-related buffers if bin count changed
        if binCount != currentBinCount {
            let bufferSize = binCount * MemoryLayout<UInt32>.stride

            guard let histogram = device.makeBuffer(length: bufferSize, options: .storageModePrivate),
                  let prefixSum = device.makeBuffer(length: bufferSize, options: .storageModePrivate),
                  let binOffsets = device.makeBuffer(length: bufferSize, options: .storageModePrivate) else {
                throw SplatRendererError.failedToCreateBuffer(length: bufferSize)
            }

            histogram.label = "CountingSort Histogram"
            prefixSum.label = "CountingSort PrefixSum"
            binOffsets.label = "CountingSort BinOffsets"

            histogramBuffer = histogram
            prefixSumBuffer = prefixSum
            binOffsetsBuffer = binOffsets

            // Allocate block sums buffer for blocked prefix sum (when binCount > maxPrefixSumBlockSize)
            let blockCount = (binCount + maxPrefixSumBlockSize - 1) / maxPrefixSumBlockSize
            if blockCount > 1 {
                let blockSumsSize = blockCount * MemoryLayout<UInt32>.stride
                guard let blockSums = device.makeBuffer(length: blockSumsSize, options: .storageModePrivate) else {
                    throw SplatRendererError.failedToCreateBuffer(length: blockSumsSize)
                }
                blockSums.label = "CountingSort BlockSums"
                blockSumsBuffer = blockSums
            } else {
                blockSumsBuffer = nil
            }

            currentBinCount = binCount
        }

        // Reallocate cached bins buffer if splat count increased
        if splatCount > currentSplatCapacity {
            // Use ushort (UInt16) to save memory - 2 bytes per splat
            let cachedBinsSize = splatCount * MemoryLayout<UInt16>.stride

            guard let cachedBins = device.makeBuffer(length: cachedBinsSize, options: .storageModePrivate) else {
                throw SplatRendererError.failedToCreateBuffer(length: cachedBinsSize)
            }

            cachedBins.label = "CountingSort CachedBins"
            cachedBinsBuffer = cachedBins
            currentSplatCapacity = splatCount
        }
    }

    /// Computes depth bounds for all splats (can be cached if splats don't change)
    internal func computeDepthBounds(
        splats: UnsafeBufferPointer<SplatRenderer.Splat>,
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool
    ) -> (min: Float, max: Float) {

        var minDepth: Float = .infinity
        var maxDepth: Float = -.infinity

        for splat in splats {
            let splatPos = SIMD3<Float>(splat.position.x, splat.position.y, splat.position.z)

            let depth: Float
            if sortByDistance {
                let delta = splatPos - cameraPosition
                depth = simd_length(delta)
            } else {
                let delta = splatPos - cameraPosition
                depth = simd_dot(delta, cameraForward)
            }

            minDepth = min(minDepth, depth)
            maxDepth = max(maxDepth, depth)
        }

        // Handle edge cases
        if minDepth.isInfinite {
            minDepth = 0
            maxDepth = 1
        }

        // Add small padding to avoid edge cases
        let range = maxDepth - minDepth
        let padding = range * 0.001
        return (minDepth - padding, maxDepth + padding)
    }

    /// Determines optimal bin count based on splat count
    /// Smaller scenes need fewer bins for efficiency
    private func optimalBinCount(for splatCount: Int) -> Int {
        if splatCount < 10_000 {
            return Self.minBinCount
        } else if splatCount < 100_000 {
            return 16384
        } else if splatCount < 1_000_000 {
            return 32768
        } else {
            return Self.defaultBinCount
        }
    }

    /// Performs counting sort on splats
    /// - Parameters:
    ///   - commandBuffer: Command buffer to encode into
    ///   - splatBuffer: Input splat buffer
    ///   - outputBuffer: Output sorted indices buffer (Int32)
    ///   - cameraPosition: Camera world position
    ///   - cameraForward: Camera forward direction
    ///   - sortByDistance: True for radial distance, false for projected distance
    ///   - splatCount: Number of splats to sort
    ///   - depthBounds: Optional pre-computed depth bounds (min, max)
    ///   - useCameraRelativeBinning: When true, allocates more precision to near-camera splats
    internal func sort(
        commandBuffer: MTLCommandBuffer,
        splatBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool,
        splatCount: Int,
        depthBounds: (min: Float, max: Float)? = nil,
        useCameraRelativeBinning: Bool = false
    ) throws {
        guard splatCount > 0 else { return }

        let binCount = optimalBinCount(for: splatCount)
        try ensureBuffers(binCount: binCount, splatCount: splatCount)

        guard let histogram = histogramBuffer,
              let prefixSum = prefixSumBuffer,
              let binOffsets = binOffsetsBuffer,
              let cachedBins = cachedBinsBuffer else {
            Self.log.error("Counting sort buffers not allocated")
            return
        }

        // Use provided bounds or compute them
        let bounds: (min: Float, max: Float)
        if let providedBounds = depthBounds {
            bounds = providedBounds
        } else {
            // For now, use a reasonable default range
            // In production, this should be computed from splat data
            bounds = (0.1, 100.0)
        }

        var params = CountingSortParams(
            minDepth: bounds.min,
            maxDepth: bounds.max,
            splatCount: splatCount,
            binCount: binCount
        )
        var cameraPos = cameraPosition
        var cameraFwd = cameraForward
        var sortByDist = sortByDistance
        var binCountVar = UInt32(binCount)

        let threadsPerGroup = min(256, histogramPipeline.maxTotalThreadsPerThreadgroup)
        let threadgroups = (splatCount + threadsPerGroup - 1) / threadsPerGroup

        // Pass 1: Reset histogram
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "CountingSort Reset"
            encoder.setComputePipelineState(resetHistogramPipeline)
            encoder.setBuffer(histogram, offset: 0, index: 0)
            encoder.setBytes(&binCountVar, length: MemoryLayout<UInt32>.size, index: 1)

            let resetThreadgroups = (binCount + 255) / 256
            encoder.dispatchThreadgroups(
                MTLSize(width: resetThreadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }

        // Pass 2: Build histogram AND cache bin indices
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            if useCameraRelativeBinning {
                // Use camera-relative weighted binning for better near-camera precision
                encoder.label = "CountingSort Histogram (Weighted)"
                encoder.setComputePipelineState(histogramWeightedPipeline)

                var binParams = computeCameraRelativeBinParams(
                    minDepth: bounds.min,
                    maxDepth: bounds.max,
                    cameraPosition: cameraPosition,
                    sortByDistance: sortByDistance,
                    binCount: binCount
                )

                encoder.setBuffer(splatBuffer, offset: 0, index: 0)
                encoder.setBuffer(histogram, offset: 0, index: 1)
                encoder.setBytes(&params, length: MemoryLayout<CountingSortParams>.size, index: 2)
                encoder.setBytes(&cameraPos, length: MemoryLayout<SIMD3<Float>>.size, index: 3)
                encoder.setBytes(&cameraFwd, length: MemoryLayout<SIMD3<Float>>.size, index: 4)
                encoder.setBytes(&sortByDist, length: MemoryLayout<Bool>.size, index: 5)
                encoder.setBuffer(cachedBins, offset: 0, index: 6)
                encoder.setBytes(&binParams, length: MemoryLayout<CameraRelativeBinParams>.size, index: 7)
            } else {
                // Standard uniform binning
                encoder.label = "CountingSort Histogram"
                encoder.setComputePipelineState(histogramPipeline)
                encoder.setBuffer(splatBuffer, offset: 0, index: 0)
                encoder.setBuffer(histogram, offset: 0, index: 1)
                encoder.setBytes(&params, length: MemoryLayout<CountingSortParams>.size, index: 2)
                encoder.setBytes(&cameraPos, length: MemoryLayout<SIMD3<Float>>.size, index: 3)
                encoder.setBytes(&cameraFwd, length: MemoryLayout<SIMD3<Float>>.size, index: 4)
                encoder.setBytes(&sortByDist, length: MemoryLayout<Bool>.size, index: 5)
                encoder.setBuffer(cachedBins, offset: 0, index: 6)
            }

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }

        // Pass 3: Prefix sum (converts histogram to starting indices)
        // Use blocked approach for large bin counts that exceed threadgroup memory
        let blockCount = (binCount + maxPrefixSumBlockSize - 1) / maxPrefixSumBlockSize

        if blockCount > 1, let blockSums = blockSumsBuffer {
            // Blocked 3-phase prefix sum for large bin counts
            var blockSizeVar = UInt32(maxPrefixSumBlockSize)

            // Phase 1: Local prefix sum per block, output block totals
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "CountingSort BlockPrefixSum"
                encoder.setComputePipelineState(blockPrefixSumPipeline)
                encoder.setBuffer(histogram, offset: 0, index: 0)
                encoder.setBuffer(prefixSum, offset: 0, index: 1)
                encoder.setBuffer(blockSums, offset: 0, index: 2)
                encoder.setBytes(&binCountVar, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&blockSizeVar, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setThreadgroupMemoryLength(maxPrefixSumBlockSize * MemoryLayout<UInt32>.stride, index: 0)
                encoder.dispatchThreadgroups(
                    MTLSize(width: blockCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(256, maxPrefixSumBlockSize), height: 1, depth: 1)
                )
                encoder.endEncoding()
            }

            // Phase 2: Prefix sum of block totals (small array, single-thread is fine)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "CountingSort BlockSumsPrefixSum"
                encoder.setComputePipelineState(blockSumsPrefixSumPipeline)
                encoder.setBuffer(blockSums, offset: 0, index: 0)
                var blockCountVar = UInt32(blockCount)
                encoder.setBytes(&blockCountVar, length: MemoryLayout<UInt32>.size, index: 1)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
                )
                encoder.endEncoding()
            }

            // Phase 3: Add block prefix to each element
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "CountingSort AddBlockPrefix"
                encoder.setComputePipelineState(addBlockPrefixPipeline)
                encoder.setBuffer(prefixSum, offset: 0, index: 0)
                encoder.setBuffer(blockSums, offset: 0, index: 1)
                encoder.setBytes(&binCountVar, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.setBytes(&blockSizeVar, length: MemoryLayout<UInt32>.size, index: 3)
                let addThreadgroups = (binCount + 255) / 256
                encoder.dispatchThreadgroups(
                    MTLSize(width: addThreadgroups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                encoder.endEncoding()
            }
        } else {
            // Single-thread prefix sum for small bin counts (fits in one block)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.label = "CountingSort PrefixSum"
                encoder.setComputePipelineState(prefixSumPipeline)
                encoder.setBuffer(histogram, offset: 0, index: 0)
                encoder.setBuffer(prefixSum, offset: 0, index: 1)
                encoder.setBytes(&binCountVar, length: MemoryLayout<UInt32>.size, index: 2)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
                )
                encoder.endEncoding()
            }
        }

        // Pass 4: Copy prefix sum to bin offsets
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "CountingSort InitOffsets"
            encoder.setComputePipelineState(initBinOffsetsPipeline)
            encoder.setBuffer(prefixSum, offset: 0, index: 0)
            encoder.setBuffer(binOffsets, offset: 0, index: 1)
            encoder.setBytes(&binCountVar, length: MemoryLayout<UInt32>.size, index: 2)

            let copyThreadgroups = (binCount + 255) / 256
            encoder.dispatchThreadgroups(
                MTLSize(width: copyThreadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }

        // Pass 5: Scatter indices to sorted positions (uses cached bin indices - no depth recomputation!)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "CountingSort Scatter"
            encoder.setComputePipelineState(scatterPipeline)
            encoder.setBuffer(cachedBins, offset: 0, index: 0)   // Use cached bin indices
            encoder.setBuffer(binOffsets, offset: 0, index: 1)
            encoder.setBuffer(outputBuffer, offset: 0, index: 2)
            encoder.setBytes(&params, length: MemoryLayout<CountingSortParams>.size, index: 3)

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }
    }

    /// Clears cached depth bounds (call when splats change)
    internal func invalidateCache() {
        cachedMinDepth = nil
        cachedMaxDepth = nil
    }
}
