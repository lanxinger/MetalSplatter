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
internal class CountingSorter {

    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "MetalSplatter",
                                    category: "CountingSorter")

    // Number of histogram bins - 16 bits provides good precision
    // while keeping buffers small
    static let defaultBinCount: Int = 65536  // 2^16

    // Minimum bin count for small scenes (saves memory)
    static let minBinCount: Int = 4096

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

    private let device: MTLDevice

    // Pipeline states
    private let histogramPipeline: MTLComputePipelineState
    private let prefixSumPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState
    private let resetHistogramPipeline: MTLComputePipelineState
    private let initBinOffsetsPipeline: MTLComputePipelineState

    // Reusable buffers (allocated once, reused across frames)
    private var histogramBuffer: MTLBuffer?
    private var prefixSumBuffer: MTLBuffer?
    private var binOffsetsBuffer: MTLBuffer?
    private var cachedBinsBuffer: MTLBuffer?  // Caches bin indices between histogram and scatter passes
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

        // Create pipeline states
        histogramPipeline = try device.makeComputePipelineState(function: histogramFunction)
        prefixSumPipeline = try device.makeComputePipelineState(function: prefixSumFunction)
        scatterPipeline = try device.makeComputePipelineState(function: scatterFunction)
        resetHistogramPipeline = try device.makeComputePipelineState(function: resetFunction)
        initBinOffsetsPipeline = try device.makeComputePipelineState(function: initOffsetsFunction)
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
    internal func sort(
        commandBuffer: MTLCommandBuffer,
        splatBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool,
        splatCount: Int,
        depthBounds: (min: Float, max: Float)? = nil
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
            encoder.label = "CountingSort Histogram"
            encoder.setComputePipelineState(histogramPipeline)
            encoder.setBuffer(splatBuffer, offset: 0, index: 0)
            encoder.setBuffer(histogram, offset: 0, index: 1)
            encoder.setBytes(&params, length: MemoryLayout<CountingSortParams>.size, index: 2)
            encoder.setBytes(&cameraPos, length: MemoryLayout<SIMD3<Float>>.size, index: 3)
            encoder.setBytes(&cameraFwd, length: MemoryLayout<SIMD3<Float>>.size, index: 4)
            encoder.setBytes(&sortByDist, length: MemoryLayout<Bool>.size, index: 5)
            encoder.setBuffer(cachedBins, offset: 0, index: 6)  // Cache bin indices for scatter pass

            encoder.dispatchThreadgroups(
                MTLSize(width: threadgroups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
            )
            encoder.endEncoding()
        }

        // Pass 3: Prefix sum (converts histogram to starting indices)
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.label = "CountingSort PrefixSum"
            encoder.setComputePipelineState(prefixSumPipeline)
            encoder.setBuffer(histogram, offset: 0, index: 0)
            encoder.setBuffer(prefixSum, offset: 0, index: 1)
            encoder.setBytes(&binCountVar, length: MemoryLayout<UInt32>.size, index: 2)

            // Single thread for simple prefix sum (sufficient for 64K bins)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            encoder.endEncoding()
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
