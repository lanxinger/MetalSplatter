import Metal
import simd
import os

/// Metal 4 GPU-accelerated radix sort using stable histogram-based approach
/// Uses 4-pass radix sort (8 bits per pass for full 32-bit coverage)
///
/// The algorithm (per pass):
/// 1. Reset histogram to zeros
/// 2. Histogram: count elements per bucket (256 buckets)
/// 3. Prefix sum: convert counts to cumulative offsets
/// 4. Stable scatter (three-phase for correctness):
///    - Phase 1: Each threadgroup counts bucket populations (no atomic claiming)
///    - Phase 2: Compute deterministic block offsets via prefix sum across threadgroups
///    - Phase 3: Each thread computes local rank and writes to pre-computed offset
///
/// Key features:
/// - Processes all 32 bits for correct float ordering
/// - Uses IEEE 754 float-to-sortable-uint transform
/// - Produces DESCENDING order (back-to-front for splat rendering)
/// - Truly stable scatter: maintains relative order from previous passes (LSD requirement)
/// - Deterministic inter-threadgroup ordering prevents stability violations
///
/// This sorter is designed for very large splat counts (>100K) where the
/// GPU parallelism benefits outweigh the multi-pass overhead.
@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
internal class Metal4Sorter {

    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "MetalSplatter",
                                    category: "Metal4Sorter")

    // Structure matching Metal shader SortingKey
    struct SortingKey {
        var depth: Float  // Actually stores sortable uint as float bits
        var originalIndex: UInt32
    }

    // 4 passes of 8 bits each = 32 bits total (full float coverage)
    static let radixPasses: Int = 4
    static let bitsPerPass: UInt32 = 8
    static let bucketsPerPass: Int = 256  // 2^8

    private let device: MTLDevice

    // Threadgroup size for scatter phase (must match shader constant)
    static let scatterThreadgroupSize: Int = 256

    // OneSweep tile size (must match ONESWEEP_TILE in OneSweepSort.metal)
    static let oneSweepTileSize: Int = 1024

    // Pipeline states
    private let buildKeysPipeline: MTLComputePipelineState
    private let resetHistogramPipeline: MTLComputePipelineState
    private let histogramPipeline: MTLComputePipelineState
    private let prefixSumPipeline: MTLComputePipelineState
    private let scatterCountPipeline: MTLComputePipelineState     // Phase 1: count per threadgroup
    private let scatterOffsetsPipeline: MTLComputePipelineState   // Phase 2: compute deterministic offsets
    private let scatterWritePipeline: MTLComputePipelineState     // Phase 3: stable write
    private let extractIndicesPipeline: MTLComputePipelineState

    // OneSweep pipelines (MSL 4.1, compiled at runtime; nil when the OS's
    // Metal compiler doesn't support MSL 4.1 or compilation fails)
    private let oneSweepHistogramPipeline: MTLComputePipelineState?
    private let oneSweepScanPipeline: MTLComputePipelineState?
    private let oneSweepResetPipeline: MTLComputePipelineState?
    private let oneSweepDigitPassPipeline: MTLComputePipelineState?

    /// Prefer the OneSweep path when its MSL 4.1 kernels are available.
    /// Set false to force the legacy multi-dispatch radix sort. OneSweep relies
    /// on decoupled lookback (tiles spin-wait on predecessors), which assumes
    /// occupancy-bound threadgroup scheduling — the standard assumption for
    /// this algorithm, but worth a kill switch.
    var useOneSweep: Bool = true

    var oneSweepAvailable: Bool {
        oneSweepHistogramPipeline != nil && oneSweepScanPipeline != nil
            && oneSweepResetPipeline != nil && oneSweepDigitPassPipeline != nil
    }

    // Reusable buffers (allocated on demand)
    private var keysBufferA: MTLBuffer?
    private var keysBufferB: MTLBuffer?
    private var histogramBuffer: MTLBuffer?  // 256 atomic uints
    private var tgBucketCountsBuffer: MTLBuffer?   // [num_threadgroups * 256] counts per TG
    private var tgBucketOffsetsBuffer: MTLBuffer?  // [num_threadgroups * 256] offsets per TG
    private var oneSweepHistBuffer: MTLBuffer?     // [4 passes * 256] digit histograms/offsets
    private var oneSweepStatusBuffer: MTLBuffer?   // [numTiles * 256] lookback status words
    private var oneSweepTicketBuffer: MTLBuffer?   // tile ticket counter
    private var allocatedCount: Int = 0
    private var allocatedThreadgroups: Int = 0
    private var allocatedTiles: Int = 0

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device

        // Create pipeline states for each kernel
        // Note: Kernel functions are at global scope in Metal (not namespaced)
        // because Metal's makeFunction(name:) doesn't support C++ namespace-qualified names
        guard let buildKeysFunction = library.makeFunction(name: "build_sorting_keys") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "build_sorting_keys")
        }
        buildKeysPipeline = try device.makeComputePipelineState(function: buildKeysFunction)

        guard let resetHistogramFunction = library.makeFunction(name: "reset_histogram") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "reset_histogram")
        }
        resetHistogramPipeline = try device.makeComputePipelineState(function: resetHistogramFunction)

        guard let histogramFunction = library.makeFunction(name: "histogram_radix_pass") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "histogram_radix_pass")
        }
        histogramPipeline = try device.makeComputePipelineState(function: histogramFunction)

        guard let prefixSumFunction = library.makeFunction(name: "prefix_sum_buckets") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "prefix_sum_buckets")
        }
        prefixSumPipeline = try device.makeComputePipelineState(function: prefixSumFunction)

        guard let scatterCountFunction = library.makeFunction(name: "scatter_count_per_threadgroup") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "scatter_count_per_threadgroup")
        }
        scatterCountPipeline = try device.makeComputePipelineState(function: scatterCountFunction)

        guard let scatterOffsetsFunction = library.makeFunction(name: "compute_scatter_offsets") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "compute_scatter_offsets")
        }
        scatterOffsetsPipeline = try device.makeComputePipelineState(function: scatterOffsetsFunction)

        guard let scatterWriteFunction = library.makeFunction(name: "scatter_write_stable") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "scatter_write_stable")
        }
        scatterWritePipeline = try device.makeComputePipelineState(function: scatterWriteFunction)

        guard let extractIndicesFunction = library.makeFunction(name: "extract_sorted_indices") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "extract_sorted_indices")
        }
        extractIndicesPipeline = try device.makeComputePipelineState(function: extractIndicesFunction)

        // The OneSweep kernels require MSL 4.1, but the build-time Metal
        // compiler pins the language version to the package's deployment
        // target, so they can never ship in the precompiled library. Compile
        // them from source at runtime on OS versions whose compiler supports
        // MSL 4.1; fall back to the legacy path otherwise.
        if let oneSweepLibrary = Self.makeOneSweepLibrary(device: device),
           let histFunction = oneSweepLibrary.makeFunction(name: "onesweep_global_histogram"),
           let scanFunction = oneSweepLibrary.makeFunction(name: "onesweep_scan_histograms"),
           let resetFunction = oneSweepLibrary.makeFunction(name: "onesweep_reset_status"),
           let digitFunction = oneSweepLibrary.makeFunction(name: "onesweep_digit_pass") {
            oneSweepHistogramPipeline = try? device.makeComputePipelineState(function: histFunction)
            oneSweepScanPipeline = try? device.makeComputePipelineState(function: scanFunction)
            oneSweepResetPipeline = try? device.makeComputePipelineState(function: resetFunction)
            oneSweepDigitPassPipeline = try? device.makeComputePipelineState(function: digitFunction)
        } else {
            oneSweepHistogramPipeline = nil
            oneSweepScanPipeline = nil
            oneSweepResetPipeline = nil
            oneSweepDigitPassPipeline = nil
        }

        if oneSweepAvailable {
            Self.log.info("Metal4Sorter initialized with OneSweep radix sort (MSL 4.1 decoupled lookback)")
        } else {
            Self.log.info("Metal4Sorter initialized with \(Self.radixPasses)-pass stable radix sort (8-bit buckets)")
        }
    }

    /// Runtime-compile the OneSweep kernels (OneSweepSort.metal, shipped as a
    /// copied resource) as MSL 4.1. Returns nil when the OS's Metal compiler
    /// predates MSL 4.1 or compilation fails.
    private static func makeOneSweepLibrary(device: MTLDevice) -> MTLLibrary? {
        guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return nil }
        guard let sourceURL = Bundle.module.url(forResource: "OneSweepSort", withExtension: "metal"),
              let source = try? String(contentsOf: sourceURL, encoding: .utf8) else {
            log.warning("OneSweepSort.metal resource missing; using legacy radix sort")
            return nil
        }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_1
        options.mathMode = .fast  // match the build-time -fmetal-math-mode=fast
        do {
            return try device.makeLibrary(source: source, options: options)
        } catch {
            log.warning("OneSweep runtime compilation failed; using legacy radix sort: \(error.localizedDescription)")
            return nil
        }
    }

    /// Ensure buffers are allocated for the given splat count
    private func ensureBuffers(count: Int, oneSweep: Bool) throws {
        let keyBufferSize = count * MemoryLayout<SortingKey>.stride

        if count > allocatedCount {
            keysBufferA = device.makeBuffer(length: keyBufferSize, options: .storageModePrivate)
            keysBufferA?.label = "Metal4Sorter Keys A"

            keysBufferB = device.makeBuffer(length: keyBufferSize, options: .storageModePrivate)
            keysBufferB?.label = "Metal4Sorter Keys B"

            allocatedCount = count
        }

        guard keysBufferA != nil, keysBufferB != nil else {
            throw SplatRendererError.failedToCreateBuffer(length: keyBufferSize)
        }

        if oneSweep {
            // Digit histograms/offsets for all 4 passes, plus lookback state
            let histSize = Self.radixPasses * Self.bucketsPerPass * MemoryLayout<UInt32>.stride
            if oneSweepHistBuffer == nil {
                oneSweepHistBuffer = device.makeBuffer(length: histSize, options: .storageModePrivate)
                oneSweepHistBuffer?.label = "Metal4Sorter OneSweep Histograms"
            }
            if oneSweepTicketBuffer == nil {
                oneSweepTicketBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModePrivate)
                oneSweepTicketBuffer?.label = "Metal4Sorter OneSweep Ticket"
            }
            let numTiles = (count + Self.oneSweepTileSize - 1) / Self.oneSweepTileSize
            if numTiles > allocatedTiles {
                let statusSize = numTiles * Self.bucketsPerPass * MemoryLayout<UInt32>.stride
                oneSweepStatusBuffer = device.makeBuffer(length: statusSize, options: .storageModePrivate)
                oneSweepStatusBuffer?.label = "Metal4Sorter OneSweep Status"
                allocatedTiles = numTiles
            }
            guard oneSweepHistBuffer != nil, oneSweepStatusBuffer != nil, oneSweepTicketBuffer != nil else {
                throw SplatRendererError.failedToCreateBuffer(length: keyBufferSize)
            }
            return
        }

        // Histogram buffer: 256 atomic uints
        let histogramSize = Self.bucketsPerPass * MemoryLayout<UInt32>.stride
        if histogramBuffer == nil || histogramBuffer!.length < histogramSize {
            histogramBuffer = device.makeBuffer(length: histogramSize, options: .storageModePrivate)
            histogramBuffer?.label = "Metal4Sorter Histogram"
        }

        // Threadgroup bucket counts and offsets buffers: [num_threadgroups * 256] each
        // Counts: stores how many elements each threadgroup has per bucket
        // Offsets: stores the computed starting position for each (threadgroup, bucket) pair
        let numThreadgroups = (count + Self.scatterThreadgroupSize - 1) / Self.scatterThreadgroupSize
        if numThreadgroups > allocatedThreadgroups {
            let tgBufferSize = numThreadgroups * Self.bucketsPerPass * MemoryLayout<UInt32>.stride
            tgBucketCountsBuffer = device.makeBuffer(length: tgBufferSize, options: .storageModePrivate)
            tgBucketCountsBuffer?.label = "Metal4Sorter TG Bucket Counts"
            tgBucketOffsetsBuffer = device.makeBuffer(length: tgBufferSize, options: .storageModePrivate)
            tgBucketOffsetsBuffer?.label = "Metal4Sorter TG Bucket Offsets"
            allocatedThreadgroups = numThreadgroups
        }

        guard histogramBuffer != nil,
              tgBucketCountsBuffer != nil, tgBucketOffsetsBuffer != nil else {
            throw SplatRendererError.failedToCreateBuffer(length: keyBufferSize)
        }
    }

    /// Sort splats by depth using GPU radix sort
    /// - Parameters:
    ///   - splats: Buffer containing Splat data
    ///   - count: Number of splats to sort
    ///   - cameraPosition: Camera world position
    ///   - cameraForward: Camera forward direction (normalized)
    ///   - sortByDistance: If true, sort by distance from camera; if false, sort by forward dot product
    ///   - outputIndices: Buffer to write sorted indices (Int32)
    ///   - commandBuffer: Metal command buffer to encode into
    func sort(
        splats: MTLBuffer,
        count: Int,
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool,
        outputIndices: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard count > 0 else { return }

        let oneSweep = useOneSweep && oneSweepAvailable
        try ensureBuffers(count: count, oneSweep: oneSweep)

        if oneSweep {
            try sortOneSweep(
                splats: splats,
                count: count,
                cameraPosition: cameraPosition,
                cameraForward: cameraForward,
                sortByDistance: sortByDistance,
                outputIndices: outputIndices,
                commandBuffer: commandBuffer
            )
            return
        }

        guard let keysA = keysBufferA,
              let keysB = keysBufferB,
              let histogram = histogramBuffer,
              let tgOffsets = tgBucketOffsetsBuffer else {
            throw SplatRendererError.failedToCreateBuffer(length: 0)
        }

        // Step 1: Build sorting keys from splat positions
        // Keys are transformed to sortable uints with descending order
        try encodeBuildKeys(
            splats: splats,
            keys: keysA,
            cameraPosition: cameraPosition,
            cameraForward: cameraForward,
            sortByDistance: sortByDistance,
            count: count,
            commandBuffer: commandBuffer
        )

        // Step 2: Multi-pass stable radix sort (LSD - least significant digit first)
        // Ping-pong between keysA and keysB
        var inputKeys = keysA
        var outputKeys = keysB

        for pass in 0..<Self.radixPasses {
            let byteIndex = UInt32(pass)  // 0, 1, 2, 3 for each byte

            // 2a. Reset histogram
            try encodeResetHistogram(
                histogram: histogram,
                commandBuffer: commandBuffer
            )

            // 2b. Build histogram for this byte
            try encodeHistogram(
                keys: inputKeys,
                histogram: histogram,
                byteIndex: byteIndex,
                count: count,
                commandBuffer: commandBuffer
            )

            // 2c. Convert histogram to prefix sum (cumulative offsets)
            try encodePrefixSum(
                histogram: histogram,
                commandBuffer: commandBuffer
            )

            // 2d. Scatter elements to sorted positions (two-phase stable scatter)
            try encodeScatter(
                inputKeys: inputKeys,
                outputKeys: outputKeys,
                histogram: histogram,
                tgOffsets: tgOffsets,
                byteIndex: byteIndex,
                count: count,
                commandBuffer: commandBuffer
            )

            // Swap buffers for next pass
            swap(&inputKeys, &outputKeys)
        }

        // After 4 passes (even), result is in keysA
        let finalKeys = keysA

        // Step 3: Extract sorted indices
        try encodeExtractIndices(
            sortedKeys: finalKeys,
            outputIndices: outputIndices,
            count: count,
            commandBuffer: commandBuffer
        )
    }

    // MARK: - OneSweep Path (MSL 4.1)

    /// OneSweep radix sort: build keys, one all-digit histogram sweep, one scan,
    /// then a single chained-lookback binning dispatch per digit. Everything is
    /// encoded into one serial compute encoder (12 dispatches, vs ~26 encoders
    /// for the legacy path).
    private func sortOneSweep(
        splats: MTLBuffer,
        count: Int,
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool,
        outputIndices: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let keysA = keysBufferA,
              let keysB = keysBufferB,
              let globalHist = oneSweepHistBuffer,
              let status = oneSweepStatusBuffer,
              let ticket = oneSweepTicketBuffer,
              let histogramPipeline = oneSweepHistogramPipeline,
              let scanPipeline = oneSweepScanPipeline,
              let resetPipeline = oneSweepResetPipeline,
              let digitPassPipeline = oneSweepDigitPassPipeline else {
            throw SplatRendererError.failedToCreateBuffer(length: 0)
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatRendererError.failedToCreateComputeEncoder
        }
        encoder.label = "OneSweep Radix Sort"

        var splatCount = UInt32(count)
        var camPos = cameraPosition
        var camFwd = cameraForward
        var byDistance = sortByDistance
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)

        // Step 1: Build sorting keys from splat positions
        encoder.setComputePipelineState(buildKeysPipeline)
        encoder.setBuffer(splats, offset: 0, index: 0)
        encoder.setBuffer(keysA, offset: 0, index: 1)
        encoder.setBytes(&camPos, length: MemoryLayout<SIMD3<Float>>.stride, index: 2)
        encoder.setBytes(&camFwd, length: MemoryLayout<SIMD3<Float>>.stride, index: 3)
        encoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&byDistance, length: MemoryLayout<Bool>.stride, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: (count + 255) / 256, height: 1, depth: 1),
                                     threadsPerThreadgroup: threadsPerGroup)

        // Step 2: Zero the 4x256 digit histograms (reset kernel also zeroes the ticket)
        var histEntryCount = UInt32(Self.radixPasses * Self.bucketsPerPass)
        encoder.setComputePipelineState(resetPipeline)
        encoder.setBuffer(globalHist, offset: 0, index: 0)
        encoder.setBuffer(ticket, offset: 0, index: 1)
        encoder.setBytes(&histEntryCount, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: Self.radixPasses, height: 1, depth: 1),
                                     threadsPerThreadgroup: threadsPerGroup)

        // Step 3: All four digit histograms in a single sweep over the keys
        encoder.setComputePipelineState(histogramPipeline)
        encoder.setBuffer(keysA, offset: 0, index: 0)
        encoder.setBuffer(globalHist, offset: 0, index: 1)
        encoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 2)
        let histThreads = (count + 3) / 4  // each thread strides over ~4 keys
        encoder.dispatchThreadgroups(MTLSize(width: max(1, (histThreads + 255) / 256), height: 1, depth: 1),
                                     threadsPerThreadgroup: threadsPerGroup)

        // Step 4: Exclusive scan of each pass's histogram (in place)
        encoder.setComputePipelineState(scanPipeline)
        encoder.setBuffer(globalHist, offset: 0, index: 0)
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: threadsPerGroup)

        // Step 5: One chained-lookback binning dispatch per digit
        let numTiles = (count + Self.oneSweepTileSize - 1) / Self.oneSweepTileSize
        var statusCount = UInt32(numTiles * Self.bucketsPerPass)
        var inputKeys = keysA
        var outputKeys = keysB

        for pass in 0..<Self.radixPasses {
            var byteIndex = UInt32(pass)

            encoder.setComputePipelineState(resetPipeline)
            encoder.setBuffer(status, offset: 0, index: 0)
            encoder.setBuffer(ticket, offset: 0, index: 1)
            encoder.setBytes(&statusCount, length: MemoryLayout<UInt32>.stride, index: 2)
            encoder.dispatchThreadgroups(MTLSize(width: (Int(statusCount) + 255) / 256, height: 1, depth: 1),
                                         threadsPerThreadgroup: threadsPerGroup)

            encoder.setComputePipelineState(digitPassPipeline)
            encoder.setBuffer(inputKeys, offset: 0, index: 0)
            encoder.setBuffer(outputKeys, offset: 0, index: 1)
            encoder.setBuffer(globalHist, offset: 0, index: 2)
            encoder.setBuffer(status, offset: 0, index: 3)
            encoder.setBuffer(ticket, offset: 0, index: 4)
            encoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&byteIndex, length: MemoryLayout<UInt32>.stride, index: 6)
            encoder.dispatchThreadgroups(MTLSize(width: numTiles, height: 1, depth: 1),
                                         threadsPerThreadgroup: threadsPerGroup)

            swap(&inputKeys, &outputKeys)
        }

        // Step 6: Extract sorted indices (after 4 passes the result is in keysA)
        encoder.setComputePipelineState(extractIndicesPipeline)
        encoder.setBuffer(keysA, offset: 0, index: 0)
        encoder.setBuffer(outputIndices, offset: 0, index: 1)
        encoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.dispatchThreadgroups(MTLSize(width: (count + 255) / 256, height: 1, depth: 1),
                                     threadsPerThreadgroup: threadsPerGroup)

        encoder.endEncoding()
    }

    // MARK: - Private Encoding Methods

    private func encodeBuildKeys(
        splats: MTLBuffer,
        keys: MTLBuffer,
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool,
        count: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatRendererError.failedToCreateComputeEncoder
        }
        encoder.label = "Build Sorting Keys"

        encoder.setComputePipelineState(buildKeysPipeline)

        var camPos = cameraPosition
        var camFwd = cameraForward
        var splatCount = UInt32(count)
        var byDistance = sortByDistance

        encoder.setBuffer(splats, offset: 0, index: 0)
        encoder.setBuffer(keys, offset: 0, index: 1)
        encoder.setBytes(&camPos, length: MemoryLayout<SIMD3<Float>>.stride, index: 2)
        encoder.setBytes(&camFwd, length: MemoryLayout<SIMD3<Float>>.stride, index: 3)
        encoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&byDistance, length: MemoryLayout<Bool>.stride, index: 5)

        let threadsPerThreadgroup = MTLSize(width: min(256, buildKeysPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: (count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    private func encodeResetHistogram(
        histogram: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatRendererError.failedToCreateComputeEncoder
        }
        encoder.label = "Reset Histogram"

        encoder.setComputePipelineState(resetHistogramPipeline)
        encoder.setBuffer(histogram, offset: 0, index: 0)

        // 256 threads to reset 256 buckets
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encodeHistogram(
        keys: MTLBuffer,
        histogram: MTLBuffer,
        byteIndex: UInt32,
        count: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatRendererError.failedToCreateComputeEncoder
        }
        encoder.label = "Histogram Pass (byte \(byteIndex))"

        encoder.setComputePipelineState(histogramPipeline)

        var splatCount = UInt32(count)
        var byte = byteIndex

        encoder.setBuffer(keys, offset: 0, index: 0)
        encoder.setBuffer(histogram, offset: 0, index: 1)
        encoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&byte, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadsPerThreadgroup = MTLSize(width: min(256, histogramPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: (count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }

    private func encodePrefixSum(
        histogram: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatRendererError.failedToCreateComputeEncoder
        }
        encoder.label = "Prefix Sum"

        encoder.setComputePipelineState(prefixSumPipeline)
        encoder.setBuffer(histogram, offset: 0, index: 0)

        // Allocate threadgroup memory for Blelloch scan (256 uints)
        let threadgroupMemorySize = 256 * MemoryLayout<UInt32>.stride
        encoder.setThreadgroupMemoryLength(threadgroupMemorySize, index: 0)

        // Single threadgroup with 256 threads
        encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    /// Three-phase stable scatter for LSD radix sort correctness.
    /// Phase 1: Count elements per bucket per threadgroup (no atomic claiming)
    /// Phase 2: Compute deterministic block offsets via prefix sum across threadgroups
    /// Phase 3: Write elements in stable order using pre-computed offsets
    ///
    /// This guarantees threadgroup ordering: TG0's elements come before TG1's, etc.
    /// which is essential for LSD radix sort stability across passes.
    private func encodeScatter(
        inputKeys: MTLBuffer,
        outputKeys: MTLBuffer,
        histogram: MTLBuffer,
        tgOffsets: MTLBuffer,
        byteIndex: UInt32,
        count: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let tgCounts = tgBucketCountsBuffer else {
            throw SplatRendererError.failedToCreateBuffer(length: 0)
        }

        var splatCount = UInt32(count)
        var byte = byteIndex

        let threadsPerThreadgroup = MTLSize(width: Self.scatterThreadgroupSize, height: 1, depth: 1)
        let numThreadgroups = (count + Self.scatterThreadgroupSize - 1) / Self.scatterThreadgroupSize
        let threadgroupsPerGrid = MTLSize(width: numThreadgroups, height: 1, depth: 1)
        var numTGs = UInt32(numThreadgroups)

        // Threadgroup memory size for local histogram (256 uints)
        let localHistogramSize = Self.bucketsPerPass * MemoryLayout<UInt32>.stride

        // Phase 1: Count elements per bucket per threadgroup
        // Stores counts to tgCounts[threadgroup_id * 256 + bucket]
        guard let countEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatRendererError.failedToCreateComputeEncoder
        }
        countEncoder.label = "Scatter Count (byte \(byteIndex))"

        countEncoder.setComputePipelineState(scatterCountPipeline)
        countEncoder.setBuffer(inputKeys, offset: 0, index: 0)
        countEncoder.setBuffer(tgCounts, offset: 0, index: 1)   // Per-threadgroup bucket counts
        countEncoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 2)
        countEncoder.setBytes(&byte, length: MemoryLayout<UInt32>.stride, index: 3)
        countEncoder.setThreadgroupMemoryLength(localHistogramSize, index: 0)

        countEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        countEncoder.endEncoding()

        // Phase 2: Compute deterministic block offsets via prefix sum across threadgroups
        // For each bucket, iterates through TGs in order to compute starting positions
        guard let offsetsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatRendererError.failedToCreateComputeEncoder
        }
        offsetsEncoder.label = "Scatter Offsets (byte \(byteIndex))"

        offsetsEncoder.setComputePipelineState(scatterOffsetsPipeline)
        offsetsEncoder.setBuffer(histogram, offset: 0, index: 0)   // Global bucket offsets (prefix sum result)
        offsetsEncoder.setBuffer(tgCounts, offset: 0, index: 1)    // Per-threadgroup bucket counts
        offsetsEncoder.setBuffer(tgOffsets, offset: 0, index: 2)   // Output: per-threadgroup bucket offsets
        offsetsEncoder.setBytes(&numTGs, length: MemoryLayout<UInt32>.stride, index: 3)

        // Single threadgroup with 256 threads (one per bucket)
        offsetsEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        offsetsEncoder.endEncoding()

        // Phase 3: Write elements in stable order using pre-computed offsets
        // Each thread computes its local rank within its threadgroup's bucket
        guard let writeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatRendererError.failedToCreateComputeEncoder
        }
        writeEncoder.label = "Scatter Write (byte \(byteIndex))"

        writeEncoder.setComputePipelineState(scatterWritePipeline)
        writeEncoder.setBuffer(inputKeys, offset: 0, index: 0)
        writeEncoder.setBuffer(outputKeys, offset: 0, index: 1)
        writeEncoder.setBuffer(tgOffsets, offset: 0, index: 2)     // Pre-computed deterministic offsets
        writeEncoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 3)
        writeEncoder.setBytes(&byte, length: MemoryLayout<UInt32>.stride, index: 4)

        // Threadgroup memory for local counts and prefix sums (2 * 256 uints)
        writeEncoder.setThreadgroupMemoryLength(localHistogramSize, index: 0)  // local_counts
        writeEncoder.setThreadgroupMemoryLength(localHistogramSize, index: 1)  // local_prefix

        writeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        writeEncoder.endEncoding()
    }

    private func encodeExtractIndices(
        sortedKeys: MTLBuffer,
        outputIndices: MTLBuffer,
        count: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SplatRendererError.failedToCreateComputeEncoder
        }
        encoder.label = "Extract Sorted Indices"

        encoder.setComputePipelineState(extractIndicesPipeline)

        var splatCount = UInt32(count)

        encoder.setBuffer(sortedKeys, offset: 0, index: 0)
        encoder.setBuffer(outputIndices, offset: 0, index: 1)
        encoder.setBytes(&splatCount, length: MemoryLayout<UInt32>.stride, index: 2)

        let threadsPerThreadgroup = MTLSize(width: min(256, extractIndicesPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: (count + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}
