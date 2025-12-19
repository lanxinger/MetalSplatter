import Metal
import simd
import os

/// Camera-relative binned precision sorting inspired by PlayCanvas
/// Allocates more sort precision to near-camera splats where visual quality matters most
internal class BinnedSorter {

    private static let log = Logger(subsystem: Bundle.main.bundleIdentifier ?? "MetalSplatter",
                                    category: "BinnedSorter")

    // Match Metal shader constants
    private static let numBins: Int = 32
    private static let chunkSize: Int = 256

    // Bin parameters structure matching Metal shader
    struct BinParameters {
        var binBase: [UInt32]     // NUM_BINS + 1 entries
        var binDivider: [UInt32]  // NUM_BINS + 1 entries

        init() {
            binBase = Array(repeating: 0, count: BinnedSorter.numBins + 1)
            binDivider = Array(repeating: 0, count: BinnedSorter.numBins + 1)
        }
    }

    private let device: MTLDevice
    private let setupBinsPipeline: MTLComputePipelineState
    private let computeDistancesPipeline: MTLComputePipelineState

    private var binParametersBuffer: MTLBuffer?

    internal init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device

        // Load compute functions
        guard let setupFunction = library.makeFunction(name: "setupCameraRelativeBins") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "setupCameraRelativeBins")
        }

        guard let computeFunction = library.makeFunction(name: "computeSplatDistancesBinned") else {
            throw SplatRendererError.failedToLoadShaderFunction(name: "computeSplatDistancesBinned")
        }

        // Create pipeline states
        setupBinsPipeline = try device.makeComputePipelineState(function: setupFunction)
        computeDistancesPipeline = try device.makeComputePipelineState(function: computeFunction)

        // Allocate bin parameters buffer
        let binParamsSize = MemoryLayout<UInt32>.stride * (Self.numBins + 1) * 2
        guard let buffer = device.makeBuffer(length: binParamsSize, options: .storageModeShared) else {
            throw SplatRendererError.failedToCreateBuffer(length: binParamsSize)
        }
        binParametersBuffer = buffer
    }

    /// Computes AABB bounds for all splats
    /// Returns (minDistance, maxDistance) based on sort mode
    internal func computeDistanceBounds(
        splats: UnsafeBufferPointer<SplatRenderer.Splat>,
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool
    ) -> (min: Float, max: Float) {

        var minDist: Float = .infinity
        var maxDist: Float = -.infinity

        for splat in splats {
            let splatPos = SIMD3<Float>(splat.position.x, splat.position.y, splat.position.z)

            let dist: Float
            if sortByDistance {
                // Radial distance from camera
                let delta = splatPos - cameraPosition
                dist = simd_length(delta)
            } else {
                // Projected distance along forward vector
                let delta = splatPos - cameraPosition
                dist = simd_dot(delta, cameraForward)
            }

            minDist = min(minDist, dist)
            maxDist = max(maxDist, dist)
        }

        // Handle edge cases
        if minDist.isInfinite {
            minDist = 0
            maxDist = 0
        }

        return (minDist, maxDist)
    }

    /// Sets up camera-relative bins with weighted precision allocation
    internal func setupBins(
        commandBuffer: MTLCommandBuffer,
        minDist: Float,
        maxDist: Float,
        cameraPosition: SIMD3<Float>,
        sortByDistance: Bool,
        compareBits: UInt32
    ) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let binBuffer = binParametersBuffer else {
            Self.log.error("Failed to create compute encoder for bin setup")
            return
        }

        var minDistVar = minDist
        var maxDistVar = maxDist
        var cameraPos = cameraPosition
        var sortByDist = sortByDistance
        var bits = compareBits

        computeEncoder.setComputePipelineState(setupBinsPipeline)
        computeEncoder.setBytes(&minDistVar, length: MemoryLayout<Float>.size, index: 0)
        computeEncoder.setBytes(&maxDistVar, length: MemoryLayout<Float>.size, index: 1)
        computeEncoder.setBytes(&cameraPos, length: MemoryLayout<SIMD3<Float>>.size, index: 2)
        computeEncoder.setBytes(&sortByDist, length: MemoryLayout<Bool>.size, index: 3)
        computeEncoder.setBytes(&bits, length: MemoryLayout<UInt32>.size, index: 4)
        computeEncoder.setBuffer(binBuffer, offset: 0, index: 5)

        // Only one thread needed for setup
        computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        computeEncoder.endEncoding()
    }

    /// Sets up bins using a chunk histogram on the CPU (closer to PlayCanvas parity).
    internal func setupBinsWithChunkHistogram(
        minDist: Float,
        maxDist: Float,
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool,
        compareBits: UInt32,
        splats: UnsafeBufferPointer<SplatRenderer.Splat>
    ) {
        guard let binBuffer = binParametersBuffer else {
            Self.log.error("Failed to access bin parameters buffer")
            return
        }

        let range = max(maxDist - minDist, 0.001)
        let bucketCount = Int((1 << compareBits) + 1)
        let binCount = Self.numBins

        var histogram = Array(repeating: 0, count: binCount)
        let chunkCount = (splats.count + Self.chunkSize - 1) / Self.chunkSize

        for chunkIndex in 0..<chunkCount {
            let start = chunkIndex * Self.chunkSize
            let end = min(start + Self.chunkSize, splats.count)

            var minPos = SIMD3<Float>(repeating: .infinity)
            var maxPos = SIMD3<Float>(repeating: -.infinity)

            for i in start..<end {
                let pos = SIMD3<Float>(splats[i].position.x, splats[i].position.y, splats[i].position.z)
                minPos = simd.min(minPos, pos)
                maxPos = simd.max(maxPos, pos)
            }

            let center = (minPos + maxPos) * 0.5
            let radius = simd_length(maxPos - minPos) * 0.5

            let dist: Float
            if sortByDistance {
                dist = simd_distance(center, cameraPosition)
            } else {
                dist = simd_dot(center - cameraPosition, cameraForward)
            }

            let binMinFloat = ((dist - radius - minDist) / range) * Float(binCount)
            let binMaxFloat = ((dist + radius - minDist) / range) * Float(binCount)
            let binMin = max(0, min(binCount - 1, Int(floor(binMinFloat))))
            let binMax = max(0, min(binCount, Int(ceil(binMaxFloat))))

            if binMax > binMin {
                for bin in binMin..<binMax {
                    histogram[bin] += 1
                }
            }
        }

        let histogramSum = histogram.reduce(0, +)
        let total = max(1, histogramSum)
        var binBase = Array(repeating: UInt32(0), count: binCount + 1)
        var binDivider = Array(repeating: UInt32(0), count: binCount + 1)

        var accumulated: UInt32 = 0
        if histogramSum == 0 {
            let baseDivider = max(1, bucketCount / binCount)
            for i in 0..<binCount {
                binDivider[i] = UInt32(baseDivider)
                binBase[i] = accumulated
                accumulated &+= binDivider[i]
            }
        } else {
            for i in 0..<binCount {
                let weight = Float(histogram[i]) / Float(total)
                let divider = max(1, Int(floor(weight * Float(bucketCount))))
                binDivider[i] = UInt32(divider)
                binBase[i] = accumulated
                accumulated &+= binDivider[i]
            }
        }

        if accumulated > UInt32(bucketCount) {
            let excess = accumulated - UInt32(bucketCount)
            if binDivider[binCount - 1] > excess {
                binDivider[binCount - 1] -= excess
            }
        }

        binBase[binCount] = binBase[binCount - 1] + binDivider[binCount - 1]
        binDivider[binCount] = 0

        let basePtr = binBuffer.contents().bindMemory(to: UInt32.self,
                                                     capacity: (binCount + 1) * 2)
        for i in 0..<(binCount + 1) {
            basePtr[i] = binBase[i]
            basePtr[binCount + 1 + i] = binDivider[i]
        }
    }

    /// Computes binned distances for all splats
    internal func computeBinnedDistances(
        commandBuffer: MTLCommandBuffer,
        splatBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool,
        splatCount: Int,
        minDist: Float,
        range: Float
    ) {
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
              let binBuffer = binParametersBuffer else {
            Self.log.error("Failed to create compute encoder for distance computation")
            return
        }

        var cameraPos = cameraPosition
        var cameraFwd = cameraForward
        var sortByDist = sortByDistance
        var count = UInt32(splatCount)
        var minDistVar = minDist
        var rangeVar = range

        computeEncoder.setComputePipelineState(computeDistancesPipeline)
        computeEncoder.setBuffer(splatBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&cameraPos, length: MemoryLayout<SIMD3<Float>>.size, index: 2)
        computeEncoder.setBytes(&cameraFwd, length: MemoryLayout<SIMD3<Float>>.size, index: 3)
        computeEncoder.setBytes(&sortByDist, length: MemoryLayout<Bool>.size, index: 4)
        computeEncoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 5)
        computeEncoder.setBytes(&minDistVar, length: MemoryLayout<Float>.size, index: 6)
        computeEncoder.setBytes(&rangeVar, length: MemoryLayout<Float>.size, index: 7)
        computeEncoder.setBuffer(binBuffer, offset: 0, index: 8)

        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (splatCount + 255) / 256, height: 1, depth: 1)

        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
    }
}
