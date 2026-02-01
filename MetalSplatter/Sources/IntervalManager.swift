import Metal
import simd
import os

/// Represents a contiguous range of splats that can be processed together.
/// Used for LOD streaming where different scene regions may be loaded/unloaded dynamically.
public struct SplatInterval: Sendable {
    /// Starting index in the global splat buffer
    public var sourceStart: Int

    /// Ending index (exclusive) in the global splat buffer
    public var sourceEnd: Int

    /// Starting index in the remapped (output) space after culling/filtering
    public var targetStart: Int

    /// Priority for rendering (higher = more important, rendered first if budget exceeded)
    public var priority: Float

    /// LOD level (0 = highest detail)
    public var lodLevel: Int

    /// Whether this interval is currently visible (after frustum culling)
    public var isVisible: Bool

    /// Number of splats in this interval
    public var count: Int {
        sourceEnd - sourceStart
    }

    public init(
        sourceStart: Int,
        sourceEnd: Int,
        targetStart: Int = 0,
        priority: Float = 1.0,
        lodLevel: Int = 0,
        isVisible: Bool = true
    ) {
        self.sourceStart = sourceStart
        self.sourceEnd = sourceEnd
        self.targetStart = targetStart
        self.priority = priority
        self.lodLevel = lodLevel
        self.isVisible = isVisible
    }
}

/// Manages interval-based remapping for splat rendering.
/// Enables partial scene updates and efficient LOD streaming by processing
/// splats in intervals rather than globally.
///
/// The interval system works as follows:
/// 1. Scene is divided into intervals (e.g., octree nodes)
/// 2. Each interval tracks its source range in the global splat buffer
/// 3. Visible intervals are remapped to a contiguous target range
/// 4. GPU uses interval texture to remap global indices to local indices
///
/// This supports:
/// - Partial scene loading/unloading without full buffer reallocation
/// - LOD-based interval skipping for performance
/// - Priority-based rendering when over budget
public class IntervalManager {

    private static let log = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "MetalSplatter",
        category: "IntervalManager"
    )

    private let device: MTLDevice

    /// All registered intervals
    private(set) var intervals: [SplatInterval] = []

    /// Active (visible) intervals after culling
    private(set) var activeIntervals: [SplatInterval] = []

    /// Total splat count across all active intervals
    private(set) var activeSplatCount: Int = 0

    /// GPU texture for interval remapping (1D texture storing interval lookup data)
    /// Format: Each texel contains (intervalIndex, localOffset) for remapping
    private var intervalTexture: MTLTexture?

    /// CPU-side interval lookup buffer (for debugging/validation)
    private var intervalLookupBuffer: MTLBuffer?

    /// Maximum number of intervals supported
    public let maxIntervals: Int

    /// Structure matching Metal shader for interval lookup
    struct IntervalLookupEntry {
        var intervalIndex: UInt32
        var sourceOffset: UInt32    // Offset from interval source start
        var targetOffset: UInt32    // Offset in remapped output space
        var reserved: UInt32        // Padding for alignment
    }

    public init(device: MTLDevice, maxIntervals: Int = 1024) {
        self.device = device
        self.maxIntervals = maxIntervals
    }

    /// Registers a new interval with the manager.
    /// - Parameter interval: The interval to register
    /// - Returns: Index of the registered interval
    @discardableResult
    public func registerInterval(_ interval: SplatInterval) -> Int {
        let index = intervals.count
        intervals.append(interval)
        return index
    }

    /// Clears all registered intervals
    public func clearIntervals() {
        intervals.removeAll()
        activeIntervals.removeAll()
        activeSplatCount = 0
    }

    /// Updates interval visibility based on frustum culling results.
    /// Call this after frustum culling to mark which intervals are visible.
    ///
    /// - Parameter visibleIndices: Set of interval indices that are visible
    public func updateVisibility(visibleIndices: Set<Int>) {
        for i in intervals.indices {
            intervals[i].isVisible = visibleIndices.contains(i)
        }
        rebuildActiveIntervals()
    }

    /// Updates all intervals to be visible (no culling)
    public func setAllVisible() {
        for i in intervals.indices {
            intervals[i].isVisible = true
        }
        rebuildActiveIntervals()
    }

    /// Rebuilds the active interval list and computes target offsets.
    /// Active intervals are sorted by priority and assigned contiguous target ranges.
    private func rebuildActiveIntervals() {
        // Filter to visible intervals
        activeIntervals = intervals.filter { $0.isVisible }

        // Sort by priority (higher priority first) then by LOD level (lower = more detail first)
        activeIntervals.sort { a, b in
            if a.priority != b.priority {
                return a.priority > b.priority
            }
            return a.lodLevel < b.lodLevel
        }

        // Assign target offsets via prefix sum
        var currentTarget = 0
        for i in activeIntervals.indices {
            activeIntervals[i].targetStart = currentTarget
            currentTarget += activeIntervals[i].count
        }

        activeSplatCount = currentTarget
    }

    /// Creates or updates the interval texture for GPU-based remapping.
    /// The texture is a 1D lookup that maps global splat indices to (interval, localOffset).
    ///
    /// - Parameter totalSplatCount: Total number of splats in the global buffer
    /// - Returns: The interval texture, or nil if creation failed
    @discardableResult
    public func buildIntervalTexture(totalSplatCount: Int) -> MTLTexture? {
        guard totalSplatCount > 0 else { return nil }

        // Create texture descriptor for 1D lookup
        // Using RG32Uint: R = interval index, G = local offset within interval
        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = .type1D
        descriptor.pixelFormat = .rg32Uint
        descriptor.width = totalSplatCount
        descriptor.usage = [.shaderRead]
        descriptor.storageMode = .shared

        // Recreate texture if size changed
        if intervalTexture?.width != totalSplatCount {
            intervalTexture = device.makeTexture(descriptor: descriptor)
            intervalTexture?.label = "Interval Remap Texture"
        }

        guard let texture = intervalTexture else {
            Self.log.error("Failed to create interval texture")
            return nil
        }

        // Build lookup data on CPU
        // For each global splat index, store which interval it belongs to and its local offset
        var lookupData = [SIMD2<UInt32>](repeating: SIMD2<UInt32>(0, 0), count: totalSplatCount)

        for (intervalIdx, interval) in activeIntervals.enumerated() {
            for i in interval.sourceStart..<interval.sourceEnd {
                guard i < totalSplatCount else { continue }
                let localOffset = i - interval.sourceStart
                lookupData[i] = SIMD2<UInt32>(UInt32(intervalIdx), UInt32(localOffset))
            }
        }

        // Upload to texture
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: totalSplatCount, height: 1, depth: 1))
        lookupData.withUnsafeBytes { ptr in
            texture.replace(region: region,
                           mipmapLevel: 0,
                           withBytes: ptr.baseAddress!,
                           bytesPerRow: totalSplatCount * MemoryLayout<SIMD2<UInt32>>.stride)
        }

        return texture
    }

    /// Creates a buffer containing interval metadata for GPU access.
    /// This allows shaders to look up interval properties (targetStart, count, etc.)
    ///
    /// - Returns: Buffer containing interval metadata, or nil if creation failed
    public func buildIntervalMetadataBuffer() -> MTLBuffer? {
        guard !activeIntervals.isEmpty else { return nil }

        struct IntervalMetadata {
            var sourceStart: UInt32
            var sourceEnd: UInt32
            var targetStart: UInt32
            var lodLevel: UInt32
        }

        var metadata = activeIntervals.map { interval in
            IntervalMetadata(
                sourceStart: UInt32(interval.sourceStart),
                sourceEnd: UInt32(interval.sourceEnd),
                targetStart: UInt32(interval.targetStart),
                lodLevel: UInt32(interval.lodLevel)
            )
        }

        let bufferSize = metadata.count * MemoryLayout<IntervalMetadata>.stride
        let buffer = device.makeBuffer(bytes: &metadata,
                                        length: bufferSize,
                                        options: .storageModeShared)
        buffer?.label = "Interval Metadata Buffer"

        return buffer
    }

    /// Remaps a global splat index to a target index based on active intervals.
    /// This is the CPU version for validation/debugging.
    ///
    /// - Parameter globalIndex: Index in the global splat buffer
    /// - Returns: Remapped index in the output buffer, or nil if not in any active interval
    public func remapIndex(_ globalIndex: Int) -> Int? {
        for interval in activeIntervals {
            if globalIndex >= interval.sourceStart && globalIndex < interval.sourceEnd {
                let localOffset = globalIndex - interval.sourceStart
                return interval.targetStart + localOffset
            }
        }
        return nil
    }

    /// Checks if a global index is within any active interval.
    ///
    /// - Parameter globalIndex: Index to check
    /// - Returns: True if the index is in an active interval
    public func isIndexActive(_ globalIndex: Int) -> Bool {
        for interval in activeIntervals {
            if globalIndex >= interval.sourceStart && globalIndex < interval.sourceEnd {
                return true
            }
        }
        return false
    }

    /// Returns statistics about current interval state
    public struct Statistics {
        public let totalIntervals: Int
        public let activeIntervals: Int
        public let activeSplatCount: Int
        public let intervalCoverage: Float  // Percentage of total splats that are active
    }

    public func getStatistics(totalSplatCount: Int) -> Statistics {
        let coverage = totalSplatCount > 0 ? Float(activeSplatCount) / Float(totalSplatCount) : 0
        return Statistics(
            totalIntervals: intervals.count,
            activeIntervals: activeIntervals.count,
            activeSplatCount: activeSplatCount,
            intervalCoverage: coverage
        )
    }
}
