import Foundation
import Metal
import MetalKit
import os
import SplatIO

#if arch(x86_64)
typealias Float16 = Float
#warning("x86_64 targets are unsupported by MetalSplatter and will fail at runtime. MetalSplatter builds on x86_64 only because Xcode builds Swift Packages as universal binaries and provides no way to override this. When Swift supports Float16 on x86_64, this may be revisited.")
#endif

// MARK: - Error Types

public enum SplatRendererError: LocalizedError {
    case metalDeviceUnavailable
    case failedToCreateBuffer(length: Int)
    case failedToCreateLibrary(underlying: Error)
    case failedToCreateDepthStencilState
    case failedToLoadShaderFunction(name: String)
    case failedToCreateComputePipelineState(functionName: String, underlying: Error)
    case failedToCreateRenderPipelineState(label: String, underlying: Error)
    case bundleIdentifierUnavailable
    case unsupportedArchitecture
    case failedToCreateRenderEncoder
    case failedToCreateComputeEncoder

    public var errorDescription: String? {
        switch self {
        case .metalDeviceUnavailable:
            return "Metal rendering is not available on this device"
        case .failedToCreateBuffer(let length):
            return "Failed to create Metal buffer with length \(length) bytes"
        case .failedToCreateLibrary(let underlying):
            return "Failed to create Metal shader library: \(underlying.localizedDescription)"
        case .failedToCreateDepthStencilState:
            return "Failed to create Metal depth stencil state"
        case .failedToLoadShaderFunction(let name):
            return "Failed to load required shader function: \"\(name)\""
        case .failedToCreateComputePipelineState(let functionName, let underlying):
            return "Failed to create compute pipeline state for function \"\(functionName)\": \(underlying.localizedDescription)"
        case .failedToCreateRenderPipelineState(let label, let underlying):
            return "Failed to create render pipeline state \"\(label)\": \(underlying.localizedDescription)"
        case .bundleIdentifierUnavailable:
            return "Bundle identifier is not available"
        case .unsupportedArchitecture:
            return "MetalSplatter is unsupported on Intel architecture (x86_64)"
        case .failedToCreateRenderEncoder:
            return "Failed to create Metal render command encoder"
        case .failedToCreateComputeEncoder:
            return "Failed to create Metal compute command encoder"
        }
    }
}

public class SplatRenderer: @unchecked Sendable {
    enum Constants {
        // Keep in sync with Shaders.metal : maxViewCount
        static let maxViewCount = 2
        // Sort by euclidian distance squared from camera position (true), or along the "forward" vector (false)
        // TODO: compare the behaviour and performance of sortByDistance
        // notes: sortByDistance introduces unstable artifacts when you get close to an object; whereas !sortByDistance introduces artifacts are you turn -- but they're a little subtler maybe?
        static let sortByDistance = true
        // Only store indices for 1024 splats; for the remainder, use instancing of these existing indices.
        // Setting to 1 uses only instancing (with a significant performance penalty); setting to a number higher than the splat count
        // uses only indexing (with a significant memory penalty for th elarge index array, and a small performance penalty
        // because that can't be cached as easiliy). Anywhere within an order of magnitude (or more?) of 1k seems to be the sweet spot,
        // with effectively no memory penalty compated to instancing, and slightly better performance than even using all indexing.
        static let maxIndexedSplatCount = 1024

        static let tileSize = MTLSize(width: 32, height: 32, depth: 1)
        
        // LOD system constants
        static let maxRenderDistance: Float = 100.0
        static let lodDistanceThresholds: [Float] = [10.0, 25.0, 50.0]
        static let lodSkipFactors: [Int] = [1, 2, 4, 8] // Skip every Nth splat based on distance
    }

    internal static let log =
        Logger(subsystem: Bundle.module.bundleIdentifier ?? "com.metalsplatter.unknown",
               category: "SplatRenderer")
    
    public struct DebugOptions: OptionSet, Sendable {
        public let rawValue: UInt32
        public init(rawValue: UInt32) { self.rawValue = rawValue }

        public static let overdraw = DebugOptions(rawValue: 1 << 0)
        public static let lodTint  = DebugOptions(rawValue: 1 << 1)
        public static let showAABB = DebugOptions(rawValue: 1 << 2)
    }
    
    public struct FrameStatistics {
        public let ready: Bool
        public let loadingCount: Int
        public let sortDuration: TimeInterval?
        public let bufferUploadCount: Int
        public let splatCount: Int
        public let frameTime: TimeInterval

        // Buffer pool statistics for performance monitoring
        public struct BufferPoolStats {
            public let availableBuffers: Int
            public let leasedBuffers: Int
            public let totalMemoryMB: Float
        }

        public let sortBufferPoolStats: BufferPoolStats?

        // Sort queue status
        public let sortJobsInFlight: Int
    }
    
    private var computeDepthsPipelineState: MTLComputePipelineState?
    private var computeDistancesPipelineState: MTLComputePipelineState?
    private var frustumCullPipelineState: MTLComputePipelineState?
    
    // Frustum culling buffers and state
    private var visibleIndicesBuffer: MTLBuffer?
    private var visibleCountBuffer: MTLBuffer?
    private var frustumCullDataBuffer: MTLBuffer?
    private var indirectDrawArgsBuffer: MTLBuffer?  // For GPU-driven indirect draw
    private var generateIndirectArgsPipelineState: MTLComputePipelineState?
    private var resetVisibleCountPipelineState: MTLComputePipelineState?
    public var frustumCullingEnabled = false  // Enable via settings
    private var lastVisibleCount: Int = 0
    
    // SIMD-group parallel bounds computation
    private var computeBoundsPipelineState: MTLComputePipelineState?
    private var resetBoundsPipelineState: MTLComputePipelineState?
    private var boundsMinBuffer: MTLBuffer?  // 3 atomic floats for min bounds
    private var boundsMaxBuffer: MTLBuffer?  // 3 atomic floats for max bounds
    
    // Cached bounds - computed once on GPU, reused until splats change
    private var cachedBounds: (min: SIMD3<Float>, max: SIMD3<Float>)?
    private var boundsDirty = true
    private var boundsComputationInProgress = false
    private let boundsLock = NSLock()
    
    // Metal 4 TensorOps batch precompute (pre-computes covariance/transforms)
    private var batchPrecomputePipelineState: MTLComputePipelineState?
    private var precomputedSplatBuffer: MTLBuffer?
    private var precomputedDataDirty = true
    private var lastPrecomputeViewMatrix: simd_float4x4?
    public var batchPrecomputeEnabled = false  // Enable for large scenes
    
    // PrecomputedSplat structure size (must match Metal shader with proper alignment)
    // Metal alignment: float4 (16) + float3 (12+4 padding) + float2 (8) + float2 (8) + float (4) + uint (4)
    // = 56 bytes, rounded to 64 due to struct's 16-byte alignment (from float4/float3)
    private static let precomputedSplatStride = 64
    
    public struct ViewportDescriptor {
        public var viewport: MTLViewport
        public var projectionMatrix: simd_float4x4
        public var viewMatrix: simd_float4x4
        public var screenSize: SIMD2<Int>

        public init(viewport: MTLViewport, projectionMatrix: simd_float4x4, viewMatrix: simd_float4x4, screenSize: SIMD2<Int>) {
            self.viewport = viewport
            self.projectionMatrix = projectionMatrix
            self.viewMatrix = viewMatrix
            self.screenSize = screenSize
        }
    }

    // Keep in sync with ShaderCommon.h : BufferIndex
    enum BufferIndex: NSInteger {
        case uniforms       = 0
        case splat          = 1
        case sortedIndices  = 2  // GPU-side sorted indices for indirect rendering
        case precomputed    = 3  // Precomputed splat data (Metal 4 TensorOps)
        case packedColors   = 4  // Optional packed colors (snorm10a2)
    }

    // Keep in sync with Shaders.metal : Uniforms
    struct Uniforms {
        var projectionMatrix: matrix_float4x4
        var viewMatrix: matrix_float4x4
        var screenSize: SIMD2<UInt32> // Size of screen in pixels

        var splatCount: UInt32
        var indexedSplatCount: UInt32
        var debugFlags: UInt32
        var lodThresholds: SIMD3<Float>
    }

    // Keep in sync with Shaders.metal : UniformsArray
    struct UniformsArray {
        // maxViewCount = 2, so we have 2 entries
        var uniforms0: Uniforms
        var uniforms1: Uniforms

        // The 256 byte aligned size of our uniform structure
        static var alignedSize: Int { (MemoryLayout<UniformsArray>.size + 0xFF) & -0x100 }

        mutating func setUniforms(index: Int, _ uniforms: Uniforms) {
            switch index {
            case 0: uniforms0 = uniforms
            case 1: uniforms1 = uniforms
            default: break
            }
        }
    }
    
    // Keep in sync with FrustumCulling.metal : FrustumCullData
    // Simplified struct using view-projection matrix directly for NDC-based culling
    struct FrustumCullData {
        var viewProjectionMatrix: matrix_float4x4
        var cameraPosition: SIMD3<Float>
        var padding1: Float
        var maxDistance: Float
        var padding2: SIMD3<Float>
        
        init() {
            viewProjectionMatrix = matrix_identity_float4x4
            cameraPosition = .zero
            padding1 = 0
            maxDistance = 10000.0  // Large default, effectively disabled
            padding2 = .zero
        }
    }

    struct PackedHalf3 {
        var x: Float16
        var y: Float16
        var z: Float16
    }

    struct PackedRGBHalf4 {
        var r: Float16
        var g: Float16
        var b: Float16
        var a: Float16
    }

    // Keep in sync with Shaders.metal : Splat
    struct Splat {
        var position: MTLPackedFloat3
        var color: PackedRGBHalf4
        var covA: PackedHalf3
        var covB: PackedHalf3
    }

    struct SplatIndexAndDepth {
        var index: UInt32
        var depth: Float
    }

    public let device: MTLDevice
    public let colorFormat: MTLPixelFormat
    public let depthFormat: MTLPixelFormat
    public let sampleCount: Int
    public let maxViewCount: Int
    public let maxSimultaneousRenders: Int

    /**
     High-quality depth takes longer, but results in a continuous, more-representative depth buffer result, which is useful for reducing artifacts during Vision Pro's frame reprojection.
     */
    public var highQualityDepth: Bool = true

    private var writeDepth: Bool {
        depthFormat != .invalid
    }

    /**
     The SplatRenderer has two shader pipelines.
     - The single stage has a vertex shader, and a fragment shader. It can produce depth (or not), but the depth it produces is the depth of the nearest splat, whether it's visible or now.
     - The multi-stage pipeline uses a set of shaders which communicate using imageblock tile memory: initialization (which clears the tile memory), draw splats (similar to the single-stage
     pipeline but the end result is tile memory, not color+depth), and a post-process stage which merely copies the tile memory (color and optionally depth) to the frame's buffers.
     This is neccessary so that the primary stage can do its own blending -- of both color and depth -- by reading the previous values and writing new ones, which isn't possible without tile
     memory. Color blending works the same as the hardcoded path, but depth blending uses color alpha and results in mostly-transparent splats contributing only slightly to the depth,
     resulting in a much more continuous and representative depth value, which is important for reprojection on Vision Pro.
     */
    internal var useMultiStagePipeline: Bool {
#if targetEnvironment(simulator)
        false
#else
        writeDepth && highQualityDepth
#endif
    }

    public var clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0)

    public var onSortStart: (() -> Void)?
    public var onSortComplete: ((TimeInterval) -> Void)?
    public var onRenderStart: (() -> Void)?
    public var onRenderComplete: ((TimeInterval) -> Void)?
    public var onFrameReady: ((FrameStatistics) -> Void)?
    
    public var debugOptions: DebugOptions = []
    public var lodThresholds: SIMD3<Float> = {
        let thresholds = Constants.lodDistanceThresholds
        return SIMD3<Float>(thresholds[0], thresholds[1], thresholds[2])
    }()

    // MARK: - Morton Ordering

    /// When true, splats added via `add(_:)` will be reordered using Morton codes
    /// for improved GPU cache coherency. This is a one-time cost at load time that
    /// can significantly improve rendering performance for large scenes.
    ///
    /// Note: This only affects newly added splats. Existing splats are not reordered.
    /// For best results, enable this before loading splat data.
    public var mortonOrderingEnabled: Bool = true

    /// Threshold for using parallel Morton code computation.
    /// Scenes with more splats than this will use parallel processing.
    public var mortonParallelThreshold: Int = 100_000

    // MARK: - Dithered Transparency (Order-Independent)

    /// When true, uses stochastic (dithered) transparency instead of sorted alpha blending.
    /// This eliminates the need for depth sorting, providing order-independent transparency.
    ///
    /// **Benefits:**
    /// - No sorting overhead - significant performance improvement
    /// - Order-independent - no popping artifacts from sort order changes
    /// - Better for VR where sorting latency is problematic
    ///
    /// **Trade-offs:**
    /// - Produces noise/stippling pattern (best paired with TAA)
    /// - May look grainy without temporal anti-aliasing
    /// - Different visual aesthetic than smooth alpha blending
    ///
    /// Note: When enabled, sorting is still performed but can be deprioritized since
    /// visual correctness no longer depends on sort order.
    public var useDitheredTransparency: Bool = false {
        didSet {
            if useDitheredTransparency != oldValue {
                // Invalidate pipeline states to rebuild with correct settings
                invalidatePipelineStates()
            }
        }
    }

    public var sortPositionEpsilon: Float = 0.01
    public var sortDirectionEpsilon: Float = 0.0001  // ~0.5-1° rotation (reduced from 0.001 to fix flickering during rotation)
    public var minimumSortInterval: TimeInterval = 0

    // MARK: - Spherical Harmonics Update Thresholds

    /// Direction change threshold for SH re-evaluation (in dot product units).
    /// SH is only re-evaluated when (1 - dot(lastDir, currentDir)) > this threshold.
    /// Default 0.001 (~2.5° rotation) balances visual quality vs. computation.
    /// Set to 0 to update every frame; larger values reduce updates but may show lighting lag.
    public var shDirectionEpsilon: Float = 0.001

    /// Minimum interval between SH updates (seconds). Set to 0 for no limit.
    public var minimumSHUpdateInterval: TimeInterval = 0

    /// Last camera direction used for SH evaluation (for threshold comparison)
    internal var lastSHCameraDirection: SIMD3<Float>?

    /// Last time SH was updated
    internal var lastSHUpdateTime: CFAbsoluteTime = 0

    /// Flag indicating SH needs update due to data change
    internal var shDirtyDueToData: Bool = true

    // MARK: - Interaction Mode (Adaptive Quality)
    
    /// When true, sort parameters are relaxed for smoother interaction (less popping)
    public private(set) var isInteracting: Bool = false
    
    /// Stored "quality" sort parameters to restore after interaction
    private var qualitySortPositionEpsilon: Float = 0.01
    private var qualitySortDirectionEpsilon: Float = 0.0001
    private var qualityMinimumSortInterval: TimeInterval = 0
    
    /// Interaction mode sort parameters (relaxed for performance)
    public var interactionSortPositionEpsilon: Float = 0.05      // 5cm during interaction
    public var interactionSortDirectionEpsilon: Float = 0.003    // ~2-3° during interaction
    public var interactionMinimumSortInterval: TimeInterval = 0.033  // Max ~30 sorts/sec
    
    /// Delay before forcing a final high-quality sort after interaction ends
    public var postInteractionSortDelay: TimeInterval = 0.1
    private var interactionEndTime: CFAbsoluteTime?

    // Performance tracking
    private var frameStartTime: CFAbsoluteTime = 0
    private var lastFrameTime: TimeInterval = 0
    public var averageFrameTime: TimeInterval = 0
    private var frameCount: Int = 0
    private var lastSortDuration: TimeInterval?
    private var frameBufferUploads: Int = 0
    private var lastSortTime: CFAbsoluteTime = 0
    private var metal4LoggedOnce: Bool = false
    private var lastSplatCountLogged: Int = 0

    internal let library: MTLLibrary
    // Single-stage pipeline
    internal var singleStagePipelineState: MTLRenderPipelineState?
    internal var singleStageDepthState: MTLDepthStencilState?
    // Dithered transparency pipeline (order-independent, no sorting required)
    private var ditheredPipelineState: MTLRenderPipelineState?
    private var ditheredDepthState: MTLDepthStencilState?
    // Multi-stage pipeline
    private var initializePipelineState: MTLRenderPipelineState?
    internal var drawSplatPipelineState: MTLRenderPipelineState?
    internal var drawSplatDepthState: MTLDepthStencilState?
    private var postprocessPipelineState: MTLRenderPipelineState?
    private var postprocessDepthState: MTLDepthStencilState?

    // Mesh Shader Pipeline (Metal 3+, Apple Silicon)
    private var meshShaderPipelineState: MTLRenderPipelineState?
    private var meshShaderDepthState: MTLDepthStencilState?
    public var meshShaderEnabled = false
    private var meshShadersSupported = false
    
    /// Returns true if mesh shaders are supported on this device
    public var isMeshShaderSupported: Bool { meshShadersSupported }

    /// Returns true if mesh shaders can be safely used without quality regression
    /// Note: useCulledDitheredPath check stays inline in render() since it's a local
    private var canUseMeshShadersSafely: Bool {
        meshShadersSupported && !useMultiStagePipeline && !useDitheredTransparency
    }

    // Debug AABB rendering
    private var debugAABBPipelineState: MTLRenderPipelineState?
    private var debugAABBDepthState: MTLDepthStencilState?
    private var aabbVertexBuffer: MTLBuffer?
    private var aabbIndexBuffer: MTLBuffer?

    // dynamicUniformBuffers contains maxSimultaneousRenders uniforms buffers,
    // which we round-robin through, one per render; this is managed by switchToNextDynamicBuffer.
    // uniforms = the i'th buffer (where i = uniformBufferIndex, which varies from 0 to maxSimultaneousRenders-1)
    var dynamicUniformBuffers: MTLBuffer
    var uniformBufferOffset = 0
    internal var uniformBufferIndex = 0
    var uniforms: UnsafeMutablePointer<UniformsArray>

    // cameraWorldPosition and Forward vectors are the latest mean camera position across all viewports
    var cameraWorldPosition: SIMD3<Float> = .zero
    var cameraWorldForward: SIMD3<Float> = .init(x: 0, y: 0, z: -1)
    // Reference camera used to drive sorting (typically viewport[0])
    private var sortCameraPosition: SIMD3<Float> = .zero
    private var sortCameraForward: SIMD3<Float> = .init(x: 0, y: 0, z: -1)
    private var lastSortedCameraPosition: SIMD3<Float>?
    private var lastSortedCameraForward: SIMD3<Float>?
    private var sortDirtyDueToData = true
    private var sortDataRevision: UInt64 = 0

    // MARK: - Color-Only Update Path
    // Separate tracking for geometry vs color changes to skip unnecessary work.
    // When only colors change (e.g., SH re-evaluation), we can skip sorting entirely
    // since sort order depends only on position.

    /// Tracks whether geometry (position/covariance) has changed and needs re-sorting
    private var geometryDirty = true

    /// Tracks whether colors have changed and need GPU buffer update
    private var colorsDirty = true

    /// Revision counter for color-only updates (allows skipping sort on color changes)
    private var colorRevision: UInt64 = 0

    // MARK: - Staged Color Updates (GPU Race Prevention)
    // Color updates are staged and applied at render start to avoid CPU/GPU data races.
    // Direct writes to splatBuffer while GPU is reading cause undefined behavior.
    // Updates are applied in FIFO order to preserve "last write wins" semantics.

    /// Staged color update types - applied in order to preserve call semantics
    private enum PendingColorUpdate {
        case full([SIMD4<Float>])
        case range([SIMD4<Float>], Range<Int>)
        case single(SIMD4<Float>, Int)
    }

    /// Ordered queue of pending color updates (FIFO)
    private var pendingColorUpdates: [PendingColorUpdate] = []

    /// Lock protecting pending color update queue
    private var pendingColorUpdateLock = os_unfair_lock()

    typealias IndexType = UInt32
    
    // Buffer pools for efficient memory management
    private let splatBufferPool: MetalBufferPool<Splat>
    internal let indexBufferPool: MetalBufferPool<UInt32>

    // Sort buffer pools for GPU sorting operations (reuse across frames)
    private let sortDistanceBufferPool: MetalBufferPool<Float>
    private let sortIndexBufferPool: MetalBufferPool<Int32>

    // splatBuffer contains one entry for each gaussian splat (static, never reordered)
    var splatBuffer: MetalBuffer<Splat>
    
    // GPU-only sorting: sorted indices buffer holds the depth-sorted order
    // Shaders use this to index into splatBuffer in the correct render order
    // This eliminates CPU readback and reordering - a major performance win
    var sortedIndicesBuffer: MetalBuffer<Int32>?
    
    // Legacy: splatBufferPrime kept for CPU fallback sorting path
    var splatBufferPrime: MetalBuffer<Splat>

    var indexBuffer: MetalBuffer<UInt32>

    public var splatCount: Int { splatBuffer.count }

    var sorting = false
    private var sortJobsInFlight: Int = 0  // Track concurrent sort operations
    private let maxConcurrentSorts: Int = 2  // Allow overlap: one sorting while one renders
    private var sortStateLock = os_unfair_lock()  // Protects sortJobsInFlight and buffer swap
    var orderAndDepthTempSort: [SplatIndexAndDepth] = []

    // Metal 4 command buffer pool for improved performance
    private var commandBufferManager: CommandBufferManager
    
    // Async compute overlap: separate queue for sorting
    private var computeCommandQueue: MTLCommandQueue?
    private var computeCommandBufferManager: CommandBufferManager?
    
    // Double-buffered sorted indices for async overlap
    // While one is being used for rendering, the other can be filled by sorting
    private var sortedIndicesBufferA: MetalBuffer<Int32>?
    private var sortedIndicesBufferB: MetalBuffer<Int32>?
    private var usingSortedBufferA: Bool = true  // Which buffer is currently used for rendering

    // Cached arrays to avoid per-frame allocations
    private var cameraPositionsTemp: [SIMD3<Float>] = []
    private var cameraForwardsTemp: [SIMD3<Float>] = []
    private var viewMappingsTemp: [MTLVertexAmplificationViewMapping] = []

    // O(n) Counting Sort - faster than MPS argSort for large splat counts
    private var countingSorter: CountingSorter?

    // Cached MPS ArgSort - avoids graph recompilation per frame (perf fix)
    private lazy var cachedMPSArgSort: MPSArgSort = {
        MPSArgSort(dataType: .float32, descending: true)
    }()

    /// When true, uses O(n) counting sort instead of O(n log n) MPS argSort.
    /// Counting sort is faster for large splat counts (>50K) and provides
    /// sufficient depth precision (16-bit quantization) for visual correctness.
    /// Set to false to use the traditional MPS-based radix sort.
    public var useCountingSort: Bool = true

    /// When true, allocates more sorting precision to splats near the camera.
    /// This uses camera-relative bin weighting (PlayCanvas-style) where:
    /// - Camera bin gets 40x precision
    /// - Adjacent bins get 20x precision
    /// - Nearby bins get 8x precision
    /// - Medium distance gets 3x precision
    /// - Far bins get 1x precision
    /// This improves visual quality for close objects while saving precision budget on distant ones.
    /// Only effective when useCountingSort is true.
    public var useCameraRelativeBinning: Bool = true

    // Metal 4 Advanced Atomics Sorter - GPU radix sort for very large scenes
    // Note: Metal4Sorter is only available on iOS 26+, macOS 26+, visionOS 26+
    // Stored as AnyObject to avoid @available restrictions on stored properties
    private var _metal4Sorter: AnyObject?

    @available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
    private var metal4Sorter: Metal4Sorter? {
        get { _metal4Sorter as? Metal4Sorter }
        set { _metal4Sorter = newValue }
    }

    /// When true and Metal 4 is available, uses atomic radix sort for scenes >100K splats.
    /// This is opt-in because it requires Metal 4 and is only beneficial for very large scenes.
    /// For smaller scenes, counting sort remains more efficient.
    public var useMetal4Sorting: Bool = false

    /// Minimum splat count to use Metal 4 sorting (below this, counting sort is faster)
    public var metal4SortingThreshold: Int = 100_000

    // snorm10a2 color packing for bandwidth optimization
    // When enabled, colors are stored in 4 bytes instead of 8 bytes (50% reduction)
    // Note: May cause visible precision loss with SH data - test before enabling
    private var packedColorBuffer: MTLBuffer?

    /// When true, uses snorm10a2 packed colors (4 bytes) instead of half4 (8 bytes).
    /// This reduces memory bandwidth by 50% for color data but may cause
    /// precision loss that's visible with spherical harmonics. Opt-in only.
    /// Requires rebuilding splat buffer after changing this setting.
    public var usePackedColors: Bool = false {
        didSet {
            if usePackedColors != oldValue {
                rebuildPackedColorBufferIfNeeded()
                // Rebuild pipeline states with updated function constants
                resetPipelineStates()
                setupMeshShaders()  // Rebuild mesh shader pipeline with new constants
            }
        }
    }

    // MARK: - Thread-safe sort state accessors

    private func incrementSortJobsInFlight() {
        os_unfair_lock_lock(&sortStateLock)
        sortJobsInFlight += 1
        os_unfair_lock_unlock(&sortStateLock)
    }

    private func decrementSortJobsInFlight() {
        os_unfair_lock_lock(&sortStateLock)
        sortJobsInFlight -= 1
        os_unfair_lock_unlock(&sortStateLock)
    }

    private func getSortJobsInFlight() -> Int {
        os_unfair_lock_lock(&sortStateLock)
        defer { os_unfair_lock_unlock(&sortStateLock) }
        return sortJobsInFlight
    }

    private func canStartNewSort() -> Bool {
        os_unfair_lock_lock(&sortStateLock)
        defer { os_unfair_lock_unlock(&sortStateLock) }
        return sortJobsInFlight < maxConcurrentSorts
    }

    /// Atomically try to start a sort operation.
    /// Returns true if sort was started, false if already sorting or too many jobs in flight.
    private func tryStartSort() -> Bool {
        os_unfair_lock_lock(&sortStateLock)
        defer { os_unfair_lock_unlock(&sortStateLock) }

        // Check both conditions atomically
        guard !sorting && sortJobsInFlight < maxConcurrentSorts else {
            return false
        }

        // Set sorting flag and increment job count atomically
        sorting = true
        sortJobsInFlight += 1
        return true
    }

    /// Atomically mark sort as complete.
    private func finishSort() {
        os_unfair_lock_lock(&sortStateLock)
        sorting = false
        sortJobsInFlight -= 1
        os_unfair_lock_unlock(&sortStateLock)
    }

    /// Thread-safe access to the current sorted indices buffer for rendering
    private func getCurrentSortedIndicesBuffer() -> MetalBuffer<Int32>? {
        os_unfair_lock_lock(&sortStateLock)
        defer { os_unfair_lock_unlock(&sortStateLock) }
        return sortedIndicesBuffer
    }

    /// Thread-safe check if currently sorting
    private var isSorting: Bool {
        os_unfair_lock_lock(&sortStateLock)
        defer { os_unfair_lock_unlock(&sortStateLock) }
        return sorting
    }

    // Deferred buffer release - wait for GPU to finish before releasing
    private var pendingReleaseBuffers: [MetalBuffer<Int32>] = []
    private var pendingReleaseLock = os_unfair_lock()

    /// Queue a buffer for deferred release (call when swapping sorted indices)
    private func deferredBufferRelease(_ buffer: MetalBuffer<Int32>) {
        os_unfair_lock_lock(&pendingReleaseLock)
        pendingReleaseBuffers.append(buffer)
        os_unfair_lock_unlock(&pendingReleaseLock)
    }

    /// Release buffers that are no longer in use by GPU (call at frame start)
    private func releasePendingBuffers() {
        os_unfair_lock_lock(&pendingReleaseLock)
        let toRelease = pendingReleaseBuffers
        pendingReleaseBuffers.removeAll()
        os_unfair_lock_unlock(&pendingReleaseLock)

        for buffer in toRelease {
            sortIndexBufferPool.release(buffer)
        }
    }

    /// Thread-safe buffer swap for double-buffered sorting
    /// Returns the old buffer that was replaced (if any) for release back to pool
    private func swapSortedIndicesBuffer(newBuffer: MetalBuffer<Int32>) -> MetalBuffer<Int32>? {
        os_unfair_lock_lock(&sortStateLock)
        defer { os_unfair_lock_unlock(&sortStateLock) }

        let oldBuffer: MetalBuffer<Int32>?
        if usingSortedBufferA {
            // Currently rendering with A, so B is safe to replace
            oldBuffer = sortedIndicesBufferB
            sortedIndicesBufferB = newBuffer
        } else {
            // Currently rendering with B, so A is safe to replace
            oldBuffer = sortedIndicesBufferA
            sortedIndicesBufferA = newBuffer
        }

        // Flip the active buffer for next render
        usingSortedBufferA = !usingSortedBufferA

        // Update the main sortedIndicesBuffer pointer
        sortedIndicesBuffer = usingSortedBufferA
            ? sortedIndicesBufferA
            : sortedIndicesBufferB

        return oldBuffer
    }

    public init(device: MTLDevice,
                colorFormat: MTLPixelFormat,
                depthFormat: MTLPixelFormat,
                sampleCount: Int,
                maxViewCount: Int,
                maxSimultaneousRenders: Int) throws {
#if arch(x86_64)
        throw SplatRendererError.unsupportedArchitecture
#endif

        self.device = device

        // Initialize command buffer manager with Metal 4 pooling support
        guard let commandQueue = device.makeCommandQueue() else {
            throw SplatRendererError.metalDeviceUnavailable
        }
        commandQueue.label = "SplatRenderer Command Queue"
        self.commandBufferManager = CommandBufferManager(commandQueue: commandQueue)
        
        // Create separate compute queue for async sorting overlap
        // This allows sorting to run in parallel with rendering
        if let computeQueue = device.makeCommandQueue() {
            computeQueue.label = "SplatRenderer Compute Queue (Async Sort)"
            self.computeCommandQueue = computeQueue
            self.computeCommandBufferManager = CommandBufferManager(commandQueue: computeQueue)
        }

        self.colorFormat = colorFormat
        self.depthFormat = depthFormat
        self.sampleCount = sampleCount
        self.maxViewCount = min(maxViewCount, Constants.maxViewCount)
        self.maxSimultaneousRenders = maxSimultaneousRenders

        let dynamicUniformBuffersSize = UniformsArray.alignedSize * maxSimultaneousRenders
        guard let dynamicUniformBuffers = device.makeBuffer(length: dynamicUniformBuffersSize,
                                                           options: .storageModeShared) else {
            throw SplatRendererError.failedToCreateBuffer(length: dynamicUniformBuffersSize)
        }
        self.dynamicUniformBuffers = dynamicUniformBuffers
        self.dynamicUniformBuffers.label = "Uniform Buffers"
        self.uniforms = UnsafeMutableRawPointer(dynamicUniformBuffers.contents()).bindMemory(to: UniformsArray.self, capacity: 1)

        // Initialize buffer pools with optimized configurations
        let splatPoolConfig = MetalBufferPool<Splat>.Configuration(
            maxPoolSize: 8,  // Allow more splat buffers for complex scenes
            maxBufferAge: 120.0,  // Keep splat buffers longer as they're expensive
            memoryPressureThreshold: 0.7  // More aggressive cleanup for large buffers
        )
        self.splatBufferPool = MetalBufferPool(device: device, configuration: splatPoolConfig)

        let indexPoolConfig = MetalBufferPool<UInt32>.Configuration(
            maxPoolSize: 12,  // Index buffers are smaller, can pool more
            maxBufferAge: 90.0
        )
        self.indexBufferPool = MetalBufferPool(device: device, configuration: indexPoolConfig)

        // Initialize sort buffer pools (optimized for frequent reuse during sorting)
        let sortDistancePoolConfig = MetalBufferPool<Float>.Configuration(
            maxPoolSize: 4,  // Keep a few sort buffers cached
            maxBufferAge: 30.0,  // Short age since sort patterns are stable
            memoryPressureThreshold: 0.75
        )
        self.sortDistanceBufferPool = MetalBufferPool(device: device, configuration: sortDistancePoolConfig)

        let sortIndexPoolConfig = MetalBufferPool<Int32>.Configuration(
            maxPoolSize: 4,  // Keep a few sort buffers cached
            maxBufferAge: 30.0,  // Short age since sort patterns are stable
            memoryPressureThreshold: 0.75
        )
        self.sortIndexBufferPool = MetalBufferPool(device: device, configuration: sortIndexPoolConfig)

        // Acquire initial buffers from pools
        self.splatBuffer = try splatBufferPool.acquire(minimumCapacity: 1)
        self.splatBufferPrime = try splatBufferPool.acquire(minimumCapacity: 1)
        self.indexBuffer = try indexBufferPool.acquire(minimumCapacity: 1)

        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
        } catch {
            throw SplatRendererError.failedToCreateLibrary(underlying: error)
        }
        
        // Initialize compute pipeline for distance calculation
        do {
            guard let computeFunction = library.makeFunction(name: "computeSplatDistances") else {
                throw SplatRendererError.failedToLoadShaderFunction(name: "computeSplatDistances")
            }
            computeDistancesPipelineState = try device.makeComputePipelineState(function: computeFunction)
        } catch {
            Self.log.error("Failed to create compute pipeline state: \(error)")
        }
        
        // Initialize frustum culling pipeline and buffers
        do {
            if let frustumFunction = library.makeFunction(name: "frustumCullSplats") {
                frustumCullPipelineState = try device.makeComputePipelineState(function: frustumFunction)
            }
            if let generateArgsFunction = library.makeFunction(name: "generateIndirectDrawArguments") {
                generateIndirectArgsPipelineState = try device.makeComputePipelineState(function: generateArgsFunction)
            }
            if let resetCountFunction = library.makeFunction(name: "resetVisibleCount") {
                resetVisibleCountPipelineState = try device.makeComputePipelineState(function: resetCountFunction)
            }
            // Create frustum culling buffers (will be resized as needed)
            frustumCullDataBuffer = device.makeBuffer(length: MemoryLayout<FrustumCullData>.stride, options: .storageModeShared)
            frustumCullDataBuffer?.label = "Frustum Cull Data"
            visibleCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
            visibleCountBuffer?.label = "Visible Count"
            // Indirect draw arguments buffer (MTLDrawIndexedPrimitivesIndirectArguments = 5 * uint32)
            indirectDrawArgsBuffer = device.makeBuffer(length: 5 * MemoryLayout<UInt32>.stride, options: .storageModePrivate)
            indirectDrawArgsBuffer?.label = "Indirect Draw Arguments"
        } catch {
            Self.log.error("Failed to create frustum culling pipeline state: \(error)")
        }
        
        // Initialize SIMD-group parallel bounds computation
        do {
            if let boundsFunction = library.makeFunction(name: "computeBoundsParallel") {
                computeBoundsPipelineState = try device.makeComputePipelineState(function: boundsFunction)
            }
            if let resetFunction = library.makeFunction(name: "resetBoundsAtomics") {
                resetBoundsPipelineState = try device.makeComputePipelineState(function: resetFunction)
            }
            // Create buffers for atomic bounds (3 floats each for x, y, z)
            boundsMinBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * 3, options: .storageModeShared)
            boundsMaxBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * 3, options: .storageModeShared)
            boundsMinBuffer?.label = "Bounds Min Atomics"
            boundsMaxBuffer?.label = "Bounds Max Atomics"
        } catch {
            Self.log.warning("Failed to create bounds compute pipeline: \(error)")
        }
        
        // Initialize Metal 4 TensorOps batch precompute pipeline
        do {
            if let precomputeFunction = library.makeFunction(name: "batchPrecomputeSplats") {
                batchPrecomputePipelineState = try device.makeComputePipelineState(function: precomputeFunction)
                Self.log.info("✅ Metal 4 TensorOps batch precompute available")
            }
        } catch {
            Self.log.warning("Failed to create batch precompute pipeline: \(error)")
        }

        // Setup mesh shaders if supported (Metal 3+, Apple Silicon)
        setupMeshShaders()

        // Setup Metal 4.0 optimizations if available
        setupMetal4Integration()

        // Initialize O(n) counting sorter for faster sorting
        do {
            countingSorter = try CountingSorter(device: device, library: library)
            Self.log.info("O(n) counting sort available")
        } catch {
            Self.log.warning("Failed to initialize counting sorter, using MPS fallback: \(error)")
        }

        // Initialize Metal 4 radix sorter for very large scenes (iOS 26+, macOS 26+)
        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
            if device.supportsFamily(.apple9) {
                do {
                    metal4Sorter = try Metal4Sorter(device: device, library: library)
                    Self.log.info("Metal 4 radix sort available for large scenes (>100K splats)")
                } catch {
                    Self.log.warning("Failed to initialize Metal 4 sorter: \(error)")
                }
            }
        }
    }
    
    /// Check if mesh shaders are supported and set up the pipeline
    private func setupMeshShaders() {
        // Mesh shaders require Metal 3+ which is available on:
        // - Apple Silicon Macs (M1+)
        // - A14+ iOS devices (iPhone 12+)
        guard device.supportsFamily(.apple7) else {
            Self.log.info("Mesh shaders not supported (requires Apple7 GPU family or later)")
            return
        }
        
        do {
            // Set function constants for packed colors
            let functionConstants = MTLFunctionConstantValues()
            var usePackedColorsValue = usePackedColors
            var hasPackedColorsBufferValue = packedColorBuffer != nil
            functionConstants.setConstantValue(&usePackedColorsValue, type: .bool, index: 10)
            functionConstants.setConstantValue(&hasPackedColorsBufferValue, type: .bool, index: 11)

            // Try to load mesh shader functions with function constants
            guard let objectFunction = library.makeFunction(name: "splatObjectShader"),
                  let meshFunction = try? library.makeFunction(name: "splatMeshShader", constantValues: functionConstants),
                  let fragmentFunction = library.makeFunction(name: "meshSplatFragmentShader") else {
                Self.log.info("Mesh shader functions not found in library")
                return
            }
            
            // Create mesh render pipeline descriptor
            let meshPipelineDescriptor = MTLMeshRenderPipelineDescriptor()
            meshPipelineDescriptor.label = "MeshShaderSplatPipeline"
            meshPipelineDescriptor.objectFunction = objectFunction
            meshPipelineDescriptor.meshFunction = meshFunction
            meshPipelineDescriptor.fragmentFunction = fragmentFunction
            
            // Configure color attachment (same as single-stage pipeline)
            let colorAttachment = meshPipelineDescriptor.colorAttachments[0]
            colorAttachment?.pixelFormat = colorFormat
            colorAttachment?.isBlendingEnabled = true
            colorAttachment?.rgbBlendOperation = .add
            colorAttachment?.alphaBlendOperation = .add
            colorAttachment?.sourceRGBBlendFactor = .one
            colorAttachment?.sourceAlphaBlendFactor = .one
            colorAttachment?.destinationRGBBlendFactor = .oneMinusSourceAlpha
            colorAttachment?.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            
            meshPipelineDescriptor.depthAttachmentPixelFormat = depthFormat
            meshPipelineDescriptor.rasterSampleCount = sampleCount
            
            // Meshlet configuration: 64 splats per meshlet (increased from 32, limited by Metal's 256 vertex max)
            meshPipelineDescriptor.maxTotalThreadsPerObjectThreadgroup = 64
            meshPipelineDescriptor.maxTotalThreadsPerMeshThreadgroup = 64
            
            // Create pipeline state
            let (pipelineState, _) = try device.makeRenderPipelineState(descriptor: meshPipelineDescriptor, options: [])
            meshShaderPipelineState = pipelineState
            
            // Create depth state (same as single-stage)
            let depthStateDescriptor = MTLDepthStencilDescriptor()
            depthStateDescriptor.depthCompareFunction = .always
            depthStateDescriptor.isDepthWriteEnabled = writeDepth
            meshShaderDepthState = device.makeDepthStencilState(descriptor: depthStateDescriptor)
            
            meshShadersSupported = true

            // Auto-enable mesh shaders only when NOT using multi-stage depth pipeline
            // Multi-stage is critical for Vision Pro depth quality
            if !useMultiStagePipeline && !useDitheredTransparency {
                meshShaderEnabled = true
                Self.log.info("✅ Mesh shaders auto-enabled (single-stage path) - geometry generated on GPU")
            } else {
                Self.log.info("✅ Mesh shaders available but not auto-enabled (multi-stage or dithered path active)")
            }
            
        } catch {
            Self.log.warning("Failed to create mesh shader pipeline: \(error)")
        }
    }
    
    deinit {
        // Return buffers to pools for reuse
        splatBufferPool.release(splatBuffer)
        splatBufferPool.release(splatBufferPrime)
        indexBufferPool.release(indexBuffer)
        // Release double-buffered sort index buffers
        if let bufferA = sortedIndicesBufferA {
            sortIndexBufferPool.release(bufferA)
        }
        if let bufferB = sortedIndicesBufferB {
            sortIndexBufferPool.release(bufferB)
        }
    }

    public func reset() {
        // Clear current buffers and return them to pools
        splatBufferPool.release(splatBuffer)
        splatBufferPool.release(splatBufferPrime)
        
        // Invalidate cached bounds
        cachedBounds = nil
        boundsDirty = true
        invalidatePrecomputedData()  // Invalidate TensorOps cache
        
        // Acquire fresh small buffers from pools
        do {
            splatBuffer = try splatBufferPool.acquire(minimumCapacity: 1)
            splatBufferPrime = try splatBufferPool.acquire(minimumCapacity: 1)
        } catch {
            Self.log.error("Failed to acquire buffers during reset: \(error)")
            // Fallback to creating new buffers if pool fails
            do {
                splatBuffer = try MetalBuffer(device: device)
                splatBufferPrime = try MetalBuffer(device: device)
            } catch {
                Self.log.error("Failed to create fallback buffers: \(error)")
            }
        }
    }
    
    /// Efficiently swaps buffers using the buffer pool to optimize memory allocation
    private func swapSplatBuffers() {
        swap(&splatBuffer, &splatBufferPrime)
        didSwapSplatBuffers()
    }

    open func didSwapSplatBuffers() {}

    /// Ensures splatBufferPrime has sufficient capacity, acquiring a new buffer from pool if needed
    private func ensurePrimeBufferCapacity(_ minimumCapacity: Int) throws {
        if splatBufferPrime.capacity < minimumCapacity {
            // Return current prime buffer to pool and acquire a larger one
            splatBufferPool.release(splatBufferPrime)
            splatBufferPrime = try splatBufferPool.acquire(minimumCapacity: minimumCapacity)
        }
        splatBufferPrime.count = 0
    }

    open func prepareForSorting(count: Int) throws {
        try ensurePrimeBufferCapacity(count)
    }

    open func appendSplatForSorting(from oldIndex: Int) {
        splatBufferPrime.append(splatBuffer, fromIndex: oldIndex)
    }

    public func read(from url: URL) async throws {
        var newPoints = SplatMemoryBuffer()
        try await newPoints.read(from: try AutodetectSceneReader(url))
        try add(newPoints.points)
    }

    private func resetPipelineStates() {
        singleStagePipelineState = nil
        ditheredPipelineState = nil
        ditheredDepthState = nil
        initializePipelineState = nil
        drawSplatPipelineState = nil
        drawSplatDepthState = nil
        postprocessPipelineState = nil
        postprocessDepthState = nil
        meshShaderPipelineState = nil  // Rebuild with updated function constants
    }

    private func invalidatePipelineStates() {
        resetPipelineStates()
    }

    private func buildSingleStagePipelineStatesIfNeeded() throws {
        guard singleStagePipelineState == nil else { return }

        singleStagePipelineState = try buildSingleStagePipelineState()
        singleStageDepthState = try buildSingleStageDepthState()
    }

    private func buildMultiStagePipelineStatesIfNeeded() throws {
        guard initializePipelineState == nil else { return }

        initializePipelineState = try buildInitializePipelineState()
        drawSplatPipelineState = try buildDrawSplatPipelineState()
        drawSplatDepthState = try buildDrawSplatDepthState()
        postprocessPipelineState = try buildPostprocessPipelineState()
        postprocessDepthState = try buildPostprocessDepthState()
    }

    private func buildSingleStagePipelineState() throws -> MTLRenderPipelineState {
        assert(!useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "SingleStagePipeline"

        // Set function constants for packed colors
        let functionConstants = MTLFunctionConstantValues()
        var usePackedColorsValue = usePackedColors
        var hasPackedColorsBufferValue = packedColorBuffer != nil
        functionConstants.setConstantValue(&usePackedColorsValue, type: .bool, index: 10)
        functionConstants.setConstantValue(&hasPackedColorsBufferValue, type: .bool, index: 11)

        pipelineDescriptor.vertexFunction = try library.makeFunction(name: "singleStageSplatVertexShader", constantValues: functionConstants)
        pipelineDescriptor.fragmentFunction = try library.makeRequiredFunction(name: "singleStageSplatFragmentShader")

        pipelineDescriptor.rasterSampleCount = sampleCount

        let colorAttachment = pipelineDescriptor.colorAttachments[0]
        colorAttachment?.pixelFormat = colorFormat
        colorAttachment?.isBlendingEnabled = true
        colorAttachment?.rgbBlendOperation = .add
        colorAttachment?.alphaBlendOperation = .add
        colorAttachment?.sourceRGBBlendFactor = .one
        colorAttachment?.sourceAlphaBlendFactor = .one
        colorAttachment?.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment?.destinationAlphaBlendFactor = .oneMinusSourceAlpha

        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildSingleStageDepthState() throws -> MTLDepthStencilState {
        assert(!useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        guard let depthState = device.makeDepthStencilState(descriptor: depthStateDescriptor) else {
            throw SplatRendererError.failedToCreateDepthStencilState
        }
        return depthState
    }

    // MARK: - Dithered Transparency Pipeline

    private func buildDitheredPipelineStatesIfNeeded() throws {
        guard ditheredPipelineState == nil else { return }

        ditheredPipelineState = try buildDitheredPipelineState()
        ditheredDepthState = try buildDitheredDepthState()
    }

    private func buildDitheredPipelineState() throws -> MTLRenderPipelineState {
        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "DitheredTransparencyPipeline"

        // Set function constants for packed colors
        let functionConstants = MTLFunctionConstantValues()
        var usePackedColorsValue = usePackedColors
        var hasPackedColorsBufferValue = packedColorBuffer != nil
        functionConstants.setConstantValue(&usePackedColorsValue, type: .bool, index: 10)
        functionConstants.setConstantValue(&hasPackedColorsBufferValue, type: .bool, index: 11)

        pipelineDescriptor.vertexFunction = try library.makeFunction(name: "singleStageSplatVertexShader", constantValues: functionConstants)
        pipelineDescriptor.fragmentFunction = try library.makeRequiredFunction(name: "singleStageSplatFragmentShaderDithered")

        pipelineDescriptor.rasterSampleCount = sampleCount

        let colorAttachment = pipelineDescriptor.colorAttachments[0]
        colorAttachment?.pixelFormat = colorFormat
        // Dithered mode: no blending, fragments are either fully opaque or discarded
        colorAttachment?.isBlendingEnabled = false

        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildDitheredDepthState() throws -> MTLDepthStencilState {
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        // Dithered mode: use depth testing for proper occlusion (since we're order-independent)
        depthStateDescriptor.depthCompareFunction = .less
        depthStateDescriptor.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depthStateDescriptor) else {
            throw SplatRendererError.failedToCreateDepthStencilState
        }
        return depthState
    }

    private func buildInitializePipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLTileRenderPipelineDescriptor()

        pipelineDescriptor.label = "InitializePipeline"
        pipelineDescriptor.tileFunction = try library.makeRequiredFunction(name: "initializeFragmentStore")
        pipelineDescriptor.threadgroupSizeMatchesTileSize = true;
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat

        return try device.makeRenderPipelineState(tileDescriptor: pipelineDescriptor, options: [], reflection: nil)
    }

    private func buildDrawSplatPipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "DrawSplatPipeline"

        // Set function constants for packed colors
        let functionConstants = MTLFunctionConstantValues()
        var usePackedColorsValue = usePackedColors
        var hasPackedColorsBufferValue = packedColorBuffer != nil
        functionConstants.setConstantValue(&usePackedColorsValue, type: .bool, index: 10)
        functionConstants.setConstantValue(&hasPackedColorsBufferValue, type: .bool, index: 11)

        pipelineDescriptor.vertexFunction = try library.makeFunction(name: "multiStageSplatVertexShader", constantValues: functionConstants)
        pipelineDescriptor.fragmentFunction = try library.makeRequiredFunction(name: "multiStageSplatFragmentShader")

        pipelineDescriptor.rasterSampleCount = sampleCount

        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildDrawSplatDepthState() throws -> MTLDepthStencilState {
        assert(useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        guard let depthState = device.makeDepthStencilState(descriptor: depthStateDescriptor) else {
            throw SplatRendererError.failedToCreateDepthStencilState
        }
        return depthState
    }

    private func buildPostprocessPipelineState() throws -> MTLRenderPipelineState {
        assert(useMultiStagePipeline)

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "PostprocessPipeline"
        pipelineDescriptor.vertexFunction =
            try library.makeRequiredFunction(name: "postprocessVertexShader")
        pipelineDescriptor.fragmentFunction =
            writeDepth
            ? try library.makeRequiredFunction(name: "postprocessFragmentShader")
            : try library.makeRequiredFunction(name: "postprocessFragmentShaderNoDepth")

        pipelineDescriptor.colorAttachments[0]?.pixelFormat = colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildPostprocessDepthState() throws -> MTLDepthStencilState {
        assert(useMultiStagePipeline)

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        guard let depthState = device.makeDepthStencilState(descriptor: depthStateDescriptor) else {
            throw SplatRendererError.failedToCreateDepthStencilState
        }
        return depthState
    }

    // MARK: - Debug AABB Pipeline

    private func buildDebugAABBPipelineStateIfNeeded() throws {
        guard debugAABBPipelineState == nil else { return }
        debugAABBPipelineState = try buildDebugAABBPipelineState()
        debugAABBDepthState = try buildDebugAABBDepthState()
    }

    private func buildDebugAABBPipelineState() throws -> MTLRenderPipelineState {
        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "DebugAABBPipeline"
        pipelineDescriptor.vertexFunction = try library.makeRequiredFunction(name: "aabbVertexShader")
        pipelineDescriptor.fragmentFunction = try library.makeRequiredFunction(name: "aabbFragmentShader")
        pipelineDescriptor.rasterSampleCount = sampleCount

        let colorAttachment = pipelineDescriptor.colorAttachments[0]
        colorAttachment?.pixelFormat = colorFormat
        colorAttachment?.isBlendingEnabled = true
        colorAttachment?.sourceRGBBlendFactor = .sourceAlpha
        colorAttachment?.destinationRGBBlendFactor = .oneMinusSourceAlpha
        colorAttachment?.sourceAlphaBlendFactor = .one
        colorAttachment?.destinationAlphaBlendFactor = .oneMinusSourceAlpha

        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat
        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildDebugAABBDepthState() throws -> MTLDepthStencilState {
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = .less
        depthStateDescriptor.isDepthWriteEnabled = false // Don't write depth, just test
        guard let depthState = device.makeDepthStencilState(descriptor: depthStateDescriptor) else {
            throw SplatRendererError.failedToCreateDepthStencilState
        }
        return depthState
    }

    private func setupAABBBuffers(min: SIMD3<Float>, max: SIMD3<Float>) {
        // Define 8 vertices of the bounding box
        let vertices: [SIMD3<Float>] = [
            SIMD3(min.x, min.y, min.z), // 0
            SIMD3(max.x, min.y, min.z), // 1
            SIMD3(max.x, max.y, min.z), // 2
            SIMD3(min.x, max.y, min.z), // 3
            SIMD3(min.x, min.y, max.z), // 4
            SIMD3(max.x, min.y, max.z), // 5
            SIMD3(max.x, max.y, max.z), // 6
            SIMD3(min.x, max.y, max.z), // 7
        ]

        // Define 12 edges (24 indices for lines)
        let indices: [UInt16] = [
            // Bottom face
            0, 1,  1, 2,  2, 3,  3, 0,
            // Top face
            4, 5,  5, 6,  6, 7,  7, 4,
            // Vertical edges
            0, 4,  1, 5,  2, 6,  3, 7
        ]

        let vertexBufferSize = vertices.count * MemoryLayout<SIMD3<Float>>.stride
        aabbVertexBuffer = device.makeBuffer(bytes: vertices, length: vertexBufferSize, options: .storageModeShared)
        aabbVertexBuffer?.label = "AABB Vertex Buffer"

        let indexBufferSize = indices.count * MemoryLayout<UInt16>.stride
        aabbIndexBuffer = device.makeBuffer(bytes: indices, length: indexBufferSize, options: .storageModeShared)
        aabbIndexBuffer?.label = "AABB Index Buffer"
    }

    public func ensureAdditionalCapacity(_ pointCount: Int) throws {
        try splatBuffer.ensureCapacity(splatBuffer.count + pointCount)
    }

    public func add(_ points: [SplatScenePoint]) throws {
        // Validate all points before adding any
        try SplatDataValidator.validatePoints(points)

        do {
            try ensureAdditionalCapacity(points.count)
        } catch {
            Self.log.error("Failed to grow buffers: \(error)")
            return
        }

        // Apply Morton ordering if enabled (improves GPU cache coherency)
        let orderedPoints: [SplatScenePoint]
        if mortonOrderingEnabled && points.count > 1 {
            let startTime = CFAbsoluteTimeGetCurrent()
            if points.count > mortonParallelThreshold {
                orderedPoints = MortonOrder.reorderParallel(points)
            } else {
                orderedPoints = MortonOrder.reorder(points)
            }
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            Self.log.info("Morton ordering \(points.count) splats took \(String(format: "%.2f", duration * 1000))ms")
        } else {
            orderedPoints = points
        }

        splatBuffer.append(orderedPoints.map { Splat($0) })
        markGeometryDirty()  // New splats affect geometry and require re-sorting
        colorsDirty = true   // New splats also have new colors

        // Initialize sorted indices with identity mapping (0, 1, 2, ...)
        // This ensures rendering works before first sort completes
        try initializeIdentitySortedIndices()
    }
    
    /// Initialize sorted indices buffer with identity mapping (0, 1, 2, ...)
    /// Called when splats are added to ensure valid render state before first sort
    private func initializeIdentitySortedIndices() throws {
        let count = splatBuffer.count
        guard count > 0 else { return }

        // Acquire new buffer and fill with identity indices
        let identityBuffer = try sortIndexBufferPool.acquire(minimumCapacity: count)
        identityBuffer.count = count
        for i in 0..<count {
            identityBuffer.values[i] = Int32(i)
        }

        // Use thread-safe swap to exchange buffers
        if let oldBuffer = swapSortedIndicesBuffer(newBuffer: identityBuffer) {
            // Defer release until GPU is done with old buffer
            deferredBufferRelease(oldBuffer)
        }
    }

    public func add(_ point: SplatScenePoint) throws {
        // Validate single point
        try SplatDataValidator.validatePoint(point)
        try add([ point ])
    }

    // MARK: - Color-Only Updates

    /// Updates only the color component of splats without triggering re-sorting.
    /// Use this when geometry (position/covariance) hasn't changed but colors need updating,
    /// such as after SH re-evaluation or color grading adjustments.
    ///
    /// This is significantly faster than full splat updates because:
    /// - Skips O(n) or O(n log n) sorting entirely
    /// - Skips bounds recalculation
    /// - Only updates the color portion of the GPU buffer
    ///
    /// Thread-safe: Updates are staged and applied at the start of the next render
    /// to avoid CPU/GPU data races on shared buffers. Call order is preserved.
    ///
    /// - Parameter colors: Array of new colors (RGBA), must match current splat count
    public func updateColorsOnly(_ colors: [SIMD4<Float>]) {
        let count = splatCount
        guard colors.count == count else {
            Self.log.warning("Color count mismatch: \(colors.count) vs \(count) splats")
            return
        }

        // Stage the update to avoid CPU/GPU race on shared buffer.
        // Full update clears pending queue since it overwrites everything.
        os_unfair_lock_lock(&pendingColorUpdateLock)
        pendingColorUpdates.removeAll()
        pendingColorUpdates.append(.full(colors))
        os_unfair_lock_unlock(&pendingColorUpdateLock)

        // Mark only colors as dirty, NOT geometry
        colorsDirty = true
        colorRevision &+= 1

        // Do NOT set these - they would trigger unnecessary work:
        // sortDirtyDueToData = true  // Skip - positions unchanged
        // boundsDirty = true         // Skip - positions unchanged
        // geometryDirty = true       // Skip - only colors changed
    }

    /// Updates colors for a range of splats without triggering re-sorting.
    /// Thread-safe: Updates are staged and applied at render start. Call order is preserved.
    /// - Parameters:
    ///   - colors: Array of new colors (RGBA)
    ///   - range: Index range to update
    public func updateColorsOnly(_ colors: [SIMD4<Float>], range: Range<Int>) {
        let count = splatCount
        guard range.lowerBound >= 0 && range.upperBound <= count else {
            Self.log.warning("Color update range out of bounds: \(range) vs \(count) splats")
            return
        }
        guard colors.count == range.count else {
            Self.log.warning("Color count mismatch: \(colors.count) vs range \(range.count)")
            return
        }

        os_unfair_lock_lock(&pendingColorUpdateLock)
        pendingColorUpdates.append(.range(colors, range))
        os_unfair_lock_unlock(&pendingColorUpdateLock)

        colorsDirty = true
        colorRevision &+= 1
    }

    /// Updates a single splat's color without triggering re-sorting.
    /// Thread-safe: Updates are staged and applied at render start. Call order is preserved.
    /// - Parameters:
    ///   - color: New color value (RGBA)
    ///   - index: Splat index to update
    public func updateColorOnly(_ color: SIMD4<Float>, at index: Int) {
        let count = splatCount
        guard index >= 0 && index < count else {
            Self.log.warning("Color update index out of bounds: \(index) vs \(count) splats")
            return
        }

        os_unfair_lock_lock(&pendingColorUpdateLock)
        pendingColorUpdates.append(.single(color, index))
        os_unfair_lock_unlock(&pendingColorUpdateLock)

        colorsDirty = true
        colorRevision &+= 1
    }

    /// Applies any pending color updates to the splat buffer in FIFO order.
    /// Must be called at render start, before any GPU work that reads from splatBuffer.
    /// This ensures CPU writes complete before GPU reads begin.
    private func applyPendingColorUpdates() {
        os_unfair_lock_lock(&pendingColorUpdateLock)
        let updates = pendingColorUpdates
        pendingColorUpdates.removeAll()
        os_unfair_lock_unlock(&pendingColorUpdateLock)

        guard !updates.isEmpty else { return }

        // Apply updates in order to preserve "last write wins" semantics
        for update in updates {
            switch update {
            case .full(let colors):
                for i in 0..<min(colors.count, splatBuffer.count) {
                    let c = colors[i]
                    splatBuffer.values[i].color = PackedRGBHalf4(
                        r: Float16(c.x), g: Float16(c.y), b: Float16(c.z), a: Float16(c.w)
                    )
                }
            case .range(let colors, let range):
                for (i, colorIndex) in range.enumerated() where colorIndex < splatBuffer.count {
                    let c = colors[i]
                    splatBuffer.values[colorIndex].color = PackedRGBHalf4(
                        r: Float16(c.x), g: Float16(c.y), b: Float16(c.z), a: Float16(c.w)
                    )
                }
            case .single(let color, let index):
                guard index < splatBuffer.count else { continue }
                splatBuffer.values[index].color = PackedRGBHalf4(
                    r: Float16(color.x), g: Float16(color.y),
                    b: Float16(color.z), a: Float16(color.w)
                )
            }
        }
    }

    /// Marks that geometry has changed and requires re-sorting and bounds update.
    /// Called internally when positions or covariance values are modified.
    private func markGeometryDirty() {
        geometryDirty = true
        sortDirtyDueToData = true
        sortDataRevision &+= 1
        boundsDirty = true
        invalidatePrecomputedData()
    }

    /// Get cached AABB bounds - returns immediately with cached value (never blocks).
    /// If bounds are dirty and no computation is in progress, triggers async GPU computation.
    /// Uses GPU SIMD-group parallel reduction, cached for subsequent calls.
    ///
    /// - Note: For callers that need guaranteed results (e.g., initial viewport setup),
    ///   use `getBoundsBlocking()` or `calculateBounds()` instead.
    public func getBounds() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        guard splatCount > 0 else { return nil }

        boundsLock.lock()
        let cached = cachedBounds
        let dirty = boundsDirty
        let inProgress = boundsComputationInProgress
        boundsLock.unlock()

        // Return cached bounds if valid
        if !dirty, let cached = cached {
            return cached
        }

        // If dirty and not already computing, start async computation
        if dirty && !inProgress {
            requestBoundsUpdateAsync()
        }

        // Return stale cached bounds while computation is in progress, or nil if none available
        return cached
    }

    /// Get AABB bounds, computing synchronously if needed (may block).
    /// Use this when you need guaranteed results, such as initial viewport setup.
    /// Uses CPU fallback for synchronous computation to avoid GPU stalls.
    public func getBoundsBlocking() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        guard splatCount > 0 else { return nil }

        boundsLock.lock()
        let cached = cachedBounds
        let dirty = boundsDirty
        boundsLock.unlock()

        // Return cached bounds if valid
        if !dirty, let cached = cached {
            return cached
        }

        // Compute synchronously using CPU (avoids GPU stalls)
        let bounds = calculateBoundsCPU()

        // Cache the result
        if let bounds = bounds {
            boundsLock.lock()
            cachedBounds = bounds
            boundsDirty = false
            boundsLock.unlock()
        }

        return bounds
    }

    /// Calculate AABB bounds - uses CPU for synchronous access.
    /// This is the backwards-compatible method that guarantees a result.
    public func calculateBounds() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        return getBoundsBlocking()
    }

    /// Request async bounds update. Non-blocking - updates cached bounds when complete.
    public func requestBoundsUpdateAsync() {
        boundsLock.lock()
        if boundsComputationInProgress {
            boundsLock.unlock()
            return
        }
        boundsComputationInProgress = true
        boundsLock.unlock()

        // First try GPU async computation
        if calculateBoundsGPUAsync() {
            return  // GPU computation started
        }

        // GPU not available, fall back to CPU on background thread
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let bounds = self.calculateBoundsCPU()

            self.boundsLock.lock()
            if let bounds = bounds {
                self.cachedBounds = bounds
                self.boundsDirty = false
            }
            self.boundsComputationInProgress = false
            self.boundsLock.unlock()
        }
    }

    /// GPU-accelerated bounds computation using SIMD-group parallel reduction (async version).
    /// Uses simd_min/simd_max for 32x fewer atomic operations than naive approach.
    /// Returns true if GPU computation was started, false if GPU is unavailable.
    private func calculateBoundsGPUAsync() -> Bool {
        guard let computePipeline = computeBoundsPipelineState,
              let resetPipeline = resetBoundsPipelineState,
              let minBuffer = boundsMinBuffer,
              let maxBuffer = boundsMaxBuffer,
              splatCount > 0 else {
            return false
        }

        guard let commandBuffer = commandBufferManager.makeCommandBuffer() else {
            Self.log.warning("Failed to create command buffer for GPU bounds computation")
            return false
        }

        // Step 1: Reset atomic bounds to initial values
        guard let resetEncoder = commandBuffer.makeComputeCommandEncoder() else {
            Self.log.warning("Failed to create compute encoder for reset bounds")
            return false
        }
        resetEncoder.label = "Reset Bounds Atomics"
        resetEncoder.setComputePipelineState(resetPipeline)
        resetEncoder.setBuffer(minBuffer, offset: 0, index: 0)
        resetEncoder.setBuffer(maxBuffer, offset: 0, index: 1)
        resetEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        resetEncoder.endEncoding()

        // Step 2: Compute bounds with SIMD-group parallel reduction
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            Self.log.warning("Failed to create compute encoder for bounds computation")
            return false
        }
        computeEncoder.label = "Compute Bounds Parallel"
        computeEncoder.setComputePipelineState(computePipeline)
        computeEncoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)

        var count = UInt32(splatCount)
        computeEncoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 1)
        computeEncoder.setBuffer(minBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(maxBuffer, offset: 0, index: 3)

        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (splatCount + 255) / 256, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()

        // Execute asynchronously with completion handler
        // Note: We capture self weakly and re-access the instance buffers in the completion handler.
        // This is safe because boundsMinBuffer/boundsMaxBuffer are instance properties with storageModeShared.
        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self = self,
                  let minBuf = self.boundsMinBuffer,
                  let maxBuf = self.boundsMaxBuffer else { return }

            // Read results from GPU
            let minPtr = minBuf.contents().bindMemory(to: Float.self, capacity: 3)
            let maxPtr = maxBuf.contents().bindMemory(to: Float.self, capacity: 3)
            let minBounds = SIMD3<Float>(minPtr[0], minPtr[1], minPtr[2])
            let maxBounds = SIMD3<Float>(maxPtr[0], maxPtr[1], maxPtr[2])

            self.boundsLock.lock()
            // Validate bounds (check for infinity which indicates no valid splats)
            if !minBounds.x.isInfinite && !maxBounds.x.isInfinite {
                self.cachedBounds = (min: minBounds, max: maxBounds)
                self.boundsDirty = false
            }
            self.boundsComputationInProgress = false
            self.boundsLock.unlock()
        }

        commandBuffer.commit()
        return true
    }
    
    /// CPU fallback for bounds computation
    private func calculateBoundsCPU() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        guard splatCount > 0 else { return nil }
        
        let splats = splatBuffer.values
        var minBounds = SIMD3<Float>(repeating: .infinity)
        var maxBounds = SIMD3<Float>(repeating: -.infinity)
        
        for i in 0..<splatCount {
            let position = SIMD3<Float>(splats[i].position.elements.0,
                                       splats[i].position.elements.1,
                                       splats[i].position.elements.2)
            minBounds = min(minBounds, position)
            maxBounds = max(maxBounds, position)
        }
        
        return (min: minBounds, max: maxBounds)
    }
    
    // MARK: - Metal 4 TensorOps Batch Precompute
    
    /// Pre-compute covariance and transforms for all splats on GPU
    /// This moves expensive per-vertex math to a one-time batch operation
    /// when camera changes, cached until next camera movement
    private func runBatchPrecompute(viewport: ViewportDescriptor, to commandBuffer: MTLCommandBuffer) {
        guard batchPrecomputeEnabled,
              let precomputePipeline = batchPrecomputePipelineState,
              splatCount > 0 else {
            return
        }
        
        // Check if we need to recompute (view matrix changed significantly)
        // Must check FULL matrix change (rotation + translation), not just translation
        if let lastMatrix = lastPrecomputeViewMatrix, !precomputedDataDirty {
            // Compare rotation (upper-left 3x3) and translation
            let rotDiff = simd_length(lastMatrix.columns.0 - viewport.viewMatrix.columns.0) +
                          simd_length(lastMatrix.columns.1 - viewport.viewMatrix.columns.1) +
                          simd_length(lastMatrix.columns.2 - viewport.viewMatrix.columns.2)
            let transDiff = simd_length(lastMatrix.columns.3 - viewport.viewMatrix.columns.3)
            if rotDiff < 0.001 && transDiff < 0.001 {
                return  // No significant camera movement or rotation, reuse cached data
            }
        }
        
        // Ensure precomputed buffer has correct size
        let requiredSize = splatCount * Self.precomputedSplatStride
        if precomputedSplatBuffer == nil || precomputedSplatBuffer!.length < requiredSize {
            precomputedSplatBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)
            precomputedSplatBuffer?.label = "Precomputed Splats"
        }
        guard let precomputedBuffer = precomputedSplatBuffer else { return }
        
        // Create uniform buffer for this computation
        var uniforms = Uniforms(
            projectionMatrix: viewport.projectionMatrix,
            viewMatrix: viewport.viewMatrix,
            screenSize: SIMD2<UInt32>(UInt32(viewport.screenSize.x), UInt32(viewport.screenSize.y)),
            splatCount: UInt32(splatCount),
            indexedSplatCount: UInt32(min(splatCount, Constants.maxIndexedSplatCount)),
            debugFlags: 0,
            lodThresholds: SIMD3<Float>(Constants.lodDistanceThresholds[0],
                                         Constants.lodDistanceThresholds[1],
                                         Constants.lodDistanceThresholds[2])
        )
        
        var splatCountValue = UInt32(splatCount)
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            Self.log.error("Failed to create compute encoder for batch precompute")
            return
        }
        computeEncoder.label = "Batch Precompute Splats"
        computeEncoder.setComputePipelineState(precomputePipeline)
        
        computeEncoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)
        computeEncoder.setBuffer(precomputedBuffer, offset: 0, index: 1)
        computeEncoder.setBytes(&uniforms, length: MemoryLayout<Uniforms>.stride, index: 2)
        computeEncoder.setBytes(&splatCountValue, length: MemoryLayout<UInt32>.stride, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (splatCount + 255) / 256, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        // Cache the view matrix to detect changes
        lastPrecomputeViewMatrix = viewport.viewMatrix
        precomputedDataDirty = false
        
        Self.log.debug("Batch precomputed \(self.splatCount) splats for current view")
    }
    
    /// Invalidate precomputed data when splats change
    private func invalidatePrecomputedData() {
        precomputedDataDirty = true
        lastPrecomputeViewMatrix = nil
    }

    // MARK: - Packed Color Buffer (snorm10a2 bandwidth optimization)

    /// Rebuild the packed color buffer from current splat data
    /// Called when usePackedColors is enabled or splat data changes
    private func rebuildPackedColorBufferIfNeeded() {
        guard usePackedColors, splatBuffer.count > 0 else {
            packedColorBuffer = nil
            return
        }

        // Each packed color is 4 bytes (uint)
        let bufferSize = splatBuffer.count * MemoryLayout<UInt32>.stride
        if packedColorBuffer == nil || packedColorBuffer!.length < bufferSize {
            packedColorBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
            packedColorBuffer?.label = "Packed Colors (snorm10a2)"
        }

        guard let buffer = packedColorBuffer else {
            Self.log.error("Failed to create packed color buffer")
            return
        }

        // Pack colors from splat buffer to packed buffer
        let packedPtr = buffer.contents().bindMemory(to: UInt32.self, capacity: splatBuffer.count)

        for i in 0..<splatBuffer.count {
            let color = splatBuffer.values[i].color
            // Pack half4 color to snorm10a2
            // Format: [A:2][B:10][G:10][R:10]
            let r = packSnorm10(Float(color.r))
            let g = packSnorm10(Float(color.g))
            let b = packSnorm10(Float(color.b))
            let a = packUnorm2(Float(color.a))

            packedPtr[i] = r | (g << 10) | (b << 20) | (a << 30)
        }

        Self.log.debug("Packed \(self.splatBuffer.count) colors to snorm10a2 format")
    }

    /// Pack a float value to signed normalized 10-bit integer
    private func packSnorm10(_ value: Float) -> UInt32 {
        let clamped = max(-1.0, min(1.0, value))
        let scaled = clamped * 511.0
        let signed = Int(scaled.rounded())
        // Convert to two's complement 10-bit representation
        if signed < 0 {
            return UInt32(bitPattern: Int32(1024 + signed)) & 0x3FF
        } else {
            return UInt32(signed) & 0x3FF
        }
    }

    /// Pack a float value to unsigned normalized 2-bit integer
    private func packUnorm2(_ value: Float) -> UInt32 {
        let clamped = max(0.0, min(1.0, value))
        return UInt32((clamped * 3.0).rounded()) & 0x3
    }

    // MARK: - Frustum Culling
    
    /// Encode frustum culling compute pass into command buffer
    /// Includes: reset count → cull splats → generate indirect draw args
    private func encodeFrustumCulling(viewport: ViewportDescriptor, to commandBuffer: MTLCommandBuffer) {
        guard frustumCullingEnabled,
              let cullPipeline = frustumCullPipelineState,
              let resetPipeline = resetVisibleCountPipelineState,
              let generateArgsPipeline = generateIndirectArgsPipelineState,
              let cullDataBuffer = frustumCullDataBuffer,
              let countBuffer = visibleCountBuffer,
              let argsBuffer = indirectDrawArgsBuffer,
              splatCount > 0 else {
            return
        }
        
        // Ensure visible indices buffer is large enough
        let requiredSize = splatCount * MemoryLayout<UInt32>.stride
        if visibleIndicesBuffer == nil || visibleIndicesBuffer!.length < requiredSize {
            visibleIndicesBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)
            visibleIndicesBuffer?.label = "Visible Indices"
        }
        guard let indicesBuffer = visibleIndicesBuffer else { return }
        
        // Prepare view-projection matrix for NDC-based culling
        let viewProjection = viewport.projectionMatrix * viewport.viewMatrix
        
        // Extract camera position from inverse view matrix
        let invView = viewport.viewMatrix.inverse
        let cameraPosition = SIMD3<Float>(invView[3][0], invView[3][1], invView[3][2])
        
        // Prepare cull data with view-projection matrix
        var cullData = FrustumCullData()
        cullData.viewProjectionMatrix = viewProjection
        cullData.cameraPosition = cameraPosition
        cullData.maxDistance = 10000.0  // Large value = effectively no distance culling
        
        // Copy cull data to buffer
        cullDataBuffer.contents().copyMemory(from: &cullData, byteCount: MemoryLayout<FrustumCullData>.stride)
        
        // === Step 1: Reset visible count on GPU ===
        guard let resetEncoder = commandBuffer.makeComputeCommandEncoder() else {
            Self.log.error("Failed to create compute encoder for reset visible count")
            return
        }
        resetEncoder.label = "Reset Visible Count"
        resetEncoder.setComputePipelineState(resetPipeline)
        resetEncoder.setBuffer(countBuffer, offset: 0, index: 0)
        resetEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        resetEncoder.endEncoding()
        
        // === Step 2: Frustum cull splats ===
        guard let cullEncoder = commandBuffer.makeComputeCommandEncoder() else {
            Self.log.error("Failed to create compute encoder for frustum culling")
            return
        }
        cullEncoder.label = "Frustum Culling"
        cullEncoder.setComputePipelineState(cullPipeline)
        cullEncoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)
        cullEncoder.setBuffer(indicesBuffer, offset: 0, index: 1)
        cullEncoder.setBuffer(countBuffer, offset: 0, index: 2)
        cullEncoder.setBuffer(cullDataBuffer, offset: 0, index: 3)
        
        var count = UInt32(splatCount)
        cullEncoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (splatCount + 255) / 256, height: 1, depth: 1)
        cullEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        cullEncoder.endEncoding()
        
        // === Step 3: Generate indirect draw arguments ===
        guard let argsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            Self.log.error("Failed to create compute encoder for indirect draw args")
            return
        }
        argsEncoder.label = "Generate Indirect Draw Args"
        argsEncoder.setComputePipelineState(generateArgsPipeline)
        argsEncoder.setBuffer(argsBuffer, offset: 0, index: 0)
        argsEncoder.setBuffer(countBuffer, offset: 0, index: 1)
        
        var indicesPerSplat: UInt32 = 6  // 2 triangles per splat
        var maxIndexed = UInt32(Constants.maxIndexedSplatCount)
        argsEncoder.setBytes(&indicesPerSplat, length: MemoryLayout<UInt32>.stride, index: 2)
        argsEncoder.setBytes(&maxIndexed, length: MemoryLayout<UInt32>.stride, index: 3)
        
        argsEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        argsEncoder.endEncoding()
    }
    
    /// Read back frustum culling results after command buffer completes
    /// Call this in the completion handler
    private func readFrustumCullingResults() {
        guard frustumCullingEnabled,
              let countBuffer = visibleCountBuffer else {
            return
        }
        
        let visibleCount = countBuffer.contents().load(as: UInt32.self)
        let previousCount = lastVisibleCount
        lastVisibleCount = Int(visibleCount)
        
        // Log when count changes by more than 5%
        let totalCount = self.splatCount
        let changeThreshold = max(totalCount / 20, 100)  // At least 100 splats or 5%
        if abs(lastVisibleCount - previousCount) > changeThreshold {
            let percentage = totalCount > 0 ? Int(Float(visibleCount) / Float(totalCount) * 100) : 0
            Self.log.info("Frustum culling: \(visibleCount)/\(totalCount) visible (\(percentage)%)")
        }
    }
    
    /// Get the last frustum culling result
    public var culledSplatCount: Int {
        frustumCullingEnabled ? lastVisibleCount : splatCount
    }
    
    // MARK: - Interaction Mode Control
    
    /// Begin interaction mode - relaxes sort parameters for smoother user experience
    /// Call this when user starts panning, pinching, or rotating
    public func beginInteraction() {
        guard !isInteracting else { return }
        
        isInteracting = true
        interactionEndTime = nil
        
        // Store current quality settings
        qualitySortPositionEpsilon = sortPositionEpsilon
        qualitySortDirectionEpsilon = sortDirectionEpsilon
        qualityMinimumSortInterval = minimumSortInterval
        
        // Apply relaxed interaction settings
        sortPositionEpsilon = interactionSortPositionEpsilon
        sortDirectionEpsilon = interactionSortDirectionEpsilon
        minimumSortInterval = interactionMinimumSortInterval
        
        Self.log.debug("Interaction mode started - sort thresholds relaxed")
    }
    
    /// End interaction mode - restores quality sort parameters and triggers final sort
    /// Call this when user ends touch interaction
    public func endInteraction() {
        guard isInteracting else { return }
        
        isInteracting = false
        interactionEndTime = CFAbsoluteTimeGetCurrent()
        
        // Restore quality settings
        sortPositionEpsilon = qualitySortPositionEpsilon
        sortDirectionEpsilon = qualitySortDirectionEpsilon
        minimumSortInterval = qualityMinimumSortInterval
        
        // Schedule a final high-quality sort after a brief delay
        // This allows the last frame to render before the sort overhead kicks in
        // Skip if using dithered transparency (order-independent, no sort needed)
        if !useDitheredTransparency {
            DispatchQueue.main.asyncAfter(deadline: .now() + postInteractionSortDelay) { [weak self] in
                guard let self = self else { return }
                // Only trigger if we haven't started interacting again
                if !self.isInteracting {
                    Self.log.debug("Interaction mode ended - triggering final sort")
                    self.sortDirtyDueToData = true  // Force a re-sort
                    self.resort(useGPU: true)
                }
            }
        }
    }
    
    // MARK: - Buffer Pool Management
    
    /// Returns statistics about buffer pool usage for monitoring and debugging
    public func getBufferPoolStatistics() -> (splatPoolAvailable: Int, splatPoolLeased: Int, splatPoolMemoryMB: Float,
                                              indexPoolAvailable: Int, indexPoolLeased: Int, indexPoolMemoryMB: Float) {
        let splatStats = splatBufferPool.getStatistics()
        let indexStats = indexBufferPool.getStatistics()
        
        return (
            splatPoolAvailable: splatStats.availableBuffers,
            splatPoolLeased: splatStats.leasedBuffers,
            splatPoolMemoryMB: splatStats.totalMemoryMB,
            indexPoolAvailable: indexStats.availableBuffers,
            indexPoolLeased: indexStats.leasedBuffers,
            indexPoolMemoryMB: indexStats.totalMemoryMB
        )
    }
    
    /// Manually triggers memory pressure cleanup on buffer pools
    public func trimBufferPools() {
        splatBufferPool.trimToMemoryPressure()
        indexBufferPool.trimToMemoryPressure()
    }

    internal func switchToNextDynamicBuffer() {
        uniformBufferIndex = (uniformBufferIndex + 1) % maxSimultaneousRenders
        uniformBufferOffset = UniformsArray.alignedSize * uniformBufferIndex
        uniforms = UnsafeMutableRawPointer(dynamicUniformBuffers.contents() + uniformBufferOffset).bindMemory(to: UniformsArray.self, capacity: 1)
    }

    private func updateSortReferenceCamera(from viewports: [ViewportDescriptor]) {
        if let reference = viewports.first {
            sortCameraPosition = Self.cameraWorldPosition(forViewMatrix: reference.viewMatrix)
            sortCameraForward = Self.cameraWorldForward(forViewMatrix: reference.viewMatrix).normalized
        } else {
            sortCameraPosition = .zero
            sortCameraForward = .init(x: 0, y: 0, z: -1)
        }
    }

    private func shouldResortForCurrentCamera() -> Bool {
        // Skip sorting entirely when using dithered transparency
        // Dithered mode is order-independent, so sort order doesn't affect visual quality
        if useDitheredTransparency {
            return false
        }
        if sortDirtyDueToData {
            return true
        }
        let now = CFAbsoluteTimeGetCurrent()
        if minimumSortInterval > 0 && (now - lastSortTime) < minimumSortInterval {
            return false
        }
        guard let lastPos = lastSortedCameraPosition,
              let lastFwd = lastSortedCameraForward else {
            return true
        }
        let positionDelta = simd_distance(sortCameraPosition, lastPos)
        let forwardDelta = 1 - simd_dot(simd_normalize(sortCameraForward), simd_normalize(lastFwd))
        return positionDelta > sortPositionEpsilon || forwardDelta > sortDirectionEpsilon
    }

    /// Determines whether spherical harmonics should be re-evaluated based on camera direction change.
    /// Returns true if SH evaluation is needed, false if cached values can be reused.
    internal func shouldUpdateSHForCurrentCamera() -> Bool {
        if shDirtyDueToData {
            return true
        }

        let now = CFAbsoluteTimeGetCurrent()
        if minimumSHUpdateInterval > 0 && (now - lastSHUpdateTime) < minimumSHUpdateInterval {
            return false
        }

        guard let lastDir = lastSHCameraDirection else {
            return true
        }

        let directionDelta = 1 - simd_dot(simd_normalize(cameraWorldForward), simd_normalize(lastDir))
        return directionDelta > shDirectionEpsilon
    }

    /// Marks that SH evaluation has completed for the current camera direction.
    /// Call this after performing SH evaluation to update the cached state.
    internal func didUpdateSHForCurrentCamera() {
        lastSHCameraDirection = cameraWorldForward
        lastSHUpdateTime = CFAbsoluteTimeGetCurrent()
        shDirtyDueToData = false
    }

    internal func updateUniforms(forViewports viewports: [ViewportDescriptor],
                                splatCount: UInt32,
                                indexedSplatCount: UInt32) {
        // Clamp to maxViewCount to avoid buffer overrun (off-by-one fix: use < not <=)
        for (i, viewport) in viewports.prefix(maxViewCount).enumerated() {
            let debugFlags = debugOptions.rawValue
            let uniforms = Uniforms(projectionMatrix: viewport.projectionMatrix,
                                    viewMatrix: viewport.viewMatrix,
                                    screenSize: SIMD2(x: UInt32(viewport.screenSize.x), y: UInt32(viewport.screenSize.y)),
                                    splatCount: splatCount,
                                    indexedSplatCount: indexedSplatCount,
                                    debugFlags: debugFlags,
                                    lodThresholds: lodThresholds)
            self.uniforms.pointee.setUniforms(index: i, uniforms)
        }
        updateSortReferenceCamera(from: viewports)
        // Use cached arrays to avoid per-frame allocations
        if cameraPositionsTemp.count != viewports.count {
            cameraPositionsTemp = Array(repeating: .zero, count: viewports.count)
            cameraForwardsTemp = Array(repeating: .zero, count: viewports.count)
        }
        for (i, viewport) in viewports.enumerated() {
            cameraPositionsTemp[i] = Self.cameraWorldPosition(forViewMatrix: viewport.viewMatrix)
            cameraForwardsTemp[i] = Self.cameraWorldForward(forViewMatrix: viewport.viewMatrix)
        }
        cameraWorldPosition = cameraPositionsTemp.mean ?? .zero
        cameraWorldForward = cameraForwardsTemp.mean?.normalized ?? .init(x: 0, y: 0, z: -1)

        // Release any buffers that were pending from previous frames
        releasePendingBuffers()

        if !isSorting && shouldResortForCurrentCamera() {
            resort()
        }
    }

    private static func cameraWorldForward(forViewMatrix view: simd_float4x4) -> simd_float3 {
        (view.inverse * SIMD4<Float>(x: 0, y: 0, z: -1, w: 0)).xyz
    }

    private static func cameraWorldPosition(forViewMatrix view: simd_float4x4) -> simd_float3 {
        (view.inverse * SIMD4<Float>(x: 0, y: 0, z: 0, w: 1)).xyz
    }

    func renderEncoder(multiStage: Bool,
                       viewports: [ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorLoadAction: MTLLoadAction = .clear,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       depthStoreAction: MTLStoreAction = .dontCare,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       for commandBuffer: MTLCommandBuffer) -> MTLRenderCommandEncoder? {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = colorLoadAction
        renderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        renderPassDescriptor.colorAttachments[0].clearColor = clearColor
        if let depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = depthStoreAction
            renderPassDescriptor.depthAttachment.clearDepth = 0.0
        }
        renderPassDescriptor.rasterizationRateMap = rasterizationRateMap
        renderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength

        renderPassDescriptor.tileWidth  = Constants.tileSize.width
        renderPassDescriptor.tileHeight = Constants.tileSize.height

        if multiStage {
            if let initializePipelineState {
                renderPassDescriptor.imageblockSampleLength = initializePipelineState.imageblockSampleLength
            } else {
                Self.log.error("initializePipeline == nil in renderEncoder()")
            }
        }

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            Self.log.error("Failed to create primary render encoder")
            return nil
        }

        renderEncoder.label = "Primary Render Encoder"

        // Clamp viewports to pipeline's maxVertexAmplificationCount to avoid Metal validation errors
        let clampedViewportCount = min(viewports.count, maxViewCount)
        let activeViewports = Array(viewports.prefix(clampedViewportCount))

        renderEncoder.setViewports(activeViewports.map(\.viewport))

        if clampedViewportCount > 1 {
            // Use cached view mappings to avoid per-frame allocations
            if viewMappingsTemp.count != clampedViewportCount {
                viewMappingsTemp = (0..<clampedViewportCount).map {
                    MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                      renderTargetArrayIndexOffset: UInt32($0))
                }
            }
            renderEncoder.setVertexAmplificationCount(clampedViewportCount, viewMappings: &viewMappingsTemp)
        }

        return renderEncoder
    }

    public func render(viewports: [ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorLoadAction: MTLLoadAction = .clear,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       depthStoreAction: MTLStoreAction = .dontCare,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        onRenderStart?()
        frameStartTime = CFAbsoluteTimeGetCurrent()
        frameBufferUploads = 0

        // Apply any pending color updates before GPU work begins.
        // This prevents CPU/GPU data races on the shared splatBuffer.
        applyPendingColorUpdates()

        let splatCount = splatBuffer.count
        guard splatBuffer.count != 0 else { return }
        let indexedSplatCount = min(splatCount, Constants.maxIndexedSplatCount)
        let instanceCount = (splatCount + indexedSplatCount - 1) / indexedSplatCount

        // Rebuild packed color buffer if enabled and splat data has changed
        if usePackedColors && colorsDirty {
            let hadPackedBuffer = packedColorBuffer != nil
            rebuildPackedColorBufferIfNeeded()
            colorsDirty = false

            // If packed buffer was just created, rebuild pipelines with updated function constants
            // This handles the case where usePackedColors was set before splats were loaded
            if !hadPackedBuffer && packedColorBuffer != nil {
                resetPipelineStates()
                setupMeshShaders()
            }
        }

        switchToNextDynamicBuffer()
        updateUniforms(forViewports: viewports, splatCount: UInt32(splatCount), indexedSplatCount: UInt32(indexedSplatCount))
        frameBufferUploads += 1 // uniforms update
        
        // GPU Frustum Culling: encode compute pass before rendering
        if frustumCullingEnabled, let firstViewport = viewports.first {
            encodeFrustumCulling(viewport: firstViewport, to: commandBuffer)
            
            // Add completion handler to read culling results
            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.readFrustumCullingResults()
            }
        }
        
        // Metal 4 TensorOps: batch precompute covariance/transforms (when enabled)
        // Only run for single-viewport rendering until per-viewport buffers are implemented
        if batchPrecomputeEnabled && viewports.count == 1, let firstViewport = viewports.first {
            runBatchPrecompute(viewport: firstViewport, to: commandBuffer)
        }

        // Log Metal 4.0 availability but use standard rendering path (only log once per scene)
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            if isMetal4OptimizationsAvailable && splatCount > 5000 {
                // Only log if this is a new scene or first time
                if !metal4LoggedOnce || abs(splatCount - lastSplatCountLogged) > 1000 {
                    Self.log.info("Metal 4.0: Enhanced pipeline active for \(splatCount) splats")
                    metal4LoggedOnce = true
                    lastSplatCountLogged = splatCount
                }
                // Continue with standard rendering but Metal 4.0 features are available
            }
        }

        // =========================================================================
        // MESH SHADER PATH (Metal 3+)
        // Generates geometry entirely on GPU - significant performance improvement
        // Note: Mesh shader path doesn't support frustum culling with dithered mode yet
        // (would need indirect dispatch with visible count)
        // =========================================================================
        let useCulledDitheredPath = useDitheredTransparency && frustumCullingEnabled
        if meshShaderEnabled && canUseMeshShadersSafely && !useCulledDitheredPath,
           let meshPipeline = meshShaderPipelineState,
           let meshDepth = meshShaderDepthState,
           let sortedIndices = getCurrentSortedIndicesBuffer() {
            
            guard let renderEncoder = renderEncoder(multiStage: false,
                                              viewports: viewports,
                                              colorTexture: colorTexture,
                                              colorLoadAction: colorLoadAction,
                                              colorStoreAction: colorStoreAction,
                                              depthTexture: depthTexture,
                                              depthStoreAction: depthStoreAction,
                                              rasterizationRateMap: rasterizationRateMap,
                                              renderTargetArrayLength: renderTargetArrayLength,
                                              for: commandBuffer) else {
                return
            }

            renderEncoder.pushDebugGroup("Mesh Shader Splats")
            renderEncoder.setRenderPipelineState(meshPipeline)
            renderEncoder.setDepthStencilState(meshDepth)
            renderEncoder.setCullMode(.none)
            
            // Set buffers for object/mesh shaders
            // BufferIndex matches: 0=uniforms, 1=splats, 2=sortedIndices
            renderEncoder.setObjectBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
            renderEncoder.setObjectBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)
            renderEncoder.setObjectBuffer(sortedIndices.buffer, offset: 0, index: BufferIndex.sortedIndices.rawValue)
            
            renderEncoder.setMeshBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
            renderEncoder.setMeshBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)
            renderEncoder.setMeshBuffer(sortedIndices.buffer, offset: 0, index: BufferIndex.sortedIndices.rawValue)

            // Bind precomputed buffer if TensorOps precompute is enabled and data is valid
            if batchPrecomputeEnabled, let precomputedBuffer = precomputedSplatBuffer, !precomputedDataDirty {
                renderEncoder.setMeshBuffer(precomputedBuffer, offset: 0, index: BufferIndex.precomputed.rawValue)
            }

            // Bind packed colors buffer (or placeholder for shader compatibility)
            if let packedColors = packedColorBuffer {
                renderEncoder.setMeshBuffer(packedColors, offset: 0, index: BufferIndex.packedColors.rawValue)
            } else {
                renderEncoder.setMeshBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.packedColors.rawValue)
            }

            // Calculate number of meshlets needed
            // Each meshlet handles 64 splats (increased from 32, limited by Metal's 256 vertex max)
            let splatsPerMeshlet: Int = 64
            let meshletCount = (splatCount + splatsPerMeshlet - 1) / splatsPerMeshlet

            // Dispatch mesh shader grid
            // Object shader threadgroups = number of meshlets
            // Each object threadgroup has 64 threads (one per potential splat)
            let objectThreadsPerGrid = MTLSize(width: splatsPerMeshlet, height: 1, depth: 1)
            let objectThreadgroupsPerGrid = MTLSize(width: meshletCount, height: 1, depth: 1)
            
            renderEncoder.drawMeshThreadgroups(objectThreadgroupsPerGrid,
                                               threadsPerObjectThreadgroup: objectThreadsPerGrid,
                                               threadsPerMeshThreadgroup: MTLSize(width: splatsPerMeshlet, height: 1, depth: 1))
            
            renderEncoder.popDebugGroup()
            renderEncoder.endEncoding()
            
            // Draw debug AABB if enabled (same as other paths)
            if debugOptions.contains(.showAABB), let bounds = calculateBounds() {
                drawDebugAABB(bounds: bounds, viewports: viewports, colorTexture: colorTexture,
                             depthTexture: depthTexture, rasterizationRateMap: rasterizationRateMap,
                             renderTargetArrayLength: renderTargetArrayLength, commandBuffer: commandBuffer)
            }
            return
        }
        
        // =========================================================================
        // TRADITIONAL VERTEX SHADER PATH (fallback)
        // =========================================================================
        let multiStage = useMultiStagePipeline
        if multiStage {
            try buildMultiStagePipelineStatesIfNeeded()
        } else if useDitheredTransparency {
            try buildDitheredPipelineStatesIfNeeded()
        } else {
            try buildSingleStagePipelineStatesIfNeeded()
        }

        guard let renderEncoder = renderEncoder(multiStage: multiStage,
                                          viewports: viewports,
                                          colorTexture: colorTexture,
                                          colorLoadAction: colorLoadAction,
                                          colorStoreAction: colorStoreAction,
                                          depthTexture: depthTexture,
                                          depthStoreAction: depthStoreAction,
                                          rasterizationRateMap: rasterizationRateMap,
                                          renderTargetArrayLength: renderTargetArrayLength,
                                          for: commandBuffer) else {
            return
        }

        let indexCount = indexedSplatCount * 6
        if indexBuffer.count < indexCount {
            do {
                // If current buffer is too small, get a larger one from pool
                if indexBuffer.capacity < indexCount {
                    indexBufferPool.release(indexBuffer)
                    indexBuffer = try indexBufferPool.acquire(minimumCapacity: indexCount)
                }
            } catch {
                Self.log.error("Failed to acquire larger index buffer: \(error)")
                return
            }
            indexBuffer.count = indexCount
            for i in 0..<indexedSplatCount {
                indexBuffer.values[i * 6 + 0] = UInt32(i * 4 + 0)
                indexBuffer.values[i * 6 + 1] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 2] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 3] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 4] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 5] = UInt32(i * 4 + 3)
            }
            frameBufferUploads += 1
        }

        if multiStage {
            guard let initializePipelineState,
                  let drawSplatPipelineState
            else { return }

            renderEncoder.pushDebugGroup("Initialize")
            renderEncoder.setRenderPipelineState(initializePipelineState)
            renderEncoder.dispatchThreadsPerTile(Constants.tileSize)
            renderEncoder.popDebugGroup()

            renderEncoder.pushDebugGroup("Draw Splats")
            renderEncoder.setRenderPipelineState(drawSplatPipelineState)
            renderEncoder.setDepthStencilState(drawSplatDepthState)
        } else if useDitheredTransparency {
            guard let ditheredPipelineState
            else { return }

            renderEncoder.pushDebugGroup("Draw Splats (Dithered)")
            renderEncoder.setRenderPipelineState(ditheredPipelineState)
            // Only set depth stencil state if we have a depth texture
            // Dithered mode benefits from depth testing for occlusion, but works without it
            if depthTexture != nil, let ditheredDepthState {
                renderEncoder.setDepthStencilState(ditheredDepthState)
            }
        } else {
            guard let singleStagePipelineState
            else { return }

            renderEncoder.pushDebugGroup("Draw Splats")
            renderEncoder.setRenderPipelineState(singleStagePipelineState)
            renderEncoder.setDepthStencilState(singleStageDepthState)
        }

        renderEncoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setVertexBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)

        // Bind packed colors buffer if enabled (or a dummy buffer for shader compatibility)
        // The shader uses function constants to decide whether to use packed colors
        if let packedColors = packedColorBuffer {
            renderEncoder.setVertexBuffer(packedColors, offset: 0, index: BufferIndex.packedColors.rawValue)
        } else {
            // Bind splat buffer as placeholder (shader won't access it when function constant is false)
            renderEncoder.setVertexBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.packedColors.rawValue)
        }

        // Dithered + Frustum Culling: use culled indices directly (order-independent)
        // This avoids sorting entirely while still benefiting from frustum culling
        if useDitheredTransparency && frustumCullingEnabled,
           let visibleIndices = visibleIndicesBuffer,
           let indirectArgs = indirectDrawArgsBuffer {
            // Use culled visible indices (unsorted is fine for dithered transparency)
            renderEncoder.setVertexBuffer(visibleIndices, offset: 0, index: BufferIndex.sortedIndices.rawValue)

            // GPU-driven indirect draw using culled count from compute pass
            renderEncoder.drawIndexedPrimitives(
                type: MTLPrimitiveType.triangle,
                indexType: MTLIndexType.uint32,
                indexBuffer: indexBuffer.buffer,
                indexBufferOffset: 0,
                indirectBuffer: indirectArgs,
                indirectBufferOffset: 0
            )
        } else {
            // Standard path: use sorted indices for correct alpha blending
            if let sortedIndices = getCurrentSortedIndicesBuffer() {
                renderEncoder.setVertexBuffer(sortedIndices.buffer, offset: 0, index: BufferIndex.sortedIndices.rawValue)
            }

            renderEncoder.drawIndexedPrimitives(type: .triangle,
                                                indexCount: indexCount,
                                                indexType: .uint32,
                                                indexBuffer: indexBuffer.buffer,
                                                indexBufferOffset: 0,
                                                instanceCount: instanceCount)
        }

        if multiStage {
            guard let postprocessPipelineState
            else { return }

            renderEncoder.popDebugGroup()

            renderEncoder.pushDebugGroup("Postprocess")
            renderEncoder.setRenderPipelineState(postprocessPipelineState)
            // Only set depth stencil state if we're actually storing depth
            if depthStoreAction == .store {
                renderEncoder.setDepthStencilState(postprocessDepthState)
            }
            renderEncoder.setCullMode(.none)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renderEncoder.popDebugGroup()
        } else {
            renderEncoder.popDebugGroup()
        }

        // Draw debug AABB wireframe if enabled
        if debugOptions.contains(.showAABB), let bounds = calculateBounds() {
            do {
                try buildDebugAABBPipelineStateIfNeeded()

                // Setup AABB buffers if needed
                if aabbVertexBuffer == nil || aabbIndexBuffer == nil {
                    setupAABBBuffers(min: bounds.min, max: bounds.max)
                }

                guard let pipeline = debugAABBPipelineState,
                      let depthState = debugAABBDepthState,
                      let vertexBuffer = aabbVertexBuffer,
                      let indexBuffer = aabbIndexBuffer else {
                    Self.log.warning("Debug AABB pipeline not initialized")
                    renderEncoder.endEncoding()
                    return
                }

                renderEncoder.pushDebugGroup("Debug AABB")
                renderEncoder.setRenderPipelineState(pipeline)
                renderEncoder.setDepthStencilState(depthState)
                renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
                renderEncoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: 1)
                renderEncoder.drawIndexedPrimitives(
                    type: .line,
                    indexCount: 24, // 12 edges * 2 vertices
                    indexType: .uint16,
                    indexBuffer: indexBuffer,
                    indexBufferOffset: 0
                )
                renderEncoder.popDebugGroup()
            } catch {
                Self.log.error("Failed to draw debug AABB: \(error)")
            }
        }

        renderEncoder.endEncoding()

        lastFrameTime = CFAbsoluteTimeGetCurrent() - frameStartTime
        frameCount += 1
        averageFrameTime += (lastFrameTime - averageFrameTime) / Double(frameCount)
        
        onRenderComplete?(lastFrameTime)

        // Only collect stats when callback is set (avoid overhead when not needed)
        if let frameReadyCallback = onFrameReady {
            let distancePoolStats = sortDistanceBufferPool.getStatistics()
            let indexPoolStats = sortIndexBufferPool.getStatistics()

            let sortBufferStats = FrameStatistics.BufferPoolStats(
                availableBuffers: distancePoolStats.availableBuffers + indexPoolStats.availableBuffers,
                leasedBuffers: distancePoolStats.leasedBuffers + indexPoolStats.leasedBuffers,
                totalMemoryMB: distancePoolStats.totalMemoryMB + indexPoolStats.totalMemoryMB
            )

            let stats = FrameStatistics(
                ready: !sorting,
                loadingCount: sorting ? 1 : 0,
                sortDuration: lastSortDuration,
                bufferUploadCount: frameBufferUploads,
                splatCount: splatCount,
                frameTime: lastFrameTime,
                sortBufferPoolStats: sortBufferStats,
                sortJobsInFlight: getSortJobsInFlight()
            )
            frameReadyCallback(stats)
        }
    }
    
    /// Helper to draw debug AABB wireframe - used by both mesh shader and traditional paths
    private func drawDebugAABB(bounds: (min: SIMD3<Float>, max: SIMD3<Float>),
                               viewports: [ViewportDescriptor],
                               colorTexture: MTLTexture,
                               depthTexture: MTLTexture?,
                               rasterizationRateMap: MTLRasterizationRateMap?,
                               renderTargetArrayLength: Int,
                               commandBuffer: MTLCommandBuffer) {
        do {
            try buildDebugAABBPipelineStateIfNeeded()
            
            if aabbVertexBuffer == nil || aabbIndexBuffer == nil {
                setupAABBBuffers(min: bounds.min, max: bounds.max)
            }
            
            guard let pipeline = debugAABBPipelineState,
                  let depthState = debugAABBDepthState,
                  let vertexBuffer = aabbVertexBuffer,
                  let indexBuffer = aabbIndexBuffer else {
                return
            }
            
            let renderPassDescriptor = MTLRenderPassDescriptor()
            renderPassDescriptor.colorAttachments[0].texture = colorTexture
            renderPassDescriptor.colorAttachments[0].loadAction = .load
            renderPassDescriptor.colorAttachments[0].storeAction = .store
            if let depthTexture = depthTexture {
                renderPassDescriptor.depthAttachment.texture = depthTexture
                renderPassDescriptor.depthAttachment.loadAction = .load
                renderPassDescriptor.depthAttachment.storeAction = .store
            }
            
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
                Self.log.warning("Failed to create render encoder for debug AABB")
                return
            }
            encoder.label = "Debug AABB Encoder"
            
            encoder.setRenderPipelineState(pipeline)
            encoder.setDepthStencilState(depthState)
            encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: 1)
            encoder.drawIndexedPrimitives(type: .line, indexCount: 24, indexType: .uint16,
                                          indexBuffer: indexBuffer, indexBufferOffset: 0)
            encoder.endEncoding()
        } catch {
            Self.log.error("Failed to draw debug AABB: \(error)")
        }
    }

    /// Completes a GPU sort by swapping buffers and updating state.
    /// Called from command buffer completion handlers to avoid blocking.
    /// Dispatches to main thread to avoid race conditions with render/update calls.
    private func finishSort(
        indexOutputBuffer: MetalBuffer<Int32>,
        sortStartTime: CFAbsoluteTime,
        cameraWorldPosition: SIMD3<Float>,
        cameraWorldForward: SIMD3<Float>,
        dataDirtySnapshot: UInt64,
        useGPU: Bool
    ) {
        // Dispatch to main thread to serialize with render/update calls
        DispatchQueue.main.async { [weak self] in
            guard let self = self else {
                // If self is deallocated, the pool is also gone. The indexOutputBuffer
                // will be freed when this closure completes (Metal buffers are refcounted).
                return
            }

            // GPU-ONLY SORTING with double-buffering for async overlap
            // Swap buffers atomically - rendering continues with old buffer
            // until this completes, then switches to new buffer
            if let oldBuffer = self.swapSortedIndicesBuffer(newBuffer: indexOutputBuffer) {
                // IMPORTANT: Defer release until next frame to avoid use-after-free.
                // The old buffer may still be referenced by in-flight render command buffers.
                // releasePendingBuffers() is called at the start of the next render.
                self.deferredBufferRelease(oldBuffer)
            }

            let elapsed = CFAbsoluteTimeGetCurrent() - sortStartTime
            self.lastSortDuration = elapsed
            self.onSortComplete?(elapsed)
            self.lastSortedCameraPosition = cameraWorldPosition
            self.lastSortedCameraForward = cameraWorldForward
            self.lastSortTime = CFAbsoluteTimeGetCurrent()
            if self.sortDataRevision == dataDirtySnapshot {
                self.sortDirtyDueToData = false
            }
            self.finishSort()

            Self.log.debug("Async sort completed in \(String(format: "%.1f", elapsed * 1000))ms")

            if self.shouldResortForCurrentCamera() {
                self.resort(useGPU: useGPU)
            }
        }
    }

    // Sort splatBuffer (read-only), storing the results in splatBuffer (write-only) then swap splatBuffer and splatBufferPrime
    public func resort(useGPU: Bool = true) {
        // Atomically check sorting flag and job count, and set if available
        guard tryStartSort() else {
            // Already sorting or too many jobs in flight
            return
        }

        onSortStart?()

        let splatCount = splatBuffer.count
        let dataDirtySnapshot = sortDataRevision

        let cameraWorldForward = sortCameraForward
        let cameraWorldPosition = sortCameraPosition
        let sortStartTime = CFAbsoluteTimeGetCurrent()
        
//        // For benchmark.
//        guard splatCount > 0 else {
//            sorting = false
//            let elapsed: TimeInterval = 0
//            Self.log.info("Sort time (\(useGPU ? "GPU" : "CPU")): \(elapsed) seconds")
//            onSortComplete?(elapsed)
//            return
//        }

        if useGPU {
            Task(priority: .high) {
                // Acquire index output buffer from pool
                let indexOutputBuffer: MetalBuffer<Int32>

                do {
                    indexOutputBuffer = try sortIndexBufferPool.acquire(minimumCapacity: splatCount)
                    indexOutputBuffer.count = splatCount
                } catch {
                    Self.log.error("Failed to acquire index output buffer from pool: \(error)")
                    self.finishSort()
                    return
                }

                // === METAL 4 RADIX SORT PATH (for very large scenes) ===
                // Uses GPU atomics-based radix sort, beneficial for >100K splats
                if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
                    if self.useMetal4Sorting,
                       splatCount > self.metal4SortingThreshold,
                       let sorter = self.metal4Sorter {
                        let sortCommandBufferManager = self.computeCommandBufferManager ?? commandBufferManager
                        guard let commandBuffer = sortCommandBufferManager.makeCommandBuffer() else {
                            Self.log.error("Failed to create compute command buffer for Metal 4 sort.")
                            sortIndexBufferPool.release(indexOutputBuffer)
                            self.finishSort()
                            return
                        }

                        do {
                            try sorter.sort(
                                splats: splatBuffer.buffer,
                                count: splatCount,
                                cameraPosition: cameraWorldPosition,
                                cameraForward: cameraWorldForward,
                                sortByDistance: Constants.sortByDistance,
                                outputIndices: indexOutputBuffer.buffer,
                                commandBuffer: commandBuffer
                            )
                        } catch {
                            Self.log.error("Metal 4 radix sort failed: \(error)")
                            sortIndexBufferPool.release(indexOutputBuffer)
                            self.finishSort()
                            return
                        }

                        // Capture pool before weak self check to ensure buffer release even if self is deallocated
                        commandBuffer.addCompletedHandler { [weak self, sortIndexBufferPool] _ in
                            guard let self = self else {
                                // Self deallocated during async sort - release buffer via captured pool
                                sortIndexBufferPool.release(indexOutputBuffer)
                                return
                            }
                            self.finishSort(
                                indexOutputBuffer: indexOutputBuffer,
                                sortStartTime: sortStartTime,
                                cameraWorldPosition: cameraWorldPosition,
                                cameraWorldForward: cameraWorldForward,
                                dataDirtySnapshot: dataDirtySnapshot,
                                useGPU: true
                            )
                        }
                        commandBuffer.commit()
                        return
                    }
                }

                // === O(n) COUNTING SORT PATH ===
                // Uses histogram-based sorting which is faster than O(n log n) radix sort
                if self.useCountingSort, let sorter = self.countingSorter {
                    // Use compute queue for sorting to allow overlap with rendering
                    let sortCommandBufferManager = self.computeCommandBufferManager ?? commandBufferManager
                    guard let commandBuffer = sortCommandBufferManager.makeCommandBuffer() else {
                        Self.log.error("Failed to create compute command buffer.")
                        sortIndexBufferPool.release(indexOutputBuffer)
                        self.finishSort()
                        return
                    }

                    // Compute depth bounds for proper bin distribution
                    // For large scenes, this could be cached and updated incrementally
                    let depthBounds: (min: Float, max: Float)?
                    if let cached = self.cachedBounds {
                        // Use cached AABB bounds as depth range estimate
                        // This is an approximation but avoids per-frame bounds computation
                        let cameraPos = cameraWorldPosition
                        let minDist = simd_distance(cameraPos, cached.min)
                        let maxDist = simd_distance(cameraPos, cached.max)
                        depthBounds = (min(0.1, minDist * 0.5), max(maxDist * 1.5, 100.0))
                    } else {
                        depthBounds = nil  // Use default range
                    }

                    do {
                        try sorter.sort(
                            commandBuffer: commandBuffer,
                            splatBuffer: splatBuffer.buffer,
                            outputBuffer: indexOutputBuffer.buffer,
                            cameraPosition: cameraWorldPosition,
                            cameraForward: cameraWorldForward,
                            sortByDistance: Constants.sortByDistance,
                            splatCount: splatCount,
                            depthBounds: depthBounds,
                            useCameraRelativeBinning: self.useCameraRelativeBinning
                        )
                    } catch {
                        Self.log.error("Counting sort failed: \(error)")
                        sortIndexBufferPool.release(indexOutputBuffer)
                        self.finishSort()
                        return
                    }

                    // Use completion handler instead of blocking waitUntilCompleted
                    // This allows the Task to return while GPU continues sorting
                    // Capture pool before weak self check to ensure buffer release even if self is deallocated
                    commandBuffer.addCompletedHandler { [weak self, sortIndexBufferPool] _ in
                        guard let self = self else {
                            // Self deallocated during async sort - release buffer via captured pool
                            sortIndexBufferPool.release(indexOutputBuffer)
                            return
                        }
                        self.finishSort(
                            indexOutputBuffer: indexOutputBuffer,
                            sortStartTime: sortStartTime,
                            cameraWorldPosition: cameraWorldPosition,
                            cameraWorldForward: cameraWorldForward,
                            dataDirtySnapshot: dataDirtySnapshot,
                            useGPU: useGPU
                        )
                    }
                    commandBuffer.commit()
                    return  // Exit Task - completion handler will finish sort

                } else {
                    // === LEGACY MPS ARGSORT PATH ===
                    // Falls back to O(n log n) MPS-based radix sort

                    let distanceBuffer: MetalBuffer<Float>
                    do {
                        distanceBuffer = try sortDistanceBufferPool.acquire(minimumCapacity: splatCount)
                        distanceBuffer.count = splatCount
                    } catch {
                        Self.log.error("Failed to acquire distance buffer from pool: \(error)")
                        sortIndexBufferPool.release(indexOutputBuffer)
                        self.finishSort()
                        return
                    }

                    // Create command buffer for distance computation using pooled manager
                    guard let commandBuffer = commandBufferManager.makeCommandBuffer() else {
                        Self.log.error("Failed to create compute command buffer.")
                        sortDistanceBufferPool.release(distanceBuffer)
                        sortIndexBufferPool.release(indexOutputBuffer)
                        self.finishSort()
                        return
                    }

                    // Standard distance computation
                    guard let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
                          let computePipelineState = computeDistancesPipelineState else {
                        Self.log.error("Failed to create compute encoder.")
                        sortDistanceBufferPool.release(distanceBuffer)
                        sortIndexBufferPool.release(indexOutputBuffer)
                        self.finishSort()
                        return
                    }

                    // Set up compute shader parameters
                    var cameraPos = cameraWorldPosition
                    var cameraFwd = cameraWorldForward
                    var sortByDist = Constants.sortByDistance
                    var count = UInt32(splatCount)

                    computeEncoder.setComputePipelineState(computePipelineState)
                    computeEncoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(distanceBuffer.buffer, offset: 0, index: 1)
                    computeEncoder.setBytes(&cameraPos, length: MemoryLayout<SIMD3<Float>>.size, index: 2)
                    computeEncoder.setBytes(&cameraFwd, length: MemoryLayout<SIMD3<Float>>.size, index: 3)
                    computeEncoder.setBytes(&sortByDist, length: MemoryLayout<Bool>.size, index: 4)
                    computeEncoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 5)

                    let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
                    let threadgroups = MTLSize(width: (splatCount + 255) / 256, height: 1, depth: 1)

                    computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
                    computeEncoder.endEncoding()

                    // === ASYNC COMPUTE OVERLAP ===
                    // Use separate compute queue so sorting doesn't block rendering
                    let sortQueue = self.computeCommandBufferManager?.queue ?? commandBufferManager.queue

                    // Use completion handler to chain distance computation -> argsort -> finish
                    commandBuffer.addCompletedHandler { [weak self] _ in
                        guard let self = self else {
                            return
                        }

                        // Run argsort (synchronous, but we're in a completion handler so not blocking main thread)
                        // Uses cached MPSArgSort to avoid graph recompilation overhead per frame
                        do {
                            try self.cachedMPSArgSort(commandQueue: sortQueue,
                                    input: distanceBuffer.buffer,
                                    output: indexOutputBuffer.buffer,
                                    count: splatCount)
                        } catch {
                            Self.log.error("MPSArgSort failed: \(error)")
                            self.sortDistanceBufferPool.release(distanceBuffer)
                            self.sortIndexBufferPool.release(indexOutputBuffer)
                            self.finishSort()
                            return
                        }

                        // Release distance buffer (only used by MPS path)
                        self.sortDistanceBufferPool.release(distanceBuffer)

                        self.finishSort(
                            indexOutputBuffer: indexOutputBuffer,
                            sortStartTime: sortStartTime,
                            cameraWorldPosition: cameraWorldPosition,
                            cameraWorldForward: cameraWorldForward,
                            dataDirtySnapshot: dataDirtySnapshot,
                            useGPU: useGPU
                        )
                    }
                    commandBuffer.commit()
                    return  // Exit Task - completion handler will finish sort
                }
            }
        } else {
            Task(priority: .high) {
                if orderAndDepthTempSort.count != splatCount {
                    orderAndDepthTempSort = Array(
                        repeating: SplatIndexAndDepth(index: .max, depth: 0),
                        count: splatCount
                    )
                }

                // Copy positions under lock to ensure pointer validity during sort
                // This avoids holding the lock during the slow sort operation
                splatBuffer.withLockedValues { values, count in
                    let actualCount = min(count, splatCount)
                    if Constants.sortByDistance {
                        for i in 0..<actualCount {
                            orderAndDepthTempSort[i].index = UInt32(i)
                            let splatPos = values[i].position.simd
                            orderAndDepthTempSort[i].depth = (splatPos - cameraWorldPosition).lengthSquared
                        }
                    } else {
                        for i in 0..<actualCount {
                            orderAndDepthTempSort[i].index = UInt32(i)
                            let splatPos = values[i].position.simd
                            orderAndDepthTempSort[i].depth = dot(splatPos, cameraWorldForward)
                        }
                    }
                }

                orderAndDepthTempSort.sort { $0.depth > $1.depth }

                // CPU fallback: populate sortedIndicesBuffer instead of reordering splats
                // This maintains consistency with GPU path - splat data stays static
                do {
                    // Acquire new buffer and fill with sorted indices
                    let cpuSortedIndices = try sortIndexBufferPool.acquire(minimumCapacity: splatCount)
                    cpuSortedIndices.count = splatCount
                    for newIndex in 0..<orderAndDepthTempSort.count {
                        cpuSortedIndices.values[newIndex] = Int32(orderAndDepthTempSort[newIndex].index)
                    }

                    // Use thread-safe swap (deferred release handles old buffer)
                    if let oldBuffer = self.swapSortedIndicesBuffer(newBuffer: cpuSortedIndices) {
                        self.deferredBufferRelease(oldBuffer)
                    }
                } catch {
                    Self.log.error("Failed to create sorted indices buffer: \(error)")
                }

                let elapsedCPU = CFAbsoluteTimeGetCurrent() - sortStartTime
                self.lastSortDuration = elapsedCPU
                self.onSortComplete?(elapsedCPU)
                self.lastSortedCameraPosition = cameraWorldPosition
                self.lastSortedCameraForward = cameraWorldForward
                self.lastSortTime = CFAbsoluteTimeGetCurrent()
                if self.sortDataRevision == dataDirtySnapshot {
                    self.sortDirtyDueToData = false
                }
                self.finishSort()
                if self.shouldResortForCurrentCamera() {
                    self.resort(useGPU: useGPU)
                }
            }
        }
    }
}

extension SplatRenderer.Splat {
    init(_ splat: SplatScenePoint) {
        self.init(position: splat.position,
                  color: .init(splat.color.asLinearFloat.sRGBToLinear, splat.opacity.asLinearFloat),
                  scale: splat.scale.asLinearFloat,
                  rotation: splat.rotation.normalized)
    }

    init(position: SIMD3<Float>,
         color: SIMD4<Float>,
         scale: SIMD3<Float>,
         rotation: simd_quatf) {
        let transform = simd_float3x3(rotation) * simd_float3x3(diagonal: scale)
        let cov3D = transform * transform.transpose
        self.init(position: MTLPackedFloat3Make(position.x, position.y, position.z),
                  color: SplatRenderer.PackedRGBHalf4(r: Float16(color.x), g: Float16(color.y), b: Float16(color.z), a: Float16(color.w)),
                  covA: SplatRenderer.PackedHalf3(x: Float16(cov3D[0, 0]), y: Float16(cov3D[0, 1]), z: Float16(cov3D[0, 2])),
                  covB: SplatRenderer.PackedHalf3(x: Float16(cov3D[1, 1]), y: Float16(cov3D[1, 2]), z: Float16(cov3D[2, 2])))
    }
}

protocol MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { get }
}

extension UInt32: MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { .uint32 }
}
extension UInt16: MTLIndexTypeProvider {
    static var asMTLIndexType: MTLIndexType { .uint16 }
}

extension Array where Element == SIMD3<Float> {
    var mean: SIMD3<Float>? {
        guard !isEmpty else { return nil }
        return reduce(.zero, +) / Float(count)
    }
}

private extension MTLPackedFloat3 {
    var simd: SIMD3<Float> {
        SIMD3(x: x, y: y, z: z)
    }
}

private extension SIMD3 where Scalar: BinaryFloatingPoint, Scalar.RawSignificand: FixedWidthInteger {
    var normalized: SIMD3<Scalar> {
        self / Scalar(sqrt(lengthSquared))
    }

    var lengthSquared: Scalar {
        x*x + y*y + z*z
    }

    func vector4(w: Scalar) -> SIMD4<Scalar> {
        SIMD4<Scalar>(x: x, y: y, z: z, w: w)
    }

    static func random(in range: Range<Scalar>) -> SIMD3<Scalar> {
        Self(x: Scalar.random(in: range), y: .random(in: range), z: .random(in: range))
    }
}

private extension SIMD3<Float> {
    var sRGBToLinear: SIMD3<Float> {
        SIMD3(x: pow(x, 2.2), y: pow(y, 2.2), z: pow(z, 2.2))
    }
}

private extension SIMD4 where Scalar: BinaryFloatingPoint {
    var xyz: SIMD3<Scalar> {
        .init(x: x, y: y, z: z)
    }
}

private extension MTLLibrary {
    func makeRequiredFunction(name: String) throws -> MTLFunction {
        guard let result = makeFunction(name: name) else {
            throw SplatRendererError.failedToLoadShaderFunction(name: name)
        }
        return result
    }
}
