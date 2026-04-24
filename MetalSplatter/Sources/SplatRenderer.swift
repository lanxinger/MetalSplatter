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
    case internalPipelineMismatch(expected: String, actual: String)

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
        case .internalPipelineMismatch(let expected, let actual):
            return "Internal pipeline mismatch: expected \(expected), but useMultiStagePipeline=\(actual)"
        }
    }
}

public class SplatRenderer: @unchecked Sendable {
    public enum SplatRenderMode: UInt32, Sendable {
        case standard = 0
        case mip = 1

        var defaultCovarianceBlur: Float {
            switch self {
            case .standard: 0.3
            case .mip: 0.1
            }
        }
    }

    /// Sorting mode selection for gaussian splat depth ordering
    /// - radial: Sort by squared distance from camera (better for rotation-heavy views like 360°)
    /// - linear: Sort by dot product with view direction (better for translation-heavy movement)
    /// - auto: Automatically select based on camera motion between frames
    public enum SortingMode: Sendable {
        case radial   // Distance-based sorting (better for rotation)
        case linear   // View-direction-based sorting (better for translation)
        case auto     // Auto-select based on camera motion
    }

    internal enum SortPath: String, Sendable {
        case metal4
        case counting
        case mps
        case cpu
    }

    internal struct SortPerformanceSample: Sendable {
        let path: SortPath
        let splatCount: Int
        let renderableCount: Int
        let wallTime: TimeInterval
        let callbackWallTime: TimeInterval?
        let gpuTime: TimeInterval?
        let mainQueueDelay: TimeInterval?
        let inFlightSortsAtStart: Int
        let inFlightSortsAtCompletion: Int
        let interactionMode: Bool
        let sortByDistance: Bool
        let status: String

        var overheadTime: TimeInterval? {
            gpuTime.map { max(0, wallTime - $0) }
        }

        var logMessage: String {
            let callbackWallMs = callbackWallTime.map { Self.formatMilliseconds($0) } ?? "n/a"
            let gpuMs = gpuTime.map { Self.formatMilliseconds($0) } ?? "n/a"
            let overheadMs = overheadTime.map { Self.formatMilliseconds($0) } ?? "n/a"
            let mainQueueMs = mainQueueDelay.map { Self.formatMilliseconds($0) } ?? "n/a"
            return "Sort performance path=\(path.rawValue) " +
                "splats=\(splatCount) renderable=\(renderableCount) " +
                "wallMs=\(Self.formatMilliseconds(wallTime)) callbackWallMs=\(callbackWallMs) " +
                "gpuMs=\(gpuMs) overheadMs=\(overheadMs) mainQueueMs=\(mainQueueMs) " +
                "inFlightStart=\(inFlightSortsAtStart) inFlightEnd=\(inFlightSortsAtCompletion) " +
                "interaction=\(interactionMode) sortByDistance=\(sortByDistance) status=\(status)"
        }

        private static func formatMilliseconds(_ duration: TimeInterval) -> String {
            String(format: "%.2f", duration * 1000)
        }
    }

    private struct SortPerformanceContext: Sendable {
        let path: SortPath
        let splatCount: Int
        let renderableCount: Int
        let inFlightSortsAtStart: Int
        let interactionMode: Bool
        let sortByDistance: Bool

        func makeSample(
            wallTime: TimeInterval,
            callbackWallTime: TimeInterval?,
            gpuTime: TimeInterval?,
            mainQueueDelay: TimeInterval?,
            inFlightSortsAtCompletion: Int,
            status: String
        ) -> SortPerformanceSample {
            SortPerformanceSample(
                path: path,
                splatCount: splatCount,
                renderableCount: renderableCount,
                wallTime: wallTime,
                callbackWallTime: callbackWallTime,
                gpuTime: gpuTime,
                mainQueueDelay: mainQueueDelay,
                inFlightSortsAtStart: inFlightSortsAtStart,
                inFlightSortsAtCompletion: inFlightSortsAtCompletion,
                interactionMode: interactionMode,
                sortByDistance: sortByDistance,
                status: status
            )
        }
    }

    enum Constants {
        // Keep in sync with Shaders.metal : maxViewCount
        static let maxViewCount = 2
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
    public var frustumCullingEnabled = false {  // Enable via settings
        didSet {
            if frustumCullingEnabled != oldValue {
                frustumCullDirtyDueToData = true
            }
        }
    }
    private var lastVisibleCount: Int = 0
    private var lastFrustumCullCameraPosition: SIMD3<Float>?
    private var lastFrustumCullCameraForward: SIMD3<Float>?
    private var lastFrustumCullProjectionMatrix: simd_float4x4?
    private var currentFrustumCullProjectionMatrix: simd_float4x4?
    private var lastFrustumCullTime: CFAbsoluteTime = 0
    private var frustumCullDirtyDueToData = true
    
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
    internal var precomputedDataDirty = true
    private var lastPrecomputeViewMatrix: simd_float4x4?
    public var batchPrecomputeEnabled = false  // Enable for large scenes
    
    // PrecomputedSplat structure size (must match Metal shader with proper alignment)
    // Metal alignment: float4 (16) + float3 (12+4 padding) + float2 (8) + float2 (8)
    // + float depth (4) + float opacityScale (4) + uint visible (4) = 60 bytes,
    // rounded to 64 due to the struct's 16-byte alignment.
    internal static let precomputedSplatStride = 64

    internal struct Metal4SIMDOutputs {
        var viewPositions: MTLBuffer
        var clipPositions: MTLBuffer
        var depths: MTLBuffer
        var count: Int
    }

    internal var metal4SIMDOutputs: Metal4SIMDOutputs?

    // Cache Metal 4 compute pipelines to avoid runtime recompilation.
    let metal4PipelineCacheLock = NSLock()
    var metal4ComputePipelineCache: [String: MTLComputePipelineState] = [:]
    
    public struct ViewportDescriptor: Sendable {
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
        case editState      = 4
        case transformIndex = 5
        case transformPalette = 6
    }

    // Keep in sync with Shaders.metal : Uniforms
    struct Uniforms {
        var projectionMatrix: matrix_float4x4
        var viewMatrix: matrix_float4x4
        var screenSize: SIMD2<UInt32> // Size of screen in pixels

        // Precomputed values for covariance projection (derived from projectionMatrix and screenSize)
        var focalX: Float             // screenSize.x * projectionMatrix[0][0] / 2
        var focalY: Float             // screenSize.y * projectionMatrix[1][1] / 2
        var tanHalfFovX: Float        // 1 / projectionMatrix[0][0]
        var tanHalfFovY: Float        // 1 / projectionMatrix[1][1]

        var splatCount: UInt32
        var indexedSplatCount: UInt32
        var debugFlags: UInt32
        var renderMode: UInt32
        var padding0: UInt32
        var padding1: UInt32
        var lodThresholds: SIMD3<Float>
        var covarianceBlur: Float
        var selectionTintColor: SIMD4<Float>
        var editingEnabled: UInt32
        var padding2: UInt32
        var padding3: UInt32
        var padding4: UInt32
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

    // Keep in sync with Shaders.metal : Splat
    struct Splat {
        var position: MTLPackedFloat3
        var packedColor: UInt32  // RGBA8 unorm
        var covA: PackedHalf3
        var covB: PackedHalf3
    }

    static func packRGBA8(_ r: Float, _ g: Float, _ b: Float, _ a: Float) -> UInt32 {
        let rb = UInt32(max(0, min(255, (r * 255).rounded())))
        let gb = UInt32(max(0, min(255, (g * 255).rounded())))
        let bb = UInt32(max(0, min(255, (b * 255).rounded())))
        let ab = UInt32(max(0, min(255, (a * 255).rounded())))
        return rb | (gb << 8) | (bb << 16) | (ab << 24)
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

    /// Rendering behavior for Brush-style covariance filtering.
    /// Updating the mode resets `covarianceBlur` to the mode default.
    public var renderMode: SplatRenderMode = .standard {
        didSet {
            covarianceBlur = renderMode.defaultCovarianceBlur
            invalidatePrecomputedData()
        }
    }

    /// Low-pass filter applied to the 2D covariance diagonal.
    /// Derived from `renderMode` by default.
    /// Override only for compatibility experiments; MIP compensation still comes from `renderMode`.
    public var covarianceBlur: Float = SplatRenderMode.standard.defaultCovarianceBlur {
        didSet {
            invalidatePrecomputedData()
        }
    }

    public var animationConfiguration: SplatAnimationConfiguration? {
        didSet {
            guard animationConfiguration != oldValue else { return }
            refreshAnimationSceneMetricsIfNeeded()
            animationDirty = true
            markRenderableSetDirty()
        }
    }

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

    // MARK: - 2DGS Rendering Mode

    /// When true, uses simplified 2D Gaussian splat rendering instead of full 3D covariance projection.
    ///
    /// **Benefits:**
    /// - Faster rendering (no Jacobian computation)
    /// - Simpler math for screen-space projection
    ///
    /// **Trade-offs:**
    /// - Less accurate for anisotropic (elongated) splats
    /// - Best suited for content with relatively uniform/circular splats
    /// - May show visual artifacts on highly stretched splats
    ///
    /// The 2DGS mode uses uniform circular splats based on the maximum covariance component,
    /// avoiding the full 3D covariance projection and eigenvalue decomposition.
    public var use2DGSMode: Bool = false {
        didSet {
            if use2DGSMode != oldValue {
                // Invalidate pipeline states to rebuild with correct settings
                invalidatePipelineStates()
            }
        }
    }

    public var sortPositionEpsilon: Float = 0.01
    public var sortDirectionEpsilon: Float = 0.0001  // ~0.5-1° rotation (reduced from 0.001 to fix flickering during rotation)
    public var minimumSortInterval: TimeInterval = 0
    public var frustumCullPositionEpsilon: Float = 0.01
    public var frustumCullDirectionEpsilon: Float = 0.0001
    public var minimumFrustumCullInterval: TimeInterval = 0

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
    private var qualityFrustumCullPositionEpsilon: Float = 0.01
    private var qualityFrustumCullDirectionEpsilon: Float = 0.0001
    private var qualityMinimumFrustumCullInterval: TimeInterval = 0
    private var qualityHighQualityDepth: Bool = true
    
    /// Interaction mode sort parameters (relaxed for performance)
    public var interactionSortPositionEpsilon: Float = 0.05      // 5cm during interaction
    public var interactionSortDirectionEpsilon: Float = 0.003    // ~2-3° during interaction
    public var interactionMinimumSortInterval: TimeInterval = 0.033  // Max ~30 sorts/sec
    public var interactionFrustumCullPositionEpsilon: Float = 0.05
    public var interactionFrustumCullDirectionEpsilon: Float = 0.003
    public var interactionMinimumFrustumCullInterval: TimeInterval = 0.033
    
    /// Delay before forcing a final high-quality sort after interaction ends
    public var postInteractionSortDelay: TimeInterval = 0.1
    private var interactionEndTime: CFAbsoluteTime?

    // MARK: - Adaptive Sort Frequency

    /// When true, sort interval adapts based on frame time to maintain target FPS
    public var adaptiveSortFrequencyEnabled: Bool = false

    /// Target frame rate for adaptive sort interval calculation
    public var targetFrameRate: Double = 60.0

    /// Computed adaptive sort interval based on recent frame performance.
    /// Respects interaction mode (minimumSortInterval is already adjusted there).
    private var effectiveMinimumSortInterval: TimeInterval {
        var interval = minimumSortInterval

        guard adaptiveSortFrequencyEnabled else { return interval }

        let targetFrameTime = 1.0 / targetFrameRate

        if averageFrameTime > targetFrameTime * 1.2 {
            // Over budget by 20%+ - sort less often to recover
            interval = max(interval, targetFrameTime * 2.0)
        } else if averageFrameTime < targetFrameTime * 0.8 {
            // Under budget by 20%+ - can sort more often
            interval = interval * 0.5
        }
        return interval
    }

    // MARK: - Sorting Mode Selection

    /// Sorting mode for depth ordering. Default is `.auto` which selects based on camera motion.
    /// - `.radial`: Always use squared distance from camera (better for rotation-heavy views)
    /// - `.linear`: Always use dot product with view direction (better for translation)
    /// - `.auto`: Automatically select based on camera motion between frames
    public var sortingMode: SortingMode = .auto

    /// Threshold for auto-selecting sorting mode. When the ratio of rotation to translation
    /// exceeds this value, radial sorting is preferred. Range: 0.0 to 1.0
    public var autoSortModeRotationBias: Float = 0.5

    /// Previous camera position for motion tracking (used in auto mode)
    private var previousCameraPosition: SIMD3<Float>?

    /// Previous camera forward for motion tracking (used in auto mode)
    private var previousCameraForward: SIMD3<Float>?

    /// Last computed effective sorting mode (for debugging/statistics)
    private(set) var lastEffectiveSortingMode: SortingMode = .radial

    /// Computes the effective sort mode based on camera motion when in auto mode.
    /// Returns true for radial sorting, false for linear sorting.
    private func computeEffectiveSortByDistance(
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>
    ) -> Bool {
        switch sortingMode {
        case .radial:
            lastEffectiveSortingMode = .radial
            return true
        case .linear:
            lastEffectiveSortingMode = .linear
            return false
        case .auto:
            // Track camera motion to determine best sorting mode
            guard let prevPos = previousCameraPosition,
                  let prevFwd = previousCameraForward else {
                // First frame: default to radial
                previousCameraPosition = cameraPosition
                previousCameraForward = cameraForward
                lastEffectiveSortingMode = .radial
                return true
            }

            // Compute translation magnitude (squared for efficiency)
            let translationDelta = cameraPosition - prevPos
            let translationMagnitudeSq = simd_dot(translationDelta, translationDelta)

            // Compute rotation magnitude via dot product deviation from 1.0
            // (1 - dot) gives 0 for no rotation, approaches 2 for 180° rotation
            let rotationDelta = 1.0 - simd_dot(cameraForward, prevFwd)

            // Update previous values for next frame
            previousCameraPosition = cameraPosition
            previousCameraForward = cameraForward

            // Normalize rotation to roughly match translation scale
            // rotation of ~0.01 (about 8°) corresponds to noticeable turn
            // Multiply by 100 to bring it to similar magnitude as typical translation
            let normalizedRotation = rotationDelta * 100.0

            // Calculate ratio: higher = more rotation relative to translation
            let total = Float(translationMagnitudeSq) + normalizedRotation
            guard total > 0.0001 else {
                // Minimal motion - keep last mode
                return lastEffectiveSortingMode == .radial
            }

            let rotationRatio = normalizedRotation / total

            // Use radial if rotation dominates, linear if translation dominates
            let useRadial = rotationRatio > autoSortModeRotationBias
            lastEffectiveSortingMode = useRadial ? .radial : .linear
            return useRadial
        }
    }

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
    internal var selectionOutlinePipelineState: MTLRenderPipelineState?
    internal var selectionOutlineDepthState: MTLDepthStencilState?
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
    private var currentSortViewMatrix: simd_float4x4?
    private var lastSortedViewMatrix: simd_float4x4?
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
    var animatedSplatBuffer: MetalBuffer<Splat>?
    internal var sourceScenePoints: [SplatScenePoint] = []
    internal var animationSceneIndices: [UInt32] = []
    internal var animationSceneCounts: [Int] = []
    internal var animationSceneMetrics: [SplatAnimationSceneMetrics] = []
    internal var animationMetricsDirty = false
    internal var animationDirty = true
    internal var lastAppliedAnimationTime: Float?
    
    // GPU-only sorting: sorted indices buffer holds the depth-sorted order
    // Shaders use this to index into splatBuffer in the correct render order
    // This eliminates CPU readback and reordering - a major performance win
    var sortedIndicesBuffer: MetalBuffer<Int32>?
    
    // Legacy: splatBufferPrime kept for CPU fallback sorting path
    var splatBufferPrime: MetalBuffer<Splat>

    var indexBuffer: MetalBuffer<UInt32>

    internal var editStateBuffer: MTLBuffer?
    internal var editTransformIndexBuffer: MTLBuffer?
    internal var editTransformPaletteBuffer: MTLBuffer?
    internal var selectionTintColor = SIMD4<Float>(0.15, 0.55, 1.0, 0.45)
    internal var selectionOutlineEnabled = true
    internal var editingEnabled = false
    private var nonZeroEditStateCount = 0
    private var selectedEditStateCount = 0
    private var hiddenOrDeletedEditStateCount = 0
    private var activeTransformIndexCount = 0
    private var previewTransformActive = false
    private var savedOptimizedEditSettings: (meshShaderEnabled: Bool, batchPrecomputeEnabled: Bool)?

    public var splatCount: Int { splatBuffer.count }
    internal var renderableSplatCountForCurrentEditState: Int {
        max(0, splatCount - hiddenOrDeletedEditStateCount)
    }
    internal var shouldDrawSelectionOutline: Bool {
        selectionOutlineEnabled && selectedEditStateCount > 0 && !isInteracting
    }

    var sorting = false
    private var sortJobsInFlight: Int = 0  // Track concurrent sort operations
    private let maxConcurrentSorts: Int = 2  // Allow overlap: one sorting while one renders
    private var sortStateLock = os_unfair_lock()  // Protects sort scheduling state and buffer swap
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

    /// When true and Metal 4 is available, uses the experimental atomic radix sorter.
    /// This stays opt-in because counting sort is faster for typical splat scenes.
    public var useMetal4Sorting: Bool = false

    /// Minimum splat count to use Metal 4 sorting (below this, counting sort is faster)
    public var metal4SortingThreshold: Int = 100_000


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

    private func getSortFrameState() -> (ready: Bool, duration: TimeInterval?, jobsInFlight: Int) {
        os_unfair_lock_lock(&sortStateLock)
        defer { os_unfair_lock_unlock(&sortStateLock) }
        return (!sorting, lastSortDuration, sortJobsInFlight)
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

    /// Atomically publish scheduling metadata and clear the in-flight sort gate.
    private func finishSortState(
        duration: TimeInterval,
        bufferReadyTime: CFAbsoluteTime,
        cameraWorldPosition: SIMD3<Float>,
        cameraWorldForward: SIMD3<Float>,
        sortViewMatrix: simd_float4x4?,
        dataDirtySnapshot: UInt64
    ) -> Int {
        os_unfair_lock_lock(&sortStateLock)
        lastSortDuration = duration
        lastSortedCameraPosition = cameraWorldPosition
        lastSortedCameraForward = cameraWorldForward
        lastSortedViewMatrix = sortViewMatrix
        lastSortTime = bufferReadyTime
        if sortDataRevision == dataDirtySnapshot {
            sortDirtyDueToData = false
        }
        sorting = false
        sortJobsInFlight -= 1
        let inFlightSortsAtCompletion = sortJobsInFlight
        os_unfair_lock_unlock(&sortStateLock)
        return inFlightSortsAtCompletion
    }

    private func markSortDataDirty() {
        os_unfair_lock_lock(&sortStateLock)
        sortDirtyDueToData = true
        sortDataRevision &+= 1
        os_unfair_lock_unlock(&sortStateLock)
    }

    private func markSortDirtyWithoutRevisionChange() {
        os_unfair_lock_lock(&sortStateLock)
        sortDirtyDueToData = true
        os_unfair_lock_unlock(&sortStateLock)
    }

    private func getSortDataRevision() -> UInt64 {
        os_unfair_lock_lock(&sortStateLock)
        defer { os_unfair_lock_unlock(&sortStateLock) }
        return sortDataRevision
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
    // Warn when pending releases build up (helps detect long GPU stalls)
    private static let maxPendingReleaseBuffers = 16

    /// Queue a buffer for deferred release (call when swapping sorted indices)
    private func deferredBufferRelease(_ buffer: MetalBuffer<Int32>) {
        os_unfair_lock_lock(&pendingReleaseLock)
        if pendingReleaseBuffers.count >= Self.maxPendingReleaseBuffers {
            Self.log.warning("Pending sorted-index releases reached \(self.pendingReleaseBuffers.count + 1); waiting for GPU completion")
        }
        pendingReleaseBuffers.append(buffer)
        os_unfair_lock_unlock(&pendingReleaseLock)
    }

    /// Drain any queued buffers that need release once GPU work completes.
    private func drainPendingReleaseBuffers() -> [MetalBuffer<Int32>] {
        os_unfair_lock_lock(&pendingReleaseLock)
        let toRelease = pendingReleaseBuffers
        pendingReleaseBuffers.removeAll()
        os_unfair_lock_unlock(&pendingReleaseLock)
        return toRelease
    }

    /// Schedule pending sorted-index buffer releases on command buffer completion.
    private func schedulePendingBufferRelease(on commandBuffer: MTLCommandBuffer) {
        let toRelease = drainPendingReleaseBuffers()
        guard !toRelease.isEmpty else { return }

        commandBuffer.addCompletedHandler { [weak self] _ in
            guard let self else { return }
            for buffer in toRelease {
                self.sortIndexBufferPool.release(buffer)
            }
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
            // Log failures - these are optional features that degrade gracefully
            frustumCullDataBuffer = device.makeBuffer(length: MemoryLayout<FrustumCullData>.stride, options: .storageModeShared)
            if frustumCullDataBuffer == nil {
                Self.log.warning("Failed to create frustum cull data buffer - frustum culling disabled")
            }
            frustumCullDataBuffer?.label = "Frustum Cull Data"
            visibleCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
            if visibleCountBuffer == nil {
                Self.log.warning("Failed to create visible count buffer - frustum culling disabled")
            }
            visibleCountBuffer?.label = "Visible Count"
            // Indirect draw arguments buffer (MTLDrawIndexedPrimitivesIndirectArguments = 5 * uint32)
            indirectDrawArgsBuffer = device.makeBuffer(length: 5 * MemoryLayout<UInt32>.stride, options: .storageModePrivate)
            if indirectDrawArgsBuffer == nil {
                Self.log.warning("Failed to create indirect draw args buffer - frustum culling disabled")
            }
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
            // Log failures - GPU bounds computation will fall back to CPU
            boundsMinBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * 3, options: .storageModeShared)
            boundsMaxBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride * 3, options: .storageModeShared)
            if boundsMinBuffer == nil || boundsMaxBuffer == nil {
                Self.log.warning("Failed to create bounds buffers - GPU bounds computation disabled")
            }
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
            // Set function constants for 2DGS mode
            let functionConstants = MTLFunctionConstantValues()
            var use2DGSValue = use2DGSMode
            functionConstants.setConstantValue(&use2DGSValue, type: .bool, index: 12)

            // Try to load mesh shader functions with function constants
            guard let objectFunction = try? library.makeFunction(name: "splatObjectShader", constantValues: functionConstants),
                  let meshFunction = try? library.makeFunction(name: "splatMeshShader", constantValues: functionConstants),
                  let fragmentFunction = try? library.makeFunction(name: "meshSplatFragmentShader", constantValues: functionConstants) else {
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
        if let animatedSplatBuffer {
            splatBufferPool.release(animatedSplatBuffer)
        }
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
        if let animatedSplatBuffer {
            splatBufferPool.release(animatedSplatBuffer)
            self.animatedSplatBuffer = nil
        }
        resetEditingTracking()
        sourceScenePoints.removeAll(keepingCapacity: false)
        animationSceneIndices.removeAll(keepingCapacity: false)
        animationSceneCounts.removeAll(keepingCapacity: false)
        animationSceneMetrics.removeAll(keepingCapacity: false)
        animationMetricsDirty = false
        animationDirty = true
        lastAppliedAnimationTime = nil
        
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
        let reader = try AutodetectSceneReader(url)
        var newPoints = SplatMemoryBuffer()
        try await newPoints.read(from: reader)
        renderMode = Self.renderMode(from: reader.renderMode)
        try add(newPoints.points)
    }

    internal func resetPipelineStates() {
        singleStagePipelineState = nil
        singleStageDepthState = nil
        selectionOutlinePipelineState = nil
        selectionOutlineDepthState = nil
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

    /// Compiles the active render pipeline(s) before first draw to avoid first-frame JIT stalls.
    public func prewarmRenderPipelines() {
        do {
            if useMultiStagePipeline {
                try buildMultiStagePipelineStatesIfNeeded()
            } else if useDitheredTransparency {
                try buildDitheredPipelineStatesIfNeeded()
            } else {
                try buildSingleStagePipelineStatesIfNeeded()
            }

            if meshShaderEnabled && meshShaderPipelineState == nil {
                setupMeshShaders()
            }
        } catch {
            Self.log.warning("Pipeline prewarm failed: \(error.localizedDescription)")
        }
    }

    @available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
    internal func updateMetal4ResidencyForFrame(commandBuffer: MTLCommandBuffer) {
        guard let manager = metal4ArgumentBufferManager else { return }

        do {
            try manager.registerSplatBuffer(splatBuffer.buffer, at: 0)
            try manager.registerUniformBuffer(dynamicUniformBuffers, at: 1)
            manager.registerAdditionalBuffer(indexBuffer.buffer)

            if let sortedIndices = sortedIndicesBuffer {
                manager.registerAdditionalBuffer(sortedIndices.buffer)
            }
            if let visibleIndicesBuffer {
                manager.registerAdditionalBuffer(visibleIndicesBuffer)
            }
            if let visibleCountBuffer {
                manager.registerAdditionalBuffer(visibleCountBuffer)
            }
            if let frustumCullDataBuffer {
                manager.registerAdditionalBuffer(frustumCullDataBuffer)
            }
            if let indirectDrawArgsBuffer {
                manager.registerAdditionalBuffer(indirectDrawArgsBuffer)
            }
            if let precomputedSplatBuffer {
                manager.registerAdditionalBuffer(precomputedSplatBuffer)
            }
            if let boundsMinBuffer {
                manager.registerAdditionalBuffer(boundsMinBuffer)
            }
            if let boundsMaxBuffer {
                manager.registerAdditionalBuffer(boundsMaxBuffer)
            }
        } catch {
            Self.log.warning("Metal 4 residency registration failed: \(error.localizedDescription)")
        }

        manager.makeResourcesResident(commandBuffer: commandBuffer)
    }

    private func buildSingleStagePipelineStatesIfNeeded() throws {
        guard singleStagePipelineState == nil else { return }

        singleStagePipelineState = try buildSingleStagePipelineState()
        singleStageDepthState = try buildSingleStageDepthState()
        selectionOutlinePipelineState = try buildSelectionOutlinePipelineState()
        selectionOutlineDepthState = try buildSelectionOutlineDepthState()
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
        guard !useMultiStagePipeline else {
            throw SplatRendererError.internalPipelineMismatch(expected: "single-stage", actual: "multi-stage")
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "SingleStagePipeline"

        // Set function constants for 2DGS mode
        let functionConstants = MTLFunctionConstantValues()
        var use2DGSValue = use2DGSMode
        functionConstants.setConstantValue(&use2DGSValue, type: .bool, index: 12)

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
        guard !useMultiStagePipeline else {
            throw SplatRendererError.internalPipelineMismatch(expected: "single-stage", actual: "multi-stage")
        }

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        guard let depthState = device.makeDepthStencilState(descriptor: depthStateDescriptor) else {
            throw SplatRendererError.failedToCreateDepthStencilState
        }
        return depthState
    }

    private func buildSelectionOutlinePipelineState() throws -> MTLRenderPipelineState {
        guard !useMultiStagePipeline else {
            throw SplatRendererError.internalPipelineMismatch(expected: "single-stage", actual: "multi-stage")
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "SelectionOutlinePipeline"

        let functionConstants = MTLFunctionConstantValues()
        var use2DGSValue = use2DGSMode
        functionConstants.setConstantValue(&use2DGSValue, type: .bool, index: 12)

        pipelineDescriptor.vertexFunction = try library.makeFunction(name: "selectedOutlineVertexShader", constantValues: functionConstants)
        pipelineDescriptor.fragmentFunction = try library.makeRequiredFunction(name: "selectedOutlineFragmentShader")
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

    private func buildSelectionOutlineDepthState() throws -> MTLDepthStencilState {
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = .lessEqual
        depthStateDescriptor.isDepthWriteEnabled = false
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

        // Set function constants for 2DGS mode
        let functionConstants = MTLFunctionConstantValues()
        var use2DGSValue = use2DGSMode
        functionConstants.setConstantValue(&use2DGSValue, type: .bool, index: 12)

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
        guard useMultiStagePipeline else {
            throw SplatRendererError.internalPipelineMismatch(expected: "multi-stage", actual: "single-stage")
        }

        let pipelineDescriptor = MTLTileRenderPipelineDescriptor()

        pipelineDescriptor.label = "InitializePipeline"
        pipelineDescriptor.tileFunction = try library.makeRequiredFunction(name: "initializeFragmentStore")
        pipelineDescriptor.threadgroupSizeMatchesTileSize = true;
        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat

        return try device.makeRenderPipelineState(tileDescriptor: pipelineDescriptor, options: [], reflection: nil)
    }

    private func buildDrawSplatPipelineState() throws -> MTLRenderPipelineState {
        guard useMultiStagePipeline else {
            throw SplatRendererError.internalPipelineMismatch(expected: "multi-stage", actual: "single-stage")
        }

        let pipelineDescriptor = MTLRenderPipelineDescriptor()

        pipelineDescriptor.label = "DrawSplatPipeline"

        // Set function constants for 2DGS mode
        let functionConstants = MTLFunctionConstantValues()
        var use2DGSValue = use2DGSMode
        functionConstants.setConstantValue(&use2DGSValue, type: .bool, index: 12)

        pipelineDescriptor.vertexFunction = try library.makeFunction(name: "multiStageSplatVertexShader", constantValues: functionConstants)
        pipelineDescriptor.fragmentFunction = try library.makeRequiredFunction(name: "multiStageSplatFragmentShader")

        pipelineDescriptor.rasterSampleCount = sampleCount

        pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
        pipelineDescriptor.depthAttachmentPixelFormat = depthFormat

        pipelineDescriptor.maxVertexAmplificationCount = maxViewCount

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func buildDrawSplatDepthState() throws -> MTLDepthStencilState {
        guard useMultiStagePipeline else {
            throw SplatRendererError.internalPipelineMismatch(expected: "multi-stage", actual: "single-stage")
        }

        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = MTLCompareFunction.always
        depthStateDescriptor.isDepthWriteEnabled = writeDepth
        guard let depthState = device.makeDepthStencilState(descriptor: depthStateDescriptor) else {
            throw SplatRendererError.failedToCreateDepthStencilState
        }
        return depthState
    }

    private func buildPostprocessPipelineState() throws -> MTLRenderPipelineState {
        guard useMultiStagePipeline else {
            throw SplatRendererError.internalPipelineMismatch(expected: "multi-stage", actual: "single-stage")
        }

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
        guard useMultiStagePipeline else {
            throw SplatRendererError.internalPipelineMismatch(expected: "multi-stage", actual: "single-stage")
        }

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
            throw error
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
        sourceScenePoints.append(contentsOf: orderedPoints)
        if animationSceneIndices.isEmpty {
            animationSceneIndices = Array(repeating: 0, count: sourceScenePoints.count)
            animationSceneCounts = [sourceScenePoints.count]
        } else {
            let targetSceneIndex = animationSceneIndices.last ?? 0
            animationSceneIndices.append(contentsOf: Array(repeating: targetSceneIndex, count: orderedPoints.count))
            if animationSceneCounts.isEmpty {
                animationSceneCounts = [sourceScenePoints.count]
            } else {
                animationSceneCounts[animationSceneCounts.count - 1] += orderedPoints.count
            }
        }
        animationSceneMetrics = Self.makeSceneMetrics(points: sourceScenePoints,
                                                      sceneIndices: animationSceneIndices,
                                                      sceneCounts: animationSceneCounts)
        animationDirty = true
        markGeometryDirty()  // New splats affect geometry and require re-sorting
        colorsDirty = true   // New splats also have new colors
        try ensureEditingResources(pointCount: splatBuffer.count)

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

    internal func ensureEditingResources(pointCount: Int) throws {
        let stateLength = max(pointCount, 1) * MemoryLayout<UInt32>.stride
        if editStateBuffer == nil || editStateBuffer?.length != stateLength {
            guard let buffer = device.makeBuffer(length: stateLength, options: .storageModeShared) else {
                throw SplatRendererError.failedToCreateBuffer(length: stateLength)
            }
            buffer.label = "Editable Splat State Buffer"
            memset(buffer.contents(), 0, stateLength)
            editStateBuffer = buffer
        }

        if editTransformIndexBuffer == nil || editTransformIndexBuffer?.length != stateLength {
            guard let buffer = device.makeBuffer(length: stateLength, options: .storageModeShared) else {
                throw SplatRendererError.failedToCreateBuffer(length: stateLength)
            }
            buffer.label = "Editable Transform Index Buffer"
            memset(buffer.contents(), 0, stateLength)
            editTransformIndexBuffer = buffer
        }

        let paletteLength = max(2, 2) * MemoryLayout<matrix_float4x4>.stride
        if editTransformPaletteBuffer == nil || editTransformPaletteBuffer?.length != paletteLength {
            guard let buffer = device.makeBuffer(length: paletteLength, options: .storageModeShared) else {
                throw SplatRendererError.failedToCreateBuffer(length: paletteLength)
            }
            buffer.label = "Editable Transform Palette Buffer"
            let palette = buffer.contents().bindMemory(to: matrix_float4x4.self, capacity: 2)
            palette[0] = matrix_identity_float4x4
            palette[1] = matrix_identity_float4x4
            editTransformPaletteBuffer = buffer
        }
    }

    internal func replaceEditingState(_ rawStates: [UInt32],
                                      transformIndices: [UInt32],
                                      transformPalette: [matrix_float4x4]) throws {
        let pointCount = rawStates.count
        try ensureEditingResources(pointCount: pointCount)

        guard let editStateBuffer,
              let editTransformIndexBuffer,
              let editTransformPaletteBuffer else {
            return
        }

        let bufferCount = max(pointCount, 1)
        let statePointer = editStateBuffer.contents().bindMemory(to: UInt32.self, capacity: bufferCount)
        memset(statePointer, 0, stateLength(for: bufferCount))
        nonZeroEditStateCount = 0
        selectedEditStateCount = 0
        hiddenOrDeletedEditStateCount = 0
        for (index, value) in rawStates.enumerated() {
            statePointer[index] = value
            updateEditStateCounters(from: 0, to: value)
        }

        let transformIndexPointer = editTransformIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: bufferCount)
        memset(transformIndexPointer, 0, stateLength(for: bufferCount))
        activeTransformIndexCount = 0
        for (index, value) in transformIndices.prefix(pointCount).enumerated() {
            transformIndexPointer[index] = value
            updateTransformIndexCounters(from: 0, to: value)
        }

        let palettePointer = editTransformPaletteBuffer.contents().bindMemory(to: matrix_float4x4.self, capacity: max(transformPalette.count, 2))
        writeTransformPalette(transformPalette, to: palettePointer)
        refreshEditingEnabled()
    }

    internal func updateEditingState(_ rawStates: [UInt32],
                                     transformIndices: [UInt32],
                                     transformPalette: [matrix_float4x4]) throws {
        try replaceEditingState(rawStates, transformIndices: transformIndices, transformPalette: transformPalette)
    }

    internal func updateEditStates(at indices: [Int], values: [UInt32]) throws {
        guard indices.count == values.count else { return }
        try ensureEditingResources(pointCount: splatCount)
        guard let editStateBuffer else { return }

        let previousRenderableCount = renderableSplatCountForCurrentEditState
        let pointer = editStateBuffer.contents().bindMemory(to: UInt32.self, capacity: max(splatCount, 1))
        var visibilityChanged = false
        for (index, value) in zip(indices, values) where index >= 0 && index < splatCount {
            let oldValue = pointer[index]
            guard oldValue != value else { continue }
            visibilityChanged = visibilityChanged || isHiddenOrDeleted(oldValue) != isHiddenOrDeleted(value)
            updateEditStateCounters(from: oldValue, to: value)
            pointer[index] = value
        }
        refreshEditingEnabled()
        if visibilityChanged {
            let currentRenderableCount = renderableSplatCountForCurrentEditState
            if currentRenderableCount < previousRenderableCount,
               compactCurrentSortedIndicesForReducedRenderableSet() {
                markRenderableSetReducedWithoutResort()
            } else {
                markRenderableSetDirty()
            }
        }
    }

    internal func updateTransformIndices(at indices: [Int], values: [UInt32]) throws {
        guard indices.count == values.count else { return }
        try ensureEditingResources(pointCount: splatCount)
        guard let editTransformIndexBuffer else { return }

        let pointer = editTransformIndexBuffer.contents().bindMemory(to: UInt32.self, capacity: max(splatCount, 1))
        for (index, value) in zip(indices, values) where index >= 0 && index < splatCount {
            let oldValue = pointer[index]
            guard oldValue != value else { continue }
            updateTransformIndexCounters(from: oldValue, to: value)
            pointer[index] = value
        }
        refreshEditingEnabled()
    }

    internal func updateTransformPalette(_ transformPalette: [matrix_float4x4]) throws {
        try ensureEditingResources(pointCount: splatCount)
        guard let editTransformPaletteBuffer else { return }
        let palettePointer = editTransformPaletteBuffer.contents().bindMemory(to: matrix_float4x4.self, capacity: max(transformPalette.count, 2))
        writeTransformPalette(transformPalette, to: palettePointer)
    }

    internal func setPreviewTransformActive(_ active: Bool) throws {
        guard previewTransformActive != active else { return }
        previewTransformActive = active

        if active {
            savedOptimizedEditSettings = (meshShaderEnabled, batchPrecomputeEnabled)
            meshShaderEnabled = false
            batchPrecomputeEnabled = false
        } else if let savedOptimizedEditSettings {
            meshShaderEnabled = savedOptimizedEditSettings.meshShaderEnabled
            batchPrecomputeEnabled = savedOptimizedEditSettings.batchPrecomputeEnabled
            self.savedOptimizedEditSettings = nil
        }
    }

    internal func updateSplats(_ points: [SplatScenePoint], at indices: [Int]) throws {
        guard !indices.isEmpty else { return }
        splatBuffer.withLockedValues { values, count in
            for index in indices where index >= 0 && index < count && index < points.count {
                values[index] = Splat(points[index])
            }
        }
        for index in indices where index >= 0 && index < sourceScenePoints.count && index < points.count {
            sourceScenePoints[index] = points[index]
            animationMetricsDirty = true
        }
        animationDirty = animationDirty || animationMetricsDirty
        markGeometryDirty()
    }

    public func replaceSceneLayers(_ layers: [SplatSceneLayer]) throws {
        let allPoints = layers.flatMap(\.points)
        let counts = layers.map { $0.points.count }
        try replaceAllSplats(with: allPoints, sceneCounts: counts)
    }

    internal func replaceAllSplats(with points: [SplatScenePoint],
                                   sceneCounts: [Int]? = nil,
                                   sceneIndices: [UInt32]? = nil) throws {
        try ensureAdditionalCapacity(points.count)
        splatBuffer.count = 0
        splatBuffer.append(points.map { Splat($0) })
        if let sceneIndices, sceneIndices.count == points.count {
            setAnimationSourcePoints(points, sceneIndices: sceneIndices)
        } else if let sceneCounts, sceneCounts.reduce(0, +) == points.count {
            setAnimationSourcePoints(points, sceneCounts: sceneCounts)
        } else {
            setAnimationSourcePoints(points)
        }
        markGeometryDirty()
        colorsDirty = true
        resetEditingTracking()
        try ensureEditingResources(pointCount: points.count)
        try initializeIdentitySortedIndices()
    }

    private func refreshEditingEnabled() {
        editingEnabled = nonZeroEditStateCount > 0 || activeTransformIndexCount > 0
    }

    private func updateEditStateCounters(from oldValue: UInt32, to newValue: UInt32) {
        if oldValue == 0, newValue != 0 {
            nonZeroEditStateCount += 1
        } else if oldValue != 0, newValue == 0 {
            nonZeroEditStateCount -= 1
        }

        let selectedMask = EditableSplatState.selected.rawValue
        let oldSelected = (oldValue & selectedMask) != 0
        let newSelected = (newValue & selectedMask) != 0
        if !oldSelected, newSelected {
            selectedEditStateCount += 1
        } else if oldSelected, !newSelected {
            selectedEditStateCount -= 1
        }

        if !isHiddenOrDeleted(oldValue), isHiddenOrDeleted(newValue) {
            hiddenOrDeletedEditStateCount += 1
        } else if isHiddenOrDeleted(oldValue), !isHiddenOrDeleted(newValue) {
            hiddenOrDeletedEditStateCount -= 1
        }
    }

    private func updateTransformIndexCounters(from oldValue: UInt32, to newValue: UInt32) {
        if oldValue == 0, newValue != 0 {
            activeTransformIndexCount += 1
        } else if oldValue != 0, newValue == 0 {
            activeTransformIndexCount -= 1
        }
    }

    private func writeTransformPalette(_ transformPalette: [matrix_float4x4],
                                       to palettePointer: UnsafeMutablePointer<matrix_float4x4>) {
        palettePointer[0] = matrix_identity_float4x4
        palettePointer[1] = matrix_identity_float4x4
        for (index, transform) in transformPalette.enumerated() where index < 2 {
            palettePointer[index] = transform
        }
    }

    private func isHiddenOrDeleted(_ value: UInt32) -> Bool {
        (value & (EditableSplatState.hidden.rawValue | EditableSplatState.deleted.rawValue)) != 0
    }

    private func stateLength(for pointCount: Int) -> Int {
        max(pointCount, 1) * MemoryLayout<UInt32>.stride
    }

    private func markRenderableSetDirty() {
        frustumCullDirtyDueToData = true
        markSortDataDirty()
    }

    private func markRenderableSetReducedWithoutResort() {
        frustumCullDirtyDueToData = true
    }

    @discardableResult
    private func compactCurrentSortedIndicesForReducedRenderableSet() -> Bool {
        guard let editStateBuffer,
              let sortedIndicesBuffer = getCurrentSortedIndicesBuffer() else {
            return false
        }

        let statePointer = editStateBuffer.contents().bindMemory(to: UInt32.self, capacity: max(splatCount, 1))
        var didCompact = false
        var compactedCount: Int?

        sortedIndicesBuffer.withLockedValues { values, count in
            guard count > 0 else { return }

            var writeIndex = 0
            for readIndex in 0..<count {
                let splatIndex = Int(values[readIndex])
                guard splatIndex >= 0 && splatIndex < splatCount else { continue }
                guard !isHiddenOrDeleted(statePointer[splatIndex]) else {
                    didCompact = true
                    continue
                }
                values[writeIndex] = values[readIndex]
                writeIndex += 1
            }

            if writeIndex != count {
                compactedCount = writeIndex
                didCompact = true
            }
        }

        if let compactedCount {
            sortedIndicesBuffer.count = compactedCount
        }

        return didCompact
    }

    private func resetEditingTracking() {
        editingEnabled = false
        nonZeroEditStateCount = 0
        selectedEditStateCount = 0
        hiddenOrDeletedEditStateCount = 0
        activeTransformIndexCount = 0
        previewTransformActive = false
        savedOptimizedEditSettings = nil
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

        // Optimization: Find last .full update and skip all updates before it
        // This preserves "last write wins" semantics while avoiding redundant work
        let startIndex: Int
        if let lastFullIndex = updates.lastIndex(where: { if case .full = $0 { return true } else { return false } }) {
            startIndex = lastFullIndex
        } else {
            startIndex = 0
        }

        // Apply updates in order to preserve "last write wins" semantics
        // Use withLockedValues to safely access buffer during potential resize operations
        splatBuffer.withLockedValues { values, bufferCount in
            for update in updates[startIndex...] {
                switch update {
                case .full(let colors):
                    for i in 0..<min(colors.count, bufferCount) {
                        let c = colors[i]
                        values[i].packedColor = SplatRenderer.packRGBA8(c.x, c.y, c.z, c.w)
                        if i < sourceScenePoints.count {
                            sourceScenePoints[i].color = .linearFloat(SIMD3<Float>(c.x, c.y, c.z))
                            sourceScenePoints[i].opacity = .linearFloat(c.w)
                        }
                    }
                case .range(let colors, let range):
                    for (i, colorIndex) in range.enumerated() where colorIndex < bufferCount {
                        let c = colors[i]
                        values[colorIndex].packedColor = SplatRenderer.packRGBA8(c.x, c.y, c.z, c.w)
                        if colorIndex < sourceScenePoints.count {
                            sourceScenePoints[colorIndex].color = .linearFloat(SIMD3<Float>(c.x, c.y, c.z))
                            sourceScenePoints[colorIndex].opacity = .linearFloat(c.w)
                        }
                    }
                case .single(let color, let index):
                    guard index < bufferCount else { continue }
                    values[index].packedColor = SplatRenderer.packRGBA8(color.x, color.y, color.z, color.w)
                    if index < sourceScenePoints.count {
                        sourceScenePoints[index].color = .linearFloat(SIMD3<Float>(color.x, color.y, color.z))
                        sourceScenePoints[index].opacity = .linearFloat(color.w)
                    }
                }
            }
        }
        animationDirty = true
    }

    /// Marks that geometry has changed and requires re-sorting and bounds update.
    /// Called internally when positions or covariance values are modified.
    private func markGeometryDirty() {
        geometryDirty = true
        frustumCullDirtyDueToData = true
        markSortDataDirty()
        boundsDirty = true
        animationDirty = true
        invalidatePrecomputedData()
    }

    internal func markAnimationDependentDataDirty() {
        frustumCullDirtyDueToData = true
        markSortDataDirty()
        boundsDirty = true
        invalidatePrecomputedData()
    }

    internal func acquireAnimatedSplatBuffer(minimumCapacity: Int) throws -> MetalBuffer<Splat> {
        try splatBufferPool.acquire(minimumCapacity: minimumCapacity)
    }

    internal func releaseAnimatedSplatBuffer(_ buffer: MetalBuffer<Splat>, on commandBuffer: MTLCommandBuffer) {
        commandBuffer.addCompletedHandler { [weak self] (_: MTLCommandBuffer) in
            self?.splatBufferPool.release(buffer)
        }
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
        computeEncoder.setBuffer(activeSplatBufferForRendering.buffer, offset: 0, index: 0)

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
        let expectedCount = splatCount
        guard expectedCount > 0 else { return nil }

        return activeSplatBufferForRendering.withLockedValues { values, count in
            let actualCount = min(count, expectedCount)
            var minBounds = SIMD3<Float>(repeating: .infinity)
            var maxBounds = SIMD3<Float>(repeating: -.infinity)

            for i in 0..<actualCount {
                let position = SIMD3<Float>(values[i].position.elements.0,
                                           values[i].position.elements.1,
                                           values[i].position.elements.2)
                minBounds = min(minBounds, position)
                maxBounds = max(maxBounds, position)
            }

            return (min: minBounds, max: maxBounds)
        }
    }

    internal static func estimateCountingSortDepthBounds(
        from bounds: (min: SIMD3<Float>, max: SIMD3<Float>),
        cameraPosition: SIMD3<Float>,
        cameraForward: SIMD3<Float>,
        sortByDistance: Bool
    ) -> (min: Float, max: Float) {
        let minimumDepth: Float
        let maximumDepth: Float

        if sortByDistance {
            let nearestPoint = SIMD3<Float>(
                min(max(cameraPosition.x, bounds.min.x), bounds.max.x),
                min(max(cameraPosition.y, bounds.min.y), bounds.max.y),
                min(max(cameraPosition.z, bounds.min.z), bounds.max.z)
            )
            minimumDepth = simd_distance(cameraPosition, nearestPoint)

            let minBounds = bounds.min
            let maxBounds = bounds.max
            let corners: [SIMD3<Float>] = [
                SIMD3<Float>(minBounds.x, minBounds.y, minBounds.z),
                SIMD3<Float>(minBounds.x, minBounds.y, maxBounds.z),
                SIMD3<Float>(minBounds.x, maxBounds.y, minBounds.z),
                SIMD3<Float>(minBounds.x, maxBounds.y, maxBounds.z),
                SIMD3<Float>(maxBounds.x, minBounds.y, minBounds.z),
                SIMD3<Float>(maxBounds.x, minBounds.y, maxBounds.z),
                SIMD3<Float>(maxBounds.x, maxBounds.y, minBounds.z),
                SIMD3<Float>(maxBounds.x, maxBounds.y, maxBounds.z)
            ]

            maximumDepth = corners.reduce(into: Float.zero) { currentMax, corner in
                currentMax = max(currentMax, simd_distance(cameraPosition, corner))
            }
        } else {
            let center = (bounds.min + bounds.max) * 0.5
            let extents = (bounds.max - bounds.min) * 0.5
            let offsetCenter = center - cameraPosition
            let projectedCenter = simd_dot(offsetCenter, cameraForward)
            let absoluteForward = SIMD3<Float>(abs(cameraForward.x), abs(cameraForward.y), abs(cameraForward.z))
            let projectedRadius = simd_dot(absoluteForward, extents)

            minimumDepth = projectedCenter - projectedRadius
            maximumDepth = projectedCenter + projectedRadius
        }

        let range = max(maximumDepth - minimumDepth, 0.001)
        let padding = max(range * 0.001, 0.001)
        let paddedMinimum = sortByDistance ? max(0, minimumDepth - padding) : minimumDepth - padding
        return (min: paddedMinimum, max: maximumDepth + padding)
    }
    
    // MARK: - Metal 4 TensorOps Batch Precompute

    /// Ensure precomputed splat buffer is allocated and large enough.
    internal func ensurePrecomputedSplatBuffer(requiredSize: Int) -> MTLBuffer? {
        var currentBuffer = precomputedSplatBuffer
        if currentBuffer == nil || currentBuffer!.length < requiredSize {
            currentBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)
            currentBuffer?.label = "Precomputed Splats"
            precomputedSplatBuffer = currentBuffer
        }
        return currentBuffer
    }

    internal func makeUniforms(for viewport: ViewportDescriptor,
                               splatCount: UInt32,
                               indexedSplatCount: UInt32,
                               debugFlags: UInt32) -> Uniforms {
        Self.makeUniforms(for: viewport,
                          splatCount: splatCount,
                          indexedSplatCount: indexedSplatCount,
                          debugFlags: debugFlags,
                          renderMode: renderMode,
                          covarianceBlur: covarianceBlur,
                          lodThresholds: lodThresholds,
                          selectionTintColor: selectionTintColor,
                          editingEnabled: editingEnabled)
    }

    static func makeUniforms(for viewport: ViewportDescriptor,
                             splatCount: UInt32,
                             indexedSplatCount: UInt32,
                             debugFlags: UInt32,
                             renderMode: SplatRenderMode,
                             covarianceBlur: Float,
                             lodThresholds: SIMD3<Float>,
                             selectionTintColor: SIMD4<Float> = SIMD4<Float>(0.15, 0.55, 1.0, 0.45),
                             editingEnabled: Bool = false) -> Uniforms {
        let proj00 = viewport.projectionMatrix[0][0]
        let proj11 = viewport.projectionMatrix[1][1]
        let focalX = Float(viewport.screenSize.x) * proj00 / 2
        let focalY = Float(viewport.screenSize.y) * proj11 / 2
        let tanHalfFovX = 1 / proj00
        let tanHalfFovY = 1 / proj11

        return Uniforms(
            projectionMatrix: viewport.projectionMatrix,
            viewMatrix: viewport.viewMatrix,
            screenSize: SIMD2(x: UInt32(viewport.screenSize.x), y: UInt32(viewport.screenSize.y)),
            focalX: focalX,
            focalY: focalY,
            tanHalfFovX: tanHalfFovX,
            tanHalfFovY: tanHalfFovY,
            splatCount: splatCount,
            indexedSplatCount: indexedSplatCount,
            debugFlags: debugFlags,
            renderMode: renderMode.rawValue,
            padding0: 0,
            padding1: 0,
            lodThresholds: lodThresholds,
            covarianceBlur: covarianceBlur,
            selectionTintColor: selectionTintColor,
            editingEnabled: editingEnabled ? 1 : 0,
            padding2: 0,
            padding3: 0,
            padding4: 0
        )
    }

    static func renderMode(from mode: AutodetectSceneReader.RenderMode) -> SplatRenderMode {
        switch mode {
        case .standard:
            return .standard
        case .mip:
            return .mip
        }
    }
    
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
        guard let precomputedBuffer = ensurePrecomputedSplatBuffer(requiredSize: requiredSize) else { return }
        
        var uniforms = makeUniforms(for: viewport,
                                    splatCount: UInt32(splatCount),
                                    indexedSplatCount: UInt32(min(splatCount, Constants.maxIndexedSplatCount)),
                                    debugFlags: 0)

        var splatCountValue = UInt32(splatCount)
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            Self.log.error("Failed to create compute encoder for batch precompute")
            return
        }
        computeEncoder.label = "Batch Precompute Splats"
        computeEncoder.setComputePipelineState(precomputePipeline)
        
        computeEncoder.setBuffer(activeSplatBufferForRendering.buffer, offset: 0, index: 0)
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
        // Use local capture to avoid TOCTOU race (buffer could be nil'd between check and use)
        let requiredSize = splatCount * MemoryLayout<UInt32>.stride
        var currentIndicesBuffer = visibleIndicesBuffer
        if currentIndicesBuffer == nil || currentIndicesBuffer!.length < requiredSize {
            currentIndicesBuffer = device.makeBuffer(length: requiredSize, options: .storageModePrivate)
            currentIndicesBuffer?.label = "Visible Indices"
            visibleIndicesBuffer = currentIndicesBuffer
        }
        guard let indicesBuffer = currentIndicesBuffer else { return }
        
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
        cullEncoder.setBuffer(activeSplatBufferForRendering.buffer, offset: 0, index: 0)
        cullEncoder.setBuffer(indicesBuffer, offset: 0, index: 1)
        cullEncoder.setBuffer(countBuffer, offset: 0, index: 2)
        cullEncoder.setBuffer(cullDataBuffer, offset: 0, index: 3)
        if let editStateBuffer {
            cullEncoder.setBuffer(editStateBuffer, offset: 0, index: 4)
        }
        
        var count = UInt32(splatCount)
        cullEncoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 5)
        
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
        let totalCount = self.renderableSplatCountForCurrentEditState
        let changeThreshold = max(totalCount / 20, 100)  // At least 100 splats or 5%
        if abs(lastVisibleCount - previousCount) > changeThreshold {
            let percentage = totalCount > 0 ? Int(Float(visibleCount) / Float(totalCount) * 100) : 0
            Self.log.info("Frustum culling: \(visibleCount)/\(totalCount) visible (\(percentage)%)")
        }
    }
    
    /// Get the last frustum culling result
    public var culledSplatCount: Int {
        frustumCullingEnabled ? lastVisibleCount : renderableSplatCountForCurrentEditState
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
        qualityFrustumCullPositionEpsilon = frustumCullPositionEpsilon
        qualityFrustumCullDirectionEpsilon = frustumCullDirectionEpsilon
        qualityMinimumFrustumCullInterval = minimumFrustumCullInterval
        qualityHighQualityDepth = highQualityDepth

        // Apply relaxed interaction settings
        sortPositionEpsilon = interactionSortPositionEpsilon
        sortDirectionEpsilon = interactionSortDirectionEpsilon
        minimumSortInterval = interactionMinimumSortInterval
        frustumCullPositionEpsilon = interactionFrustumCullPositionEpsilon
        frustumCullDirectionEpsilon = interactionFrustumCullDirectionEpsilon
        minimumFrustumCullInterval = interactionMinimumFrustumCullInterval
        highQualityDepth = false

        Self.log.debug("Interaction mode started - sort thresholds relaxed, using single-stage depth path")
    }
    
    /// End interaction mode - restores quality sort parameters and triggers final sort
    /// Call this when user ends touch interaction
    public func endInteraction() {
        guard isInteracting else { return }
        
        isInteracting = false
        let endTime = CFAbsoluteTimeGetCurrent()
        interactionEndTime = endTime
        
        // Restore quality settings
        sortPositionEpsilon = qualitySortPositionEpsilon
        sortDirectionEpsilon = qualitySortDirectionEpsilon
        minimumSortInterval = qualityMinimumSortInterval
        frustumCullPositionEpsilon = qualityFrustumCullPositionEpsilon
        frustumCullDirectionEpsilon = qualityFrustumCullDirectionEpsilon
        minimumFrustumCullInterval = qualityMinimumFrustumCullInterval
        highQualityDepth = qualityHighQualityDepth
        
        // Schedule a final quality-threshold check after a brief delay. If an
        // interaction sort already matches the settled camera, avoid forcing work.
        if !useDitheredTransparency {
            DispatchQueue.main.asyncAfter(deadline: .now() + postInteractionSortDelay) { [weak self] in
                guard let self = self else { return }
                // Only the latest interaction end owns the delayed quality check.
                guard !self.isInteracting,
                      self.interactionEndTime == endTime else {
                    return
                }

                self.interactionEndTime = nil
                if self.shouldResortForCurrentCamera() {
                    Self.log.debug("Interaction mode ended - triggering final sort")
                    self.resort(useGPU: true)
                } else {
                    Self.log.debug("Interaction mode ended - final sort skipped; current order is fresh")
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

    private func shouldResortForCurrentCamera() -> Bool {
        // Skip sorting entirely when using dithered transparency
        // Dithered mode is order-independent, so sort order doesn't affect visual quality
        if useDitheredTransparency {
            return false
        }
        let now = CFAbsoluteTimeGetCurrent()
        if let interactionEndTime,
           !isInteracting,
           now - interactionEndTime < postInteractionSortDelay {
            return false
        }

        os_unfair_lock_lock(&sortStateLock)
        let sortDirty = sortDirtyDueToData
        let previousSortTime = lastSortTime
        let previousSortPosition = lastSortedCameraPosition
        let previousSortForward = lastSortedCameraForward
        os_unfair_lock_unlock(&sortStateLock)

        if sortDirty {
            return true
        }
        if effectiveMinimumSortInterval > 0 && (now - previousSortTime) < effectiveMinimumSortInterval {
            return false
        }
        return Self.shouldRunCameraDrivenUpdate(
            dirty: false,
            now: now,
            lastUpdateTime: previousSortTime,
            minimumInterval: effectiveMinimumSortInterval,
            currentPosition: sortCameraPosition,
            currentForward: sortCameraForward,
            lastPosition: previousSortPosition,
            lastForward: previousSortForward,
            positionEpsilon: sortPositionEpsilon,
            directionEpsilon: sortDirectionEpsilon
        )
    }

    private func shouldUpdateFrustumCullingForCurrentCamera() -> Bool {
        if Self.projectionMatrixChanged(
            currentFrustumCullProjectionMatrix,
            lastFrustumCullProjectionMatrix
        ) {
            return true
        }

        return Self.shouldRunCameraDrivenUpdate(
            dirty: frustumCullDirtyDueToData,
            now: CFAbsoluteTimeGetCurrent(),
            lastUpdateTime: lastFrustumCullTime,
            minimumInterval: minimumFrustumCullInterval,
            currentPosition: sortCameraPosition,
            currentForward: sortCameraForward,
            lastPosition: lastFrustumCullCameraPosition,
            lastForward: lastFrustumCullCameraForward,
            positionEpsilon: frustumCullPositionEpsilon,
            directionEpsilon: frustumCullDirectionEpsilon
        )
    }

    internal static func shouldRunCameraDrivenUpdate(
        dirty: Bool,
        now: CFAbsoluteTime,
        lastUpdateTime: CFAbsoluteTime,
        minimumInterval: TimeInterval,
        currentPosition: SIMD3<Float>,
        currentForward: SIMD3<Float>,
        lastPosition: SIMD3<Float>?,
        lastForward: SIMD3<Float>?,
        positionEpsilon: Float,
        directionEpsilon: Float
    ) -> Bool {
        if dirty {
            return true
        }
        if minimumInterval > 0 && (now - lastUpdateTime) < minimumInterval {
            return false
        }
        guard let lastPosition, let lastForward else {
            return true
        }

        let positionDelta = simd_distance(currentPosition, lastPosition)
        let forwardDelta = 1 - simd_dot(simd_normalize(currentForward), simd_normalize(lastForward))
        return positionDelta > positionEpsilon || forwardDelta > directionEpsilon
    }

    private static func projectionMatrixChanged(
        _ current: simd_float4x4?,
        _ previous: simd_float4x4?,
        epsilon: Float = 0.001
    ) -> Bool {
        guard let current else { return false }
        guard let previous else { return true }

        let totalDelta =
            simd_length(current.columns.0 - previous.columns.0) +
            simd_length(current.columns.1 - previous.columns.1) +
            simd_length(current.columns.2 - previous.columns.2) +
            simd_length(current.columns.3 - previous.columns.3)
        return totalDelta > epsilon
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
            let uniforms = makeUniforms(for: viewport,
                                        splatCount: splatCount,
                                        indexedSplatCount: indexedSplatCount,
                                        debugFlags: debugFlags)
            self.uniforms.pointee.setUniforms(index: i, uniforms)
        }
        // Use cached arrays to avoid per-frame allocations
        if cameraPositionsTemp.count != viewports.count {
            cameraPositionsTemp = Array(repeating: .zero, count: viewports.count)
            cameraForwardsTemp = Array(repeating: .zero, count: viewports.count)
        }
        for (i, viewport) in viewports.enumerated() {
            // Extract both position and forward from a single matrix inverse
            // (avoids computing the expensive 4x4 inverse twice per viewport)
            let invView = viewport.viewMatrix.inverse
            cameraPositionsTemp[i] = (invView * SIMD4<Float>(x: 0, y: 0, z: 0, w: 1)).xyz
            cameraForwardsTemp[i] = (invView * SIMD4<Float>(x: 0, y: 0, z: -1, w: 0)).xyz
        }
        cameraWorldPosition = cameraPositionsTemp.mean ?? .zero
        cameraWorldForward = cameraForwardsTemp.mean?.normalized ?? .init(x: 0, y: 0, z: -1)
        sortCameraPosition = cameraPositionsTemp.first ?? .zero
        sortCameraForward = cameraForwardsTemp.first?.normalized ?? .init(x: 0, y: 0, z: -1)
        currentSortViewMatrix = viewports.first?.viewMatrix
        currentFrustumCullProjectionMatrix = viewports.first?.projectionMatrix

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
        updateAnimatedSplatsIfNeeded(to: commandBuffer)
        schedulePendingBufferRelease(on: commandBuffer)

        let splatCount = splatBuffer.count
        guard splatCount != 0 else { return }
        let drawSplatCount = renderableSplatCountForCurrentEditState
        guard drawSplatCount > 0 else { return }
        let indexedSplatCount = min(drawSplatCount, Constants.maxIndexedSplatCount)
        let instanceCount = (drawSplatCount + indexedSplatCount - 1) / indexedSplatCount

        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            updateMetal4ResidencyForFrame(commandBuffer: commandBuffer)
        }

        switchToNextDynamicBuffer()
        updateUniforms(forViewports: viewports, splatCount: UInt32(drawSplatCount), indexedSplatCount: UInt32(indexedSplatCount))
        frameBufferUploads += 1 // uniforms update
        
        // GPU Frustum Culling: encode compute pass before rendering
        if frustumCullingEnabled,
           shouldUpdateFrustumCullingForCurrentCamera(),
           let firstViewport = viewports.first {
            lastFrustumCullCameraPosition = sortCameraPosition
            lastFrustumCullCameraForward = sortCameraForward
            lastFrustumCullProjectionMatrix = firstViewport.projectionMatrix
            lastFrustumCullTime = CFAbsoluteTimeGetCurrent()
            frustumCullDirtyDueToData = false
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
            if isMetal4OptimizationsAvailable && drawSplatCount > 5000 {
                // Only log if this is a new scene or first time
                if !metal4LoggedOnce || abs(drawSplatCount - lastSplatCountLogged) > 1000 {
                    Self.log.info("Metal 4.0: Enhanced pipeline active for \(drawSplatCount) splats")
                    metal4LoggedOnce = true
                    lastSplatCountLogged = drawSplatCount
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
            renderEncoder.setObjectBuffer(activeSplatBufferForRendering.buffer, offset: 0, index: BufferIndex.splat.rawValue)
            renderEncoder.setObjectBuffer(sortedIndices.buffer, offset: 0, index: BufferIndex.sortedIndices.rawValue)
            
            renderEncoder.setMeshBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
            renderEncoder.setMeshBuffer(activeSplatBufferForRendering.buffer, offset: 0, index: BufferIndex.splat.rawValue)
            renderEncoder.setMeshBuffer(sortedIndices.buffer, offset: 0, index: BufferIndex.sortedIndices.rawValue)

            // Bind precomputed buffer if TensorOps precompute is enabled and data is valid
            if batchPrecomputeEnabled, let precomputedBuffer = precomputedSplatBuffer, !precomputedDataDirty {
                renderEncoder.setMeshBuffer(precomputedBuffer, offset: 0, index: BufferIndex.precomputed.rawValue)
            }

            // Calculate number of meshlets needed
            // Each meshlet handles 64 splats (increased from 32, limited by Metal's 256 vertex max)
            let splatsPerMeshlet: Int = 64
            let meshletCount = (drawSplatCount + splatsPerMeshlet - 1) / splatsPerMeshlet

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
        defer {
            renderEncoder.endEncoding()
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
        renderEncoder.setVertexBuffer(activeSplatBufferForRendering.buffer, offset: 0, index: BufferIndex.splat.rawValue)
        if let editStateBuffer {
            renderEncoder.setVertexBuffer(editStateBuffer, offset: 0, index: BufferIndex.editState.rawValue)
        }
        if let editTransformIndexBuffer {
            renderEncoder.setVertexBuffer(editTransformIndexBuffer, offset: 0, index: BufferIndex.transformIndex.rawValue)
        }
        if let editTransformPaletteBuffer {
            renderEncoder.setVertexBuffer(editTransformPaletteBuffer, offset: 0, index: BufferIndex.transformPalette.rawValue)
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

        if !multiStage,
           shouldDrawSelectionOutline,
           let selectionOutlinePipelineState,
           let selectionOutlineDepthState {
            renderEncoder.pushDebugGroup("Draw Selection Outline")
            renderEncoder.setRenderPipelineState(selectionOutlinePipelineState)
            renderEncoder.setDepthStencilState(selectionOutlineDepthState)
            renderEncoder.drawIndexedPrimitives(type: .triangle,
                                                indexCount: indexCount,
                                                indexType: .uint32,
                                                indexBuffer: indexBuffer.buffer,
                                                indexBufferOffset: 0,
                                                instanceCount: instanceCount)
            renderEncoder.popDebugGroup()
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

        lastFrameTime = CFAbsoluteTimeGetCurrent() - frameStartTime
        frameCount += 1
        // Use exponential moving average so recent frames have meaningful weight.
        // A cumulative average (dividing by frameCount) becomes unresponsive after
        // thousands of frames, making adaptive sort frequency ineffective.
        let emaAlpha = 0.05  // ~20-frame smoothing window
        averageFrameTime += (lastFrameTime - averageFrameTime) * emaAlpha
        
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
            let sortFrameState = getSortFrameState()

            let stats = FrameStatistics(
                ready: sortFrameState.ready,
                loadingCount: sortFrameState.ready ? 0 : 1,
                sortDuration: sortFrameState.duration,
                bufferUploadCount: frameBufferUploads,
                splatCount: splatCount,
                frameTime: lastFrameTime,
                sortBufferPoolStats: sortBufferStats,
                sortJobsInFlight: sortFrameState.jobsInFlight
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

    /// Completes a sort by publishing the sorted-index buffer and scheduling state
    /// immediately, then dispatching external callbacks/logging to main.
    private func finishSort(
        indexOutputBuffer: MetalBuffer<Int32>,
        sortStartTime: CFAbsoluteTime,
        cameraWorldPosition: SIMD3<Float>,
        cameraWorldForward: SIMD3<Float>,
        sortViewMatrix: simd_float4x4?,
        dataDirtySnapshot: UInt64,
        performanceContext: SortPerformanceContext,
        completionHandlerTime: CFAbsoluteTime?,
        gpuTime: TimeInterval?,
        commandBufferStatus: String
    ) {
        let bufferReadyTime = completionHandlerTime ?? CFAbsoluteTimeGetCurrent()
        let elapsed = bufferReadyTime - sortStartTime
        let callbackWallTime = completionHandlerTime.map { max(0, $0 - sortStartTime) }

        // GPU-only sorting uses double buffering. Publish the newly sorted indices
        // under the same lock used by render reads, so the next frame can consume
        // them without waiting for a main-queue turn.
        if let oldBuffer = swapSortedIndicesBuffer(newBuffer: indexOutputBuffer) {
            // IMPORTANT: Defer release until a later render command buffer completes.
            // The old buffer may still be referenced by in-flight GPU work.
            deferredBufferRelease(oldBuffer)
        }

        let inFlightSortsAtCompletion = finishSortState(
            duration: elapsed,
            bufferReadyTime: bufferReadyTime,
            cameraWorldPosition: cameraWorldPosition,
            cameraWorldForward: cameraWorldForward,
            sortViewMatrix: sortViewMatrix,
            dataDirtySnapshot: dataDirtySnapshot
        )

        // Keep external observers and logging on main. Scheduling metadata above is
        // already published so frame-aligned main-queue delays do not keep sorting gated.
        DispatchQueue.main.async { [weak self] in
            guard let self = self else {
                return
            }

            let mainApplyTime = CFAbsoluteTimeGetCurrent()
            let mainQueueDelay = max(0, mainApplyTime - bufferReadyTime)
            self.onSortComplete?(elapsed)
            let sample = performanceContext.makeSample(
                wallTime: elapsed,
                callbackWallTime: callbackWallTime,
                gpuTime: gpuTime,
                mainQueueDelay: mainQueueDelay,
                inFlightSortsAtCompletion: inFlightSortsAtCompletion,
                status: commandBufferStatus
            )

            Self.log.debug("\(sample.logMessage, privacy: .public)")
        }
    }

    private static func gpuDuration(for commandBuffer: MTLCommandBuffer) -> TimeInterval? {
        let duration = commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        return duration > 0 ? duration : nil
    }

    private static func combinedGPUTime(_ times: TimeInterval?...) -> TimeInterval? {
        let validTimes = times.compactMap { $0 }
        guard !validTimes.isEmpty else { return nil }
        return validTimes.reduce(0, +)
    }

    private static func statusDescription(for commandBuffer: MTLCommandBuffer) -> String {
        switch commandBuffer.status {
        case .notEnqueued:
            return "notEnqueued"
        case .enqueued:
            return "enqueued"
        case .committed:
            return "committed"
        case .scheduled:
            return "scheduled"
        case .completed:
            return "completed"
        case .error:
            return "error"
        @unknown default:
            return "unknown"
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
        let renderableCount = renderableSplatCountForCurrentEditState
        let dataDirtySnapshot = getSortDataRevision()

        guard renderableCount > 0 else {
            finishSort()
            return
        }

        let cameraWorldForward = sortCameraForward
        let cameraWorldPosition = sortCameraPosition
        let sortViewMatrix = currentSortViewMatrix
        let sortStartTime = CFAbsoluteTimeGetCurrent()

        // Compute effective sort mode based on camera motion (auto mode tracks rotation vs translation)
        let effectiveSortByDistance = computeEffectiveSortByDistance(
            cameraPosition: cameraWorldPosition,
            cameraForward: cameraWorldForward
        )
        let sortJobsInFlightAtStart = getSortJobsInFlight()
        let interactionModeAtStart = isInteracting
        
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
                    indexOutputBuffer.count = max(renderableCount, 0)
                } catch {
                    Self.log.error("Failed to acquire index output buffer from pool: \(error)")
                    self.finishSort()
                    return
                }

                // === METAL 4 RADIX SORT PATH (for very large scenes) ===
                // Uses GPU atomics-based radix sort, beneficial for >100K splats
                if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
                    if self.useMetal4Sorting,
                       self.hiddenOrDeletedEditStateCount == 0,
                       splatCount > self.metal4SortingThreshold,
                       let sorter = self.metal4Sorter {
                        let performanceContext = SortPerformanceContext(
                            path: .metal4,
                            splatCount: splatCount,
                            renderableCount: renderableCount,
                            inFlightSortsAtStart: sortJobsInFlightAtStart,
                            interactionMode: interactionModeAtStart,
                            sortByDistance: effectiveSortByDistance
                        )
                        let sortCommandBufferManager = self.computeCommandBufferManager ?? commandBufferManager
                        guard let commandBuffer = sortCommandBufferManager.makeCommandBuffer() else {
                            Self.log.error("Failed to create compute command buffer for Metal 4 sort.")
                            sortIndexBufferPool.release(indexOutputBuffer)
                            self.finishSort()
                            return
                        }

                        do {
                            try sorter.sort(
                                splats: activeSplatBufferForRendering.buffer,
                                count: splatCount,
                                cameraPosition: cameraWorldPosition,
                                cameraForward: cameraWorldForward,
                                sortByDistance: effectiveSortByDistance,
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
                        commandBuffer.addCompletedHandler { [weak self, sortIndexBufferPool] buffer in
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
                                sortViewMatrix: sortViewMatrix,
                                dataDirtySnapshot: dataDirtySnapshot,
                                performanceContext: performanceContext,
                                completionHandlerTime: CFAbsoluteTimeGetCurrent(),
                                gpuTime: Self.gpuDuration(for: buffer),
                                commandBufferStatus: Self.statusDescription(for: buffer)
                            )
                        }
                        commandBuffer.commit()
                        return
                    }
                }

                // === O(n) COUNTING SORT PATH ===
                // Uses histogram-based sorting which is faster than O(n log n) radix sort
                if self.useCountingSort, let sorter = self.countingSorter {
                    let performanceContext = SortPerformanceContext(
                        path: .counting,
                        splatCount: splatCount,
                        renderableCount: renderableCount,
                        inFlightSortsAtStart: sortJobsInFlightAtStart,
                        interactionMode: interactionModeAtStart,
                        sortByDistance: effectiveSortByDistance
                    )
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
                    let depthBounds = (self.getBounds() ?? self.getBoundsBlocking()).map {
                        Self.estimateCountingSortDepthBounds(
                            from: $0,
                            cameraPosition: cameraWorldPosition,
                            cameraForward: cameraWorldForward,
                            sortByDistance: effectiveSortByDistance
                        )
                    }

                    do {
                            try sorter.sort(
                                commandBuffer: commandBuffer,
                                splatBuffer: activeSplatBufferForRendering.buffer,
                                editStateBuffer: self.editStateBuffer,
                                outputBuffer: indexOutputBuffer.buffer,
                                cameraPosition: cameraWorldPosition,
                                cameraForward: cameraWorldForward,
                            sortByDistance: effectiveSortByDistance,
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
                    commandBuffer.addCompletedHandler { [weak self, sortIndexBufferPool] buffer in
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
                            sortViewMatrix: sortViewMatrix,
                            dataDirtySnapshot: dataDirtySnapshot,
                            performanceContext: performanceContext,
                            completionHandlerTime: CFAbsoluteTimeGetCurrent(),
                            gpuTime: Self.gpuDuration(for: buffer),
                            commandBufferStatus: Self.statusDescription(for: buffer)
                        )
                    }
                    commandBuffer.commit()
                    return  // Exit Task - completion handler will finish sort

                } else {
                    // === LEGACY MPS ARGSORT PATH ===
                    // Falls back to O(n log n) MPS-based radix sort
                    let performanceContext = SortPerformanceContext(
                        path: .mps,
                        splatCount: splatCount,
                        renderableCount: renderableCount,
                        inFlightSortsAtStart: sortJobsInFlightAtStart,
                        interactionMode: interactionModeAtStart,
                        sortByDistance: effectiveSortByDistance
                    )

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
                    var sortByDist = effectiveSortByDistance
                    var count = UInt32(splatCount)

                    computeEncoder.setComputePipelineState(computePipelineState)
                    computeEncoder.setBuffer(activeSplatBufferForRendering.buffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(distanceBuffer.buffer, offset: 0, index: 1)
                    if let editStateBuffer = self.editStateBuffer {
                        computeEncoder.setBuffer(editStateBuffer, offset: 0, index: 2)
                    }
                    computeEncoder.setBytes(&cameraPos, length: MemoryLayout<SIMD3<Float>>.size, index: 3)
                    computeEncoder.setBytes(&cameraFwd, length: MemoryLayout<SIMD3<Float>>.size, index: 4)
                    computeEncoder.setBytes(&sortByDist, length: MemoryLayout<Bool>.size, index: 5)
                    computeEncoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 6)

                    let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
                    let threadgroups = MTLSize(width: (splatCount + 255) / 256, height: 1, depth: 1)

                    computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
                    computeEncoder.endEncoding()

                    // === ASYNC COMPUTE OVERLAP ===
                    // Use separate compute queue so sorting doesn't block rendering
                    let sortQueue = self.computeCommandBufferManager?.queue ?? commandBufferManager.queue

                    // Use completion handler to chain distance computation -> argsort -> finish
                    commandBuffer.addCompletedHandler { [weak self] distanceCommandBuffer in
                        guard let self = self else {
                            return
                        }
                        let distanceGPUTime = Self.gpuDuration(for: distanceCommandBuffer)
                        let distanceStatus = Self.statusDescription(for: distanceCommandBuffer)

                        guard let argSortCommandBuffer = sortQueue.makeCommandBuffer() else {
                            Self.log.error("Failed to create MPS arg sort command buffer.")
                            self.sortDistanceBufferPool.release(distanceBuffer)
                            self.sortIndexBufferPool.release(indexOutputBuffer)
                            self.finishSort()
                            return
                        }
                        argSortCommandBuffer.addCompletedHandler { [weak self] buffer in
                            guard let self = self else { return }

                            self.sortDistanceBufferPool.release(distanceBuffer)

                            if buffer.status != .completed {
                                Self.log.error("MPSArgSort command buffer failed: \(String(describing: buffer.error))")
                                self.sortIndexBufferPool.release(indexOutputBuffer)
                                self.finishSort()
                                return
                            }

                            self.finishSort(
                                indexOutputBuffer: indexOutputBuffer,
                                sortStartTime: sortStartTime,
                                cameraWorldPosition: cameraWorldPosition,
                                cameraWorldForward: cameraWorldForward,
                                sortViewMatrix: sortViewMatrix,
                                dataDirtySnapshot: dataDirtySnapshot,
                                performanceContext: performanceContext,
                                completionHandlerTime: CFAbsoluteTimeGetCurrent(),
                                gpuTime: Self.combinedGPUTime(distanceGPUTime, Self.gpuDuration(for: buffer)),
                                commandBufferStatus: "distance:\(distanceStatus),argsort:\(Self.statusDescription(for: buffer))"
                            )
                        }
                        self.cachedMPSArgSort.encode(
                            commandBuffer: argSortCommandBuffer,
                            input: distanceBuffer.buffer,
                            output: indexOutputBuffer.buffer,
                            count: splatCount
                        )
                        argSortCommandBuffer.commit()
                    }
                    commandBuffer.commit()
                    return  // Exit Task - completion handler will finish sort
                }
            }
        } else {
            Task(priority: .high) {
                let performanceContext = SortPerformanceContext(
                    path: .cpu,
                    splatCount: splatCount,
                    renderableCount: renderableCount,
                    inFlightSortsAtStart: sortJobsInFlightAtStart,
                    interactionMode: interactionModeAtStart,
                    sortByDistance: effectiveSortByDistance
                )
                var actualCount = 0

                // Copy positions under lock to ensure pointer validity during sort
                // This avoids holding the lock during the slow sort operation
                activeSplatBufferForRendering.withLockedValues { values, count in
                    actualCount = count
                    if orderAndDepthTempSort.count != actualCount {
                        orderAndDepthTempSort = Array(
                            repeating: SplatIndexAndDepth(index: .max, depth: 0),
                            count: actualCount
                        )
                    }
                    guard actualCount > 0 else { return }
                    if effectiveSortByDistance {
                        for i in 0..<actualCount {
                            orderAndDepthTempSort[i].index = UInt32(i)
                            let splatPos = values[i].position.simd
                            orderAndDepthTempSort[i].depth = (splatPos - cameraWorldPosition).lengthSquared
                        }
                    } else {
                        for i in 0..<actualCount {
                            orderAndDepthTempSort[i].index = UInt32(i)
                            let splatPos = values[i].position.simd
                            orderAndDepthTempSort[i].depth = dot(splatPos - cameraWorldPosition, cameraWorldForward)
                        }
                    }
                }

                orderAndDepthTempSort.sort { $0.depth > $1.depth }

                // CPU fallback: populate sortedIndicesBuffer instead of reordering splats
                // This maintains consistency with GPU path - splat data stays static
                do {
                    // Acquire new buffer and fill with sorted indices
                    let cpuSortedIndices = try sortIndexBufferPool.acquire(minimumCapacity: max(actualCount, 1))
                    cpuSortedIndices.count = actualCount
                    for newIndex in 0..<actualCount {
                        cpuSortedIndices.values[newIndex] = Int32(orderAndDepthTempSort[newIndex].index)
                    }

                    // finishSort publishes scheduling state under sortStateLock, matching
                    // the GPU completion path while keeping external callbacks on main.
                    self.finishSort(
                        indexOutputBuffer: cpuSortedIndices,
                        sortStartTime: sortStartTime,
                        cameraWorldPosition: cameraWorldPosition,
                        cameraWorldForward: cameraWorldForward,
                        sortViewMatrix: sortViewMatrix,
                        dataDirtySnapshot: dataDirtySnapshot,
                        performanceContext: performanceContext,
                        completionHandlerTime: nil,
                        gpuTime: nil,
                        commandBufferStatus: "completed"
                    )
                } catch {
                    Self.log.error("Failed to create sorted indices buffer: \(error)")
                    self.finishSort()
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
                  packedColor: SplatRenderer.packRGBA8(color.x, color.y, color.z, color.w),
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
