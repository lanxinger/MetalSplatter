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
        }
    }
}

public class SplatRenderer {
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
    
    private var computeDepthsPipelineState: MTLComputePipelineState?
    private var computeDistancesPipelineState: MTLComputePipelineState?
    private var frustumCullPipelineState: MTLComputePipelineState?
    
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

    // Keep in sync with Shaders.metal : BufferIndex
    enum BufferIndex: NSInteger {
        case uniforms = 0
        case splat    = 1
    }

    // Keep in sync with Shaders.metal : Uniforms
    struct Uniforms {
        var projectionMatrix: matrix_float4x4
        var viewMatrix: matrix_float4x4
        var screenSize: SIMD2<UInt32> // Size of screen in pixels

        var splatCount: UInt32
        var indexedSplatCount: UInt32
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
    
    // Performance tracking
    private var frameStartTime: CFAbsoluteTime = 0
    private var lastFrameTime: TimeInterval = 0
    public var averageFrameTime: TimeInterval = 0
    private var frameCount: Int = 0
    private var metal4LoggedOnce: Bool = false
    private var lastSplatCountLogged: Int = 0

    internal let library: MTLLibrary
    // Single-stage pipeline
    internal var singleStagePipelineState: MTLRenderPipelineState?
    internal var singleStageDepthState: MTLDepthStencilState?
    // Multi-stage pipeline
    private var initializePipelineState: MTLRenderPipelineState?
    internal var drawSplatPipelineState: MTLRenderPipelineState?
    internal var drawSplatDepthState: MTLDepthStencilState?
    private var postprocessPipelineState: MTLRenderPipelineState?
    private var postprocessDepthState: MTLDepthStencilState?

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

    typealias IndexType = UInt32
    
    // Buffer pools for efficient memory management
    private let splatBufferPool: MetalBufferPool<Splat>
    private let indexBufferPool: MetalBufferPool<UInt32>
    
    // splatBuffer contains one entry for each gaussian splat
    var splatBuffer: MetalBuffer<Splat>
    // splatBufferPrime is a copy of splatBuffer, which is not currenly in use for rendering.
    // We use this for sorting, and when we're done, swap it with splatBuffer.
    // Multiple buffers from the pool ensure we're never actively sorting a buffer still in use for rendering
    var splatBufferPrime: MetalBuffer<Splat>

    var indexBuffer: MetalBuffer<UInt32>

    public var splatCount: Int { splatBuffer.count }

    var sorting = false
    var orderAndDepthTempSort: [SplatIndexAndDepth] = []
    
    // Metal 4 command buffer pool for improved performance
    private var commandBufferManager: CommandBufferManager

    public init(device: MTLDevice,
                colorFormat: MTLPixelFormat,
                depthFormat: MTLPixelFormat,
                sampleCount: Int,
                maxViewCount: Int,
                maxSimultaneousRenders: Int) throws {
#if arch(x86_64)
        fatalError("MetalSplatter is unsupported on Intel architecture (x86_64)")
#endif

        self.device = device

        // Initialize command buffer manager with Metal 4 pooling support
        guard let commandQueue = device.makeCommandQueue() else {
            throw SplatRendererError.metalDeviceUnavailable
        }
        commandQueue.label = "SplatRenderer Command Queue"
        self.commandBufferManager = CommandBufferManager(commandQueue: commandQueue)

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
        
        // Initialize frustum culling pipeline
        do {
            if let frustumFunction = library.makeFunction(name: "frustumCullSplats") {
                frustumCullPipelineState = try device.makeComputePipelineState(function: frustumFunction)
            }
        } catch {
            Self.log.error("Failed to create frustum culling pipeline state: \(error)")
        }
        
        // Setup Metal 4.0 optimizations if available
        setupMetal4Integration()
    }
    
    deinit {
        // Return buffers to pools for reuse
        splatBufferPool.release(splatBuffer)
        splatBufferPool.release(splatBufferPrime)
        indexBufferPool.release(indexBuffer)
    }

    public func reset() {
        // Clear current buffers and return them to pools
        splatBufferPool.release(splatBuffer)
        splatBufferPool.release(splatBufferPrime)
        
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
    }
    
    /// Ensures splatBufferPrime has sufficient capacity, acquiring a new buffer from pool if needed
    private func ensurePrimeBufferCapacity(_ minimumCapacity: Int) throws {
        if splatBufferPrime.capacity < minimumCapacity {
            // Return current prime buffer to pool and acquire a larger one
            splatBufferPool.release(splatBufferPrime)
            splatBufferPrime = try splatBufferPool.acquire(minimumCapacity: minimumCapacity)
        }
        splatBufferPrime.count = 0
    }

    public func read(from url: URL) async throws {
        var newPoints = SplatMemoryBuffer()
        try await newPoints.read(from: try AutodetectSceneReader(url))
        try add(newPoints.points)
    }

    private func resetPipelineStates() {
        singleStagePipelineState = nil
        initializePipelineState = nil
        drawSplatPipelineState = nil
        drawSplatDepthState = nil
        postprocessPipelineState = nil
        postprocessDepthState = nil
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
        pipelineDescriptor.vertexFunction = try library.makeRequiredFunction(name: "singleStageSplatVertexShader")
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
        pipelineDescriptor.vertexFunction = try library.makeRequiredFunction(name: "multiStageSplatVertexShader")
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

        splatBuffer.append(points.map { Splat($0) })
    }

    public func add(_ point: SplatScenePoint) throws {
        // Validate single point
        try SplatDataValidator.validatePoint(point)
        try add([ point ])
    }
    
    public func calculateBounds() -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
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

    internal func updateUniforms(forViewports viewports: [ViewportDescriptor],
                                splatCount: UInt32,
                                indexedSplatCount: UInt32) {
        for (i, viewport) in viewports.enumerated() where i <= maxViewCount {
            let uniforms = Uniforms(projectionMatrix: viewport.projectionMatrix,
                                    viewMatrix: viewport.viewMatrix,
                                    screenSize: SIMD2(x: UInt32(viewport.screenSize.x), y: UInt32(viewport.screenSize.y)),
                                    splatCount: splatCount,
                                    indexedSplatCount: indexedSplatCount)
            self.uniforms.pointee.setUniforms(index: i, uniforms)
        }

        cameraWorldPosition = viewports.map { Self.cameraWorldPosition(forViewMatrix: $0.viewMatrix) }.mean ?? .zero
        cameraWorldForward = viewports.map { Self.cameraWorldForward(forViewMatrix: $0.viewMatrix) }.mean?.normalized ?? .init(x: 0, y: 0, z: -1)

        if !sorting {
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
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       for commandBuffer: MTLCommandBuffer) -> MTLRenderCommandEncoder {
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = colorLoadAction
        renderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        renderPassDescriptor.colorAttachments[0].clearColor = clearColor
        if let depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
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
            fatalError("Failed to create render encoder")
        }

        renderEncoder.label = "Primary Render Encoder"

        renderEncoder.setViewports(viewports.map(\.viewport))

        if viewports.count > 1 {
            var viewMappings = (0..<viewports.count).map {
                MTLVertexAmplificationViewMapping(viewportArrayIndexOffset: UInt32($0),
                                                  renderTargetArrayIndexOffset: UInt32($0))
            }
            renderEncoder.setVertexAmplificationCount(viewports.count, viewMappings: &viewMappings)
        }

        return renderEncoder
    }

    public func render(viewports: [ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorLoadAction: MTLLoadAction = .clear,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        let splatCount = splatBuffer.count
        guard splatBuffer.count != 0 else { return }
        let indexedSplatCount = min(splatCount, Constants.maxIndexedSplatCount)
        let instanceCount = (splatCount + indexedSplatCount - 1) / indexedSplatCount

        switchToNextDynamicBuffer()
        updateUniforms(forViewports: viewports, splatCount: UInt32(splatCount), indexedSplatCount: UInt32(indexedSplatCount))

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

        let multiStage = useMultiStagePipeline
        if multiStage {
            try buildMultiStagePipelineStatesIfNeeded()
        } else {
            try buildSingleStagePipelineStatesIfNeeded()
        }

        let renderEncoder = renderEncoder(multiStage: multiStage,
                                          viewports: viewports,
                                          colorTexture: colorTexture,
                                          colorLoadAction: colorLoadAction,
                                          colorStoreAction: colorStoreAction,
                                          depthTexture: depthTexture,
                                          rasterizationRateMap: rasterizationRateMap,
                                          renderTargetArrayLength: renderTargetArrayLength,
                                          for: commandBuffer)

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
        } else {
            guard let singleStagePipelineState
            else { return }

            renderEncoder.pushDebugGroup("Draw Splats")
            renderEncoder.setRenderPipelineState(singleStagePipelineState)
            renderEncoder.setDepthStencilState(singleStageDepthState)
        }

        renderEncoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        renderEncoder.setVertexBuffer(splatBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)

        renderEncoder.drawIndexedPrimitives(type: .triangle,
                                            indexCount: indexCount,
                                            indexType: .uint32,
                                            indexBuffer: indexBuffer.buffer,
                                            indexBufferOffset: 0,
                                            instanceCount: instanceCount)

        if multiStage {
            guard let postprocessPipelineState
            else { return }

            renderEncoder.popDebugGroup()

            renderEncoder.pushDebugGroup("Postprocess")
            renderEncoder.setRenderPipelineState(postprocessPipelineState)
            renderEncoder.setDepthStencilState(postprocessDepthState)
            renderEncoder.setCullMode(.none)
            renderEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renderEncoder.popDebugGroup()
        } else {
            renderEncoder.popDebugGroup()
        }

        renderEncoder.endEncoding()
    }

    // Sort splatBuffer (read-only), storing the results in splatBuffer (write-only) then swap splatBuffer and splatBufferPrime
    public func resort(useGPU: Bool = true) {
        guard !sorting else { return }
        sorting = true
        onSortStart?()

        let splatCount = splatBuffer.count
        
        let cameraWorldForward = cameraWorldForward
        let cameraWorldPosition = cameraWorldPosition
        
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
//                let startTime = Date()

                // Allocate a GPU buffer for storing distances.
                guard let distanceBuffer = device.makeBuffer(
                    length: MemoryLayout<Float>.size * splatCount,
                    options: .storageModeShared
                ) else {
                    Self.log.error("Failed to create distance buffer.")
                    self.sorting = false
                    return
                }

                // Create command buffer for distance computation using pooled manager
                guard let commandBuffer = commandBufferManager.makeCommandBuffer(),
                      let computeEncoder = commandBuffer.makeComputeCommandEncoder(),
                      let computePipelineState = computeDistancesPipelineState else {
                    Self.log.error("Failed to create compute command buffer or encoder.")
                    self.sorting = false
                    return
                }
                
                // Set up compute shader parameters
                var cameraPos = cameraWorldPosition
                var cameraFwd = cameraWorldForward
                var sortByDist = Constants.sortByDistance
                var count = UInt32(splatCount)
                
                computeEncoder.setComputePipelineState(computePipelineState)
                computeEncoder.setBuffer(splatBuffer.buffer, offset: 0, index: 0)
                computeEncoder.setBuffer(distanceBuffer, offset: 0, index: 1)
                computeEncoder.setBytes(&cameraPos, length: MemoryLayout<SIMD3<Float>>.size, index: 2)
                computeEncoder.setBytes(&cameraFwd, length: MemoryLayout<SIMD3<Float>>.size, index: 3)
                computeEncoder.setBytes(&sortByDist, length: MemoryLayout<Bool>.size, index: 4)
                computeEncoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 5)
                
                let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
                let threadgroups = MTLSize(width: (splatCount + 255) / 256, height: 1, depth: 1)
                
                computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
                computeEncoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()

                // Allocate a GPU buffer for the ArgSort output indices
                guard let indexOutputBuffer = device.makeBuffer(
                    length: MemoryLayout<Int32>.size * splatCount,
                    options: .storageModeShared
                ) else {
                    Self.log.error("Failed to create output indices buffer.")
                    self.sorting = false
                    return
                }

                // Run argsort, in decending order.
                let argSort = MPSArgSort(dataType: .float32, descending: true)
                argSort(commandQueue: commandBufferManager.queue,
                        input: distanceBuffer,
                        output: indexOutputBuffer,
                        count: splatCount)

                // Read back the sorted indices and reorder splats on the CPU.
                let sortedIndices = indexOutputBuffer.contents().bindMemory(to: Int32.self, capacity: splatCount)

                do {
                    try self.ensurePrimeBufferCapacity(splatCount)
                    for newIndex in 0 ..< splatCount {
                        let oldIndex = Int(sortedIndices[newIndex])
                        splatBufferPrime.append(splatBuffer, fromIndex: oldIndex)
                    }
                    self.swapSplatBuffers()
                } catch {
                    Self.log.error("Failed to reorder splats with pooled buffers: \(error)")
                }

//                let elapsed = Date().timeIntervalSince(startTime)
//                Self.log.info("Sort time (GPU): \(elapsed) seconds")
//                self.onSortComplete?(elapsed)
                self.sorting = false
            }
        } else {
            Task(priority: .high) {
//                let cpuStart = Date()
                if orderAndDepthTempSort.count != splatCount {
                    orderAndDepthTempSort = Array(
                        repeating: SplatIndexAndDepth(index: .max, depth: 0),
                        count: splatCount
                    )
                }

                if Constants.sortByDistance {
                    for i in 0 ..< splatCount {
                        orderAndDepthTempSort[i].index = UInt32(i)
                        let splatPos = splatBuffer.values[i].position.simd
                        orderAndDepthTempSort[i].depth = (splatPos - cameraWorldPosition).lengthSquared
                    }
                } else {
                    for i in 0 ..< splatCount {
                        orderAndDepthTempSort[i].index = UInt32(i)
                        let splatPos = splatBuffer.values[i].position.simd
                        orderAndDepthTempSort[i].depth = dot(splatPos, cameraWorldForward)
                    }
                }

                orderAndDepthTempSort.sort { $0.depth > $1.depth }

                do {
                    try ensurePrimeBufferCapacity(splatCount)
                    for newIndex in 0..<orderAndDepthTempSort.count {
                        let oldIndex = Int(orderAndDepthTempSort[newIndex].index)
                        splatBufferPrime.append(splatBuffer, fromIndex: oldIndex)
                    }

                    swapSplatBuffers()
                } catch {
                    Self.log.error("Failed to reorder splats with pooled buffers: \(error)")
                }

//                let elapsedCPU = -cpuStart.timeIntervalSinceNow
//                Self.log.info("Sort time (CPU): \(elapsedCPU) seconds")
//                onSortComplete?(elapsedCPU)
                self.sorting = false
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
