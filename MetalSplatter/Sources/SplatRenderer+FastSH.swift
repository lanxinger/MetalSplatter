import Foundation
import Metal
import MetalKit
import SplatIO
import simd

// Use typealias to reference the SampleApp version when available
#if canImport(SampleApp)
import SampleApp
#else
// Fallback definition for when SampleApp is not available
public struct ModelRendererViewportDescriptor {
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
#endif

// MARK: - Fast SH Support for SplatRenderer

extension SplatRenderer {
    
    /// Extended Splat structure that includes SH information
    struct SplatSH {
        var position: MTLPackedFloat3
        var baseColor: UInt32              // Base color (DC term) + opacity, RGBA8 unorm
        var tintColor: SIMD4<Float>        // Animation tint/alpha multiplier in linear space
        var rotation: simd_float4          // Quaternion (x,y,z,w)
        var covA: PackedHalf3
        var covB: PackedHalf3
        var shPaletteIndex: UInt32         // Index into SH palette (for SOGS)
        var shDegree: UInt16               // SH degree (0-3)
        var padding: UInt16 = 0            // Alignment padding

        /// Convert from regular Splat + SH info
        init(splat: Splat, rotation: simd_quatf, shIndex: UInt32 = 0, shDegree: UInt16 = 0) {
            self.position = splat.position
            self.baseColor = splat.packedColor
            self.tintColor = SIMD4<Float>(1, 1, 1, 1)
            self.rotation = rotation.vector
            self.covA = splat.covA
            self.covB = splat.covB
            self.shPaletteIndex = shIndex
            self.shDegree = shDegree
        }
    }
}

// MARK: - Fast SH Renderer Extension

public class FastSHSplatRenderer: SplatRenderer, @unchecked Sendable {
    
    /// Configuration for fast SH rendering
    public struct FastSHConfiguration {
        /// Enable fast SH evaluation (vs per-splat evaluation)
        public var enabled: Bool = true

        /// Maximum number of unique SH coefficient sets (palette size)
        public var maxPaletteSize: Int = 65536

        /// Update SH evaluation every N frames (1 = every frame)
        /// Note: This is superseded by shDirectionEpsilon threshold-based updates.
        @available(*, deprecated, message: "Use shDirectionEpsilon for threshold-based updates instead")
        public var updateFrequency: Int = 1

        public init() {}
    }

    // Fast SH specific properties
    public var fastSHConfig = FastSHConfiguration() {
        didSet {
            if fastSHConfig.enabled != oldValue.enabled { invalidateRender() }
        }
    }

    /// When false, SH evaluation is completely disabled and only base color is used.
    /// This can provide significant performance gains at the cost of view-dependent lighting.
    public var shRenderingEnabled: Bool = true {
        didSet {
            if shRenderingEnabled != oldValue { invalidateRender() }
        }
    }
    private var shPaletteBuffer: MTLBuffer?
    private var fastSHPipelineState: MTLRenderPipelineState?
    private var fastSHEditingPipelineState: MTLRenderPipelineState?
    
    // SH data storage
    public private(set) var shCoefficients: [[SIMD3<Float>]] = []
    private var shPaletteMap: [Int: UInt32] = [:] // Maps from splat index to palette index
    public private(set) var shDegree: Int = 0
    private var shCoefficientsPerEntry: Int = 0
    
    // Extended splat buffer for SH
    private var splatSHBuffer: MetalBuffer<SplatSH>
    private var splatSHBufferPrime: MetalBuffer<SplatSH>
    private var animatedSplatSHBuffer: MetalBuffer<SplatSH>?
    private var lastAppliedAnimationTimeSH: Float?
    // nil also means a load/edit rebuild failed and must be retried before drawing.
    private var lastFastSHAppliedColorRevision: UInt64?
    private let splatSHBufferPool: MetalBufferPool<SplatSH>

    private struct FastSHShaderParameters {
        var coeffsPerEntry: UInt32 = 0
        var paletteSize: UInt32 = 0
        var degree: UInt32 = 0
        var skipSHEvaluation: UInt32 = 0  // When non-zero, skip SH evaluation and use base color
    }

    private var shaderParameters = FastSHShaderParameters()

    static func rendererSupportedSHCoefficients(_ coefficients: [SIMD3<Float>], degree: Int) -> [SIMD3<Float>] {
        let supportedCount = SphericalHarmonicsEvaluator.coefficientCountForDegree(degree)
        var result = Array(coefficients.prefix(supportedCount))
        if result.count < supportedCount {
            result.append(contentsOf: repeatElement(.zero, count: supportedCount - result.count))
        }
        return result
    }

    internal override func resetPipelineStates() {
        super.resetPipelineStates()
        fastSHPipelineState = nil
        fastSHEditingPipelineState = nil
    }

    public override func prepareForSorting(count: Int) throws {
        try super.prepareForSorting(count: count)
        if splatSHBufferPrime.capacity < count {
            splatSHBufferPool.release(splatSHBufferPrime)
            splatSHBufferPrime = try splatSHBufferPool.acquire(minimumCapacity: count)
        }
        splatSHBufferPrime.count = 0
    }

    public override func appendSplatForSorting(from oldIndex: Int) {
        super.appendSplatForSorting(from: oldIndex)
        if oldIndex < splatSHBuffer.count {
            splatSHBufferPrime.append(splatSHBuffer, fromIndex: oldIndex)
        }
    }

    public override func didSwapSplatBuffers() {
        swap(&splatSHBuffer, &splatSHBufferPrime)
    }

    internal override func renderSpecialized(viewports: [ViewportDescriptor],
                                             colorTexture: MTLTexture,
                                             colorLoadAction: MTLLoadAction,
                                             colorStoreAction: MTLStoreAction,
                                             depthTexture: MTLTexture?,
                                             depthStoreAction: MTLStoreAction,
                                             rasterizationRateMap: MTLRasterizationRateMap?,
                                             renderTargetArrayLength: Int,
                                             animatedSplatsUpdated: Bool,
                                             to commandBuffer: MTLCommandBuffer) throws -> Bool {
        // The FastSH fragment shader implements sorted alpha blending, not
        // tile-based depth compositing or stochastic transparency. Preserve
        // those explicitly selected base modes, using their existing DC path.
        guard fastSHConfig.enabled, shRenderingEnabled,
              !useMultiStagePipeline, !useDitheredTransparency else { return false }

        do {
            let colorsChanged = try refreshFastSHColorsIfNeeded(to: commandBuffer)
            try updateAnimatedSHSplatsIfNeeded(to: commandBuffer,
                                              forceRebuild: animatedSplatsUpdated || colorsChanged)
        } catch {
            // The base lifecycle acknowledges the captured frame revision.
            // Keep failed buffer rebuilds dirty so a later frame can retry.
            invalidateRender()
            throw error
        }
        guard shPaletteBuffer != nil, shDegree > 0, shCoefficientsPerEntry > 0 else { return false }

        if fastSHPipelineState == nil || fastSHEditingPipelineState == nil {
            setupFastSHPipeline(library: library)
        }
        let bindEditingResources = shouldBindEditingResources
            && editStateBuffer != nil && editTransformIndexBuffer != nil
            && editTransformPaletteBuffer != nil
        guard let pipeline = bindEditingResources ? fastSHEditingPipelineState : fastSHPipelineState else { return false }
        try buildSingleStagePipelineStatesIfNeeded()
        try renderWithFastSH(viewports: viewports,
                             colorTexture: colorTexture,
                             colorLoadAction: colorLoadAction,
                             colorStoreAction: colorStoreAction,
                             depthTexture: depthTexture,
                             depthStoreAction: depthStoreAction,
                             rasterizationRateMap: rasterizationRateMap,
                             renderTargetArrayLength: renderTargetArrayLength,
                             commandBuffer: commandBuffer,
                             pipelineState: pipeline)
        return true
    }

    public override init(device: MTLDevice,
                        colorFormat: MTLPixelFormat,
                        depthFormat: MTLPixelFormat,
                        sampleCount: Int,
                        maxViewCount: Int,
                        maxSimultaneousRenders: Int) throws {
        
        // Initialize SH buffer pool
        let shPoolConfig = MetalBufferPool<SplatSH>.Configuration(
            maxPoolSize: 8,
            maxBufferAge: 120.0,
            memoryPressureThreshold: 0.7
        )
        self.splatSHBufferPool = MetalBufferPool(device: device, configuration: shPoolConfig)
        self.splatSHBuffer = try splatSHBufferPool.acquire(minimumCapacity: 1)
        self.splatSHBufferPrime = try splatSHBufferPool.acquire(minimumCapacity: 1)
        
        try super.init(device: device,
                      colorFormat: colorFormat,
                      depthFormat: depthFormat,
                      sampleCount: sampleCount,
                      maxViewCount: maxViewCount,
                      maxSimultaneousRenders: maxSimultaneousRenders)
        
        // Initialize fast SH render pipeline
        do {
            let library = try device.makeDefaultLibrary(bundle: Bundle.module)
            setupFastSHPipeline(library: library)
        } catch {
            print("Failed to initialize fast SH pipeline: \(error)")
        }
    }

    deinit {
        if let animatedSplatSHBuffer {
            splatSHBufferPool.release(animatedSplatSHBuffer)
        }
        splatSHBufferPool.release(splatSHBuffer)
        splatSHBufferPool.release(splatSHBufferPrime)
    }
    
    private func setupFastSHPipeline(library: MTLLibrary) {
        do {
            // Set function constants (required for shaders including ShaderCommon.h)
            let functionConstants = MTLFunctionConstantValues()
            var use2DGSValue = use2DGSMode
            functionConstants.setConstantValue(&use2DGSValue, type: .bool, index: 12)

            // Buffer-based fast SH pipeline
            let vertexFunction = try library.makeFunction(name: "fastSHSplatVertexShader", constantValues: functionConstants)
            let editingVertexFunction = try library.makeFunction(name: "fastSHSplatVertexShaderEditing", constantValues: functionConstants)
            let fragmentFunction = try library.makeFunction(name: "fastSHSplatFragmentShader", constantValues: functionConstants)

            let pipelineDescriptor = MTLRenderPipelineDescriptor()
            pipelineDescriptor.label = "Fast SH Splat Pipeline"
            pipelineDescriptor.vertexFunction = vertexFunction
            pipelineDescriptor.fragmentFunction = fragmentFunction
            pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
            pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
            pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
            pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
            pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            pipelineDescriptor.depthAttachmentPixelFormat = depthFormat
            if #available(iOS 16.0, macOS 13.0, visionOS 1.0, *) {
                pipelineDescriptor.rasterSampleCount = sampleCount
            } else {
                pipelineDescriptor.sampleCount = sampleCount
            }

            if #available(iOS 17.0, macOS 14.0, *) {
                pipelineDescriptor.maxVertexAmplificationCount = maxViewCount
            }

            fastSHPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)

            let editingPipelineDescriptor = pipelineDescriptor.copy() as! MTLRenderPipelineDescriptor
            editingPipelineDescriptor.label = "Fast SH Splat Editing Pipeline"
            editingPipelineDescriptor.vertexFunction = editingVertexFunction
            fastSHEditingPipelineState = try device.makeRenderPipelineState(descriptor: editingPipelineDescriptor)
        } catch {
            print("Failed to create fast SH pipeline: \(error)")
        }
    }
    
    /// Load splats with SH coefficients support
    private func loadSplatsWithSHSynchronously(_ splats: [SplatScenePoint]) throws {
        try replaceAllSplats(with: splats)
    }

    private func rebuildFastSHBuffers(from splats: [SplatScenePoint]) throws {
        lastFastSHAppliedColorRevision = nil
        // Extract unique SH coefficient sets and build palette
        var uniqueSHSets: [[SIMD3<Float>]] = []
        var shSetToIndex: [[SIMD3<Float>]: UInt32] = [:]
        var paletteMap: [Int: UInt32] = [:]
        var degree = 0
        
        // Layers may mix DC-only and higher-degree points. Use a uniform
        // maximum supported degree so ordering cannot discard later bands.
        for splat in splats {
            if case let .sphericalHarmonic(coeffs) = splat.color, !coeffs.isEmpty {
                degree = max(degree, SphericalHarmonicsEvaluator.degreeFromCoefficientCount(coeffs.count))
            }
        }
        
        // Build palette of unique SH coefficient sets
        for (index, splat) in splats.enumerated() {
            if case let .sphericalHarmonic(coeffs) = splat.color, !coeffs.isEmpty {
                let rendererCoeffs = Self.rendererSupportedSHCoefficients(coeffs, degree: degree)
                if let paletteIndex = shSetToIndex[rendererCoeffs] {
                    paletteMap[index] = paletteIndex
                } else if uniqueSHSets.count < fastSHConfig.maxPaletteSize {
                    let newIndex = UInt32(uniqueSHSets.count)
                    uniqueSHSets.append(rendererCoeffs)
                    shSetToIndex[rendererCoeffs] = newIndex
                    paletteMap[index] = newIndex
                }
                // Unmapped points retain their base color when the palette is full.
            }
        }
        
        // Create SH palette buffer if we have SH data
        // Uses half-precision (Float16) storage — 50% less memory and bandwidth vs float32.
        // Keep the existing half-precision storage policy.
        let paletteBuffer: MTLBuffer?
        let coeffsPerEntry = uniqueSHSets.isEmpty ? 0 : SphericalHarmonicsEvaluator.coefficientCountForDegree(degree)
        if !uniqueSHSets.isEmpty {
            let paletteSizeBytes = uniqueSHSets.count * coeffsPerEntry * MemoryLayout<SIMD3<Float16>>.stride

            print("Fast SH: Creating half-precision palette buffer - \(uniqueSHSets.count) sets × \(coeffsPerEntry) coeffs = \(paletteSizeBytes) bytes")

            guard let buffer = device.makeBuffer(length: paletteSizeBytes, options: .storageModeShared) else {
                throw SplatRendererError.failedToCreateBuffer(length: paletteSizeBytes)
            }
            buffer.label = "SH Palette Buffer (half)"
            paletteBuffer = buffer

            // Fill palette buffer with half-precision coefficients
            let contents = buffer.contents().bindMemory(to: SIMD3<Float16>.self, capacity: uniqueSHSets.count * coeffsPerEntry)
            for (setIndex, coeffSet) in uniqueSHSets.enumerated() {
                let offset = setIndex * coeffsPerEntry
                for coeffIndex in 0..<coeffsPerEntry {
                    let f = coeffSet[coeffIndex]
                    contents[offset + coeffIndex] = SIMD3<Float16>(Float16(f.x), Float16(f.y), Float16(f.z))
                }
            }
        } else {
            paletteBuffer = nil
        }

        // Create extended splat buffer with SH info
        try ensureSHBufferCapacity(splats.count)
        
        for (index, splat) in splats.enumerated() {
            let baseSplat = Splat(splat)
            let shIndex = paletteMap[index] ?? UInt32.max
            let shDeg = (paletteMap[index] != nil) ? UInt16(degree) : 0

            let splatSH = SplatSH(splat: baseSplat,
                                   rotation: splat.rotation.normalized,
                                   shIndex: shIndex,
                                   shDegree: shDeg)
            splatSHBuffer.append(splatSH)
        }
        // Publish and acknowledge only a completely built palette and splat buffer.
        shPaletteBuffer = paletteBuffer
        shPaletteMap = paletteMap
        shDegree = degree
        shCoefficients = uniqueSHSets
        shCoefficientsPerEntry = coeffsPerEntry
        shaderParameters = FastSHShaderParameters(
            coeffsPerEntry: UInt32(coeffsPerEntry),
            paletteSize: UInt32(uniqueSHSets.count),
            degree: UInt32(degree),
            skipSHEvaluation: 0
        )
        shDirtyDueToData = true
        lastFastSHAppliedColorRevision = appliedColorRevision
    }

    private func refreshFastSHColorsIfNeeded(to commandBuffer: MTLCommandBuffer) throws -> Bool {
        guard lastFastSHAppliedColorRevision != appliedColorRevision else { return false }
        // Color-only updates are applied by the base frame lifecycle first.
        // They replace source SH with constant color. Rebuild metadata into new
        // buffers so earlier GPU frames can keep using their existing buffers.
        let replacement = try splatSHBufferPool.acquire(minimumCapacity: max(splatCount, 1))
        let replacementPrime: MetalBuffer<SplatSH>
        do {
            replacementPrime = try splatSHBufferPool.acquire(minimumCapacity: max(splatCount, 1))
        } catch {
            splatSHBufferPool.release(replacement)
            throw error
        }
        let previous = splatSHBuffer
        let previousPrime = splatSHBufferPrime
        splatSHBuffer = replacement
        splatSHBufferPrime = replacementPrime
        do {
            try rebuildFastSHBuffers(from: sourceScenePoints)
        } catch {
            splatSHBuffer = previous
            splatSHBufferPrime = previousPrime
            splatSHBufferPool.release(replacement)
            splatSHBufferPool.release(replacementPrime)
            throw error
        }
        commandBuffer.addCompletedHandler { [weak self] _ in
            self?.splatSHBufferPool.release(previous)
            self?.splatSHBufferPool.release(previousPrime)
        }
        // The base color API accepts already-linear RGB. Reuse its packed
        // result rather than applying the import transfer curve a second time.
        for index in 0..<min(splatSHBuffer.count, splatBuffer.count) {
            splatSHBuffer.values[index].baseColor = splatBuffer.values[index].packedColor
        }
        return true
    }

    public func loadSplatsWithSH(_ splats: [SplatScenePoint]) async throws {
        try loadSplatsWithSHSynchronously(splats)
    }
    
    private func ensureSHBufferCapacity(_ capacity: Int) throws {
        if splatSHBuffer.capacity < capacity {
            splatSHBuffer = try splatSHBufferPool.acquire(minimumCapacity: capacity)
        }
        if splatSHBufferPrime.capacity < capacity {
            splatSHBufferPrime = try splatSHBufferPool.acquire(minimumCapacity: capacity)
        }
        splatSHBuffer.count = 0
        splatSHBufferPrime.count = 0
    }
    
    /// Override read method to use Fast SH loading pipeline
    public override func read(from url: URL) async throws {
        let reader = try AutodetectSceneReader(url)
        var newPoints = SplatMemoryBuffer()
        try await newPoints.read(from: reader)
        renderMode = Self.renderMode(from: reader.renderMode)
        try await loadSplatsWithSH(newPoints.points)
    }

    public override func replaceSceneLayers(_ layers: [SplatSceneLayer]) throws {
        let allPoints = layers.flatMap(\.points)
        if let animatedSplatSHBuffer {
            splatSHBufferPool.release(animatedSplatSHBuffer)
            self.animatedSplatSHBuffer = nil
            lastAppliedAnimationTimeSH = nil
        }
        try replaceAllSplats(with: allPoints, sceneCounts: layers.map { $0.points.count })
    }

    internal override func updateSplats(_ points: [SplatScenePoint], at indices: [Int]) throws {
        try super.updateSplats(points, at: indices)
        try rebuildFastSHBuffers(from: sourceScenePoints)
        if let animatedSplatSHBuffer {
            splatSHBufferPool.release(animatedSplatSHBuffer)
            self.animatedSplatSHBuffer = nil
            lastAppliedAnimationTimeSH = nil
        }
    }

    internal override func replaceAllSplats(with points: [SplatScenePoint],
                                            sceneCounts: [Int]? = nil,
                                            sceneIndices: [UInt32]? = nil) throws {
        try super.replaceAllSplats(with: points, sceneCounts: sceneCounts, sceneIndices: sceneIndices)
        try rebuildFastSHBuffers(from: sourceScenePoints)
        if let animatedSplatSHBuffer {
            splatSHBufferPool.release(animatedSplatSHBuffer)
            self.animatedSplatSHBuffer = nil
            lastAppliedAnimationTimeSH = nil
        }
    }

    public func replaceSceneLayersWithSH(_ layers: [SplatSceneLayer]) async throws {
        try replaceSceneLayers(layers)
    }
    
}

// MARK: - ModelRenderer-compatible rendering with Fast SH
extension FastSHSplatRenderer {
    public func render(viewports: [ModelRendererViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       depthStoreAction: MTLStoreAction = .dontCare,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {

        // Keep the compatibility API, but enter the same lifecycle and dynamic
        // draw hook used by callers holding a SplatRenderer reference.
        let splatViewports = viewports.map {
            ViewportDescriptor(viewport: $0.viewport,
                               projectionMatrix: $0.projectionMatrix,
                               viewMatrix: $0.viewMatrix,
                               screenSize: $0.screenSize)
        }
        try render(viewports: splatViewports,
                   colorTexture: colorTexture,
                   colorLoadAction: .clear,
                   colorStoreAction: colorStoreAction,
                   depthTexture: depthTexture,
                   depthStoreAction: depthStoreAction,
                   rasterizationRateMap: rasterizationRateMap,
                   renderTargetArrayLength: renderTargetArrayLength,
                   to: commandBuffer)
    }

    private func renderWithFastSH(viewports: [ViewportDescriptor],
                                 colorTexture: MTLTexture,
                                 colorLoadAction: MTLLoadAction,
                                 colorStoreAction: MTLStoreAction,
                                 depthTexture: MTLTexture?,
                                 depthStoreAction: MTLStoreAction = .dontCare,
                                 rasterizationRateMap: MTLRasterizationRateMap?,
                                 renderTargetArrayLength: Int,
                                 commandBuffer: MTLCommandBuffer,
                                 pipelineState: MTLRenderPipelineState) throws {
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
            throw SplatRendererError.failedToCreateRenderEncoder
        }
        defer { renderEncoder.endEncoding() }
        renderEncoder.label = "Fast SH Splat Render"
        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setDepthStencilState(singleStageDepthState)

        // Set buffers
        renderEncoder.setVertexBuffer(activeSplatSHBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)
        renderEncoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        let bindEditingResources = shouldBindEditingResources
            && editStateBuffer != nil
            && editTransformIndexBuffer != nil
            && editTransformPaletteBuffer != nil

        if bindEditingResources, let editStateBuffer {
            renderEncoder.setVertexBuffer(editStateBuffer, offset: 0, index: BufferIndex.editState.rawValue)
        }
        if bindEditingResources, let editTransformIndexBuffer {
            renderEncoder.setVertexBuffer(editTransformIndexBuffer, offset: 0, index: BufferIndex.transformIndex.rawValue)
        }
        if bindEditingResources, let editTransformPaletteBuffer {
            renderEncoder.setVertexBuffer(editTransformPaletteBuffer, offset: 0, index: BufferIndex.transformPalette.rawValue)
        }

        // GPU-only sorting: pass sorted indices buffer to shader
        if let sortedIndices = getCurrentSortedIndicesBuffer() {
            renderEncoder.setVertexBuffer(sortedIndices.buffer, offset: 0, index: BufferIndex.sortedIndices.rawValue)
        }

        // SH is evaluated in the vertex shader; no evaluated-color cache
        // exists. Every rendered frame must evaluate it, including a static
        // camera, translations and model edits.
        if let paletteBuffer = shPaletteBuffer {
            renderEncoder.setVertexBuffer(paletteBuffer, offset: 0, index: 7)
            var params = shaderParameters
            params.skipSHEvaluation = 0
            renderEncoder.setVertexBytes(&params, length: MemoryLayout<FastSHShaderParameters>.stride, index: 8)
        }

        let drawCount = renderableSplatCountForCurrentEditState
        let indexedCount = min(drawCount, Constants.maxIndexedSplatCount)
        let instanceCount = (drawCount + indexedCount - 1) / indexedCount

        // Ensure index buffer is properly sized and filled (FastSH path may be called independently)
        let requiredIndexCount = indexedCount * 6
        if indexBuffer.count < requiredIndexCount {
            if indexBuffer.capacity < requiredIndexCount {
                indexBufferPool.release(indexBuffer)
                indexBuffer = try indexBufferPool.acquire(minimumCapacity: requiredIndexCount)
            }
            indexBuffer.count = requiredIndexCount
            for i in 0..<indexedCount {
                indexBuffer.values[i * 6 + 0] = UInt32(i * 4 + 0)
                indexBuffer.values[i * 6 + 1] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 2] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 3] = UInt32(i * 4 + 1)
                indexBuffer.values[i * 6 + 4] = UInt32(i * 4 + 2)
                indexBuffer.values[i * 6 + 5] = UInt32(i * 4 + 3)
            }
        }

        renderEncoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: requiredIndexCount,
            indexType: UInt32.asMTLIndexType,
            indexBuffer: indexBuffer.buffer,
            indexBufferOffset: 0,
            instanceCount: instanceCount,
            baseVertex: 0,
            baseInstance: 0
        )

        if shouldDrawSelectionOutline,
           let selectionOutlinePipelineState,
           let selectionOutlineDepthState {
            // The outline vertex shader consumes the base Splat layout.
            renderEncoder.setVertexBuffer(activeSplatBufferForRendering.buffer, offset: 0, index: BufferIndex.splat.rawValue)
            renderEncoder.setRenderPipelineState(selectionOutlinePipelineState)
            renderEncoder.setDepthStencilState(selectionOutlineDepthState)
            renderEncoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: requiredIndexCount,
                indexType: UInt32.asMTLIndexType,
                indexBuffer: indexBuffer.buffer,
                indexBufferOffset: 0,
                instanceCount: instanceCount,
                baseVertex: 0,
                baseInstance: 0
            )
        }

    }

    private var activeSplatSHBuffer: MetalBuffer<SplatSH> {
        animationEnabled ? (animatedSplatSHBuffer ?? splatSHBuffer) : splatSHBuffer
    }

    private func updateAnimatedSHSplatsIfNeeded(to commandBuffer: MTLCommandBuffer, forceRebuild: Bool) throws {
        guard animationEnabled, let configuration = animationConfiguration else {
            if let animatedSplatSHBuffer {
                self.animatedSplatSHBuffer = nil
                lastAppliedAnimationTimeSH = nil
                commandBuffer.addCompletedHandler { [weak self] _ in
                    self?.splatSHBufferPool.release(animatedSplatSHBuffer)
                }
            }
            return
        }

        if !forceRebuild, let animatedSplatSHBuffer, lastAppliedAnimationTimeSH == configuration.time, animatedSplatSHBuffer.count == sourceScenePoints.count {
            return
        }

        // A forced rebuild may follow a changed SH palette. Never draw the old
        // animated buffer against that palette if allocation fails, or accept
        // it as current on the following frame.
        lastAppliedAnimationTimeSH = nil
        let animatedSplatSHBuffer = try splatSHBufferPool.acquire(minimumCapacity: max(sourceScenePoints.count, 1))
        animatedSplatSHBuffer.count = 0

        for (index, point) in sourceScenePoints.enumerated() {
            let sceneIndex = index < animationSceneIndices.count ? Int(animationSceneIndices[index]) : 0
            let sample = SplatAnimationEngine.apply(
                point: point,
                globalIndex: index,
                sceneIndex: sceneIndex,
                sceneMetrics: animationSceneMetrics,
                configuration: configuration
            )

            let baseSplat = Splat(sample.point)
            let shIndex = shPaletteMap[index] ?? UInt32.max
            let shDeg = shPaletteMap[index] != nil ? UInt16(shDegree) : 0
            var splatSH = SplatSH(splat: baseSplat,
                                  rotation: sample.point.rotation.normalized,
                                  shIndex: shIndex,
                                  shDegree: shDeg)
            splatSH.tintColor = SIMD4<Float>(sample.tint.x, sample.tint.y, sample.tint.z, 1)
            animatedSplatSHBuffer.append(splatSH)
        }

        let previousAnimatedBuffer = self.animatedSplatSHBuffer
        self.animatedSplatSHBuffer = animatedSplatSHBuffer
        lastAppliedAnimationTimeSH = configuration.time
        if let previousAnimatedBuffer {
            commandBuffer.addCompletedHandler { [weak self] _ in
                self?.splatSHBufferPool.release(previousAnimatedBuffer)
            }
        }
    }
}
