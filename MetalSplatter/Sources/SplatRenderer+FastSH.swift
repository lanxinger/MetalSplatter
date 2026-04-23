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
    public var fastSHConfig = FastSHConfiguration()

    /// When false, SH evaluation is completely disabled and only base color is used.
    /// This can provide significant performance gains at the cost of view-dependent lighting.
    public var shRenderingEnabled: Bool = true
    private var shPaletteBuffer: MTLBuffer?
    private var fastSHPipelineState: MTLRenderPipelineState?
    
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
    private let splatSHBufferPool: MetalBufferPool<SplatSH>

    private struct FastSHShaderParameters {
        var coeffsPerEntry: UInt32 = 0
        var paletteSize: UInt32 = 0
        var degree: UInt32 = 0
        var skipSHEvaluation: UInt32 = 0  // When non-zero, skip SH evaluation and use base color
    }

    private var shaderParameters = FastSHShaderParameters()

    internal override func resetPipelineStates() {
        super.resetPipelineStates()
        fastSHPipelineState = nil
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
        } catch {
            print("Failed to create fast SH pipeline: \(error)")
        }
    }
    
    /// Load splats with SH coefficients support
    private func loadSplatsWithSHSynchronously(_ splats: [SplatScenePoint]) throws {
        // Extract unique SH coefficient sets and build palette
        var uniqueSHSets: [[SIMD3<Float>]] = []
        var shSetToIndex: [[SIMD3<Float>]: UInt32] = [:]
        shPaletteMap.removeAll()
        
        // Determine SH degree from first splat with SH
        for splat in splats {
            if case let .sphericalHarmonic(coeffs) = splat.color, !coeffs.isEmpty {
                shDegree = SphericalHarmonicsEvaluator.degreeFromCoefficientCount(coeffs.count)
                print("Fast SH: Detected \(coeffs.count) coefficients, mapped to degree \(shDegree)")
                break
            }
        }
        
        // Build palette of unique SH coefficient sets
        for (index, splat) in splats.enumerated() {
            if case let .sphericalHarmonic(coeffs) = splat.color, !coeffs.isEmpty {
                if let paletteIndex = shSetToIndex[coeffs] {
                    shPaletteMap[index] = paletteIndex
                } else if uniqueSHSets.count < fastSHConfig.maxPaletteSize {
                    let newIndex = UInt32(uniqueSHSets.count)
                    uniqueSHSets.append(coeffs)
                    shSetToIndex[coeffs] = newIndex
                    shPaletteMap[index] = newIndex
                }
                // If we exceed maxPaletteSize, splats will use index 0 (fallback)
            }
        }
        
        // Create SH palette buffer if we have SH data
        // Uses half-precision (Float16) storage — 50% less memory and bandwidth vs float32.
        // Half precision is sufficient for SH coefficients (low-frequency lighting).
        if !uniqueSHSets.isEmpty {
            // Use actual coefficient count from the data, not theoretical count
            let actualCoeffsPerEntry = uniqueSHSets[0].count
            let coeffsPerEntry = actualCoeffsPerEntry // Use actual data structure
            let paletteSizeBytes = uniqueSHSets.count * coeffsPerEntry * MemoryLayout<SIMD3<Float16>>.stride

            print("Fast SH: Creating half-precision palette buffer - \(uniqueSHSets.count) sets × \(coeffsPerEntry) coeffs = \(paletteSizeBytes) bytes")

            shPaletteBuffer = device.makeBuffer(length: paletteSizeBytes, options: .storageModeShared)
            shPaletteBuffer?.label = "SH Palette Buffer (half)"

            // Fill palette buffer with half-precision coefficients
            if let buffer = shPaletteBuffer {
                let contents = buffer.contents().bindMemory(to: SIMD3<Float16>.self, capacity: uniqueSHSets.count * coeffsPerEntry)
                for (setIndex, coeffSet) in uniqueSHSets.enumerated() {
                    let offset = setIndex * coeffsPerEntry
                    // Safety check: only copy available coefficients
                    let coeffsToCopy = min(coeffSet.count, coeffsPerEntry)
                    for coeffIndex in 0..<coeffsToCopy {
                        let f = coeffSet[coeffIndex]
                        contents[offset + coeffIndex] = SIMD3<Float16>(Float16(f.x), Float16(f.y), Float16(f.z))
                    }
                    // Pad with zeros if coefficient set is shorter than expected
                    for coeffIndex in coeffsToCopy..<coeffsPerEntry {
                        contents[offset + coeffIndex] = SIMD3<Float16>(0, 0, 0)
                    }
                }
            }
            
            shCoefficients = uniqueSHSets
            print("Created SH palette with \(uniqueSHSets.count) unique sets, degree \(shDegree)")
            shCoefficientsPerEntry = coeffsPerEntry
            shaderParameters = FastSHShaderParameters(
                coeffsPerEntry: UInt32(coeffsPerEntry),
                paletteSize: UInt32(uniqueSHSets.count),
                degree: UInt32(shDegree),
                skipSHEvaluation: 0
            )
        }
        else {
            shPaletteBuffer = nil
            shCoefficients = []
            shCoefficientsPerEntry = 0
            shaderParameters = FastSHShaderParameters()
        }

        // Mark SH as dirty so it gets evaluated on first render
        shDirtyDueToData = true

        // Create extended splat buffer with SH info
        try ensureSHBufferCapacity(splats.count)
        
        for (index, splat) in splats.enumerated() {
            let baseSplat = Splat(splat)
            let shIndex = shPaletteMap[index] ?? UInt32.max
            let shDeg = (shPaletteMap[index] != nil) ? UInt16(shDegree) : 0

            let splatSH = SplatSH(splat: baseSplat,
                                   rotation: splat.rotation.normalized,
                                   shIndex: shIndex,
                                   shDegree: shDeg)
            splatSHBuffer.append(splatSH)
        }
        
        // Also fill regular splat buffer for sorting/culling compatibility
        try replaceAllSplats(with: splats)
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
        splatSHBuffer.count = 0
        splatSHBufferPrime.count = 0
        if let animatedSplatSHBuffer {
            splatSHBufferPool.release(animatedSplatSHBuffer)
            self.animatedSplatSHBuffer = nil
            lastAppliedAnimationTimeSH = nil
        }
        try loadSplatsWithSHSynchronously(allPoints)
        try replaceAllSplats(with: allPoints, sceneCounts: layers.map { $0.points.count })
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

        // Update camera direction for SH evaluation
        // Convert to SplatRenderer.ViewportDescriptor
        let splatViewports = viewports.map { viewport -> SplatRenderer.ViewportDescriptor in
            SplatRenderer.ViewportDescriptor(
                viewport: viewport.viewport,
                projectionMatrix: viewport.projectionMatrix,
                viewMatrix: viewport.viewMatrix,
                screenSize: viewport.screenSize
            )
        }

        let didUpdateAnimatedSplats = updateAnimatedSplatsIfNeeded(to: commandBuffer)
        updateAnimatedSHSplatsIfNeeded(to: commandBuffer, forceRebuild: didUpdateAnimatedSplats)

        // Use fast SH pipeline if enabled and available
        if fastSHConfig.enabled,
           let pipeline = fastSHPipelineState,
           shPaletteBuffer != nil,
           shDegree > 0,
           shCoefficientsPerEntry > 0 {
            switchToNextDynamicBuffer()
            updateUniforms(forViewports: splatViewports,
                           splatCount: UInt32(splatCount),
                           indexedSplatCount: UInt32(min(splatCount, Constants.maxIndexedSplatCount)))

            // Render using fast SH pipeline
            try renderWithFastSH(viewports: viewports,
                                colorTexture: colorTexture,
                                colorStoreAction: colorStoreAction,
                                depthTexture: depthTexture,
                                depthStoreAction: depthStoreAction,
                                rasterizationRateMap: rasterizationRateMap,
                                renderTargetArrayLength: renderTargetArrayLength,
                                commandBuffer: commandBuffer,
                                pipelineState: pipeline)
        } else {
            // Fall back to regular SplatRenderer rendering
            try render(viewports: splatViewports,
                      colorTexture: colorTexture,
                      colorStoreAction: colorStoreAction,
                      depthTexture: depthTexture,
                      depthStoreAction: depthStoreAction,
                      rasterizationRateMap: rasterizationRateMap,
                      renderTargetArrayLength: renderTargetArrayLength,
                      to: commandBuffer)
        }
    }
    
    private func renderWithFastSH(viewports: [ModelRendererViewportDescriptor],
                                 colorTexture: MTLTexture,
                                 colorStoreAction: MTLStoreAction,
                                 depthTexture: MTLTexture?,
                                 depthStoreAction: MTLStoreAction = .dontCare,
                                 rasterizationRateMap: MTLRasterizationRateMap?,
                                 renderTargetArrayLength: Int,
                                 commandBuffer: MTLCommandBuffer,
                                 pipelineState: MTLRenderPipelineState) throws {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            updateMetal4ResidencyForFrame(commandBuffer: commandBuffer)
        }

        // Create render pass descriptor
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)

        if let depthTexture = depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = depthStoreAction
            renderPassDescriptor.depthAttachment.clearDepth = 1.0
        }

        if let rateMap = rasterizationRateMap {
            renderPassDescriptor.rasterizationRateMap = rateMap
        }

        renderPassDescriptor.renderTargetArrayLength = renderTargetArrayLength

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            throw SplatRendererError.failedToCreateRenderPipelineState(label: "Fast SH Render Encoder", underlying: NSError(domain: "MetalSplatter", code: 1))
        }

        renderEncoder.label = "Fast SH Splat Render"
        renderEncoder.setRenderPipelineState(pipelineState)

        // Set buffers
        renderEncoder.setVertexBuffer(activeSplatSHBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)
        renderEncoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        if let editStateBuffer {
            renderEncoder.setVertexBuffer(editStateBuffer, offset: 0, index: BufferIndex.editState.rawValue)
        }
        if let editTransformIndexBuffer {
            renderEncoder.setVertexBuffer(editTransformIndexBuffer, offset: 0, index: BufferIndex.transformIndex.rawValue)
        }
        if let editTransformPaletteBuffer {
            renderEncoder.setVertexBuffer(editTransformPaletteBuffer, offset: 0, index: BufferIndex.transformPalette.rawValue)
        }

        // GPU-only sorting: pass sorted indices buffer to shader
        if let sortedIndices = sortedIndicesBuffer {
            renderEncoder.setVertexBuffer(sortedIndices.buffer, offset: 0, index: BufferIndex.sortedIndices.rawValue)
        }

        // Check if SH needs re-evaluation based on camera movement threshold
        let shouldUpdateSH = shRenderingEnabled && shouldUpdateSHForCurrentCamera()

        // Set SH palette data with skip flag based on threshold
        if let paletteBuffer = shPaletteBuffer {
            renderEncoder.setVertexBuffer(paletteBuffer, offset: 0, index: 7)
            var params = shaderParameters
            // Skip SH evaluation if disabled or camera hasn't moved enough
            params.skipSHEvaluation = (shRenderingEnabled && shouldUpdateSH) ? 0 : 1
            renderEncoder.setVertexBytes(&params, length: MemoryLayout<FastSHShaderParameters>.stride, index: 8)

            // Mark SH as updated if we evaluated this frame
            if shouldUpdateSH {
                didUpdateSHForCurrentCamera()
            }
        }

        // Set up viewports and draw
        for (viewportIndex, viewport) in viewports.prefix(maxViewCount).enumerated() {
            let splatViewport = ViewportDescriptor(viewport: viewport.viewport,
                                                   projectionMatrix: viewport.projectionMatrix,
                                                   viewMatrix: viewport.viewMatrix,
                                                   screenSize: viewport.screenSize)
            uniforms.pointee.setUniforms(index: viewportIndex, makeUniforms(
                for: splatViewport,
                splatCount: UInt32(splatCount),
                indexedSplatCount: UInt32(min(splatCount, Constants.maxIndexedSplatCount)),
                debugFlags: debugOptions.rawValue
            ))
        }

        // Draw splats
        let indexedCount = min(splatCount, Constants.maxIndexedSplatCount)
        let instanceCount = (splatCount + Constants.maxIndexedSplatCount - 1) / Constants.maxIndexedSplatCount

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

        renderEncoder.endEncoding()
    }

    private var activeSplatSHBuffer: MetalBuffer<SplatSH> {
        animationEnabled ? (animatedSplatSHBuffer ?? splatSHBuffer) : splatSHBuffer
    }

    private func updateAnimatedSHSplatsIfNeeded(to commandBuffer: MTLCommandBuffer, forceRebuild: Bool) {
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

        let animatedSplatSHBuffer: MetalBuffer<SplatSH>
        do {
            animatedSplatSHBuffer = try splatSHBufferPool.acquire(minimumCapacity: max(sourceScenePoints.count, 1))
        } catch {
            Self.log.error("Failed to allocate animated SH splat buffer: \(error)")
            return
        }
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
