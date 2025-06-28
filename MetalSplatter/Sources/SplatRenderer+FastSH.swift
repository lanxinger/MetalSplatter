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
        var baseColor: PackedRGBHalf4      // Base color (DC term) + opacity
        var covA: PackedHalf3
        var covB: PackedHalf3
        var shPaletteIndex: UInt16         // Index into SH palette (for SOGS)
        var shDegree: UInt16               // SH degree (0-3)
        
        /// Convert from regular Splat + SH info
        init(splat: Splat, shIndex: UInt16 = 0, shDegree: UInt16 = 0) {
            self.position = splat.position
            self.baseColor = splat.color
            self.covA = splat.covA
            self.covB = splat.covB
            self.shPaletteIndex = shIndex
            self.shDegree = shDegree
        }
    }
}

// MARK: - Fast SH Renderer Extension

public class FastSHSplatRenderer: SplatRenderer {
    
    /// Configuration for fast SH rendering
    public struct FastSHConfiguration {
        /// Enable fast SH evaluation (vs per-splat evaluation)
        public var enabled: Bool = true
        
        /// Use texture-based evaluation for better edge accuracy
        public var useTextureEvaluation: Bool = false
        
        /// Maximum number of unique SH coefficient sets (palette size)
        public var maxPaletteSize: Int = 65536
        
        /// Update SH evaluation every N frames (1 = every frame)
        public var updateFrequency: Int = 1
        
        public init() {}
    }
    
    // Fast SH specific properties
    public var fastSHConfig = FastSHConfiguration()
    private var shEvaluator: SphericalHarmonicsEvaluator?
    private var shPaletteBuffer: MTLBuffer?
    private var evaluatedSHBuffer: MTLBuffer?
    private var evaluatedSHTexture: MTLTexture?
    private var fastSHPipelineState: MTLRenderPipelineState?
    private var textureSHPipelineState: MTLRenderPipelineState?
    private var framesSinceLastSHUpdate = 0
    
    // SH data storage
    public private(set) var shCoefficients: [[SIMD3<Float>]] = []
    private var shPaletteMap: [Int: UInt16] = [:] // Maps from splat index to palette index
    public private(set) var shDegree: Int = 0
    
    // Extended splat buffer for SH
    private var splatSHBuffer: MetalBuffer<SplatSH>
    private var splatSHBufferPrime: MetalBuffer<SplatSH>
    private let splatSHBufferPool: MetalBufferPool<SplatSH>
    
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
        
        // Initialize SH evaluator
        do {
            let library = try device.makeDefaultLibrary(bundle: Bundle.module)
            self.shEvaluator = try SphericalHarmonicsEvaluator(device: device, library: library)
            
            // Create fast SH render pipeline
            setupFastSHPipeline(library: library)
        } catch {
            print("Failed to initialize fast SH support: \(error)")
        }
    }
    
    private func setupFastSHPipeline(library: MTLLibrary) {
        do {
            // Buffer-based fast SH pipeline
            let vertexFunction = library.makeFunction(name: "fastSHSplatVertexShader")
            let fragmentFunction = library.makeFunction(name: "fastSHSplatFragmentShader")
            
            if let vertexFunction = vertexFunction, let fragmentFunction = fragmentFunction {
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
                pipelineDescriptor.sampleCount = sampleCount
                
                if #available(iOS 17.0, macOS 14.0, *) {
                    pipelineDescriptor.maxVertexAmplificationCount = maxViewCount
                }
                
                fastSHPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            }
            
            // Texture-based SH pipeline
            let textureVertexFunction = library.makeFunction(name: "textureSHSplatVertexShader")
            
            if let vertexFunction = textureVertexFunction, let fragmentFunction = fragmentFunction {
                let pipelineDescriptor = MTLRenderPipelineDescriptor()
                pipelineDescriptor.label = "Texture SH Splat Pipeline"
                pipelineDescriptor.vertexFunction = vertexFunction
                pipelineDescriptor.fragmentFunction = fragmentFunction
                pipelineDescriptor.colorAttachments[0].pixelFormat = colorFormat
                pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
                pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .one
                pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
                pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
                pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
                pipelineDescriptor.depthAttachmentPixelFormat = depthFormat
                pipelineDescriptor.sampleCount = sampleCount
                
                if #available(iOS 17.0, macOS 14.0, *) {
                    pipelineDescriptor.maxVertexAmplificationCount = maxViewCount
                }
                
                textureSHPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
            }
            
        } catch {
            print("Failed to create fast SH pipeline: \(error)")
        }
    }
    
    /// Load splats with SH coefficients support
    public func loadSplatsWithSH(_ splats: [SplatScenePoint]) async throws {
        // Extract unique SH coefficient sets and build palette
        var uniqueSHSets: [[SIMD3<Float>]] = []
        var shSetToIndex: [[SIMD3<Float>]: UInt16] = [:]
        shPaletteMap.removeAll()
        
        // Determine SH degree from first splat with SH
        for splat in splats {
            if case let .sphericalHarmonic(coeffs) = splat.color, !coeffs.isEmpty {
                shDegree = SphericalHarmonicsEvaluator.degreeFromCoefficientCount(coeffs.count)
                break
            }
        }
        
        // Build palette of unique SH coefficient sets
        for (index, splat) in splats.enumerated() {
            if case let .sphericalHarmonic(coeffs) = splat.color, coeffs.count > 1 {
                if let paletteIndex = shSetToIndex[coeffs] {
                    shPaletteMap[index] = paletteIndex
                } else {
                    let newIndex = UInt16(uniqueSHSets.count)
                    uniqueSHSets.append(coeffs)
                    shSetToIndex[coeffs] = newIndex
                    shPaletteMap[index] = newIndex
                }
            }
        }
        
        // Create SH palette buffer if we have SH data
        if !uniqueSHSets.isEmpty {
            let coeffsPerEntry = SphericalHarmonicsEvaluator.coefficientCountForDegree(shDegree)
            let paletteSize = uniqueSHSets.count * coeffsPerEntry * MemoryLayout<SIMD3<Float>>.stride
            
            shPaletteBuffer = device.makeBuffer(length: paletteSize, options: .storageModeShared)
            shPaletteBuffer?.label = "SH Palette Buffer"
            
            // Fill palette buffer
            if let buffer = shPaletteBuffer {
                let contents = buffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: uniqueSHSets.count * coeffsPerEntry)
                for (setIndex, coeffSet) in uniqueSHSets.enumerated() {
                    let offset = setIndex * coeffsPerEntry
                    for (coeffIndex, coeff) in coeffSet.enumerated() {
                        contents[offset + coeffIndex] = coeff
                    }
                }
            }
            
            shCoefficients = uniqueSHSets
            print("Created SH palette with \(uniqueSHSets.count) unique sets, degree \(shDegree)")
        }
        
        // Create extended splat buffer with SH info
        try ensureSHBufferCapacity(splats.count)
        
        for (index, splat) in splats.enumerated() {
            let baseSplat = Splat(splat)
            let shIndex = shPaletteMap[index] ?? 0
            let shDeg = (shPaletteMap[index] != nil) ? UInt16(shDegree) : 0
            
            let splatSH = SplatSH(splat: baseSplat, shIndex: shIndex, shDegree: shDeg)
            splatSHBuffer.append(splatSH)
        }
        
        // Also fill regular splat buffer for compatibility
        try add(splats)
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
    
    /// Update SH evaluation based on current camera direction
    private func updateSHEvaluation(viewDirection: SIMD3<Float>, commandBuffer: MTLCommandBuffer) {
        guard fastSHConfig.enabled,
              let evaluator = shEvaluator,
              let paletteBuffer = shPaletteBuffer,
              !shCoefficients.isEmpty else { return }
        
        // Check if we should update this frame
        framesSinceLastSHUpdate += 1
        if framesSinceLastSHUpdate < fastSHConfig.updateFrequency {
            return
        }
        framesSinceLastSHUpdate = 0
        
        let paletteSize = shCoefficients.count
        
        if fastSHConfig.useTextureEvaluation {
            // Calculate texture size to fit palette
            let textureWidth = min(paletteSize, 256) // Limit width for cache efficiency
            let textureHeight = (paletteSize + textureWidth - 1) / textureWidth
            let textureSize = MTLSize(width: textureWidth, height: textureHeight, depth: 1)
            
            evaluatedSHTexture = evaluator.evaluateToTexture(
                shPalette: paletteBuffer,
                textureSize: textureSize,
                degree: shDegree,
                viewDirection: viewDirection,
                commandBuffer: commandBuffer
            )
        } else {
            evaluatedSHBuffer = evaluator.evaluatePalette(
                shPalette: paletteBuffer,
                paletteSize: paletteSize,
                degree: shDegree,
                viewDirection: viewDirection,
                commandBuffer: commandBuffer
            )
        }
    }
    
}

// MARK: - ModelRenderer-compatible rendering with Fast SH
extension FastSHSplatRenderer {
    public func render(viewports: [ModelRendererViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        
        // Update camera direction for SH evaluation
        if fastSHConfig.enabled && !viewports.isEmpty {
            // Calculate mean camera forward direction
            let cameraForwards = viewports.map { viewport -> SIMD3<Float> in
                let viewMatrix = viewport.viewMatrix
                return -SIMD3<Float>(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2])
            }
            let meanForward = cameraForwards.mean ?? SIMD3<Float>(0, 0, -1)
            
            // Update SH evaluation
            updateSHEvaluation(viewDirection: meanForward, commandBuffer: commandBuffer)
        }
        
        // Convert to SplatRenderer.ViewportDescriptor
        let splatViewports = viewports.map { viewport -> SplatRenderer.ViewportDescriptor in
            SplatRenderer.ViewportDescriptor(
                viewport: viewport.viewport,
                projectionMatrix: viewport.projectionMatrix,
                viewMatrix: viewport.viewMatrix,
                screenSize: viewport.screenSize
            )
        }
        
        // Use fast SH pipeline if enabled and available
        if fastSHConfig.enabled,
           let pipeline = fastSHConfig.useTextureEvaluation ? textureSHPipelineState : fastSHPipelineState,
           (evaluatedSHBuffer != nil || evaluatedSHTexture != nil) {
            
            // Render using fast SH pipeline
            try renderWithFastSH(viewports: viewports,
                                colorTexture: colorTexture,
                                colorStoreAction: colorStoreAction,
                                depthTexture: depthTexture,
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
                      rasterizationRateMap: rasterizationRateMap,
                      renderTargetArrayLength: renderTargetArrayLength,
                      to: commandBuffer)
        }
    }
    
    private func renderWithFastSH(viewports: [ModelRendererViewportDescriptor],
                                 colorTexture: MTLTexture,
                                 colorStoreAction: MTLStoreAction,
                                 depthTexture: MTLTexture?,
                                 rasterizationRateMap: MTLRasterizationRateMap?,
                                 renderTargetArrayLength: Int,
                                 commandBuffer: MTLCommandBuffer,
                                 pipelineState: MTLRenderPipelineState) throws {
        
        // Create render pass descriptor
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = colorTexture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = colorStoreAction
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        
        if let depthTexture = depthTexture {
            renderPassDescriptor.depthAttachment.texture = depthTexture
            renderPassDescriptor.depthAttachment.loadAction = .clear
            renderPassDescriptor.depthAttachment.storeAction = .store
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
        renderEncoder.setVertexBuffer(splatSHBuffer.buffer, offset: 0, index: BufferIndex.splat.rawValue)
        renderEncoder.setVertexBuffer(dynamicUniformBuffers, offset: uniformBufferOffset, index: BufferIndex.uniforms.rawValue)
        
        // Set evaluated SH data
        if fastSHConfig.useTextureEvaluation, let texture = evaluatedSHTexture {
            renderEncoder.setVertexTexture(texture, index: 0)
        } else if let buffer = evaluatedSHBuffer {
            renderEncoder.setVertexBuffer(buffer, offset: 0, index: 3)
        }
        
        // Set up viewports and draw
        for (viewportIndex, viewport) in viewports.prefix(maxViewCount).enumerated() {
            uniforms.pointee.setUniforms(index: viewportIndex, Uniforms(
                projectionMatrix: viewport.projectionMatrix,
                viewMatrix: viewport.viewMatrix,
                screenSize: SIMD2<UInt32>(UInt32(viewport.screenSize.x), UInt32(viewport.screenSize.y)),
                splatCount: UInt32(splatCount),
                indexedSplatCount: UInt32(min(splatCount, Constants.maxIndexedSplatCount))
            ))
        }
        
        // Draw splats
        let indexedCount = min(splatCount, Constants.maxIndexedSplatCount)
        let instanceCount = (splatCount + Constants.maxIndexedSplatCount - 1) / Constants.maxIndexedSplatCount
        
        renderEncoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexBuffer.count,
            indexType: UInt32.asMTLIndexType,
            indexBuffer: indexBuffer.buffer,
            indexBufferOffset: 0,
            instanceCount: instanceCount,
            baseVertex: 0,
            baseInstance: 0
        )
        
        renderEncoder.endEncoding()
    }
}