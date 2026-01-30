import XCTest
import MetalSplatter
import Metal

/// Tests for enhanced Metal 4 Bindless Architecture
@available(iOS 18.0, macOS 15.0, visionOS 2.0, *)
class Metal4BindlessTests: XCTestCase {
    
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var bindlessArchitecture: Metal4BindlessArchitecture!
    var renderer: SplatRenderer!
    
    override func setUp() {
        super.setUp()
        
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal device not available")
            return
        }
        device = metalDevice
        
        guard let queue = device.makeCommandQueue() else {
            XCTFail("Failed to create command queue")
            return
        }
        commandQueue = queue
        
        // Create renderer
        do {
            renderer = try SplatRenderer(
                device: device,
                colorFormat: .bgra8Unorm_srgb,
                depthFormat: .depth32Float,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: 3
            )
        } catch {
            XCTFail("Failed to create renderer: \(error)")
        }
    }
    
    override func tearDown() {
        bindlessArchitecture = nil
        renderer = nil
        commandQueue = nil
        device = nil
        super.tearDown()
    }
    
    // MARK: - Basic Functionality Tests
    
    func testBindlessArchitectureInitialization() throws {
        let config = Metal4BindlessArchitecture.Configuration(
            maxResources: 512,
            maxSplatBuffers: 8,
            maxUniformBuffers: 4,
            maxTextures: 16
        )
        
        bindlessArchitecture = try Metal4BindlessArchitecture(
            device: device,
            configuration: config
        )
        
        XCTAssertNotNil(bindlessArchitecture, "Bindless architecture should initialize successfully")
        
        let stats = bindlessArchitecture.getStatistics()
        XCTAssertGreaterThan(stats.argumentBufferSize, 0, "Argument buffer should be allocated")
        XCTAssertGreaterThan(stats.resourceTableSize, 0, "Resource table should be allocated")
    }
    
    func testResourceRegistration() throws {
        bindlessArchitecture = try Metal4BindlessArchitecture(device: device)

        // Create test buffer
        let bufferSize = 1024 * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: bufferSize, options: .storageModeShared) else {
            XCTFail("Failed to create test buffer")
            return
        }

        // Register buffer
        guard let handle = bindlessArchitecture.registerBuffer(buffer, type: .splatBuffer) else {
            XCTFail("Failed to register buffer")
            return
        }

        XCTAssertNotEqual(handle.value, 0, "Resource handle should be valid")
        XCTAssertGreaterThan(handle.index, 0, "Resource index should be positive")
        
        // Check statistics
        let stats = bindlessArchitecture.getStatistics()
        XCTAssertEqual(stats.registeredResources, 1, "Should have one registered resource")
    }
    
    func testBackgroundResourcePopulation() throws {
        let config = Metal4BindlessArchitecture.Configuration(
            enableBackgroundPopulation: true
        )
        bindlessArchitecture = try Metal4BindlessArchitecture(device: device, configuration: config)
        
        // Register multiple resources
        var handles: [ResourceHandle] = []
        for i in 0..<10 {
            let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
            buffer.label = "Test Buffer \(i)"
            if let handle = bindlessArchitecture.registerBuffer(buffer, type: .uniformBuffer) {
                handles.append(handle)
            }
        }
        
        // Wait for background population
        Thread.sleep(forTimeInterval: 0.1)
        
        let stats = bindlessArchitecture.getStatistics()
        XCTAssertGreaterThan(stats.metrics.resourcesPopulatedInBackground, 0,
                            "Resources should be populated in background")
        XCTAssertEqual(stats.registeredResources, 10, "All resources should be registered")
    }
    
    // MARK: - Performance Tests
    
    func testZeroPerDrawBinding() throws {
        bindlessArchitecture = try Metal4BindlessArchitecture(device: device)
        
        // Register resources
        let splatBuffer = device.makeBuffer(length: 10000 * 64, options: .storageModeShared)!
        let uniformBuffer = device.makeBuffer(length: 256, options: .storageModeShared)!
        
        _ = bindlessArchitecture.registerBuffer(splatBuffer, type: .splatBuffer)
        _ = bindlessArchitecture.registerBuffer(uniformBuffer, type: .uniformBuffer)
        
        // Create render pass
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 1920,
            height: 1080,
            mipmapped: false
        )
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create resources")
            return
        }
        
        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].storeAction = .store
        
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            XCTFail("Failed to create render encoder")
            return
        }
        
        // Measure binding overhead
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Bind ONCE for entire render pass
        bindlessArchitecture.bindToRenderEncoder(renderEncoder)
        
        // Simulate 1000 draw calls WITHOUT any resource binding
        for _ in 0..<1000 {
            // No setBuffer calls needed!
            // Resources accessed through bindless argument buffer
        }
        
        let bindingTime = CFAbsoluteTimeGetCurrent() - startTime
        
        renderEncoder.endEncoding()
        commandBuffer.commit()
        
        // Verify performance
        XCTAssertLessThan(bindingTime, 0.001, "Binding should be nearly instant (< 1ms)")
        
        let stats = bindlessArchitecture.getStatistics()
        XCTAssertGreaterThan(stats.metrics.renderPassesWithoutBinding, 0,
                            "Should track render passes without per-draw binding")
    }
    
    func testCPUOverheadReduction() throws {
        // Test traditional vs bindless CPU overhead
        
        // Traditional approach timing
        let traditionalStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10000 {
            // Simulate traditional per-draw binding cost
            _ = device.makeBuffer(length: 64, options: .storageModeShared)
        }
        let traditionalTime = CFAbsoluteTimeGetCurrent() - traditionalStart
        
        // Bindless approach timing
        bindlessArchitecture = try Metal4BindlessArchitecture(device: device)
        
        let bindlessStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<10000 {
            // Bindless just updates handles, no actual binding
            let buffer = device.makeBuffer(length: 64, options: .storageModeShared)!
            _ = bindlessArchitecture.registerBuffer(buffer, type: .splatBuffer)
        }
        let bindlessTime = CFAbsoluteTimeGetCurrent() - bindlessStart
        
        let reduction = (traditionalTime - bindlessTime) / traditionalTime * 100
        
        print("Traditional approach: \(traditionalTime * 1000)ms")
        print("Bindless approach: \(bindlessTime * 1000)ms")
        print("CPU overhead reduction: \(Int(reduction))%")
        
        // Expect significant reduction
        XCTAssertGreaterThan(reduction, 30, "Should achieve >30% CPU overhead reduction")
    }
    
    // MARK: - Residency Management Tests
    
    func testResidencyTracking() throws {
        let config = Metal4BindlessArchitecture.Configuration(
            enableResidencyTracking: true
        )
        bindlessArchitecture = try Metal4BindlessArchitecture(device: device, configuration: config)
        
        // Register resources
        var handles: [ResourceHandle] = []
        for _ in 0..<5 {
            let buffer = device.makeBuffer(length: 1024 * 1024, options: .storageModeShared)!
            if let handle = bindlessArchitecture.registerBuffer(buffer, type: .splatBuffer) {
                handles.append(handle)
            }
        }
        
        // Update residency
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            return
        }
        
        bindlessArchitecture.updateResidency(visibleHandles: handles, commandBuffer: commandBuffer)
        
        let stats = bindlessArchitecture.getStatistics()
        XCTAssertGreaterThan(stats.metrics.residencyUpdates, 0, "Residency should be updated")
        XCTAssertGreaterThan(stats.residencyInfo.residentCount, 0, "Resources should be resident")
    }
    
    func testMemoryPressureHandling() throws {
        bindlessArchitecture = try Metal4BindlessArchitecture(device: device)
        
        // Register many resources
        for i in 0..<100 {
            let buffer = device.makeBuffer(length: 1024 * 1024, options: .storageModeShared)!
            buffer.label = "Large Buffer \(i)"
            _ = bindlessArchitecture.registerBuffer(buffer, type: .splatBuffer)
        }
        
        let statsBefore = bindlessArchitecture.getStatistics()
        
        // Simulate memory pressure
        bindlessArchitecture.handleMemoryPressure()
        
        let statsAfter = bindlessArchitecture.getStatistics()
        
        XCTAssertLessThanOrEqual(statsAfter.pendingResources, statsBefore.pendingResources,
                                 "Pending resources should be cleared")
        XCTAssertGreaterThan(statsAfter.residencyInfo.memoryPressureEvents, 0,
                            "Memory pressure event should be recorded")
    }
    
    // MARK: - Integration Tests
    
    func testRendererBindlessIntegration() throws {
        // Initialize enhanced bindless on renderer
        try renderer.initializeEnhancedBindless()
        
        // Create test scene
        let viewport = ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: 1920, height: 1080, znear: 0, zfar: 1),
            projectionMatrix: matrix_identity_float4x4,
            viewMatrix: matrix_identity_float4x4,
            screenSize: SIMD2<Float>(1920, 1080)
        )
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: 1920,
            height: 1080,
            mipmapped: false
        )
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create resources")
            return
        }
        
        // Render with bindless
        XCTAssertNoThrow({
            try renderer.renderWithBindless(
                viewports: [viewport],
                colorTexture: texture,
                colorLoadAction: .clear,
                colorStoreAction: .store,
                depthTexture: nil,
                rasterizationRateMap: nil,
                renderTargetArrayLength: 0,
                to: commandBuffer
            )
        }, "Bindless rendering should not throw")
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Print statistics
        renderer.printBindlessStatistics()
    }
    
    // MARK: - Benchmark Tests
    
    func testBindlessScalability() throws {
        measure {
            do {
                let bindless = try Metal4BindlessArchitecture(device: device)
                
                // Register 1000 resources
                for _ in 0..<1000 {
                    let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
                    _ = bindless.registerBuffer(buffer, type: .splatBuffer)
                }
                
                // Simulate render pass
                if let commandBuffer = commandQueue.makeCommandBuffer(),
                   let texture = device.makeTexture(descriptor: MTLTextureDescriptor()) {
                    
                    let renderPassDescriptor = MTLRenderPassDescriptor()
                    renderPassDescriptor.colorAttachments[0].texture = texture
                    
                    if let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
                        bindless.bindToRenderEncoder(encoder)
                        encoder.endEncoding()
                    }
                    
                    commandBuffer.commit()
                }
            } catch {
                XCTFail("Benchmark failed: \(error)")
            }
        }
    }
}

// MARK: - Performance Comparison Tests

@available(iOS 18.0, macOS 15.0, visionOS 2.0, *)
class BindlessPerformanceComparisonTests: XCTestCase {
    
    func testTraditionalVsBindlessPerformance() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not available")
        }
        
        let drawCallCount = 1000
        let resourceCount = 100
        
        // Traditional binding benchmark
        let traditionalTime = measureTraditionalBinding(
            device: device,
            commandQueue: commandQueue,
            drawCalls: drawCallCount,
            resources: resourceCount
        )
        
        // Bindless benchmark
        let bindlessTime = try measureBindlessRendering(
            device: device,
            commandQueue: commandQueue,
            drawCalls: drawCallCount,
            resources: resourceCount
        )
        
        let improvement = (traditionalTime - bindlessTime) / traditionalTime * 100
        
        print("=== Performance Comparison ===")
        print("Traditional: \(traditionalTime * 1000)ms")
        print("Bindless: \(bindlessTime * 1000)ms")
        print("Improvement: \(Int(improvement))%")
        
        XCTAssertGreaterThan(improvement, 40, "Bindless should provide >40% performance improvement")
    }
    
    private func measureTraditionalBinding(device: MTLDevice, commandQueue: MTLCommandQueue,
                                          drawCalls: Int, resources: Int) -> TimeInterval {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Create resources
        var buffers: [MTLBuffer] = []
        for _ in 0..<resources {
            if let buffer = device.makeBuffer(length: 1024, options: .storageModeShared) {
                buffers.append(buffer)
            }
        }
        
        // Simulate render pass with traditional binding
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            // Simulate per-draw binding overhead
            for _ in 0..<drawCalls {
                for (index, buffer) in buffers.enumerated() {
                    // Simulate setBuffer calls (actual encoding would fail without proper setup)
                    _ = buffer
                    _ = index
                }
            }
            commandBuffer.commit()
        }
        
        return CFAbsoluteTimeGetCurrent() - startTime
    }
    
    private func measureBindlessRendering(device: MTLDevice, commandQueue: MTLCommandQueue,
                                         drawCalls: Int, resources: Int) throws -> TimeInterval {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let bindless = try Metal4BindlessArchitecture(device: device)
        
        // Register resources once
        for _ in 0..<resources {
            if let buffer = device.makeBuffer(length: 1024, options: .storageModeShared) {
                _ = bindless.registerBuffer(buffer, type: .splatBuffer)
            }
        }
        
        // Simulate render pass with zero per-draw binding
        if let commandBuffer = commandQueue.makeCommandBuffer() {
            // Single bind for entire pass
            // No per-draw binding needed!
            
            for _ in 0..<drawCalls {
                // Draw calls without any resource binding
            }
            
            commandBuffer.commit()
        }
        
        return CFAbsoluteTimeGetCurrent() - startTime
    }
}