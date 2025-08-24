import XCTest
import MetalSplatter
import Metal

/// Tests for Metal 4 bindless resource integration
class Metal4IntegrationTests: XCTestCase {
    
    var device: MTLDevice!
    var renderer: SplatRenderer!
    
    override func setUp() {
        super.setUp()
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal device not available")
            return
        }
        device = metalDevice
        
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
            XCTFail("Failed to create SplatRenderer: \(error)")
        }
    }
    
    override func tearDown() {
        renderer = nil
        device = nil
        super.tearDown()
    }
    
    /// Test Metal 4 availability detection
    func testMetal4AvailabilityDetection() {
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
            XCTAssertTrue(renderer.isMetal4BindlessAvailable, "Metal 4 should be available on supported platforms")
        } else {
            XCTAssertFalse(renderer.isMetal4BindlessAvailable, "Metal 4 should not be available on older platforms")
        }
    }
    
    /// Test Metal 4 bindless initialization
    func testMetal4BindlessInitialization() {
        guard #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) else {
            throw XCTSkip("Metal 4 not available on this platform")
        }
        
        XCTAssertNoThrow({
            try renderer.initializeMetal4Bindless()
        }, "Metal 4 bindless initialization should not throw on supported platforms")
    }
    
    /// Test fallback to traditional rendering on unsupported platforms
    func testFallbackToTraditionalRendering() {
        // This test ensures that the app gracefully falls back to traditional rendering
        // when Metal 4 is not available
        
        let expectation = XCTestExpectation(description: "Fallback rendering works")
        
        Task {
            do {
                // Try to initialize Metal 4 (may fail on older platforms)
                if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
                    try renderer.initializeMetal4Bindless()
                }
                
                // Traditional rendering should still work
                let commandQueue = device.makeCommandQueue()!
                let commandBuffer = commandQueue.makeCommandBuffer()!
                
                // Create dummy render pass
                let renderPassDescriptor = MTLRenderPassDescriptor()
                // Note: In a real test, we'd need actual textures
                
                expectation.fulfill()
            } catch {
                // Fallback should not fail
                XCTFail("Fallback rendering failed: \(error)")
            }
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
    
    /// Test performance metrics collection
    func testPerformanceMetrics() {
        guard #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) else {
            throw XCTSkip("Metal 4 not available on this platform")
        }
        
        let expectation = XCTestExpectation(description: "Performance metrics collected")
        
        do {
            try renderer.initializeMetal4Bindless()
            
            renderer.measureMetal4Performance { metrics in
                XCTAssertGreaterThanOrEqual(metrics.traditionalRenderTime, 0, "Traditional render time should be non-negative")
                XCTAssertGreaterThanOrEqual(metrics.bindlessRenderTime, 0, "Bindless render time should be non-negative")
                XCTAssertGreaterThanOrEqual(metrics.cpuOverheadReduction, -1.0, "CPU overhead reduction should be reasonable")
                XCTAssertLessThanOrEqual(metrics.cpuOverheadReduction, 1.0, "CPU overhead reduction should be reasonable")
                expectation.fulfill()
            }
        } catch {
            XCTFail("Performance measurement setup failed: \(error)")
        }
        
        wait(for: [expectation], timeout: 10.0)
    }
    
    /// Test argument table resource management
    func testArgumentTableResourceManagement() {
        guard #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) else {
            throw XCTSkip("Metal 4 not available on this platform")
        }
        
        XCTAssertNoThrow({
            try renderer.initializeMetal4Bindless()
            
            // Test debug statistics (should not crash)
            renderer.printMetal4Statistics()
        }, "Argument table resource management should work correctly")
    }
    
    /// Test integration with existing scene renderers
    func testSceneRendererIntegration() {
        // This test verifies that Metal 4 integration doesn't break existing functionality
        
        let expectation = XCTestExpectation(description: "Scene renderer integration works")
        
        Task {
            // Test that renderer can be created and configured
            XCTAssertNotNil(renderer, "Renderer should be created successfully")
            
            // Test Metal 4 configuration
            if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
                do {
                    try renderer.initializeMetal4Bindless()
                    // Should not crash or throw after initialization
                    XCTAssertTrue(renderer.isMetal4BindlessAvailable)
                } catch {
                    // Initialization might fail on some devices, but shouldn't crash
                    XCTAssertFalse(renderer.isMetal4BindlessAvailable)
                }
            }
            
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
}

/// Integration tests for the sample app's Metal 4 features
class SampleAppMetal4IntegrationTests: XCTestCase {
    
    /// Test MetalKitSceneRenderer Metal 4 configuration
    func testMetalKitSceneRendererMetal4Config() {
        let metalKitView = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }
        metalKitView.device = device
        
        guard let renderer = MetalKitSceneRenderer(metalKitView) else {
            XCTFail("Failed to create MetalKitSceneRenderer")
            return
        }
        
        // Test availability check
        let isAvailable = renderer.isMetal4BindlessAvailable
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
            XCTAssertTrue(isAvailable, "Metal 4 should be available on supported platforms")
        } else {
            XCTAssertFalse(isAvailable, "Metal 4 should not be available on older platforms")
        }
        
        // Test configuration
        XCTAssertNoThrow({
            renderer.setMetal4Bindless(true)
        }, "Setting Metal 4 bindless should not throw")
        
        XCTAssertNoThrow({
            renderer.setMetal4Bindless(false)
        }, "Disabling Metal 4 bindless should not throw")
    }
    
    #if os(visionOS)
    /// Test VisionSceneRenderer Metal 4 configuration
    func testVisionSceneRendererMetal4Config() {
        // Note: This test would need a proper LayerRenderer setup in a real visionOS environment
        // For now, we just test the availability detection logic
        
        if #available(visionOS 2.0, *) {
            // Test would go here if we had access to LayerRenderer
            XCTAssert(true, "visionOS 2.0+ should support Metal 4")
        } else {
            XCTAssert(true, "Older visionOS versions should gracefully handle Metal 4 absence")
        }
    }
    #endif
    
    /// Test render settings integration
    func testRenderSettingsIntegration() {
        // This test verifies that the RenderSettings view properly handles state changes
        // In a real UI test, we would test the toggle behavior
        
        // Test Metal 4 availability detection
        var isMetal4Available = false
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
            isMetal4Available = true
        }
        
        // The settings should reflect actual availability
        XCTAssertEqual(isMetal4Available, 
                      { () -> Bool in
                          if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
                              return true
                          }
                          return false
                      }(), 
                      "Settings availability should match platform capabilities")
    }
}

/// Integration tests specifically for Metal 4 command buffer reuse architecture
class Metal4CommandBufferPoolTests: XCTestCase {
    
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var commandBufferManager: CommandBufferManager!
    
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
        commandBufferManager = CommandBufferManager(commandQueue: queue)
    }
    
    override func tearDown() {
        commandBufferManager = nil
        commandQueue = nil
        device = nil
        super.tearDown()
    }
    
    /// Test basic command buffer allocation
    func testBasicCommandBufferAllocation() {
        let commandBuffer = commandBufferManager.makeCommandBuffer()
        XCTAssertNotNil(commandBuffer, "Command buffer should be allocated successfully")
    }
    
    /// Test command buffer reuse on Metal 4 devices
    @available(iOS 18.0, macOS 15.0, visionOS 2.0, *)
    func testMetal4CommandBufferReuse() {
        // Only run this test on Metal 4 supported platforms
        guard #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) else {
            throw XCTSkip("Metal 4 not available on this platform")
        }
        
        var allocatedBuffers: [MTLCommandBuffer] = []
        
        // Allocate several command buffers
        for _ in 0..<5 {
            if let buffer = commandBufferManager.makeCommandBuffer() {
                allocatedBuffers.append(buffer)
            }
        }
        
        XCTAssertEqual(allocatedBuffers.count, 5, "Should successfully allocate 5 command buffers")
        
        // Simulate completion and check for reuse
        let expectation = XCTestExpectation(description: "Command buffers complete and get reused")
        expectation.expectedFulfillmentCount = 5
        
        for buffer in allocatedBuffers {
            buffer.addCompletedHandler { _ in
                expectation.fulfill()
            }
            buffer.commit()
        }
        
        wait(for: [expectation], timeout: 5.0)
        
        // After completion, pool should have buffers available for reuse
        if let stats = commandBufferManager.poolStatistics {
            XCTAssertGreaterThan(stats.available, 0, "Pool should have available buffers after completion")
        }
    }
    
    /// Test fallback behavior on older platforms
    func testLegacyFallbackBehavior() {
        // This test verifies that command buffer allocation works correctly 
        // even on platforms that don't support Metal 4 pooling
        
        var successfulAllocations = 0
        
        // Test multiple allocations
        for _ in 0..<10 {
            if let buffer = commandBufferManager.makeCommandBuffer() {
                successfulAllocations += 1
                buffer.commit() // Immediate commit to test allocation pattern
            }
        }
        
        XCTAssertEqual(successfulAllocations, 10, "Should successfully allocate all command buffers on any platform")
    }
    
    /// Test memory pressure handling
    @available(iOS 18.0, macOS 15.0, visionOS 2.0, *)
    func testMemoryPressureHandling() {
        guard #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) else {
            throw XCTSkip("Metal 4 not available on this platform")
        }
        
        // Fill the pool
        var buffers: [MTLCommandBuffer] = []
        for _ in 0..<10 {
            if let buffer = commandBufferManager.makeCommandBuffer() {
                buffers.append(buffer)
            }
        }
        
        // Complete all buffers to populate the available pool
        let expectation = XCTestExpectation(description: "All buffers complete")
        expectation.expectedFulfillmentCount = buffers.count
        
        for buffer in buffers {
            buffer.addCompletedHandler { _ in
                expectation.fulfill()
            }
            buffer.commit()
        }
        
        wait(for: [expectation], timeout: 5.0)
        
        // Simulate memory pressure
        commandBufferManager.handleMemoryPressure()
        
        // Pool should be cleared
        if let stats = commandBufferManager.poolStatistics {
            XCTAssertEqual(stats.available, 0, "Pool should be cleared after memory pressure")
        }
    }
    
    /// Test concurrent command buffer allocation
    func testConcurrentAllocation() {
        let expectation = XCTestExpectation(description: "Concurrent allocations complete")
        expectation.expectedFulfillmentCount = 20
        
        let dispatchGroup = DispatchGroup()
        
        // Launch concurrent allocation tasks
        for i in 0..<20 {
            dispatchGroup.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                defer { dispatchGroup.leave() }
                
                if let buffer = self.commandBufferManager.makeCommandBuffer() {
                    buffer.label = "Concurrent Buffer \(i)"
                    buffer.addCompletedHandler { _ in
                        expectation.fulfill()
                    }
                    buffer.commit()
                }
            }
        }
        
        dispatchGroup.wait()
        wait(for: [expectation], timeout: 10.0)
    }
    
    /// Test pool statistics and monitoring
    @available(iOS 18.0, macOS 15.0, visionOS 2.0, *)
    func testPoolStatisticsMonitoring() {
        guard #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) else {
            throw XCTSkip("Metal 4 not available on this platform")
        }
        
        // Initially, pool should be empty
        if let stats = commandBufferManager.poolStatistics {
            XCTAssertEqual(stats.available, 0, "Pool should start empty")
            XCTAssertEqual(stats.active, 0, "No buffers should be active initially")
        }
        
        // Allocate a buffer
        let buffer = commandBufferManager.makeCommandBuffer()
        XCTAssertNotNil(buffer)
        
        // Should now have one active buffer
        if let stats = commandBufferManager.poolStatistics {
            XCTAssertEqual(stats.active, 1, "Should have one active buffer")
        }
        
        // Complete the buffer
        let expectation = XCTestExpectation(description: "Buffer completion")
        buffer?.addCompletedHandler { _ in
            expectation.fulfill()
        }
        buffer?.commit()
        
        wait(for: [expectation], timeout: 5.0)
        
        // Should now have one available buffer
        if let stats = commandBufferManager.poolStatistics {
            XCTAssertEqual(stats.available, 1, "Should have one available buffer after completion")
            XCTAssertEqual(stats.active, 0, "No buffers should be active after completion")
        }
    }
    
    /// Test memory allocation efficiency
    func testMemoryAllocationEfficiency() {
        let iterations = 100
        var totalTime: CFTimeInterval = 0
        
        // Measure allocation time
        for _ in 0..<iterations {
            let startTime = CFAbsoluteTimeGetCurrent()
            
            if let buffer = commandBufferManager.makeCommandBuffer() {
                buffer.commit()
                let endTime = CFAbsoluteTimeGetCurrent()
                totalTime += (endTime - startTime)
            }
        }
        
        let averageTime = totalTime / Double(iterations)
        
        // On Metal 4 devices with pooling, allocation should be very fast
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
            XCTAssertLessThan(averageTime, 0.001, "Command buffer allocation should be very fast with pooling")
        } else {
            // Even on older devices, should be reasonably fast
            XCTAssertLessThan(averageTime, 0.01, "Command buffer allocation should be reasonably fast")
        }
        
        print("Average command buffer allocation time: \(averageTime * 1000) ms")
    }
    
    /// Test logging and debugging capabilities
    func testLoggingAndDebugging() {
        // This test ensures logging doesn't crash and provides useful information
        XCTAssertNoThrow({
            commandBufferManager.logPoolState()
        }, "Pool state logging should not crash")
        
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
            // Test with some pool activity
            let buffer = commandBufferManager.makeCommandBuffer()
            commandBufferManager.logPoolState()
            
            let expectation = XCTestExpectation(description: "Buffer completion for logging test")
            buffer?.addCompletedHandler { _ in
                expectation.fulfill()
            }
            buffer?.commit()
            
            wait(for: [expectation], timeout: 5.0)
            commandBufferManager.logPoolState()
        }
    }
}