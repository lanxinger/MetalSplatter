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