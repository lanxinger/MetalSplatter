#if os(visionOS)

import CompositorServices
import Metal
import MetalSplatter
import os
import SampleBoxRenderer
import simd
import Spatial
import SwiftUI

extension LayerRenderer.Clock.Instant.Duration {
    var timeInterval: TimeInterval {
        let nanoseconds = TimeInterval(components.attoseconds / 1_000_000_000)
        return TimeInterval(components.seconds) + (nanoseconds / TimeInterval(NSEC_PER_SEC))
    }
}

enum RendererError: LocalizedError {
    case failedToCreateCommandQueue
    case failedToCreateRenderer(underlying: Error)
    
    var errorDescription: String? {
        switch self {
        case .failedToCreateCommandQueue:
            return "Failed to create Metal command queue"
        case .failedToCreateRenderer(let underlying):
            return "Failed to create renderer: \(underlying.localizedDescription)"
        }
    }
}

class VisionSceneRenderer {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.metalsplatter.sampleapp",
               category: "VisionSceneRenderer")

    let layerRenderer: LayerRenderer
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let commandBufferManager: CommandBufferManager

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?
    
    // Metal 4 Bindless Support
    var useMetal4Bindless: Bool = true // Default to enabled

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    var lastRotationUpdateTimestamp: Date? = nil
    var rotation: Angle = .zero

    let arSession: ARKitSession
    let worldTracking: WorldTrackingProvider

    init(_ layerRenderer: LayerRenderer) throws {
        self.layerRenderer = layerRenderer
        self.device = layerRenderer.device
        guard let commandQueue = self.device.makeCommandQueue() else {
            throw RendererError.failedToCreateCommandQueue
        }
        self.commandQueue = commandQueue
        self.commandBufferManager = CommandBufferManager(commandQueue: commandQueue)

        worldTracking = WorldTrackingProvider()
        arSession = ARKitSession()
    }

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        modelRenderer = nil
        switch model {
        case .gaussianSplat(let url):
            // Get cached model data
            let cachedModel = try await ModelCache.shared.getModel(.gaussianSplat(url))
            
            let splat = try SplatRenderer(device: device,
                                          colorFormat: layerRenderer.configuration.colorFormat,
                                          depthFormat: layerRenderer.configuration.depthFormat,
                                          sampleCount: 1,
                                          maxViewCount: layerRenderer.properties.viewCount,
                                          maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            try splat.add(cachedModel.points)
            modelRenderer = splat
            
            // Initialize Metal 4 bindless resources if available and enabled
            if useMetal4Bindless {
                if #available(visionOS 26.0, *) {
                    do {
                        try splat.initializeMetal4Bindless()
                        Self.log.info("Initialized Metal 4 bindless resources for Gaussian Splat model")
                    } catch {
                        Self.log.warning("Failed to initialize Metal 4 bindless resources: \(error.localizedDescription)")
                        // Continue with traditional rendering
                    }
                } else {
                    Self.log.info("Metal 4 bindless resources not available on this platform (requires visionOS 26+)")
                }
            }
        case .sampleBox:
            do {
                modelRenderer = try SampleBoxRenderer(device: device,
                                                     colorFormat: layerRenderer.configuration.colorFormat,
                                                     depthFormat: layerRenderer.configuration.depthFormat,
                                                     sampleCount: 1,
                                                     maxViewCount: layerRenderer.properties.viewCount,
                                                     maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            } catch {
                throw RendererError.failedToCreateRenderer(underlying: error)
            }
        case .none:
            break
        }
    }

    func startRenderLoop() {
        Task {
            do {
                try await arSession.run([worldTracking])
            } catch {
                fatalError("Failed to initialize ARSession")
            }

            let renderThread = Thread {
                self.renderLoop()
            }
            renderThread.name = "Render Thread"
            renderThread.start()
        }
    }

    private func viewports(drawable: LayerRenderer.Drawable, deviceAnchor: DeviceAnchor?) -> [ModelRendererViewportDescriptor] {
        let rotationMatrix = matrix4x4_rotation(radians: Float(rotation.radians),
                                                axis: Constants.rotationAxis)
        let translationMatrix = matrix4x4_translation(0.0, 0.0, Constants.modelCenterZ)
        // Coordinate system calibration based on file format
        // SOG coordinate system: x=right, y=up, z=back (−z is forward)
        //
        // For Vision Pro, we use the same model translation approach as MetalKitSceneRenderer,
        // so we need the same calibration rotations.
        let modelDescription = model?.description ?? ""
        let descriptionLowercased = modelDescription.lowercased()
        let isSOGS = descriptionLowercased.contains("meta.json") || descriptionLowercased.contains(".zip")
        let isSPZ = descriptionLowercased.contains(".spz") || descriptionLowercased.contains(".spx")
        let isSOGSv2 = descriptionLowercased.contains(".sog")
        
        // SPZ files are already correctly oriented like SOGS v1 files
        // SOGS v2 (.sog) files need 180° X rotation (equivalent to web's 180° Z with our camera setup)
        // PLY files need 180° rotation around Z axis to be right-side up
        let commonUpCalibration: simd_float4x4
        if isSOGSv2 {
            commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(1, 0, 0)) // 180° around X for SOGS v2
        } else if isSOGS || isSPZ {
            commonUpCalibration = matrix_identity_float4x4 // No rotation for SOGS v1 and SPZ
        } else {
            commonUpCalibration = matrix4x4_rotation(radians: .pi, axis: SIMD3<Float>(0, 0, 1)) // 180° around Z for PLY
        }

        let simdDeviceAnchor = deviceAnchor?.originFromAnchorTransform ?? matrix_identity_float4x4

        return drawable.views.map { view in
            let userViewpointMatrix = (simdDeviceAnchor * view.transform).inverse
            let projectionMatrix = ProjectiveTransform3D(leftTangent: Double(view.tangents[0]),
                                                         rightTangent: Double(view.tangents[1]),
                                                         topTangent: Double(view.tangents[2]),
                                                         bottomTangent: Double(view.tangents[3]),
                                                         nearZ: Double(drawable.depthRange.y),
                                                         farZ: Double(drawable.depthRange.x),
                                                         reverseZ: true)
            let screenSize = SIMD2(x: Int(view.textureMap.viewport.width),
                                   y: Int(view.textureMap.viewport.height))
            return ModelRendererViewportDescriptor(viewport: view.textureMap.viewport,
                                                   projectionMatrix: .init(projectionMatrix),
                                                   viewMatrix: userViewpointMatrix * translationMatrix * rotationMatrix * commonUpCalibration,
                                                   screenSize: screenSize)
        }
    }

    private func updateRotation() {
        guard Constants.rotationPerSecond.degrees != 0 else { return }
        
        let now = Date()
        defer {
            lastRotationUpdateTimestamp = now
        }

        guard let lastRotationUpdateTimestamp else { return }
        rotation += Constants.rotationPerSecond * now.timeIntervalSince(lastRotationUpdateTimestamp)
    }

    func renderFrame() {
        guard let frame = layerRenderer.queryNextFrame() else { return }

        frame.startUpdate()
        frame.endUpdate()

        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)

        guard let commandBuffer = commandBufferManager.makeCommandBuffer() else {
            fatalError("Failed to create command buffer")
        }

        guard let drawable = frame.queryDrawable() else { return }

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        frame.startSubmission()

        let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime).timeInterval
        let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: time)

        drawable.deviceAnchor = deviceAnchor

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        updateRotation()

        let viewports = self.viewports(drawable: drawable, deviceAnchor: deviceAnchor)

        do {
            try modelRenderer?.render(viewports: viewports,
                                      colorTexture: drawable.colorTextures[0],
                                      colorStoreAction: .store,
                                      depthTexture: drawable.depthTextures[0],
                                      rasterizationRateMap: drawable.rasterizationRateMaps.first,
                                      renderTargetArrayLength: layerRenderer.configuration.layout == .layered ? drawable.views.count : 1,
                                      to: commandBuffer)
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
        }

        drawable.encodePresent(commandBuffer: commandBuffer)

        commandBuffer.commit()

        frame.endSubmission()
    }

    func renderLoop() {
        while true {
            if layerRenderer.state == .invalidated {
                Self.log.warning("Layer is invalidated")
                return
            } else if layerRenderer.state == .paused {
                layerRenderer.waitUntilRunning()
                continue
            } else {
                autoreleasepool {
                    self.renderFrame()
                }
            }
        }
    }
    
    // MARK: - Metal 4 Configuration
    
    /// Enable or disable Metal 4 bindless rendering
    func setMetal4Bindless(_ enabled: Bool) {
        useMetal4Bindless = enabled
        
        // If we already have a renderer, try to initialize Metal 4
        if enabled, let splat = modelRenderer as? SplatRenderer {
            if #available(visionOS 26.0, *) {
                do {
                    try splat.initializeMetal4Bindless()
                    Self.log.info("Enabled Metal 4 bindless resources for current model")
                } catch {
                    Self.log.warning("Failed to enable Metal 4 bindless: \(error.localizedDescription)")
                }
            }
        }
    }
    
    /// Check if Metal 4 bindless is available on this device
    var isMetal4BindlessAvailable: Bool {
        if #available(visionOS 26.0, *) {
            return device.supportsFamily(.apple9) // Requires Apple 9 GPU family
        }
        return false
    }
}

#endif // os(visionOS)
