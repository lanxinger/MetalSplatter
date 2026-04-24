#if os(iOS) || os(macOS)

@preconcurrency import Metal
@preconcurrency import MetalKit
@preconcurrency import MetalSplatter
#if os(iOS) && canImport(MetalFX)
import MetalFX
#endif
import os
import SampleBoxRenderer
import simd
import SplatIO
import SwiftUI

// Fast SH is only available when MetalSplatter includes it
#if canImport(MetalSplatter)
// FastSHSettings and FastSHSplatRenderer should be available
#endif

// Helper function for linear interpolation
private func lerp<T: FloatingPoint>(_ a: T, _ b: T, _ t: T) -> T {
    return a + (b - a) * t
}

// Helper function for Angle interpolation
private func lerp(_ a: Angle, _ b: Angle, _ t: Double) -> Angle {
    .radians(lerp(a.radians, b.radians, t))
}

// Helper function for SIMD2<Float> interpolation
private func lerp(_ a: SIMD2<Float>, _ b: SIMD2<Float>, _ t: Float) -> SIMD2<Float> {
    return a + (b - a) * t
}

// Simple ease-in-out easing function
private func smoothStep<T: FloatingPoint>(_ t: T) -> T {
    return t * t * (3 - 2 * t)
}

@MainActor
class MetalKitSceneRenderer: NSObject, MTKViewDelegate {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.metalsplatter.sampleapp",
               category: "MetalKitSceneRenderer")

    let metalKitView: MTKView
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    private let commandBufferManager: CommandBufferManager

    var model: ModelIdentifier?
    var modelRenderer: (any ModelRenderer)?
    private var splatEditor: SplatEditor?
    private var currentLoadID: UUID?
    private var metal4BindlessAttemptedLoadID: UUID?
    private var meshShaderCapabilityCheckedLoadID: UUID?
    
    // Track last logged model to avoid spam
    private var lastLoggedModel: String?
    
    // Fast SH Support
    var fastSHSettings = FastSHSettings()
    
    // Metal 4 Bindless Support
    var useMetal4Bindless: Bool = true // Default to enabled
    
    // Optional frame-time hook for UI-level adaptive quality controls.
    var onFrameTimeUpdate: ((TimeInterval) -> Void)? {
        didSet {
            updateFrameTimeBridge()
        }
    }

    private struct ProfilingConfiguration {
        let logInterval: Int

        static var current: ProfilingConfiguration {
            let arguments = CommandLine.arguments
            if let intervalIndex = arguments.firstIndex(of: "--profile-log-interval"),
               arguments.indices.contains(arguments.index(after: intervalIndex)),
               let interval = Int(arguments[arguments.index(after: intervalIndex)]) {
                return ProfilingConfiguration(logInterval: max(0, interval))
            }

            let environment = ProcessInfo.processInfo.environment
            let interval = Int(environment["METALSPLATTER_PROFILE_LOG_INTERVAL"] ?? "") ?? 0
            return ProfilingConfiguration(logInterval: max(0, interval))
        }

        var isEnabled: Bool { logInterval > 0 }
    }

    private struct InteractionBenchmarkConfiguration {
        let targetZoom: Float
        let interactionWarmupFrames: Int
        let interactionSampleFrames: Int
        let settledWarmupFrames: Int
        let settledSampleFrames: Int
        let outputPath: String?

        static var current: InteractionBenchmarkConfiguration? {
            let arguments = CommandLine.arguments

            func value(after flag: String) -> String? {
                guard let index = arguments.firstIndex(of: flag),
                      arguments.indices.contains(arguments.index(after: index)) else {
                    return nil
                }
                return arguments[arguments.index(after: index)]
            }

            guard let zoomValue = value(after: "--interaction-benchmark-zoom"),
                  let targetZoom = Float(zoomValue) else {
                return nil
            }

            let interactionWarmupFrames = Int(value(after: "--interaction-benchmark-warmup-frames") ?? "") ?? 15
            let interactionSampleFrames = Int(value(after: "--interaction-benchmark-sample-frames") ?? "") ?? 45
            let settledWarmupFrames = Int(value(after: "--interaction-benchmark-settle-warmup-frames") ?? "") ?? 20
            let settledSampleFrames = Int(value(after: "--interaction-benchmark-settle-sample-frames") ?? "") ?? 45
            let defaultOutputPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
                .first?
                .appendingPathComponent("interaction-benchmark.txt")
                .path

            return InteractionBenchmarkConfiguration(
                targetZoom: max(0.1, targetZoom),
                interactionWarmupFrames: max(0, interactionWarmupFrames),
                interactionSampleFrames: max(1, interactionSampleFrames),
                settledWarmupFrames: max(0, settledWarmupFrames),
                settledSampleFrames: max(1, settledSampleFrames),
                outputPath: value(after: "--interaction-benchmark-output") ?? defaultOutputPath
            )
        }
    }

    private struct InteractionBenchmarkAccumulator {
        var frameCount = 0
        var totalFrameTime: TimeInterval = 0
        var totalSortTime: TimeInterval = 0
        var lastSplatCount = 0

        mutating func record(_ stats: SplatRenderer.FrameStatistics) {
            frameCount += 1
            totalFrameTime += stats.frameTime
            totalSortTime += stats.sortDuration ?? 0
            lastSplatCount = stats.splatCount
        }

        func result() -> InteractionBenchmarkResult {
            let sampleCount = max(frameCount, 1)
            return InteractionBenchmarkResult(
                sampleCount: frameCount,
                averageFrameMs: totalFrameTime / Double(sampleCount) * 1000,
                averageSortMs: totalSortTime / Double(sampleCount) * 1000,
                splatCount: lastSplatCount
            )
        }
    }

    private struct InteractionBenchmarkResult {
        let sampleCount: Int
        let averageFrameMs: Double
        let averageSortMs: Double
        let splatCount: Int
    }

    private enum InteractionBenchmarkPhase {
        case interactionWarmup(remainingFrames: Int)
        case interactionMeasure(InteractionBenchmarkAccumulator)
        case settledWarmup(remainingFrames: Int, interactionResult: InteractionBenchmarkResult)
        case settledMeasure(InteractionBenchmarkAccumulator, interactionResult: InteractionBenchmarkResult)
        case completed
    }

    private let profilingConfiguration = ProfilingConfiguration.current
    private let interactionBenchmarkConfiguration = InteractionBenchmarkConfiguration.current
    private var profiledFrameSampleCount = 0
    private var profiledFrameTimeAccumulator: TimeInterval = 0
    private var interactionBenchmarkPhase: InteractionBenchmarkPhase?

    let inFlightSemaphore = DispatchSemaphore(value: Constants.maxSimultaneousRenders)

    var lastRotationUpdateTimestamp: Date? = nil
    var rotation: Angle = .zero
    var zoom: Float = 1.0
    private var userIsInteracting = false
    // Add vertical rotation (pitch)
    var verticalRotation: Float = 0.0
    // Add roll rotation (2-finger twist)
    var rollRotation: Float = 0.0
    // Add translation for panning
    var translation: SIMD2<Float> = .zero
    private var modelScale: Float = 1.0
    private var autoFitEnabled: Bool = true

    var drawableSize: CGSize = .zero
    
#if os(iOS)
    // iOS: render splats at a fixed internal resolution, then upscale with MetalFX spatial scaler.
    private var internalRenderScale: CGFloat = 0.66
    private var internalColorTexture: MTLTexture?
    private var internalDepthTexture: MTLTexture?
    private var upscaledOutputTexture: MTLTexture?
    private var internalRenderSize: CGSize = .zero
#if canImport(MetalFX)
    private var metalFXScaler: (any MTLFXSpatialScaler)?
    private var metalFXOutputSize: CGSize = .zero
#endif
#endif

    // Animation State for Reset
    private var isAnimatingReset: Bool = false
    private var animationStartTime: Date? = nil
    private let animationDuration: TimeInterval = 0.3 // seconds
    private var startRotation: Angle = .zero
    private var startVerticalRotation: Float = 0.0
    private var startRollRotation: Float = 0.0
    private var startZoom: Float = 1.0
    private var startTranslation: SIMD2<Float> = .zero
    private var splatAnimationTemplate: SplatAnimationConfiguration?
    private var splatAnimationTimeOffset: Float = 0
    private var splatAnimationReferenceTime: CFTimeInterval?
    private var splatAnimationPlaying = true

    private nonisolated static func prepareGaussianSplatRenderer(
        device: MTLDevice,
        colorFormat: MTLPixelFormat,
        depthFormat: MTLPixelFormat,
        sampleCount: Int,
        points: [SplatScenePoint],
        maxSimultaneousRenders: Int,
        useFastSH: Bool,
        fastSHEnabled: Bool,
        fastSHMaxPaletteSize: Int,
        fastSHDirectionEpsilon: Float
    ) async throws -> SplatRenderer {
        try await Task.detached(priority: .userInitiated) {
            let renderer: SplatRenderer
            if useFastSH {
                let fastRenderer = try FastSHSplatRenderer(
                    device: device,
                    colorFormat: colorFormat,
                    depthFormat: depthFormat,
                    sampleCount: sampleCount,
                    maxViewCount: 1,
                    maxSimultaneousRenders: maxSimultaneousRenders
                )
                fastRenderer.fastSHConfig.enabled = fastSHEnabled
                fastRenderer.fastSHConfig.maxPaletteSize = fastSHMaxPaletteSize
                fastRenderer.shDirectionEpsilon = fastSHDirectionEpsilon
                try await fastRenderer.loadSplatsWithSH(points)
                renderer = fastRenderer
            } else {
                let standardRenderer = try SplatRenderer(
                    device: device,
                    colorFormat: colorFormat,
                    depthFormat: depthFormat,
                    sampleCount: sampleCount,
                    maxViewCount: 1,
                    maxSimultaneousRenders: maxSimultaneousRenders
                )
                try standardRenderer.add(points)
                renderer = standardRenderer
            }

            renderer.prewarmRenderPipelines()
            return renderer
        }.value
    }

    init?(_ metalKitView: MTKView) {
        guard let device = metalKitView.device else { return nil }
        self.device = device
        guard let queue = self.device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
        self.commandBufferManager = CommandBufferManager(commandQueue: queue)
        self.metalKitView = metalKitView
        metalKitView.colorPixelFormat = MTLPixelFormat.bgra8Unorm_srgb
        // Enable depth buffer for dithered transparency support
        // Dithered transparency uses stochastic alpha testing + depth for order-independence
        metalKitView.depthStencilPixelFormat = MTLPixelFormat.depth32Float
        metalKitView.sampleCount = 1
        metalKitView.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    }

    func load(_ model: ModelIdentifier?) async throws {
        guard model != self.model else { return }
        self.model = model

        // Generate unique load ID to detect stale loads from rapid model switches
        let loadID = UUID()
        currentLoadID = loadID
        metal4BindlessAttemptedLoadID = nil
        meshShaderCapabilityCheckedLoadID = nil
        interactionBenchmarkPhase = nil

        modelRenderer = nil
        splatEditor = nil

        switch model {
        case .gaussianSplat(let url):
            // Get cached model data
            let cachedModel = try await ModelCache.shared.getModel(.gaussianSplat(url))

            // Check if another load started while we were awaiting
            guard currentLoadID == loadID else {
                Self.log.info("Model load cancelled - another load started")
                return
            }
            
            // Capture needed values from main actor context
            let deviceRef = device
            let colorPixelFormat = metalKitView.colorPixelFormat
            let depthStencilPixelFormat = metalKitView.depthStencilPixelFormat
            let sampleCount = metalKitView.sampleCount
            let maxSimultaneousRenders = Constants.maxSimultaneousRenders

            // Auto-detect SH data in the loaded model — use FastSH when present
            let hasSHData = cachedModel.points.contains { point in
                if case .sphericalHarmonic(let coeffs) = point.color, coeffs.count > 1 {
                    return true
                }
                return false
            }
            let useFastSH = hasSHData || fastSHSettings.enabled
            
            // Create and load splat entirely in nonisolated context
            let points = cachedModel.points // Explicit copy for isolation
            let splat = try await Self.prepareGaussianSplatRenderer(
                device: deviceRef,
                colorFormat: colorPixelFormat,
                depthFormat: depthStencilPixelFormat,
                sampleCount: sampleCount,
                points: points,
                maxSimultaneousRenders: maxSimultaneousRenders,
                useFastSH: useFastSH,
                fastSHEnabled: fastSHSettings.enabled,
                fastSHMaxPaletteSize: fastSHSettings.maxPaletteSize,
                fastSHDirectionEpsilon: fastSHSettings.shDirectionEpsilon
            )

            if cachedModel.renderMode == .mip {
                splat.renderMode = .mip
                Self.log.info("Applied Brush MIP render mode")
            }

            // Check again after renderer creation - another load may have started
            guard currentLoadID == loadID else {
                Self.log.info("Model load cancelled after renderer creation - another load started")
                return
            }

            modelRenderer = splat
            do {
                splatEditor = try await SplatEditor(points: points, renderer: splat)
            } catch {
                splatEditor = nil
                Self.log.warning("Failed to initialize splat editor: \(error.localizedDescription)")
            }
            
            updateFrameTimeBridge()

            // Initialize Metal 4 bindless resources if available and enabled
            if useMetal4Bindless && isMetal4BindlessAvailable {
                if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
                    do {
                        try splat.initializeMetal4Bindless()
                        metal4BindlessAttemptedLoadID = loadID
                        Self.log.info("Initialized Metal 4 bindless resources for Gaussian Splat model")
                    } catch {
                        Self.log.warning("Failed to initialize Metal 4 bindless resources: \(error.localizedDescription)")
                        // Continue with traditional rendering
                    }
                }
            }

            // Configure Fast SH if using FastSHSplatRenderer
            if let fastRenderer = splat as? FastSHSplatRenderer {
                // Analyze model and update settings
                let splatCount = fastRenderer.splatCount
                let uniqueShSets = fastRenderer.shCoefficients.count
                let shDegree = fastRenderer.shDegree

                await MainActor.run {
                    fastSHSettings.analyzeAndConfigure(
                        splatCount: splatCount,
                        uniqueShSets: uniqueShSets,
                        shDegree: shDegree
                    )
                }

                applyFastSHSettings(to: fastRenderer)

                print("Fast SH configured for \(url.lastPathComponent): enabled=\(fastSHSettings.enabled), palette=\(uniqueShSets), degree=\(shDegree)")
            }
            
            if autoFitEnabled {
                await optimizeViewportForModel(splat)
            }
            beginInteractionBenchmarkIfNeeded()
        case .sampleBox:
            do {
                modelRenderer = try SampleBoxRenderer(device: device,
                                                     colorFormat: metalKitView.colorPixelFormat,
                                                     depthFormat: metalKitView.depthStencilPixelFormat,
                                                     sampleCount: metalKitView.sampleCount,
                                                     maxViewCount: 1,
                                                     maxSimultaneousRenders: Constants.maxSimultaneousRenders)
            } catch {
                Self.log.error("Failed to create SampleBoxRenderer: \(error)")
                return
            }
        case .none:
            break
        }
    }

    func currentEditorSnapshot() async -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        return await splatEditor.snapshot()
    }

    func selectEditableSplats(query: SplatSelectionQuery,
                              mode: SelectionCombineMode,
                              renderSize: CGSize) async throws -> SplatEditorSnapshot? {
        guard let splatEditor, modelRenderer is SplatRenderer else { return nil }
        let viewport = makeSplatViewport(for: renderSize)
        try await splatEditor.select(query, mode: mode, viewport: viewport)
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func selectEditableSplats(plane: SplatCutPlane,
                              side: SplatCutPlaneSide,
                              mode: SelectionCombineMode) async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.select(plane: plane, side: side, mode: mode)
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func selectEditableSplats(query: SplatSelectionQuery,
                              mode: SelectionCombineMode) async throws -> SplatEditorSnapshot? {
        try await selectEditableSplats(query: query, mode: mode, renderSize: metalKitView.bounds.size)
    }

    func pickEditablePoint(screenPoint: CGPoint,
                           renderSize: CGSize) async throws -> SplatScenePoint? {
        guard let splatEditor, modelRenderer is SplatRenderer else { return nil }
        let viewport = makeSplatViewport(for: renderSize)
        let normalized = SIMD2<Float>(
            Float(min(max(screenPoint.x / max(renderSize.width, 1), 0), 1)),
            Float(min(max(screenPoint.y / max(renderSize.height, 1), 0), 1))
        )
        return try await splatEditor.pickPoint(
            normalized: normalized,
            radius: 0.04,
            viewport: viewport
        )
    }

    func selectEditableFlood(screenPoint: CGPoint,
                             threshold: Float,
                             mode: SelectionCombineMode,
                             renderSize: CGSize) async throws -> SplatEditorSnapshot? {
        guard let splatEditor, modelRenderer is SplatRenderer else { return nil }
        let viewport = makeSplatViewport(for: renderSize)
        let normalized = SIMD2<Float>(
            Float(min(max(screenPoint.x / max(renderSize.width, 1), 0), 1)),
            Float(min(max(screenPoint.y / max(renderSize.height, 1), 0), 1))
        )
        try await splatEditor.selectFloodFill(
            normalized: normalized,
            threshold: threshold,
            mode: mode,
            viewport: viewport
        )
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func selectEditableColorMatch(screenPoint: CGPoint,
                                  threshold: Float,
                                  mode: SelectionCombineMode,
                                  renderSize: CGSize) async throws -> SplatEditorSnapshot? {
        guard let splatEditor, modelRenderer is SplatRenderer else { return nil }
        let viewport = makeSplatViewport(for: renderSize)
        let normalized = SIMD2<Float>(
            Float(min(max(screenPoint.x / max(renderSize.width, 1), 0), 1)),
            Float(min(max(screenPoint.y / max(renderSize.height, 1), 0), 1))
        )
        try await splatEditor.selectColorMatch(
            normalized: normalized,
            threshold: threshold,
            mode: mode,
            viewport: viewport
        )
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func hideSelectedEditableSplats() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.hideSelection()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func lockSelectedEditableSplats() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.lockSelection()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func selectAllEditableSplats() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.selectAll()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func clearEditableSelection() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.clearSelection()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func invertEditableSelection() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.invertSelection()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func unhideAllEditableSplats() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.unhideAll()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func unlockAllEditableSplats() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.unlockAll()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func restoreDeletedEditableSplats() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.restoreDeleted()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func deleteSelectedEditableSplats() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.deleteSelection()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func cutEditableSplats(plane: SplatCutPlane,
                           side: SplatCutPlaneSide) async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.cut(plane: plane, side: side)
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func duplicateSelectedEditableSplats() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.duplicateSelection()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func separateSelectedEditableSplats() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.separateSelection()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func undoEditableChange() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.undo()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func redoEditableChange() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.redo()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func beginEditableTransformIfPossible() async -> Bool {
        guard let splatEditor else { return false }
        guard let pivot = await currentSelectionPivot() else { return false }
        await splatEditor.beginPreviewTransform(pivot: pivot)
        requestRedraw()
        return true
    }

    func updateEditableTranslation(screenDelta: CGPoint, renderSize: CGSize) async throws {
        guard let splatEditor, let pivot = await currentSelectionPivot() else { return }
        let translation = cameraPlaneTranslation(for: screenDelta, around: pivot, renderSize: renderSize)
        try await splatEditor.updatePreviewTransform(
            SplatEditTransform(translation: translation)
        )
        requestRedraw()
    }

    func updateEditableScale(_ factor: Float) async throws {
        guard let splatEditor else { return }
        let clampedFactor = max(0.05, factor)
        try await splatEditor.updatePreviewTransform(
            SplatEditTransform(scale: SIMD3<Float>(repeating: clampedFactor))
        )
        requestRedraw()
    }

    func updateEditableRotation(angle: Float, renderSize: CGSize) async throws {
        guard let splatEditor else { return }
        let axis = cameraForwardAxis(renderSize: renderSize)
        let rotation = simd_quatf(angle: -angle, axis: axis)
        try await splatEditor.updatePreviewTransform(
            SplatEditTransform(rotation: rotation)
        )
        requestRedraw()
    }

    func commitEditableTransform() async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.commitPreviewTransform()
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func currentEditableAlignmentBounds() async -> SplatSelectionBounds? {
        guard let splatEditor else { return nil }
        return await splatEditor.alignmentBounds()
    }

    func applyEditableTransform(_ transform: SplatEditTransform,
                                pivot: SIMD3<Float>) async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.applyTransform(transform, pivot: pivot)
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func applyEditableAlignmentTransform(_ transform: SplatEditTransform,
                                         pivot: SIMD3<Float>) async throws -> SplatEditorSnapshot? {
        guard let splatEditor else { return nil }
        try await splatEditor.applyAlignmentTransform(transform, pivot: pivot)
        requestRedraw()
        return await splatEditor.snapshot()
    }

    func cancelEditableTransform() async {
        guard let splatEditor else { return }
        await splatEditor.cancelPreviewTransform()
        requestRedraw()
    }

    func exportEditedScene() async throws -> URL? {
        guard let splatEditor else { return nil }
        let points = try await splatEditor.exportVisiblePoints()
        let baseName = model?.description
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: "-")
            .lowercased() ?? "edited-splats"
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(baseName)-edited-\(UUID().uuidString.prefix(8))")
            .appendingPathExtension("ply")
        let writer = try SplatPLYSceneWriter(toFileAtPath: outputURL.path, append: false)
        try writer.start(binary: true, pointCount: points.count)
        try writer.write(points)
        try writer.close()
        return outputURL
    }

    private func makeViewport(for renderSize: CGSize) -> ModelRendererViewportDescriptor {
        // Guard against zero height to prevent NaN/inf in projection matrix
        let safeAspectRatio: Float = renderSize.height > 0
            ? Float(renderSize.width / renderSize.height)
            : 1.0
        let projectionMatrix = matrix_perspective_right_hand(fovyRadians: Float(Constants.fovy.radians) / zoom,
                                                             aspectRatio: safeAspectRatio,
                                                             nearZ: 0.1,
                                                             farZ: 100.0)

        let rotationMatrix = matrix4x4_rotation(radians: Float(rotation.radians),
                                                axis: Constants.rotationAxis)
        // Add vertical rotation (pitch) around X axis
        let verticalMatrix = matrix4x4_rotation(radians: verticalRotation, axis: SIMD3<Float>(1, 0, 0))
        // Add roll rotation (2-finger twist) around Z axis
        let rollMatrix = matrix4x4_rotation(radians: rollRotation, axis: SIMD3<Float>(0, 0, 1))
        // Add translation for panning
        let panMatrix = matrix4x4_translation(translation.x, translation.y, 0)
        let scaleMatrix = matrix4x4_scale(modelScale, modelScale, modelScale)
        let translationMatrix = matrix4x4_translation(0.0, 0.0, Constants.modelCenterZ)
        // Coordinate system calibration based on file format
        // SOG coordinate system: x=right, y=up, z=back (−z is forward)
        //
        // Camera position difference from web/Plinth viewers:
        // - Web/Plinth: camera at -Z looking toward origin (after framing)
        // - MetalSplatter: camera at origin looking toward -Z (model at z=-8)
        //
        // Since the camera orientations differ by 180° around Y, we need:
        // - Web/Plinth use 180° Z rotation
        // - MetalSplatter needs 180° Z + 180° Y = 180° X rotation
        // This produces the same visual result from the user's perspective.
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
        
        // Log coordinate calibration decision only when model changes
        if lastLoggedModel != modelDescription {
            lastLoggedModel = modelDescription
            print("MetalKitSceneRenderer: model='\(modelDescription)', isSOGS=\(isSOGS), isSOGSv2=\(isSOGSv2), isSPZ=\(isSPZ), rotation=\(isSOGSv2 ? "180°X" : (isSOGS || isSPZ ? "none" : "180°Z"))")
        }

        let viewport = MTLViewport(originX: 0, originY: 0, width: renderSize.width, height: renderSize.height, znear: 0, zfar: 1)

        return ModelRendererViewportDescriptor(viewport: viewport,
                                               projectionMatrix: projectionMatrix,
                                               viewMatrix: translationMatrix * panMatrix * rotationMatrix * verticalMatrix * rollMatrix * scaleMatrix * commonUpCalibration,
                                               screenSize: SIMD2(x: Int(renderSize.width), y: Int(renderSize.height)))
    }

    private func makeSplatViewport(for renderSize: CGSize) -> SplatRenderer.ViewportDescriptor {
        let viewport = makeViewport(for: renderSize)
        return SplatRenderer.ViewportDescriptor(
            viewport: viewport.viewport,
            projectionMatrix: viewport.projectionMatrix,
            viewMatrix: viewport.viewMatrix,
            screenSize: viewport.screenSize
        )
    }

    func projectEditableGuidePoint(_ worldPoint: SIMD3<Float>,
                                   renderSize: CGSize) -> CGPoint? {
        let viewport = makeSplatViewport(for: renderSize)
        let viewSpace = viewport.viewMatrix * SIMD4<Float>(worldPoint.x, worldPoint.y, worldPoint.z, 1)
        guard viewSpace.z < -0.05 else { return nil }

        let clip = viewport.projectionMatrix * viewSpace
        guard abs(clip.w) > .ulpOfOne else { return nil }

        let ndc = clip / clip.w
        let safeWidth = max(renderSize.width, 1)
        let safeHeight = max(renderSize.height, 1)

        return CGPoint(
            x: CGFloat((ndc.x + 1) * 0.5) * safeWidth,
            y: CGFloat(1 - ((ndc.y + 1) * 0.5)) * safeHeight
        )
    }

    private func currentSelectionPivot() async -> SIMD3<Float>? {
        guard let snapshot = await currentEditorSnapshot(),
              let bounds = snapshot.selectionBounds else {
            return nil
        }
        return bounds.center
    }

    private func cameraPlaneTranslation(for screenDelta: CGPoint,
                                        around worldPoint: SIMD3<Float>,
                                        renderSize: CGSize) -> SIMD3<Float> {
        let safeWidth = max(renderSize.width, 1)
        let safeHeight = max(renderSize.height, 1)
        let viewport = makeSplatViewport(for: renderSize)
        let viewSpace = viewport.viewMatrix * SIMD4<Float>(worldPoint.x, worldPoint.y, worldPoint.z, 1)
        let depth = max(abs(viewSpace.z), 0.1)
        let aspectRatio = Float(safeWidth / safeHeight)
        let fovY = Float(Constants.fovy.radians) / zoom
        let visibleHeight = 2 * tan(fovY * 0.5) * depth
        let visibleWidth = visibleHeight * aspectRatio

        let dx = Float(screenDelta.x / safeWidth) * visibleWidth
        let dy = Float(screenDelta.y / safeHeight) * visibleHeight

        let inverseView = viewport.viewMatrix.inverse
        let right = simd_normalize(SIMD3<Float>(inverseView.columns.0.x, inverseView.columns.0.y, inverseView.columns.0.z))
        let up = simd_normalize(SIMD3<Float>(inverseView.columns.1.x, inverseView.columns.1.y, inverseView.columns.1.z))
        return (right * dx) - (up * dy)
    }

    private func cameraForwardAxis(renderSize: CGSize) -> SIMD3<Float> {
        let inverseView = makeSplatViewport(for: renderSize).viewMatrix.inverse
        let forward = -SIMD3<Float>(inverseView.columns.2.x, inverseView.columns.2.y, inverseView.columns.2.z)
        if simd_length_squared(forward) <= .ulpOfOne {
            return SIMD3<Float>(0, 0, -1)
        }
        return simd_normalize(forward)
    }

    private func requestRedraw() {
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }

    private func applyFastSHSettings(to fastRenderer: FastSHSplatRenderer) {
        fastRenderer.fastSHConfig.enabled = fastSHSettings.enabled
        fastRenderer.fastSHConfig.maxPaletteSize = fastSHSettings.maxPaletteSize
        fastRenderer.shDirectionEpsilon = fastSHSettings.shDirectionEpsilon
    }

    func syncFastSHSettings() {
        guard let fastRenderer = modelRenderer as? FastSHSplatRenderer else { return }
        applyFastSHSettings(to: fastRenderer)
        requestRedraw()
    }

    private func updateRotation() {
        guard !userIsInteracting else { return }
        guard Constants.rotationPerSecond.degrees != 0 else { return }
        let now = Date()
        defer {
            lastRotationUpdateTimestamp = now
        }

        guard let lastRotationUpdateTimestamp else { return }
        rotation += Constants.rotationPerSecond * now.timeIntervalSince(lastRotationUpdateTimestamp)
    }

    func draw(in view: MTKView) {
        guard let modelRenderer else { return }

        syncSplatAnimation()

        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)

        guard let drawable = view.currentDrawable else {
            inFlightSemaphore.signal()
            return
        }

        guard let commandBuffer = commandBufferManager.makeCommandBuffer() else {
            inFlightSemaphore.signal()
            return
        }

        let semaphore = inFlightSemaphore
        commandBuffer.addCompletedHandler { @Sendable (_ commandBuffer)-> Swift.Void in
            semaphore.signal()
        }

        // --- Handle Reset Animation ---
        if isAnimatingReset, let startTime = animationStartTime {
            let timeElapsed = Date().timeIntervalSince(startTime)
            let progress = min(timeElapsed / animationDuration, 1.0)
            let t = smoothStep(Float(progress)) // Eased progress

            // Interpolate view properties
            rotation = lerp(startRotation, .zero, Double(t))
            verticalRotation = lerp(startVerticalRotation, 0.0, t)
            rollRotation = lerp(startRollRotation, 0.0, t)
            zoom = lerp(startZoom, 1.0, t)
            translation = lerp(startTranslation, .zero, t)

            if progress >= 1.0 {
                // Animation finished
                isAnimatingReset = false
                animationStartTime = nil
                userIsInteracting = false // Allow auto-rotate again
                lastRotationUpdateTimestamp = nil // Reset timestamp for smooth auto-rotate start
            } else {
                // Request next frame
                #if os(macOS)
                metalKitView.setNeedsDisplay(metalKitView.bounds)
                #else
                metalKitView.setNeedsDisplay()
                #endif
            }
        } else {
            if isAnimatingReset {
                // Recover from inconsistent reset state without stranding an in-flight slot.
                isAnimatingReset = false
                animationStartTime = nil
                userIsInteracting = false
                lastRotationUpdateTimestamp = nil
            }
            // Only update auto-rotation if not animating reset and not interacting
            updateRotation()
        }
        // --- End Animation Handling ---

        #if os(iOS) && canImport(MetalFX)
        if #available(iOS 16.0, *),
           internalRenderScale < 0.999,
           prepareMetalFXResourcesIfNeeded(view: view, drawable: drawable) {
            let internalViewport = makeViewport(for: internalRenderSize)
            renderTraditional(modelRenderer: modelRenderer,
                              viewport: internalViewport,
                              colorTexture: internalColorTexture,
                              depthTexture: internalDepthTexture,
                              commandBuffer: commandBuffer)
            encodeMetalFXUpscaleIfPossible(drawable: drawable, commandBuffer: commandBuffer)
        } else {
            let outputViewport = makeViewport(for: CGSize(width: drawable.texture.width, height: drawable.texture.height))
            renderTraditional(modelRenderer: modelRenderer,
                              viewport: outputViewport,
                              colorTexture: drawable.texture,
                              depthTexture: view.depthStencilTexture,
                              commandBuffer: commandBuffer)
        }
        #elseif os(iOS)
        let outputViewport = makeViewport(for: CGSize(width: drawable.texture.width, height: drawable.texture.height))
        renderTraditional(modelRenderer: modelRenderer,
                          viewport: outputViewport,
                          colorTexture: drawable.texture,
                          depthTexture: view.depthStencilTexture,
                          commandBuffer: commandBuffer)
        #else
        let outputViewport = makeViewport(for: CGSize(width: drawable.texture.width, height: drawable.texture.height))
        renderTraditional(modelRenderer: modelRenderer,
                          viewport: outputViewport,
                          colorTexture: drawable.texture,
                          depthTexture: view.depthStencilTexture,
                          commandBuffer: commandBuffer)
        #endif

        commandBuffer.present(drawable)

        commandBuffer.commit()

        if splatAnimationTemplate != nil && splatAnimationPlaying {
            requestRedraw()
        }
    }
    
    // MARK: - Render Methods
    
    /// Traditional full-resolution rendering
    private func renderTraditional(modelRenderer: any ModelRenderer,
                                   viewport: ModelRendererViewportDescriptor,
                                   colorTexture: MTLTexture?,
                                   depthTexture: MTLTexture?,
                                   commandBuffer: MTLCommandBuffer) {
        guard let colorTexture else { return }
        do {
            try modelRenderer.render(viewports: [viewport],
                                   colorTexture: colorTexture,
                                   colorStoreAction: .store,
                                   depthTexture: depthTexture,
                                   depthStoreAction: .dontCare,
                                   rasterizationRateMap: nil,
                                   renderTargetArrayLength: 0,
                                   to: commandBuffer)
        } catch {
            Self.log.error("Unable to render scene: \(error.localizedDescription)")
        }
    }

    private func updateFrameTimeBridge() {
        guard let splat = modelRenderer as? SplatRenderer else { return }
        let frameTimeCallback = onFrameTimeUpdate
        let interactionBenchmarkActive: Bool
        if interactionBenchmarkConfiguration == nil {
            interactionBenchmarkActive = false
        } else if case .completed = interactionBenchmarkPhase {
            interactionBenchmarkActive = false
        } else {
            interactionBenchmarkActive = true
        }
        guard frameTimeCallback != nil
                || profilingConfiguration.isEnabled
                || interactionBenchmarkActive else {
            splat.onFrameReady = nil
            return
        }

        let loadID = currentLoadID
        profiledFrameSampleCount = 0
        profiledFrameTimeAccumulator = 0
        splat.onFrameReady = { [weak self] stats in
            Task { @MainActor [weak self] in
                guard let self else { return }
                guard self.currentLoadID == loadID else { return }
                frameTimeCallback?(stats.frameTime)
                self.recordProfilingSample(stats)
            }
        }
    }

    private func recordProfilingSample(_ stats: SplatRenderer.FrameStatistics) {
        recordInteractionBenchmarkSample(stats)

        guard profilingConfiguration.isEnabled else { return }

        profiledFrameSampleCount += 1
        profiledFrameTimeAccumulator += stats.frameTime

        guard profiledFrameSampleCount >= profilingConfiguration.logInterval else { return }

        let averageFrameTime = profiledFrameTimeAccumulator / Double(profiledFrameSampleCount)
        let averageFPS = averageFrameTime > 0 ? 1.0 / averageFrameTime : 0
        let averageFrameMs = averageFrameTime * 1000
        let sortMs = (stats.sortDuration ?? 0) * 1000

        Self.log.info(
            "PROFILE avgFrameMs=\(String(format: "%.2f", averageFrameMs)), fps=\(String(format: "%.1f", averageFPS)), splats=\(stats.splatCount), sortMs=\(String(format: "%.2f", sortMs)), uploads=\(stats.bufferUploadCount), sortJobs=\(stats.sortJobsInFlight)"
        )

        profiledFrameSampleCount = 0
        profiledFrameTimeAccumulator = 0
    }

    private func beginInteractionBenchmarkIfNeeded() {
        guard let configuration = interactionBenchmarkConfiguration else { return }
        guard modelRenderer is SplatRenderer else { return }
        guard interactionBenchmarkPhase == nil else { return }

        interactionBenchmarkPhase = .interactionWarmup(remainingFrames: configuration.interactionWarmupFrames)
        resetInteractionBenchmarkOutputIfNeeded()
        emitInteractionBenchmarkLog(
            "BENCHMARK start zoom=\(String(format: "%.2f", configuration.targetZoom)) interactionWarmupFrames=\(configuration.interactionWarmupFrames) interactionSampleFrames=\(configuration.interactionSampleFrames) settledWarmupFrames=\(configuration.settledWarmupFrames) settledSampleFrames=\(configuration.settledSampleFrames)"
        )
        setUserZoom(configuration.targetZoom)
        requestRedraw()
    }

    private func recordInteractionBenchmarkSample(_ stats: SplatRenderer.FrameStatistics) {
        guard let configuration = interactionBenchmarkConfiguration,
              let phase = interactionBenchmarkPhase else { return }

        switch phase {
        case .interactionWarmup(let remainingFrames):
            if remainingFrames > 0 {
                interactionBenchmarkPhase = .interactionWarmup(remainingFrames: remainingFrames - 1)
                return
            }

            interactionBenchmarkPhase = .interactionMeasure(InteractionBenchmarkAccumulator())
            emitInteractionBenchmarkLog("BENCHMARK phase=interaction status=measuring")

        case .interactionMeasure(var accumulator):
            accumulator.record(stats)
            if accumulator.frameCount < configuration.interactionSampleFrames {
                interactionBenchmarkPhase = .interactionMeasure(accumulator)
                return
            }

            let interactionResult = accumulator.result()
            emitInteractionBenchmarkLog(
                "BENCHMARK phase=interaction avgFrameMs=\(String(format: "%.2f", interactionResult.averageFrameMs)) avgSortMs=\(String(format: "%.2f", interactionResult.averageSortMs)) samples=\(interactionResult.sampleCount) splats=\(interactionResult.splatCount)"
            )
            endUserInteraction()
            lastRotationUpdateTimestamp = nil
            interactionBenchmarkPhase = .settledWarmup(
                remainingFrames: configuration.settledWarmupFrames,
                interactionResult: interactionResult
            )
            requestRedraw()

        case .settledWarmup(let remainingFrames, let interactionResult):
            if remainingFrames > 0 {
                interactionBenchmarkPhase = .settledWarmup(
                    remainingFrames: remainingFrames - 1,
                    interactionResult: interactionResult
                )
                return
            }

            interactionBenchmarkPhase = .settledMeasure(
                InteractionBenchmarkAccumulator(),
                interactionResult: interactionResult
            )
            emitInteractionBenchmarkLog("BENCHMARK phase=settled status=measuring")

        case .settledMeasure(var accumulator, let interactionResult):
            accumulator.record(stats)
            if accumulator.frameCount < configuration.settledSampleFrames {
                interactionBenchmarkPhase = .settledMeasure(accumulator, interactionResult: interactionResult)
                return
            }

            let settledResult = accumulator.result()
            let savingsMs = settledResult.averageFrameMs - interactionResult.averageFrameMs
            let savingsPercent = settledResult.averageFrameMs > 0
                ? savingsMs / settledResult.averageFrameMs * 100
                : 0
            emitInteractionBenchmarkLog(
                "BENCHMARK phase=settled avgFrameMs=\(String(format: "%.2f", settledResult.averageFrameMs)) avgSortMs=\(String(format: "%.2f", settledResult.averageSortMs)) samples=\(settledResult.sampleCount) splats=\(settledResult.splatCount)"
            )
            emitInteractionBenchmarkLog(
                "BENCHMARK summary zoom=\(String(format: "%.2f", configuration.targetZoom)) interactionAvgFrameMs=\(String(format: "%.2f", interactionResult.averageFrameMs)) settledAvgFrameMs=\(String(format: "%.2f", settledResult.averageFrameMs)) frameTimeSavingsMs=\(String(format: "%.2f", savingsMs)) interactionFasterPercent=\(String(format: "%.1f", savingsPercent)) interactionAvgSortMs=\(String(format: "%.2f", interactionResult.averageSortMs)) settledAvgSortMs=\(String(format: "%.2f", settledResult.averageSortMs))"
            )
            interactionBenchmarkPhase = .completed
            updateFrameTimeBridge()

        case .completed:
            return
        }
    }

    private func resetInteractionBenchmarkOutputIfNeeded() {
        guard let outputPath = interactionBenchmarkConfiguration?.outputPath else { return }
        try? "".write(toFile: outputPath, atomically: true, encoding: .utf8)
    }

    private func emitInteractionBenchmarkLog(_ message: String) {
        print(message)

        guard let outputPath = interactionBenchmarkConfiguration?.outputPath else { return }
        let outputURL = URL(fileURLWithPath: outputPath)
        let line = "\(message)\n"

        if FileManager.default.fileExists(atPath: outputURL.path),
           let handle = try? FileHandle(forWritingTo: outputURL) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(line.utf8))
            return
        }

        try? line.write(to: outputURL, atomically: true, encoding: .utf8)
    }

#if os(iOS) && canImport(MetalFX)
    @available(iOS 16.0, *)
    private func prepareMetalFXResourcesIfNeeded(view: MTKView, drawable: CAMetalDrawable) -> Bool {
        let outputWidth = drawable.texture.width
        let outputHeight = drawable.texture.height
        guard outputWidth > 0, outputHeight > 0 else { return false }
        
        let scale = max(0.55, min(internalRenderScale, 1.0))
        let inputWidth = max(1, Int((CGFloat(outputWidth) * scale).rounded(.down)))
        let inputHeight = max(1, Int((CGFloat(outputHeight) * scale).rounded(.down)))
        let newInternalSize = CGSize(width: inputWidth, height: inputHeight)
        let newOutputSize = CGSize(width: outputWidth, height: outputHeight)
        
        let needsTextureRebuild = internalColorTexture == nil
            || Int(internalRenderSize.width) != inputWidth
            || Int(internalRenderSize.height) != inputHeight
            || internalColorTexture?.pixelFormat != view.colorPixelFormat
            || internalDepthTexture?.pixelFormat != view.depthStencilPixelFormat
        
        if needsTextureRebuild {
            let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: view.colorPixelFormat,
                                                                           width: inputWidth,
                                                                           height: inputHeight,
                                                                           mipmapped: false)
            colorDescriptor.storageMode = .private
            colorDescriptor.usage = [.renderTarget, .shaderRead]
            internalColorTexture = device.makeTexture(descriptor: colorDescriptor)
            
            if view.depthStencilPixelFormat != .invalid {
                let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: view.depthStencilPixelFormat,
                                                                               width: inputWidth,
                                                                               height: inputHeight,
                                                                               mipmapped: false)
                depthDescriptor.storageMode = .private
                depthDescriptor.usage = [.renderTarget]
                internalDepthTexture = device.makeTexture(descriptor: depthDescriptor)
            } else {
                internalDepthTexture = nil
            }
            
            internalRenderSize = newInternalSize
        }
        
        let needsScalerRebuild = metalFXScaler == nil
            || Int(metalFXOutputSize.width) != outputWidth
            || Int(metalFXOutputSize.height) != outputHeight
            || needsTextureRebuild
        
        if needsScalerRebuild {
            let descriptor = MTLFXSpatialScalerDescriptor()
            descriptor.colorTextureFormat = view.colorPixelFormat
            descriptor.outputTextureFormat = drawable.texture.pixelFormat
            descriptor.inputWidth = inputWidth
            descriptor.inputHeight = inputHeight
            descriptor.outputWidth = outputWidth
            descriptor.outputHeight = outputHeight
            descriptor.colorProcessingMode = .perceptual
            metalFXScaler = descriptor.makeSpatialScaler(device: device)
            metalFXOutputSize = newOutputSize
        }
        
        let needsUpscaledOutputRebuild = upscaledOutputTexture == nil
            || upscaledOutputTexture?.width != outputWidth
            || upscaledOutputTexture?.height != outputHeight
            || upscaledOutputTexture?.pixelFormat != drawable.texture.pixelFormat
        
        if needsUpscaledOutputRebuild {
            let outputDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: drawable.texture.pixelFormat,
                                                                            width: outputWidth,
                                                                            height: outputHeight,
                                                                            mipmapped: false)
            outputDescriptor.storageMode = .private
            outputDescriptor.usage = [.renderTarget, .shaderWrite, .shaderRead]
            upscaledOutputTexture = device.makeTexture(descriptor: outputDescriptor)
        }
        
        return internalColorTexture != nil && metalFXScaler != nil && upscaledOutputTexture != nil
    }
    
    @available(iOS 16.0, *)
    private func encodeMetalFXUpscaleIfPossible(drawable: CAMetalDrawable, commandBuffer: MTLCommandBuffer) {
        guard let scaler = metalFXScaler,
              let inputTexture = internalColorTexture,
              let outputTexture = upscaledOutputTexture else { return }
        
        scaler.colorTexture = inputTexture
        scaler.inputContentWidth = inputTexture.width
        scaler.inputContentHeight = inputTexture.height
        scaler.outputTexture = outputTexture
        scaler.encode(commandBuffer: commandBuffer)
        
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.label = "MetalFX Upscale Copy"
            let copySize = MTLSize(width: min(outputTexture.width, drawable.texture.width),
                                   height: min(outputTexture.height, drawable.texture.height),
                                   depth: 1)
            blit.copy(from: outputTexture,
                      sourceSlice: 0,
                      sourceLevel: 0,
                      sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                      sourceSize: copySize,
                      to: drawable.texture,
                      destinationSlice: 0,
                      destinationLevel: 0,
                      destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
            blit.endEncoding()
        }
    }
#endif

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        drawableSize = size
    }

    // MARK: - Metal 4 Configuration
    
    /// Enable or disable Metal 4 bindless rendering
    func setMetal4Bindless(_ enabled: Bool) {
        let previousValue = useMetal4Bindless
        useMetal4Bindless = enabled
        let loadID = currentLoadID

        guard let splat = modelRenderer as? SplatRenderer else {
            guard previousValue != enabled else { return }
            requestRedraw()
            return
        }

        guard enabled else {
            guard previousValue != enabled else { return }
            requestRedraw()
            return
        }

        guard isMetal4BindlessAvailable else {
            if metal4BindlessAttemptedLoadID != loadID || previousValue != enabled {
                metal4BindlessAttemptedLoadID = loadID
                Self.log.warning("Failed to enable Metal 4 bindless: Metal rendering is not available on this device")
            }
            return
        }

        let alreadyInitialized = splat.metal4ArgumentBufferManager != nil
        if alreadyInitialized {
            guard previousValue != enabled else { return }
            requestRedraw()
            return
        }

        guard metal4BindlessAttemptedLoadID != loadID || previousValue != enabled else { return }
        metal4BindlessAttemptedLoadID = loadID

        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            do {
                try splat.initializeMetal4Bindless()
                Self.log.info("Enabled Metal 4 bindless resources for current model")
                requestRedraw()
            } catch {
                Self.log.warning("Failed to enable Metal 4 bindless: \(error.localizedDescription)")
            }
        }
    }

    /// Check if Metal 4 bindless is available on this device
    var isMetal4BindlessAvailable: Bool {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            return device.supportsFamily(.apple9) // Requires Apple 9 GPU family
        }
        return false
    }
    
    /// Enable or disable debug AABB visualization
    /// This tests the GPU SIMD-group parallel bounds computation
    func setDebugAABB(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            let wasEnabled = splat.debugOptions.contains(.showAABB)
            guard wasEnabled != enabled else { return }
            if enabled {
                splat.debugOptions.insert(.showAABB)
            } else {
                splat.debugOptions.remove(.showAABB)
            }
            Self.log.info("Debug AABB \(enabled ? "enabled" : "disabled")")
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    
    /// Enable or disable GPU frustum culling
    /// When enabled, splats outside the camera's view frustum are filtered out before rendering
    func setFrustumCulling(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            guard splat.frustumCullingEnabled != enabled else { return }
            splat.frustumCullingEnabled = enabled
            Self.log.info("Frustum culling \(enabled ? "enabled" : "disabled")")
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    
    /// Enable or disable mesh shader rendering (Metal 3+)
    /// When enabled, geometry is generated entirely on GPU - 1 computation per splat instead of 4
    func setMeshShader(_ enabled: Bool) {
        guard let splat = modelRenderer as? SplatRenderer else { return }

        if enabled && !splat.isMeshShaderSupported {
            if meshShaderCapabilityCheckedLoadID != currentLoadID {
                meshShaderCapabilityCheckedLoadID = currentLoadID
                Self.log.info("Mesh shaders not supported on this device")
            }
            return
        }

        guard splat.meshShaderEnabled != enabled else { return }
        meshShaderCapabilityCheckedLoadID = currentLoadID
        splat.meshShaderEnabled = enabled
        if enabled {
            Self.log.info("Mesh shader rendering enabled - geometry generated on GPU")
        } else {
            Self.log.info("Mesh shader rendering disabled - using vertex shader path")
        }
        requestRedraw()
    }
    
    /// Enable or disable Metal 4 TensorOps batch precompute
    /// Pre-computes covariance/transforms for all splats when camera changes
    /// Best for large scenes (50k+ splats) where camera movement is intermittent
    func setBatchPrecompute(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            guard splat.batchPrecomputeEnabled != enabled else { return }
            splat.batchPrecomputeEnabled = enabled
            if enabled {
                Self.log.info("Metal 4 TensorOps batch precompute enabled")
            } else {
                Self.log.info("Metal 4 TensorOps batch precompute disabled")
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }

    /// Enable or disable dithered (stochastic) transparency
    /// When enabled, uses order-independent transparency via stochastic alpha testing
    /// Best paired with TAA for noise reduction - eliminates need for depth sorting
    func setDitheredTransparency(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            guard splat.useDitheredTransparency != enabled else { return }
            splat.useDitheredTransparency = enabled
            // Dithered transparency requires single-stage pipeline, which is only used when
            // highQualityDepth is false. Multi-stage pipeline (used with highQualityDepth=true)
            // bypasses the dithered path entirely.
            splat.highQualityDepth = !enabled
            if enabled {
                Self.log.info("Dithered transparency enabled - order-independent, no sorting needed")
            } else {
                Self.log.info("Dithered transparency disabled - using sorted alpha blending")
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }

    /// Enable or disable spherical harmonics (SH) evaluation
    /// When disabled, only base color is used - significant performance gain but no view-dependent lighting
    func setSHRendering(_ enabled: Bool) {
        if let fastSH = modelRenderer as? FastSHSplatRenderer {
            guard fastSH.shRenderingEnabled != enabled else { return }
            fastSH.shRenderingEnabled = enabled
            if enabled {
                Self.log.info("SH rendering enabled - view-dependent lighting active")
            } else {
                Self.log.info("SH rendering disabled - using base color only (~50% faster)")
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }

    /// Enable or disable the experimental Metal 4 GPU radix sorter.
    /// Counting sort remains the default because it is faster for typical splat scenes.
    func setMetal4Sorting(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            guard splat.useMetal4Sorting != enabled else { return }
            splat.useMetal4Sorting = enabled
            if enabled {
                Self.log.info("Experimental Metal 4 GPU radix sorting enabled")
            } else {
                Self.log.info("Metal 4 GPU radix sorting disabled - using counting sort/MPS")
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }


#if os(iOS)
    /// Set fixed internal render scale used by MetalFX spatial upscaling.
    func setInternalRenderScale(_ scale: CGFloat) {
        let clamped = max(0.55, min(scale, 1.0))
        guard abs(clamped - internalRenderScale) > 0.001 else { return }
        internalRenderScale = clamped
        internalColorTexture = nil
        internalDepthTexture = nil
        upscaledOutputTexture = nil
#if canImport(MetalFX)
        metalFXScaler = nil
        metalFXOutputSize = .zero
#endif
        internalRenderSize = .zero
        metalKitView.setNeedsDisplay()
    }
#endif

    /// Set sorting mode: auto, radial (distance-based), or linear (view-direction-based)
    /// - auto: Automatically selects based on camera motion (rotation vs translation)
    /// - radial: Better for rotation-heavy camera movement (turntable, 360°)
    /// - linear: Better for translation-heavy movement (walking through scene)
    func setSortingMode(_ mode: SplatRenderer.SortingMode) {
        if let splat = modelRenderer as? SplatRenderer {
            guard splat.sortingMode != mode else { return }
            splat.sortingMode = mode
            switch mode {
            case .auto:
                Self.log.info("Sorting mode: auto (adapts to camera motion)")
            case .radial:
                Self.log.info("Sorting mode: radial (distance-based, best for rotation)")
            case .linear:
                Self.log.info("Sorting mode: linear (view-direction, best for translation)")
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }

    /// Enable or disable 2DGS rendering mode
    /// 2DGS uses proper oriented elliptical splats with normal extraction from covariance.
    /// Best for scenes trained with 2D Gaussian Splatting (flat/planar splats).
    func set2DGSMode(_ enabled: Bool) {
        if let splat = modelRenderer as? SplatRenderer {
            guard splat.use2DGSMode != enabled else { return }
            splat.use2DGSMode = enabled
            if enabled {
                Self.log.info("2DGS mode enabled - oriented splats with normal extraction")
            } else {
                Self.log.info("2DGS mode disabled - standard 3DGS rendering")
            }
        }

        // Request redraw
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }

    func setSplatAnimation(effect: SplatAnimationEffect?,
                           isPlaying: Bool,
                           speed: Float,
                           intensity: Float) {
        let clampedSpeed = max(speed, 0.05)
        let clampedIntensity = max(intensity, 0)
        let previousSpeed = max(splatAnimationTemplate?.speed ?? clampedSpeed, 0.05)
        let currentPhaseTime = currentSplatAnimationTime() * previousSpeed
        let previousPlaying = splatAnimationPlaying
        let previousTemplate = splatAnimationTemplate

        splatAnimationTimeOffset = currentPhaseTime / clampedSpeed
        splatAnimationReferenceTime = CACurrentMediaTime()
        splatAnimationPlaying = isPlaying

        guard let effect else {
            splatAnimationTemplate = nil
            if let splat = modelRenderer as? SplatRenderer {
                if splat.animationConfiguration != nil {
                    splat.animationConfiguration = nil
                }
            }
            if previousTemplate != nil || previousPlaying != isPlaying {
                requestRedraw()
            }
            return
        }

        if splatAnimationTemplate == nil {
            splatAnimationTimeOffset = 0
        }

        var template = splatAnimationTemplate ?? SplatAnimationConfiguration(effect: effect)
        template.effect = effect
        template.speed = clampedSpeed
        template.intensity = clampedIntensity
        splatAnimationTemplate = template

        if previousPlaying != isPlaying && isPlaying {
            splatAnimationReferenceTime = CACurrentMediaTime()
        }

        syncSplatAnimation()
        requestRedraw()
    }

    func resetSplatAnimation() {
        guard splatAnimationTemplate != nil else { return }
        splatAnimationTimeOffset = 0
        splatAnimationReferenceTime = CACurrentMediaTime()
        syncSplatAnimation()
        requestRedraw()
    }

    // MARK: - User Interaction API
    #if os(iOS) || os(macOS)
    
    /// Notify renderer that user interaction has started (for adaptive sort quality)
    private func notifyInteractionBegan() {
        if let splat = modelRenderer as? SplatRenderer {
            splat.beginInteraction()
        }
    }
    
    /// Notify renderer that user interaction has ended (triggers quality sort)
    private func notifyInteractionEnded() {
        if let splat = modelRenderer as? SplatRenderer {
            splat.endInteraction()
        }
    }
    
    func setUserRotation(_ newRotation: Angle, vertical: Float) {
        if !userIsInteracting {
            notifyInteractionBegan()
        }
        userIsInteracting = true
        rotation = newRotation
        verticalRotation = vertical
        lastRotationUpdateTimestamp = nil // Reset timestamp for smooth resumption of auto-rotation
    }

    func setUserZoom(_ newZoom: Float) {
        if !userIsInteracting {
            notifyInteractionBegan()
        }
        userIsInteracting = true
        zoom = newZoom
        lastRotationUpdateTimestamp = nil
    }

    func setUserRollRotation(_ rollRotation: Float) {
        if !userIsInteracting {
            notifyInteractionBegan()
        }
        userIsInteracting = true
        self.rollRotation = rollRotation
        lastRotationUpdateTimestamp = nil
    }

    func setUserTranslation(_ translation: SIMD2<Float>) {
        if !userIsInteracting {
            notifyInteractionBegan()
        }
        userIsInteracting = true
        self.translation = translation
        lastRotationUpdateTimestamp = nil
    }

    func endUserInteraction() {
        if userIsInteracting {
            notifyInteractionEnded()
        }
        userIsInteracting = false
        lastRotationUpdateTimestamp = Date()
    }
    
    /// Reset view to default state with smooth animation
    func resetView() {
        // If already animating, do nothing
        guard !isAnimatingReset else { return }
        
        // Store starting values for animation
        startRotation = rotation
        startVerticalRotation = verticalRotation
        startRollRotation = rollRotation
        startZoom = zoom
        startTranslation = translation
        
        // Start animation
        isAnimatingReset = true
        animationStartTime = Date()
        userIsInteracting = true // Prevent auto-rotate during animation
        
        // Trigger first animation frame
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    #endif

    /// Optimizes the viewport (scale, position) to fit the loaded model
    private func optimizeViewportForModel(_ renderer: any ModelRenderer) async {
        // Calculate model bounds
        guard let bounds = await calculateModelBounds(renderer) else {
            return
        }
        
        // Calculate model dimensions
        let size = bounds.max - bounds.min
        let maxDimension = max(size.x, size.y, size.z)
        
        // Adaptive scaling strategy based on model size
        let newScale: Float
        
        if maxDimension > 0 {
            // Only scale up small models - never scale down anything
            if maxDimension < 0.5 {
                // Very tiny models - scale up significantly
                let targetSize: Float = 3.0
                newScale = targetSize / maxDimension
                modelScale = max(1.0, min(newScale, 25.0))
            } else if maxDimension < 2.0 {
                // Small models - scale up moderately  
                let targetSize: Float = 4.0
                newScale = targetSize / maxDimension
                modelScale = max(1.0, min(newScale, 8.0))
            } else {
                // Everything else (>= 2 units) - leave completely unchanged
                modelScale = 1.0
            }
        } else {
            modelScale = 1.0
        }
        
        // Trigger redraw with new scale
        #if os(macOS)
        metalKitView.setNeedsDisplay(metalKitView.bounds)
        #else
        metalKitView.setNeedsDisplay()
        #endif
    }
    
    private func calculateModelBounds(_ renderer: any ModelRenderer) async -> (min: SIMD3<Float>, max: SIMD3<Float>)? {
        if let splatRenderer = renderer as? SplatRenderer {
            return splatRenderer.calculateBounds()
        } else if let fastSHRenderer = renderer as? FastSHSplatRenderer {
            return fastSHRenderer.calculateBounds()
        }
        
        return nil
    }

    private func currentSplatAnimationTime() -> Float {
        guard splatAnimationTemplate != nil else { return 0 }
        guard splatAnimationPlaying, let referenceTime = splatAnimationReferenceTime else {
            return splatAnimationTimeOffset
        }
        return splatAnimationTimeOffset + Float(CACurrentMediaTime() - referenceTime)
    }

    private func syncSplatAnimation() {
        guard let splat = modelRenderer as? SplatRenderer else { return }
        guard var configuration = splatAnimationTemplate else {
            if splat.animationConfiguration != nil {
                splat.animationConfiguration = nil
            }
            return
        }

        if splatAnimationReferenceTime == nil, splatAnimationPlaying {
            splatAnimationReferenceTime = CACurrentMediaTime()
        }
        configuration.time = currentSplatAnimationTime()
        if splat.animationConfiguration != configuration {
            splat.animationConfiguration = configuration
        }
    }
}

#endif // os(iOS) || os(macOS)
