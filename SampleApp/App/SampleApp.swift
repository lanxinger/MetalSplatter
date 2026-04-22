#if os(visionOS)
import CompositorServices
#endif
import SwiftUI
import os

@main
struct SampleApp: App {
    private static let log =
        Logger(subsystem: Bundle.main.bundleIdentifier ?? "com.metalsplatter.sampleapp",
               category: "SampleApp")

    private static func bundledModelIdentifier(named resourceName: String) -> ModelIdentifier? {
        let resourceURL = URL(fileURLWithPath: resourceName)
        let baseName = resourceURL.deletingPathExtension().lastPathComponent
        let fileExtension = resourceURL.pathExtension.isEmpty ? nil : resourceURL.pathExtension

        guard let bundledURL = Bundle.main.url(forResource: baseName, withExtension: fileExtension) else {
            return nil
        }
        return .gaussianSplat(bundledURL)
    }

    private let startupModelIdentifier: ModelIdentifier? = {
        let arguments = CommandLine.arguments
        if let bundledSceneIndex = arguments.firstIndex(of: "--startup-bundled-scene"),
           arguments.indices.contains(arguments.index(after: bundledSceneIndex)) {
            let bundledSceneName = arguments[arguments.index(after: bundledSceneIndex)]
            if let bundledModel = bundledModelIdentifier(named: bundledSceneName) {
                return bundledModel
            }
        }

        if let startupScenePathIndex = arguments.firstIndex(of: "--startup-scene-path"),
           arguments.indices.contains(arguments.index(after: startupScenePathIndex)) {
            let startupPath = arguments[arguments.index(after: startupScenePathIndex)]
            let startupURL = URL(fileURLWithPath: startupPath)
            if FileManager.default.fileExists(atPath: startupURL.path) {
                return .gaussianSplat(startupURL)
            }
        }

        let environment = ProcessInfo.processInfo.environment
        if let bundledSceneName = environment["METALSPLATTER_BUNDLED_SCENE_NAME"],
           !bundledSceneName.isEmpty {
            if let bundledModel = bundledModelIdentifier(named: bundledSceneName) {
                return bundledModel
            }
        }

        if let startupPath = environment["METALSPLATTER_STARTUP_SCENE_PATH"],
           !startupPath.isEmpty {
            let startupURL = URL(fileURLWithPath: startupPath)
            if FileManager.default.fileExists(atPath: startupURL.path) {
                return .gaussianSplat(startupURL)
            }
        }

        return nil
    }()

#if os(visionOS)
    @State private var visionSceneRenderer: VisionSceneRenderer?
#endif

    var body: some Scene {
        WindowGroup("MetalSplatter Sample App", id: "main") {
            if let startupModelIdentifier {
                let description = startupModelIdentifier.description
                MetalKitSceneView(modelIdentifier: startupModelIdentifier)
                    .navigationTitle(description)
                    .task {
                        Self.log.info("Launching directly into startup scene: \(description, privacy: .public)")
                    }
            } else {
                ContentView()
            }
        }

#if os(macOS)
        WindowGroup(for: ModelIdentifier.self) { modelIdentifier in
            MetalKitSceneView(modelIdentifier: modelIdentifier.wrappedValue)
                .navigationTitle(modelIdentifier.wrappedValue?.description ?? "No Model")
        }
#endif // os(macOS)

#if os(visionOS)
        ImmersiveSpace(for: ModelIdentifier.self) { modelIdentifier in
            CompositorLayer(configuration: ContentStageConfiguration()) { layerRenderer in
                Task {
                    do {
                        if let existingRenderer = visionSceneRenderer,
                           existingRenderer.layerRenderer === layerRenderer,
                           existingRenderer.model == modelIdentifier.wrappedValue {
                            return
                        }

                        visionSceneRenderer?.stopRenderLoop()

                        let renderer = try VisionSceneRenderer(layerRenderer)
                        try await renderer.load(modelIdentifier.wrappedValue)
                        visionSceneRenderer = renderer
                        renderer.startRenderLoop()
                    } catch {
                        print("Error initializing or loading renderer: \(error.localizedDescription)")
                    }
                }
            }
            .onDisappear {
                visionSceneRenderer?.stopRenderLoop()
                visionSceneRenderer = nil
            }
        }
        .immersionStyle(selection: .constant(immersionStyle), in: immersionStyle)
#endif // os(visionOS)
    }

#if os(visionOS)
    var immersionStyle: ImmersionStyle {
        if #available(visionOS 2, *) {
            .mixed
        } else {
            .full
        }
    }
#endif // os(visionOS)
}
