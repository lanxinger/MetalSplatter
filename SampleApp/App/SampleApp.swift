#if os(visionOS)
import CompositorServices
#endif
import SwiftUI

@main
struct SampleApp: App {
#if os(visionOS)
    @State private var visionSceneRenderer: VisionSceneRenderer?
#endif

    var body: some Scene {
        WindowGroup("MetalSplatter Sample App", id: "main") {
            ContentView()
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
