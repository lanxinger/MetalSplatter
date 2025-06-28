#if os(iOS) || os(macOS)

import SwiftUI
import MetalKit

/// Enhanced MetalKitSceneView with Fast SH controls
struct FastSHMetalKitSceneView: View {
    var modelIdentifier: ModelIdentifier?
    @StateObject private var fastSHSettings = FastSHSettings()
    @State private var showingSettings = false
    
    var body: some View {
        ZStack {
            MetalKitSceneViewWithSettings(
                modelIdentifier: modelIdentifier,
                fastSHSettings: fastSHSettings
            )
            
            // Settings overlay
            VStack {
                HStack {
                    Spacer()
                    
                    // Settings button
                    Button(action: { showingSettings.toggle() }) {
                        Image(systemName: "gear")
                            .foregroundColor(.white)
                            .font(.title2)
                            .padding(8)
                            .background(Color.black.opacity(0.6))
                            .clipShape(Circle())
                    }
                    .padding(.trailing)
                }
                .padding(.top, 8)
                
                Spacer()
                
                // Fast SH status indicator
                if fastSHSettings.isActive {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Image(systemName: "bolt.fill")
                                .foregroundColor(.green)
                                .font(.caption)
                            Text("Fast SH Active")
                                .font(.caption)
                                .foregroundColor(.white)
                        }
                        
                        if fastSHSettings.paletteSize > 0 {
                            Text("Palette: \(fastSHSettings.paletteSize) • Degree: \(fastSHSettings.shDegree)")
                                .font(.caption2)
                                .foregroundColor(.gray)
                        }
                        
                        if !fastSHSettings.performanceGain.isEmpty {
                            Text(fastSHSettings.performanceGain)
                                .font(.caption2)
                                .foregroundColor(.green)
                        }
                    }
                    .padding(8)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(8)
                    .padding(.bottom)
                    .padding(.leading)
                }
            }
        }
        .sheet(isPresented: $showingSettings) {
            FastSHSettingsView(settings: fastSHSettings)
        }
    }
}

/// Settings view for Fast SH configuration
struct FastSHSettingsView: View {
    @ObservedObject var settings: FastSHSettings
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Fast Spherical Harmonics")) {
                    Toggle("Enable Fast SH", isOn: $settings.enabled)
                        .help("Pre-compute spherical harmonics once per frame instead of per-splat")
                    
                    if settings.enabled {
                        Toggle("Use Texture Evaluation", isOn: $settings.useTextureEvaluation)
                            .help("Use texture-based evaluation for better edge accuracy (more GPU memory)")
                        
                        Stepper("Update Frequency: \(settings.updateFrequency)", 
                               value: $settings.updateFrequency, 
                               in: 1...10)
                            .help("Update SH evaluation every N frames (higher = better performance)")
                        
                        VStack(alignment: .leading) {
                            Text("Max Palette Size: \(settings.maxPaletteSize)")
                            Slider(value: Binding(
                                get: { Double(settings.maxPaletteSize) },
                                set: { settings.maxPaletteSize = Int($0) }
                            ), in: 1024...131072, step: 1024)
                        }
                        .help("Maximum number of unique SH coefficient sets")
                    }
                }
                
                if settings.isActive {
                    Section(header: Text("Performance Info")) {
                        HStack {
                            Text("Status")
                            Spacer()
                            Text("Active")
                                .foregroundColor(.green)
                        }
                        
                        if settings.paletteSize > 0 {
                            HStack {
                                Text("Palette Size")
                                Spacer()
                                Text("\(settings.paletteSize)")
                            }
                            
                            HStack {
                                Text("SH Degree")
                                Spacer()
                                Text("\(settings.shDegree)")
                            }
                        }
                        
                        if !settings.performanceGain.isEmpty {
                            HStack {
                                Text("Estimated Gain")
                                Spacer()
                                Text(settings.performanceGain)
                                    .foregroundColor(.green)
                            }
                        }
                    }
                }
                
                Section(header: Text("About Fast SH")) {
                    Text("Fast SH pre-computes spherical harmonics lighting once per frame using the camera direction, instead of evaluating it for each gaussian splat. This provides significant performance improvements with minimal visual quality loss.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Fast SH Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

/// Modified MetalKitSceneView that accepts external settings
private struct MetalKitSceneViewWithSettings: ViewRepresentable {
    var modelIdentifier: ModelIdentifier?
    @ObservedObject var fastSHSettings: FastSHSettings
    
    class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var renderer: MetalKitSceneRenderer?
        var fastSHSettings: FastSHSettings?
        // Store camera interaction state
        var lastPanLocation: CGPoint?
        var lastRotation: Angle = .zero
        var lastRollRotation: Float = 0.0
        var zoom: Float = 1.0
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
#if os(macOS)
    func makeNSView(context: NSViewRepresentableContext<MetalKitSceneViewWithSettings>) -> MTKView {
        makeView(context.coordinator)
    }
#elseif os(iOS)
    func makeUIView(context: UIViewRepresentableContext<MetalKitSceneViewWithSettings>) -> MTKView {
        makeView(context.coordinator)
    }
#endif
    
    private func makeView(_ coordinator: Coordinator) -> MTKView {
        let metalKitView = MTKView()
        
        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }
        
        let renderer = MetalKitSceneRenderer(metalKitView)
        coordinator.renderer = renderer
        coordinator.fastSHSettings = fastSHSettings
        metalKitView.delegate = renderer
        
        // Connect fast SH settings to renderer
        syncSettings(renderer: renderer, settings: fastSHSettings)
        
        // --- Interactivity: Pan (rotation) and Pinch (zoom) ---
        #if os(iOS)
        setupGestures(metalKitView: metalKitView, coordinator: coordinator)
        #endif
        
        Task {
            do {
                try await renderer.load(modelIdentifier)
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }
        
        return metalKitView
    }
    
    private func syncSettings(renderer: MetalKitSceneRenderer, settings: FastSHSettings) {
        renderer.fastSHSettings.enabled = settings.enabled
        renderer.fastSHSettings.useTextureEvaluation = settings.useTextureEvaluation
        renderer.fastSHSettings.updateFrequency = settings.updateFrequency
        renderer.fastSHSettings.maxPaletteSize = settings.maxPaletteSize
    }
    
    #if os(iOS)
    private func setupGestures(metalKitView: MTKView, coordinator: Coordinator) {
        let panGesture = UIPanGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePan(_:)))
        panGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(panGesture)
        
        let pinchGesture = UIPinchGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePinch(_:)))
        pinchGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(pinchGesture)
        
        let rotationGesture = UIRotationGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleRotation(_:)))
        rotationGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(rotationGesture)
        
        let doubleTapGesture = UITapGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleDoubleTap(_:)))
        doubleTapGesture.numberOfTapsRequired = 2
        doubleTapGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(doubleTapGesture)
    }
    #endif
    
#if os(macOS)
    func updateNSView(_ view: MTKView, context: NSViewRepresentableContext<MetalKitSceneViewWithSettings>) {
        updateView(context.coordinator)
    }
#elseif os(iOS)
    func updateUIView(_ view: MTKView, context: UIViewRepresentableContext<MetalKitSceneViewWithSettings>) {
        updateView(context.coordinator)
    }
#endif
    
    private func updateView(_ coordinator: Coordinator) {
        guard let renderer = coordinator.renderer else { return }
        
        // Sync settings changes
        syncSettings(renderer: renderer, settings: fastSHSettings)
        
        Task {
            do {
                try await renderer.load(modelIdentifier)
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - Gesture handling extension (reuse from original)
#if os(iOS)
extension MetalKitSceneViewWithSettings.Coordinator {
    private static let verticalRotationKey = UnsafeRawPointer(bitPattern: "verticalRotation".hashValue)!
    private static let pan2TranslationKey = UnsafeRawPointer(bitPattern: "pan2Translation".hashValue)!
    
    @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
        guard let renderer = renderer else { return }
        let location = gesture.location(in: gesture.view)
        
        if gesture.state == .ended || gesture.state == .cancelled {
            renderer.endUserInteraction()
            lastPanLocation = nil
            objc_setAssociatedObject(self, MetalKitSceneViewWithSettings.Coordinator.verticalRotationKey, nil, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            objc_setAssociatedObject(self, MetalKitSceneViewWithSettings.Coordinator.pan2TranslationKey, nil, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            return
        }
        
        switch gesture.numberOfTouches {
        case 1:
            switch gesture.state {
            case .began:
                lastPanLocation = location
                lastRotation = renderer.rotation
                let vert = renderer.verticalRotation
                objc_setAssociatedObject(self, MetalKitSceneViewWithSettings.Coordinator.verticalRotationKey, vert, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            case .changed:
                guard let lastLocation = lastPanLocation else { return }
                let deltaX = Float(location.x - lastLocation.x)
                let deltaY = Float(location.y - lastLocation.y)
                let newRotation = lastRotation + Angle(degrees: Double(deltaX) * 0.2)
                var newVertical: Float = 0
                if let vert = objc_getAssociatedObject(self, MetalKitSceneViewWithSettings.Coordinator.verticalRotationKey) as? Float {
                    newVertical = vert + deltaY * 0.01
                    newVertical = max(-.pi/2, min(.pi/2, newVertical))
                }
                renderer.setUserRotation(newRotation, vertical: newVertical)
            default:
                break
            }
        case 2:
            switch gesture.state {
            case .began:
                let initial = renderer.translation
                objc_setAssociatedObject(self, MetalKitSceneViewWithSettings.Coordinator.pan2TranslationKey, initial, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            case .changed:
                if let initial = objc_getAssociatedObject(self, MetalKitSceneViewWithSettings.Coordinator.pan2TranslationKey) as? SIMD2<Float> {
                    let translation = gesture.translation(in: gesture.view)
                    let dx = Float(translation.x) * 0.01
                    let dy = Float(translation.y) * 0.01
                    renderer.setUserTranslation(SIMD2<Float>(initial.x + dx, initial.y - dy))
                }
            default:
                break
            }
        default:
            break
        }
    }
    
    @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard let renderer = renderer else { return }
        
        if gesture.state == .ended || gesture.state == .cancelled {
            renderer.endUserInteraction()
            return
        }
        
        switch gesture.state {
        case .began:
            zoom = renderer.zoom
        case .changed:
            let newZoom = max(0.1, min(zoom * Float(gesture.scale), 20.0))
            renderer.setUserZoom(newZoom)
        default:
            break
        }
    }
    
    @objc func handleRotation(_ gesture: UIRotationGestureRecognizer) {
        guard let renderer = renderer else { return }
        
        if gesture.state == .ended || gesture.state == .cancelled {
            renderer.endUserInteraction()
            return
        }
        
        switch gesture.state {
        case .began:
            lastRollRotation = renderer.rollRotation
        case .changed:
            let newRollRotation = lastRollRotation - Float(gesture.rotation)
            renderer.setUserRollRotation(newRollRotation)
        default:
            break
        }
    }
    
    @objc func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
        guard let renderer = renderer else { return }
        renderer.resetView()
    }
    
    func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
        return true
    }
}
#endif

#endif // os(iOS) || os(macOS)