#if os(iOS) || os(macOS)

import SwiftUI
import MetalKit
#if os(iOS)
import ARKit
#endif

#if os(macOS)
private typealias ViewRepresentable = NSViewRepresentable
#elseif os(iOS)
private typealias ViewRepresentable = UIViewRepresentable
#endif

struct MetalKitSceneView: View {
    var modelIdentifier: ModelIdentifier?
    @State private var showARUnavailableAlert = false
    @State private var navigateToAR = false
    @State private var showSettings = false
    @State private var fastSHEnabled = true
    @State private var metal4BindlessEnabled = true // Default to enabled
    @State private var showDebugAABB = false // Debug: visualize GPU-computed bounds
    @State private var frustumCullingEnabled = false // GPU frustum culling
    @State private var meshShaderEnabled = true // Metal 3+ mesh shader rendering - enabled by default
    @State private var ditheredTransparencyEnabled = false // Stochastic transparency - disabled by default
    
    var body: some View {
        ZStack {
            // The actual Metal view
            MetalKitRendererView(
                modelIdentifier: modelIdentifier,
                fastSHEnabled: $fastSHEnabled,
                metal4BindlessEnabled: $metal4BindlessEnabled,
                showDebugAABB: $showDebugAABB,
                frustumCullingEnabled: $frustumCullingEnabled,
                meshShaderEnabled: $meshShaderEnabled,
                ditheredTransparencyEnabled: $ditheredTransparencyEnabled
            )
            .ignoresSafeArea()
            
            // Control overlay
            VStack {
                // Settings button at top-right
                HStack {
                    Spacer()
                    Button(action: {
                        showSettings.toggle()
                    }) {
                        Image(systemName: "gearshape.fill")
                            .font(.title2)
                            .foregroundColor(.white)
                            .frame(width: 44, height: 44)
                            .background(Color.black.opacity(0.5))
                            .clipShape(Circle())
                    }
                    .padding()
                }
                
                Spacer()
                
                // AR toggle button overlay (iOS only)
                #if os(iOS)
                HStack {
                    Spacer()
                    
                    Button(action: {
                        toggleARMode()
                    }) {
                        VStack(spacing: 4) {
                            Image(systemName: "arkit")
                                .font(.title2)
                                .foregroundColor(.white)
                            Text("AR")
                                .font(.caption2)
                                .foregroundColor(.white)
                        }
                        .frame(width: 60, height: 60)
                        .background(Color.blue.opacity(0.8))
                        .clipShape(Circle())
                        .shadow(radius: 5)
                    }
                    .padding(.trailing, 20)
                    .padding(.bottom, 20)
                }
                #endif
            }
            
            // Settings overlay
            if showSettings {
                ZStack {
                    // Invisible background to capture taps
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture {
                            showSettings = false
                        }
                    
                    VStack {
                        HStack {
                        VStack(alignment: .leading, spacing: 12) {
                            // Header with close button
                            HStack {
                                Text("Render Settings")
                                    .font(.headline)
                                
                                Spacer()
                                
                                Button(action: {
                                    showSettings = false
                                }) {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.title3)
                                        .foregroundColor(.secondary)
                                }
                            }
                            .padding(.bottom, 4)
                            
                            // Fast SH Toggle
                            Toggle(isOn: $fastSHEnabled) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Fast Spherical Harmonics")
                                        .font(.subheadline)
                                    Text("Optimized SH evaluation for better performance")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            Divider()
                            
                            // Debug AABB Toggle - tests GPU SIMD-group bounds computation
                            Toggle(isOn: $showDebugAABB) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Show Bounding Box")
                                        .font(.subheadline)
                                    Text("Visualize AABB computed by GPU SIMD-group reduction")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            Divider()
                            
                            // Frustum Culling Toggle
                            Toggle(isOn: $frustumCullingEnabled) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Frustum Culling")
                                        .font(.subheadline)
                                    Text("GPU pre-filters splats outside camera view")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }

                            Divider()

                            // Dithered Transparency Toggle
                            Toggle(isOn: $ditheredTransparencyEnabled) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Dithered Transparency")
                                        .font(.subheadline)
                                    Text("Order-independent, no sorting needed (best with TAA)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }

                            Divider()

                            // Mesh Shader Toggle (Metal 3+)
                            Toggle(isOn: $meshShaderEnabled) {
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack {
                                        Text("Mesh Shaders")
                                            .font(.subheadline)
                                        Text("(Metal 3+)")
                                            .font(.caption2)
                                            .foregroundColor(.blue)
                                    }
                                    Text("Generate geometry on GPU - compute once per splat")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            Divider()
                            
                            // Metal 4 Bindless Toggle
                            Toggle(isOn: $metal4BindlessEnabled) {
                                VStack(alignment: .leading, spacing: 2) {
                                    HStack {
                                        Text("Metal 4 Bindless Resources")
                                            .font(.subheadline)
                                        Text("(Full Feature Set)")
                                            .font(.caption2)
                                            .foregroundColor(.green)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Color.blue.opacity(0.1))
                                            .cornerRadius(4)
                                    }
                                    Text("50-80% CPU overhead reduction for large scenes (iOS 26+)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            if metal4BindlessEnabled {
                                VStack(alignment: .leading, spacing: 4) {
                                    Label("Metal 4 Active", systemImage: "checkmark.circle.fill")
                                        .font(.caption)
                                        .foregroundColor(.green)
                                    Text("• Argument tables enabled\n• Residency sets active\n• Bindless rendering mode")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                        .padding(.leading, 20)
                                }
                                .padding(.top, 4)
                            }
                            
                        }
                        .padding()
#if os(iOS)
                        .background(Color(UIColor.systemBackground).opacity(0.9))
#else
                        .background(Color(NSColor.controlBackgroundColor).opacity(0.9))
#endif
                        .cornerRadius(10)
                        .shadow(radius: 5)
                        .frame(maxWidth: 400)
                        .onTapGesture {
                            // Prevent tap-through to background
                        }
                        
                        Spacer()
                    }
                    .padding()
                    
                    Spacer()
                }
                }
                .transition(.opacity)
            }
        }
#if os(iOS)
        .navigationDestination(isPresented: $navigateToAR) {
            ARContentView(model: modelIdentifier)
                .navigationTitle("AR \(modelIdentifier?.description ?? "View")")
        }
#endif
        .alert("AR Not Available", isPresented: $showARUnavailableAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("AR features are not available on this device or require iOS 17.0+")
        }
    }
    
    private func toggleARMode() {
        #if os(iOS)
        // Check if AR is available
        guard ARWorldTrackingConfiguration.isSupported else {
            showARUnavailableAlert = true
            return
        }
        
        // Navigate to AR view with current model
        navigateToAR = true
        #endif
    }
}

struct MetalKitRendererView: ViewRepresentable {
    var modelIdentifier: ModelIdentifier?
    @Binding var fastSHEnabled: Bool
    @Binding var metal4BindlessEnabled: Bool
    @Binding var showDebugAABB: Bool
    @Binding var frustumCullingEnabled: Bool
    @Binding var meshShaderEnabled: Bool
    @Binding var ditheredTransparencyEnabled: Bool

    class Coordinator: NSObject {
        var renderer: MetalKitSceneRenderer?
        var parent: MetalKitRendererView
        // Store camera interaction state
        var lastPanLocation: CGPoint?
        var lastRotation: Angle = .zero
        var lastRollRotation: Float = 0.0
        var zoom: Float = 1.0
        
        init(_ parent: MetalKitRendererView) {
            self.parent = parent
        }
        
        @MainActor
        func updateSettings() {
            // Update renderer settings silently
            renderer?.fastSHSettings.enabled = parent.fastSHEnabled
            renderer?.setMetal4Bindless(parent.metal4BindlessEnabled)
            renderer?.setDebugAABB(parent.showDebugAABB)
            renderer?.setFrustumCulling(parent.frustumCullingEnabled)
            renderer?.setMeshShader(parent.meshShaderEnabled)
            renderer?.setDitheredTransparency(parent.ditheredTransparencyEnabled)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

#if os(macOS)
    func makeNSView(context: NSViewRepresentableContext<MetalKitRendererView>) -> MTKView {
        makeView(context.coordinator)
    }
    
    func updateNSView(_ view: MTKView, context: NSViewRepresentableContext<MetalKitRendererView>) {
        updateView(context.coordinator)
    }
#elseif os(iOS)
    func makeUIView(context: UIViewRepresentableContext<MetalKitRendererView>) -> MTKView {
        makeView(context.coordinator)
    }
    
    func updateUIView(_ view: MTKView, context: UIViewRepresentableContext<MetalKitRendererView>) {
        updateView(context.coordinator)
    }
#endif

    private func makeView(_ coordinator: Coordinator) -> MTKView {
        let metalKitView = MTKView()

        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }

        let renderer = MetalKitSceneRenderer(metalKitView)
        coordinator.renderer = renderer
        metalKitView.delegate = renderer
        
        // Apply initial settings
        renderer?.fastSHSettings.enabled = fastSHEnabled
        renderer?.setMetal4Bindless(metal4BindlessEnabled)
        renderer?.setDebugAABB(showDebugAABB)
        renderer?.setFrustumCulling(frustumCullingEnabled)
        renderer?.setMeshShader(meshShaderEnabled)
        renderer?.setDitheredTransparency(ditheredTransparencyEnabled)

        // --- Interactivity: Pan (rotation) and Pinch (zoom) ---
        #if os(iOS)
        let panGesture = UIPanGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePan(_:)))
        panGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(panGesture)
        let pinchGesture = UIPinchGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePinch(_:)))
        pinchGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(pinchGesture)
        // 2-finger rotation (roll around Z-axis)
        let rotationGesture = UIRotationGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleRotation(_:)))
        rotationGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(rotationGesture)
        // Double-tap to reset view
        let doubleTapGesture = UITapGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleDoubleTap(_:)))
        doubleTapGesture.numberOfTapsRequired = 2
        doubleTapGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(doubleTapGesture)
        #elseif os(macOS)
        // Pan gesture for rotation (drag)
        let panGesture = NSPanGestureRecognizer(target: coordinator, action: #selector(Coordinator.handlePan(_:)))
        panGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(panGesture)
        // Magnification gesture for zoom (pinch on trackpad)
        let magnifyGesture = NSMagnificationGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleMagnify(_:)))
        magnifyGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(magnifyGesture)
        // Rotation gesture for roll (two-finger rotate on trackpad)
        let rotationGesture = NSRotationGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleRotation(_:)))
        rotationGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(rotationGesture)
        #endif

        Task {
            do {
                try await renderer?.load(modelIdentifier)
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }

        return metalKitView
    }

    private func updateView(_ coordinator: Coordinator) {
        guard let renderer = coordinator.renderer else { return }
        
        // Update coordinator's parent reference to get latest state
        coordinator.parent = self
        
        // Update settings when the view updates
        coordinator.updateSettings()
        
        Task {
            do {
                try await renderer.load(modelIdentifier)
            } catch {
                print("Error loading model: \(error.localizedDescription)")
            }
        }
    }
}

#if os(iOS)
    // MARK: - Coordinator Gesture Handling
    extension MetalKitRendererView.Coordinator: UIGestureRecognizerDelegate {
        // Static keys for associated objects (stable addresses, no force unwrap)
        private static nonisolated(unsafe) var verticalRotationKey: UInt8 = 0
        private static nonisolated(unsafe) var pan2TranslationKey: UInt8 = 0

        @MainActor @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let renderer = renderer else { return }
            let location = gesture.location(in: gesture.view)

            // --- Call endUserInteraction on gesture end ---
            if gesture.state == .ended || gesture.state == .cancelled {
                renderer.endUserInteraction()
                lastPanLocation = nil // Reset last location
                // Clear associated objects used for tracking state during the pan
                objc_setAssociatedObject(self, &MetalKitRendererView.Coordinator.verticalRotationKey, nil, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                objc_setAssociatedObject(self, &MetalKitRendererView.Coordinator.pan2TranslationKey, nil, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                return // Don't process further if ended/cancelled
            }
            // --- End change ---

            switch gesture.numberOfTouches {
            case 1:
                // ROTATION (single finger)
                switch gesture.state {
                case .began:
                    lastPanLocation = location
                    lastRotation = renderer.rotation
                    let vert = renderer.verticalRotation
                    objc_setAssociatedObject(self, &MetalKitRendererView.Coordinator.verticalRotationKey, vert, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                case .changed:
                    guard let lastLocation = lastPanLocation else { return }
                    let deltaX = Float(location.x - lastLocation.x)
                    let deltaY = Float(location.y - lastLocation.y)
                    let newRotation = lastRotation + Angle(degrees: Double(deltaX) * 0.2)
                    var newVertical: Float = 0
                    if let vert = objc_getAssociatedObject(self, &MetalKitRendererView.Coordinator.verticalRotationKey) as? Float {
                        newVertical = vert + deltaY * 0.01
                        newVertical = max(-.pi/2, min(.pi/2, newVertical))
                    }
                    renderer.setUserRotation(newRotation, vertical: newVertical)
                default:
                    break // .ended/.cancelled handled above
                }
            case 2:
                // PANNING (two fingers)
                switch gesture.state {
                case .began:
                    let initial = renderer.translation
                    objc_setAssociatedObject(self, &MetalKitRendererView.Coordinator.pan2TranslationKey, initial, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                case .changed:
                    if let initial = objc_getAssociatedObject(self, &MetalKitRendererView.Coordinator.pan2TranslationKey) as? SIMD2<Float> {
                        let translation = gesture.translation(in: gesture.view)
                        let dx = Float(translation.x) * 0.01
                        let dy = Float(translation.y) * 0.01
                        renderer.setUserTranslation(SIMD2<Float>(initial.x + dx, initial.y - dy))
                    }
                default:
                    break // .ended/.cancelled handled above
                }
            default:
                break
            }
        }

        @MainActor @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let renderer = renderer else { return }

            // --- Call endUserInteraction on gesture end ---
            if gesture.state == .ended || gesture.state == .cancelled {
                renderer.endUserInteraction()
                return // Don't process further if ended/cancelled
            }
            // --- End change ---

            switch gesture.state {
            case .began:
                zoom = renderer.zoom
            case .changed:
                let newZoom = max(0.1, min(zoom * Float(gesture.scale), 20.0))
                renderer.setUserZoom(newZoom)
            default:
                break // .ended/.cancelled handled above
            }
        }

        @MainActor @objc func handleRotation(_ gesture: UIRotationGestureRecognizer) {
            guard let renderer = renderer else { return }

            // --- Call endUserInteraction on gesture end ---
            if gesture.state == .ended || gesture.state == .cancelled {
                renderer.endUserInteraction()
                return // Don't process further if ended/cancelled
            }
            // --- End change ---

            switch gesture.state {
            case .began:
                lastRollRotation = renderer.rollRotation
            case .changed:
                let newRollRotation = lastRollRotation - Float(gesture.rotation)
                renderer.setUserRollRotation(newRollRotation)
            default:
                break // .ended/.cancelled handled above
            }
        }

        @MainActor @objc func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
            guard let renderer = renderer else { return }
            renderer.resetView()
        }
        
        // MARK: - UIGestureRecognizerDelegate
        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            // Allow rotation gesture to work with pan gesture for 2-finger interactions
            // Allow pinch gesture to work with rotation gesture
            return true
        }
    }
#elseif os(macOS)
    // MARK: - macOS Coordinator Gesture Handling
    extension MetalKitRendererView.Coordinator: NSGestureRecognizerDelegate {
        @MainActor @objc func handlePan(_ gesture: NSPanGestureRecognizer) {
            guard let renderer = renderer else { return }
            let location = gesture.location(in: gesture.view)

            if gesture.state == .ended || gesture.state == .cancelled {
                renderer.endUserInteraction()
                lastPanLocation = nil
                return
            }

            switch gesture.state {
            case .began:
                lastPanLocation = location
            case .changed:
                guard let lastLocation = lastPanLocation else {
                    lastPanLocation = location
                    return
                }
                let deltaX = Float(location.x - lastLocation.x)
                let deltaY = Float(location.y - lastLocation.y)

                // Horizontal drag rotates around Y axis, vertical drag rotates around X axis
                let sensitivity: Float = 0.01
                let currentRotation = renderer.rotation
                let newRotation = Angle(radians: currentRotation.radians + Double(deltaX * sensitivity))
                let currentVertical = renderer.verticalRotation
                let newVertical = currentVertical - deltaY * sensitivity

                renderer.setUserRotation(newRotation, vertical: newVertical)

                lastPanLocation = location
            default:
                break
            }
        }

        @MainActor @objc func handleMagnify(_ gesture: NSMagnificationGestureRecognizer) {
            guard let renderer = renderer else { return }

            if gesture.state == .ended || gesture.state == .cancelled {
                renderer.endUserInteraction()
                return
            }

            switch gesture.state {
            case .began:
                zoom = renderer.zoom
            case .changed:
                let newZoom = max(0.1, min(zoom * Float(1.0 + gesture.magnification), 20.0))
                renderer.setUserZoom(newZoom)
            default:
                break
            }
        }

        @MainActor @objc func handleRotation(_ gesture: NSRotationGestureRecognizer) {
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

        // MARK: - NSGestureRecognizerDelegate
        func gestureRecognizer(_ gestureRecognizer: NSGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: NSGestureRecognizer) -> Bool {
            return true
        }
    }
#endif

#endif // os(iOS) || os(macOS)
