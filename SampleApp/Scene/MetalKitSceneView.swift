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
    
    var body: some View {
        ZStack {
            // The actual Metal view
            MetalKitRendererView(modelIdentifier: modelIdentifier)
                .ignoresSafeArea()
            
            // AR toggle button overlay (iOS only)
            #if os(iOS)
            VStack {
                Spacer()
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
            }
            #endif
        }
        .navigationDestination(isPresented: $navigateToAR) {
            ARContentView(model: modelIdentifier)
                .navigationTitle("AR \(modelIdentifier?.description ?? "View")")
        }
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

    class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var renderer: MetalKitSceneRenderer?
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
    func makeNSView(context: NSViewRepresentableContext<MetalKitRendererView>) -> MTKView {
        makeView(context.coordinator)
    }
#elseif os(iOS)
    func makeUIView(context: UIViewRepresentableContext<MetalKitRendererView>) -> MTKView {
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
        metalKitView.delegate = renderer
        
        // Fast SH is configured in the renderer itself

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

#if os(macOS)
    func updateNSView(_ view: MTKView, context: NSViewRepresentableContext<MetalKitRendererView>) {
        updateView(context.coordinator)
    }
#elseif os(iOS)
    func updateUIView(_ view: MTKView, context: UIViewRepresentableContext<MetalKitRendererView>) {
        updateView(context.coordinator)
    }
#endif

    private func updateView(_ coordinator: Coordinator) {
        guard let renderer = coordinator.renderer else { return }
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
    extension MetalKitRendererView.Coordinator {
        private static let verticalRotationKey = UnsafeRawPointer(bitPattern: "verticalRotation".hashValue)!
        private static let pan2TranslationKey = UnsafeRawPointer(bitPattern: "pan2Translation".hashValue)!
        
        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let renderer = renderer else { return }
            let location = gesture.location(in: gesture.view)

            // --- Call endUserInteraction on gesture end ---
            if gesture.state == .ended || gesture.state == .cancelled {
                renderer.endUserInteraction()
                lastPanLocation = nil // Reset last location
                // Clear associated objects used for tracking state during the pan
                objc_setAssociatedObject(self, MetalKitRendererView.Coordinator.verticalRotationKey, nil, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                objc_setAssociatedObject(self, MetalKitRendererView.Coordinator.pan2TranslationKey, nil, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
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
                    objc_setAssociatedObject(self, MetalKitRendererView.Coordinator.verticalRotationKey, vert, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                case .changed:
                    guard let lastLocation = lastPanLocation else { return }
                    let deltaX = Float(location.x - lastLocation.x)
                    let deltaY = Float(location.y - lastLocation.y)
                    let newRotation = lastRotation + Angle(degrees: Double(deltaX) * 0.2)
                    var newVertical: Float = 0
                    if let vert = objc_getAssociatedObject(self, MetalKitRendererView.Coordinator.verticalRotationKey) as? Float {
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
                    objc_setAssociatedObject(self, MetalKitRendererView.Coordinator.pan2TranslationKey, initial, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
                case .changed:
                    if let initial = objc_getAssociatedObject(self, MetalKitRendererView.Coordinator.pan2TranslationKey) as? SIMD2<Float> {
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

        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
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

        @objc func handleRotation(_ gesture: UIRotationGestureRecognizer) {
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

        @objc func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
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
#endif

#endif // os(iOS) || os(macOS)
