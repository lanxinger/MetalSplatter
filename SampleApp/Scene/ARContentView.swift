#if os(iOS)

import ARKit
import AVFoundation
import SwiftUI
import MetalKit

class TouchDebugMTKView: MTKView {
    weak var renderer: ARSceneRenderer?
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        print("AR: TouchDebugMTKView - touchesBegan")
        super.touchesBegan(touches, with: event)
        
        if let touch = touches.first {
            let location = touch.location(in: self)
            print("AR: Touch began at \(location)")
            renderer?.handleTap(at: location)
        }
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        print("AR: TouchDebugMTKView - touchesEnded")
        super.touchesEnded(touches, with: event)
    }
}

struct ARContentView: View {
    @State private var arSceneRenderer: ARSceneRenderer?
    @State private var isARSessionActive = false
    @State private var showingPermissionAlert = false
    @State private var permissionDenied = false
    @State private var arError: String?
    
    let model: ModelIdentifier?
    
    var body: some View {
        ZStack {
            // AR Metal view
            ARMetalKitView(renderer: $arSceneRenderer, model: model, isARActive: $isARSessionActive)
                .ignoresSafeArea()
                .onReceive(NotificationCenter.default.publisher(for: .init("ARRendererReady"))) { _ in
                    // This will be triggered when the renderer is ready
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        if !permissionDenied && arSceneRenderer != nil {
                            print("AR: Starting AR session after renderer ready")
                            arSceneRenderer?.startARSession()
                            isARSessionActive = true
                        }
                    }
                }
            
            // AR Controls overlay
            VStack {
                // Top instruction text
                if isARSessionActive {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("AR Interactions:")
                                .font(.headline)
                                .foregroundColor(.white)
                            Text("• Tap to place")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.8))
                            Text("• Pinch to scale")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.8))
                            Text("• Pan to move")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.8))
                            Text("• Rotate to turn")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.8))
                        }
                        Spacer()
                    }
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(10)
                    .padding()
                    .allowsHitTesting(false) // Don't block touches to the AR view
                }
                
                Spacer()
                
                HStack {
                    Spacer()
                    
                    // AR Session toggle button
                    Button(action: toggleARSession) {
                        VStack(spacing: 4) {
                            Image(systemName: isARSessionActive ? "stop.fill" : "play.fill")
                                .font(.title2)
                                .foregroundColor(.white)
                            Text(isARSessionActive ? "Stop" : "Start")
                                .font(.caption2)
                                .foregroundColor(.white)
                        }
                        .frame(width: 70, height: 70)
                        .background(isARSessionActive ? Color.red.opacity(0.8) : Color.green.opacity(0.8))
                        .clipShape(RoundedRectangle(cornerRadius: 15))
                        .shadow(radius: 5)
                    }
                    .disabled(permissionDenied)
                    
                    Spacer()
                }
                .padding(.bottom, 50)
            }
            
            // Error overlay
            if let error = arError {
                VStack {
                    Spacer()
                    Text(error)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.red.opacity(0.8))
                        .cornerRadius(10)
                        .padding()
                    Spacer()
                }
            }
            
            // Permission denied overlay
            if permissionDenied {
                VStack {
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "camera.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.white)
                        
                        Text("Camera Access Required")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        
                        Text("AR features require camera access. Please enable camera permissions in Settings.")
                            .font(.body)
                            .foregroundColor(.white)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                        
                        Button("Open Settings") {
                            if let settingsUrl = URL(string: UIApplication.openSettingsURLString) {
                                UIApplication.shared.open(settingsUrl)
                            }
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding(30)
                    .background(Color.black.opacity(0.8))
                    .cornerRadius(20)
                    .padding()
                    Spacer()
                }
            }
        }
        .navigationTitle("AR Splats")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            print("AR: ARContentView appeared")
            checkARAvailability()
        }
        .onDisappear {
            arSceneRenderer?.stopARSession()
            isARSessionActive = false
        }
        .alert("Camera Permission Required", isPresented: $showingPermissionAlert) {
            Button("Cancel", role: .cancel) {
                permissionDenied = true
            }
            Button("Allow") {
                requestCameraPermission()
            }
        } message: {
            Text("This app needs camera access to provide AR functionality.")
        }
    }
    
    private func checkARAvailability() {
        guard ARWorldTrackingConfiguration.isSupported else {
            arError = "AR is not supported on this device"
            return
        }
        
        // Check camera permission
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            print("AR: Camera permission already granted")
            // Permission already granted
            break
        case .notDetermined:
            print("AR: Camera permission not determined, requesting...")
            requestCameraPermission()
        case .denied, .restricted:
            print("AR: Camera permission denied or restricted")
            permissionDenied = true
        @unknown default:
            print("AR: Camera permission unknown status")
            permissionDenied = true
        }
    }
    
    private func requestCameraPermission() {
        print("AR: Requesting camera permission...")
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                print("AR: Camera permission result: \(granted)")
                if granted {
                    self.showingPermissionAlert = false
                    self.permissionDenied = false
                    // Try to start AR session now that we have permission
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
                        if self.arSceneRenderer != nil {
                            print("AR: Starting AR session after permission granted")
                            self.arSceneRenderer?.startARSession()
                            self.isARSessionActive = true
                        }
                    }
                } else {
                    self.permissionDenied = true
                }
            }
        }
    }
    
    private func toggleARSession() {
        guard !permissionDenied else { return }
        
        if isARSessionActive {
            arSceneRenderer?.stopARSession()
            isARSessionActive = false
        } else {
            arSceneRenderer?.startARSession()
            isARSessionActive = true
        }
    }
}

struct ARMetalKitView: UIViewRepresentable {
    @Binding var renderer: ARSceneRenderer?
    let model: ModelIdentifier?
    @Binding var isARActive: Bool
    
    func makeUIView(context: Context) -> MTKView {
        print("AR: Creating MTKView...")
        let metalKitView = MTKView()
        metalKitView.device = MTLCreateSystemDefaultDevice()
        metalKitView.backgroundColor = UIColor.clear
        metalKitView.isOpaque = false
        
        print("AR: Metal device: \(metalKitView.device?.name ?? "nil")")
        
        guard let renderer = ARSceneRenderer(metalKitView) else {
            print("AR: Failed to create ARSceneRenderer!")
            return metalKitView
        }
        
        print("AR: Successfully created ARSceneRenderer")
        metalKitView.delegate = renderer
        
        // Update the binding
        DispatchQueue.main.async {
            self.renderer = renderer
            print("AR: Renderer binding updated")
            // Notify that renderer is ready
            NotificationCenter.default.post(name: .init("ARRendererReady"), object: nil)
        }
        
        // Add gesture recognizers AFTER setting up the view
        DispatchQueue.main.async {
            self.setupGestureRecognizers(for: metalKitView, renderer: renderer, context: context)
        }
        
        return metalKitView
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {
        Task { @MainActor in
            if let model = model {
                do {
                    try await renderer?.load(model)
                    print("AR: Successfully loaded model \(model)")
                } catch {
                    print("AR: Failed to load model \(model): \(error)")
                }
            } else {
                print("AR: No model provided, using sample box")
            }
        }
    }
    
    private func setupGestureRecognizers(for view: MTKView, renderer: ARSceneRenderer?, context: Context) {
        print("AR: Setting up gesture recognizers...")
        
        // Enable user interaction
        view.isUserInteractionEnabled = true
        print("AR: User interaction enabled: \(view.isUserInteractionEnabled)")
        
        // Clear any existing gesture recognizers
        view.gestureRecognizers?.removeAll()
        
        // Simple tap gesture using target-action pattern
        let tapGesture = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleTap(_:)))
        tapGesture.numberOfTapsRequired = 1
        tapGesture.numberOfTouchesRequired = 1
        
        // Store renderer reference in coordinator
        context.coordinator.renderer = renderer
        context.coordinator.metalView = view
        
        view.addGestureRecognizer(tapGesture)
        print("AR: Added tap gesture recognizer with target-action")
        
        // Add pinch gesture for scaling
        let pinchGesture = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
        context.coordinator.metalView = view
        view.addGestureRecognizer(pinchGesture)
        print("AR: Added pinch gesture recognizer")
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, UIGestureRecognizerDelegate {
        weak var renderer: ARSceneRenderer?
        weak var metalView: MTKView?
        
        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let view = metalView else {
                print("AR: No metal view for tap gesture")
                return
            }
            let location = gesture.location(in: view)
            print("AR: Coordinator handleTap at \(location)")
            renderer?.handleTap(at: location)
        }
        
        @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            let scale = Float(gesture.scale)
            print("AR: Coordinator handlePinch scale: \(scale)")
            renderer?.handlePinch(scale: CGFloat(scale), velocity: 0)
            gesture.scale = 1.0 // Reset for next gesture
        }
        
        func gestureRecognizer(_ gestureRecognizer: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith otherGestureRecognizer: UIGestureRecognizer) -> Bool {
            return true
        }
    }
}

// MARK: - UIGestureRecognizer Extensions

extension UITapGestureRecognizer {
    convenience init(handler: @escaping (UITapGestureRecognizer) -> Void) {
        self.init()
        addTarget(GestureTarget(handler: handler), action: #selector(GestureTarget.handle))
    }
}

extension UIPanGestureRecognizer {
    convenience init(handler: @escaping (UIPanGestureRecognizer) -> Void) {
        self.init()
        addTarget(GestureTarget(handler: handler), action: #selector(GestureTarget.handle))
    }
}

extension UIPinchGestureRecognizer {
    convenience init(handler: @escaping (UIPinchGestureRecognizer) -> Void) {
        self.init()
        addTarget(GestureTarget(handler: handler), action: #selector(GestureTarget.handle))
    }
}

extension UIRotationGestureRecognizer {
    convenience init(handler: @escaping (UIRotationGestureRecognizer) -> Void) {
        self.init()
        addTarget(GestureTarget(handler: handler), action: #selector(GestureTarget.handle))
    }
}

private class GestureTarget<T: UIGestureRecognizer>: NSObject {
    let handler: (T) -> Void
    
    init(handler: @escaping (T) -> Void) {
        self.handler = handler
    }
    
    @objc func handle(gesture: UIGestureRecognizer) {
        handler(gesture as! T)
    }
}

#endif // os(iOS)