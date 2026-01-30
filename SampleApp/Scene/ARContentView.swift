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
    @State private var modelLoadingError: String?
    @State private var isARTrackingReady = false
    @State private var isWaitingForSurfaceDetection = false
    @State private var showARInstructions = false
    
    let model: ModelIdentifier?
    
    var body: some View {
        ZStack {
            // AR Metal view
            ARMetalKitView(
                renderer: $arSceneRenderer, 
                model: model, 
                isARActive: $isARSessionActive,
                modelLoadingError: $modelLoadingError
            )
                .ignoresSafeArea()
                .onReceive(NotificationCenter.default.publisher(for: .init("ARRendererReady"))) { _ in
                    // Don't start AR session here - wait for model to load
                    print("AR: Renderer ready notification received, waiting for model load")
                }
                .onReceive(Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()) { _ in
                    // Check AR tracking status periodically
                    if isARSessionActive, let renderer = arSceneRenderer {
                        isARTrackingReady = renderer.isARTrackingNormal()
                        isWaitingForSurfaceDetection = renderer.isWaitingForSurfaceDetection()
                    }
                }
            
            // Help button overlay
            VStack {
                Spacer()
                
                HStack {
                    Spacer()
                    
                    if isARSessionActive {
                        Button(action: {
                            showARInstructions = true
                        }) {
                            Image(systemName: "questionmark.circle.fill")
                                .font(.title)
                                .foregroundColor(.white)
                                .frame(width: 50, height: 50)
                                .background(Color.black.opacity(0.6))
                                .clipShape(Circle())
                                .shadow(radius: 3)
                        }
                        .padding(.bottom, 30)
                        .padding(.trailing, 20)
                    }
                }
            }
            
            // Loading overlay while AR is initializing
            if isARSessionActive && !isARTrackingReady {
                VStack {
                    Spacer()
                    
                    VStack(spacing: 20) {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.5)
                        
                        Text("Initializing AR...")
                            .font(.headline)
                            .foregroundColor(.white)
                        
                        Text("Move your device slowly to scan the environment")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.8))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .padding(30)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(20)
                    
                    Spacer()
                }
            }
            
            // Surface detection overlay
            if isARSessionActive && isARTrackingReady && isWaitingForSurfaceDetection {
                VStack {
                    Spacer()
                    
                    VStack(spacing: 20) {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.5)
                        
                        Text("Detecting Surfaces...")
                            .font(.headline)
                            .foregroundColor(.white)
                        
                        Text("Point your device at horizontal surfaces like tables or floors")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.8))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                    .padding(30)
                    .background(Color.black.opacity(0.7))
                    .cornerRadius(20)
                    
                    Spacer()
                }
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
            
            // Model loading error overlay
            if let error = modelLoadingError {
                VStack {
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 60))
                            .foregroundColor(.orange)
                        
                        Text("Model Loading Error")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.white)
                        
                        Text(error)
                            .font(.body)
                            .foregroundColor(.white)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                        
                        HStack(spacing: 12) {
                            Button("Try Sample Box") {
                                loadFallbackModel()
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.green)
                            
                            Button("Dismiss") {
                                modelLoadingError = nil
                            }
                            .buttonStyle(.bordered)
                            .tint(.gray)
                        }
                    }
                    .padding(30)
                    .background(Color.black.opacity(0.8))
                    .cornerRadius(20)
                    .padding()
                    Spacer()
                }
            }
            
            // AR Instructions overlay
            if showARInstructions {
                ZStack {
                    // Semi-transparent background
                    Color.black.opacity(0.6)
                        .ignoresSafeArea()
                        .onTapGesture {
                            showARInstructions = false
                        }
                    
                    VStack {
                        Spacer()
                        
                        VStack(spacing: 16) {
                            HStack {
                                Text("AR Interactions")
                                    .font(.headline)
                                    .foregroundColor(.white)
                                
                                Spacer()
                                
                                Button(action: {
                                    showARInstructions = false
                                }) {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.title3)
                                        .foregroundColor(.white.opacity(0.8))
                                }
                            }
                            
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    Image(systemName: "hand.tap")
                                        .foregroundColor(.white.opacity(0.8))
                                        .frame(width: 20)
                                    Text("Tap to place splat")
                                        .foregroundColor(.white)
                                }
                                
                                HStack {
                                    Image(systemName: "hand.pinch")
                                        .foregroundColor(.white.opacity(0.8))
                                        .frame(width: 20)
                                    Text("Pinch to scale")
                                        .foregroundColor(.white)
                                }
                                
                                HStack {
                                    Image(systemName: "hand.draw")
                                        .foregroundColor(.white.opacity(0.8))
                                        .frame(width: 20)
                                    Text("Two-finger drag to move")
                                        .foregroundColor(.white)
                                }
                                
                                HStack {
                                    Image(systemName: "rotate.3d")
                                        .foregroundColor(.white.opacity(0.8))
                                        .frame(width: 20)
                                    Text("Two-finger twist to rotate")
                                        .foregroundColor(.white)
                                }
                            }
                            .padding(.vertical, 8)
                        }
                        .padding(24)
                        .background(Color.black.opacity(0.8))
                        .cornerRadius(16)
                        .padding(.horizontal, 20)
                        
                        Spacer()
                    }
                }
            }
        }
        .navigationTitle("AR Splats")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            print("AR: ARContentView appeared")
            // Reset state when view appears
            permissionDenied = false
            arError = nil
            checkARAvailability()
        }
        .onDisappear {
            print("AR: View disappearing, cleaning up AR session")
            arSceneRenderer?.stopARSession()
            isARSessionActive = false
            isARTrackingReady = false
            isWaitingForSurfaceDetection = false
            showARInstructions = false
            // Clear any error states for clean restart
            arError = nil
            // Clear the renderer reference to force complete cleanup
            arSceneRenderer = nil
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
            // Permission already granted - wait for renderer to be ready
            // Don't start AR session here, let the renderer ready notification handle it
            print("AR: Will wait for renderer to be ready before starting AR session")
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
                            print("AR: Permission granted but waiting for model to load before starting AR session")
                            // AR session will be started after model loads
                        }
                    }
                } else {
                    self.permissionDenied = true
                }
            }
        }
    }
    
    private func loadFallbackModel() {
        guard let renderer = arSceneRenderer else { return }
        
        Task {
            do {
                print("AR: Loading fallback sample box model")
                try await renderer.load(.sampleBox)
                print("AR: Successfully loaded fallback model")
                
                // Clear error message
                DispatchQueue.main.async {
                    self.modelLoadingError = nil
                }
                
                // Start AR session if we have permission
                if AVCaptureDevice.authorizationStatus(for: .video) == .authorized {
                    print("AR: Starting AR session with fallback model")
                    renderer.startARSession()
                    self.isARSessionActive = true
                }
            } catch {
                print("AR: Fallback model loading failed: \(error)")
                DispatchQueue.main.async {
                    self.modelLoadingError = "Even the sample model failed to load: \(error.localizedDescription)"
                }
            }
        }
    }
    
}

struct ARMetalKitView: UIViewRepresentable {
    @Binding var renderer: ARSceneRenderer?
    let model: ModelIdentifier?
    @Binding var isARActive: Bool
    @Binding var modelLoadingError: String?
    
    func makeUIView(context: Context) -> MTKView {
        print("AR: Creating MTKView - this should happen when entering AR view")
        let metalKitView = MTKView()
        metalKitView.device = MTLCreateSystemDefaultDevice()
        metalKitView.backgroundColor = UIColor.clear
        metalKitView.isOpaque = false
        
        print("AR: Metal device: \(metalKitView.device?.name ?? "nil")")
        
        // Create fresh renderer each time view is made
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
        print("AR: updateUIView called - renderer: \(renderer != nil ? "exists" : "nil")")
        
        // Only proceed if renderer exists and AR is not already active
        guard let renderer = renderer, !isARActive else { 
            print("AR: Skipping update - renderer: \(renderer != nil), isARActive: \(isARActive)")
            return 
        }
        
        Task { @MainActor in
            do {
                if let model = model {
                    print("AR: Loading model \(model)")
                    try await renderer.load(model)
                    print("AR: Successfully loaded model \(model)")
                } else {
                    print("AR: No model provided, loading sample box")
                    try await renderer.load(.sampleBox)
                    print("AR: Successfully loaded sample box")
                }
                
                // Now that model is loaded, start AR session if we have permission
                if AVCaptureDevice.authorizationStatus(for: .video) == .authorized {
                    print("AR: Starting AR session after successful model load")
                    renderer.startARSession()
                    self.isARActive = true
                } else {
                    print("AR: Cannot start AR session - camera permission not granted")
                }
            } catch {
                print("AR: Failed to load model: \(error)")
                
                // Set error message for user feedback
                DispatchQueue.main.async {
                    self.modelLoadingError = "Failed to load model: \(error.localizedDescription)"
                }
                
                // Try to load fallback model
                do {
                    print("AR: Attempting to load fallback sample box")
                    try await renderer.load(.sampleBox)
                    print("AR: Successfully loaded fallback model")
                    
                    // Start AR session if we have permission
                    if AVCaptureDevice.authorizationStatus(for: .video) == .authorized {
                        print("AR: Starting AR session with fallback model")
                        renderer.startARSession()
                        self.isARActive = true
                        
                        // Clear error since fallback worked
                        DispatchQueue.main.async {
                            self.modelLoadingError = nil
                        }
                    }
                } catch {
                    print("AR: Even fallback model failed to load: \(error)")
                    DispatchQueue.main.async {
                        self.modelLoadingError = "Cannot load any model. AR unavailable."
                    }
                }
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
        
        // Add rotation gesture for rotating splats
        let rotationGesture = UIRotationGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleRotation(_:)))
        view.addGestureRecognizer(rotationGesture)
        print("AR: Added rotation gesture recognizer")
        
        // Add pan gesture for moving splats (two-finger drag)
        let panGesture = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePan(_:)))
        panGesture.minimumNumberOfTouches = 2
        panGesture.maximumNumberOfTouches = 2
        view.addGestureRecognizer(panGesture)
        print("AR: Added pan gesture recognizer")
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
        
        @objc func handleRotation(_ gesture: UIRotationGestureRecognizer) {
            let rotation = Float(gesture.rotation)
            print("AR: Coordinator handleRotation rotation: \(rotation)")
            renderer?.handleRotation(rotation: CGFloat(rotation), velocity: 0)
            gesture.rotation = 0.0 // Reset for next gesture
        }
        
        @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let view = metalView else { return }
            let translation = gesture.translation(in: view)
            let velocity = gesture.velocity(in: view)
            print("AR: Coordinator handlePan translation: \(translation), velocity: \(velocity)")
            renderer?.handlePan(translation: translation, velocity: velocity)
            gesture.setTranslation(.zero, in: view) // Reset for next gesture
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
        guard let typedGesture = gesture as? T else { return }
        handler(typedGesture)
    }
}

#endif // os(iOS)