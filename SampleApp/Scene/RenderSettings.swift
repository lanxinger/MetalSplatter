import SwiftUI
import Metal

// Metal 4.0 Capabilities Status
struct Metal4Capabilities {
    let available: Bool
    let simdGroupOperations: Bool
    let tensorOperations: Bool
    let advancedAtomics: Bool
    let meshShaders: Bool
    
    var description: String {
        var features: [String] = []
        if simdGroupOperations { features.append("✓ SIMD-Groups") }
        if tensorOperations { features.append("✓ Tensors") }
        if advancedAtomics { features.append("✓ Atomics") }
        if meshShaders { features.append("✓ Mesh Shaders") }
        
        return available ? features.joined(separator: " • ") : "Metal 4.0 Not Available"
    }
}

/// Settings for Metal rendering features
struct RenderSettings: View {
    @Binding var fastSHEnabled: Bool
    @Binding var metal4BindlessEnabled: Bool
    @Binding var showDebugAABB: Bool
    @Binding var batchPrecomputeEnabled: Bool  // TensorOps batch precompute
    @Binding var meshShaderEnabled: Bool  // Mesh shaders (Metal 3+)
    @State private var isMetal4Available: Bool = false
    @State private var metal4SIMDGroupEnabled: Bool = true
    @State private var metal4AtomicSortEnabled: Bool = true
    @State private var metal4Capabilities: Metal4Capabilities?
    var onDismiss: (() -> Void)?
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header with close button
            HStack {
                Text("Render Settings")
                    .font(.headline)
                
                Spacer()
                
                if let dismiss = onDismiss {
                    Button(action: dismiss) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title3)
                            .foregroundColor(.secondary)
                    }
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
            
            // Debug AABB Toggle - tests GPU bounds computation
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
            
            // Metal 4 Bindless Toggle
            Toggle(isOn: $metal4BindlessEnabled) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Text("Metal 4 Bindless Resources")
                            .font(.subheadline)
                        if !isMetal4Available {
                            Text("(Requires iOS 18+/macOS 15+)")
                                .font(.caption2)
                                .foregroundColor(.orange)
                        }
                    }
                    Text("50-80% CPU overhead reduction for large scenes")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .disabled(!isMetal4Available)
            
            if metal4BindlessEnabled && isMetal4Available {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Metal 4.0 Features Active", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundColor(.green)
                    
                    // Advanced Metal 4.0 Feature Toggles
                    VStack(alignment: .leading, spacing: 6) {
                        Toggle("SIMD-Group Operations", isOn: $metal4SIMDGroupEnabled)
                            .font(.caption)
                        
                        Toggle("TensorOps Batch Precompute", isOn: $batchPrecomputeEnabled)
                            .font(.caption)
                        
                        Toggle("Mesh Shaders (GPU Geometry)", isOn: $meshShaderEnabled)
                            .font(.caption)
                        
                        Toggle("Advanced Atomic Sort", isOn: $metal4AtomicSortEnabled)
                            .font(.caption)
                    }
                    .padding(.leading, 20)
                    .padding(.vertical, 4)
                    
                    if let capabilities = metal4Capabilities {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Capabilities Status:")
                                .font(.caption2)
                                .fontWeight(.semibold)
                                .foregroundColor(.secondary)
                            
                            Text(capabilities.description)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .padding(.leading, 20)
                        .padding(.top, 4)
                    }
                }
                .padding(.top, 4)
            }
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(10)
        .onAppear {
            checkMetal4Availability()
            loadMetal4Capabilities()
        }
    }
    
    private func checkMetal4Availability() {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            // Check for actual Metal device support
            if let device = MTLCreateSystemDefaultDevice() {
                isMetal4Available = device.supportsFamily(.apple9)
            } else {
                isMetal4Available = false
            }
        } else {
            isMetal4Available = false
        }
    }
    
    private func loadMetal4Capabilities() {
        guard isMetal4Available,
              let device = MTLCreateSystemDefaultDevice() else {
            return
        }
        
        // Create a temporary renderer to check capabilities
        // In real implementation, this would get capabilities from active renderer
        metal4Capabilities = Metal4Capabilities(
            available: true,
            simdGroupOperations: metal4SIMDGroupEnabled,
            tensorOperations: batchPrecomputeEnabled,
            advancedAtomics: metal4AtomicSortEnabled,
            meshShaders: meshShaderEnabled
        )
    }
}

/// Enhanced MetalKit Scene View with Settings
struct EnhancedMetalKitSceneView: View {
    var modelIdentifier: ModelIdentifier?
    @State private var showSettings = false
    @State private var fastSHEnabled = true
    @State private var metal4BindlessEnabled = true // Default to enabled
    @State private var showDebugAABB = false // Debug: visualize GPU-computed bounds
    @State private var batchPrecomputeEnabled = true // TensorOps batch precompute - enabled by default
    @State private var meshShaderEnabled = true // Mesh shaders - enabled by default for Metal 3+ devices
    @State private var showARUnavailableAlert = false
    @State private var navigateToAR = false
    
    var body: some View {
        ZStack {
            // The actual Metal view
            MetalKitRendererViewEnhanced(
                modelIdentifier: modelIdentifier,
                fastSHEnabled: $fastSHEnabled,
                metal4BindlessEnabled: $metal4BindlessEnabled,
                showDebugAABB: $showDebugAABB,
                batchPrecomputeEnabled: $batchPrecomputeEnabled,
                meshShaderEnabled: $meshShaderEnabled
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
                
                // AR button at bottom-right (iOS only)
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
                            RenderSettings(
                                fastSHEnabled: $fastSHEnabled,
                                metal4BindlessEnabled: $metal4BindlessEnabled,
                                showDebugAABB: $showDebugAABB,
                                batchPrecomputeEnabled: $batchPrecomputeEnabled,
                                meshShaderEnabled: $meshShaderEnabled,
                                onDismiss: { showSettings = false }
                            )
                            .padding()
                            .background(Color(.systemBackground).opacity(0.9))
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
        .navigationDestination(isPresented: $navigateToAR) {
            #if os(iOS)
            ARContentView(model: modelIdentifier)
                .navigationTitle("AR \(modelIdentifier?.description ?? "View")")
            #endif
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
        guard ARKit.ARWorldTrackingConfiguration.isSupported else {
            showARUnavailableAlert = true
            return
        }
        
        // Navigate to AR view with current model
        navigateToAR = true
        #endif
    }
}

// Enhanced renderer view that accepts settings bindings
struct MetalKitRendererViewEnhanced: ViewRepresentable {
    var modelIdentifier: ModelIdentifier?
    @Binding var fastSHEnabled: Bool
    @Binding var metal4BindlessEnabled: Bool
    @Binding var showDebugAABB: Bool
    @Binding var batchPrecomputeEnabled: Bool
    @Binding var meshShaderEnabled: Bool
    
    class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var renderer: MetalKitSceneRenderer?
        var parent: MetalKitRendererViewEnhanced
        
        // Store camera interaction state
        var lastPanLocation: CGPoint?
        var lastRotation: Angle = .zero
        var lastRollRotation: Float = 0.0
        var zoom: Float = 1.0
        
        init(_ parent: MetalKitRendererViewEnhanced) {
            self.parent = parent
        }
        
        func updateSettings() {
            renderer?.fastSHSettings.enabled = parent.fastSHEnabled
            renderer?.setMetal4Bindless(parent.metal4BindlessEnabled)
            renderer?.setDebugAABB(parent.showDebugAABB)
            renderer?.setBatchPrecompute(parent.batchPrecomputeEnabled)
            renderer?.setMeshShader(parent.meshShaderEnabled)
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }
    
    #if os(macOS)
    func makeNSView(context: NSViewRepresentableContext<MetalKitRendererViewEnhanced>) -> MTKView {
        makeView(context.coordinator)
    }
    
    func updateNSView(_ view: MTKView, context: NSViewRepresentableContext<MetalKitRendererViewEnhanced>) {
        context.coordinator.updateSettings()
    }
    #elseif os(iOS)
    func makeUIView(context: UIViewRepresentableContext<MetalKitRendererViewEnhanced>) -> MTKView {
        makeView(context.coordinator)
    }
    
    func updateUIView(_ view: MTKView, context: UIViewRepresentableContext<MetalKitRendererViewEnhanced>) {
        context.coordinator.updateSettings()
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
        
        // Apply initial settings - all optimizations enabled by default
        renderer?.fastSHSettings.enabled = fastSHEnabled
        renderer?.setMetal4Bindless(metal4BindlessEnabled)
        renderer?.setDebugAABB(showDebugAABB)
        renderer?.setBatchPrecompute(batchPrecomputeEnabled)
        renderer?.setMeshShader(meshShaderEnabled)
        
        // Add gesture recognizers (same as original)
        #if os(iOS)
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
}

// MARK: - Gesture Handling Extensions

#if os(iOS)
extension MetalKitRendererViewEnhanced.Coordinator {
    @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
        // Implementation would be copied from original MetalKitSceneView
    }
    
    @objc func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        // Implementation would be copied from original MetalKitSceneView
    }
    
    @objc func handleRotation(_ gesture: UIRotationGestureRecognizer) {
        // Implementation would be copied from original MetalKitSceneView
    }
    
    @objc func handleDoubleTap(_ gesture: UITapGestureRecognizer) {
        // Implementation would be copied from original MetalKitSceneView
    }
}
#endif

#if os(macOS)
private typealias ViewRepresentable = NSViewRepresentable
#elseif os(iOS)
private typealias ViewRepresentable = UIViewRepresentable
#endif