#if os(iOS) || os(macOS)

import SwiftUI
import MetalKit
import MetalSplatter
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
    @StateObject private var editingController = SceneEditingController()
    @State private var showARUnavailableAlert = false
    @State private var navigateToAR = false
    @State private var showSettings = false
    @State private var fastSHEnabled = true
    @State private var metal4BindlessEnabled = true // Default to enabled
    @State private var showDebugAABB = false // Debug: visualize GPU-computed bounds
    @State private var frustumCullingEnabled = true // GPU frustum culling - enabled by default for AR performance
    @State private var meshShaderEnabled = true // Metal 3+ mesh shader rendering - enabled by default
    @State private var ditheredTransparencyEnabled = false // Stochastic transparency - disabled by default
    @State private var metal4SortingEnabled = true // Metal 4 GPU radix sort - enabled by default
    @State private var use2DGSMode = false // 2DGS planar rendering - disabled by default
    @State private var splatAnimationEnabled = false
    @State private var splatAnimationEffectRawValue = SplatAnimationEffect.spread.rawValue
    @State private var splatAnimationPlaying = true
    @State private var splatAnimationSpeed = 1.0
    @State private var splatAnimationIntensity = 1.0
    @State private var splatAnimationResetCounter = 0
    @State private var renderScale: CGFloat = 0.66 // iOS fill-rate control: 66% scale ~= 44% pixels
    @State private var adaptiveRenderScaleEnabled = false // Opt-in to avoid visible resolution pumping artifacts

    var body: some View {
        ZStack {
            // The actual Metal view
            MetalKitRendererView(
                modelIdentifier: modelIdentifier,
                editingController: editingController,
                fastSHEnabled: $fastSHEnabled,
                metal4BindlessEnabled: $metal4BindlessEnabled,
                showDebugAABB: $showDebugAABB,
                frustumCullingEnabled: $frustumCullingEnabled,
                meshShaderEnabled: $meshShaderEnabled,
                ditheredTransparencyEnabled: $ditheredTransparencyEnabled,
                metal4SortingEnabled: $metal4SortingEnabled,
                use2DGSMode: $use2DGSMode,
                splatAnimationEnabled: $splatAnimationEnabled,
                splatAnimationEffectRawValue: $splatAnimationEffectRawValue,
                splatAnimationPlaying: $splatAnimationPlaying,
                splatAnimationSpeed: $splatAnimationSpeed,
                splatAnimationIntensity: $splatAnimationIntensity,
                splatAnimationResetCounter: $splatAnimationResetCounter,
                renderScale: $renderScale,
                adaptiveRenderScaleEnabled: $adaptiveRenderScaleEnabled
            )
            .ignoresSafeArea()

            #if os(iOS)
            if editingController.isOverlayVisible {
                SplatEditingOverlay(controller: editingController)
                    .ignoresSafeArea()
            }
            #endif
            
            // Control overlay
            VStack {
                #if os(iOS)
                HStack {
                    if editingController.isEditorAvailable {
                        SplatEditingStatusChip(controller: editingController)
                    }
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.top, 12)
                #endif

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

                #if os(iOS)
                if editingController.isEditorAvailable {
                    SplatEditingToolbar(controller: editingController)
                        .padding(.horizontal)
                        .padding(.bottom, 12)
                }
                #endif
                
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
                            SplatSettingsPanel(
                                fastSHEnabled: $fastSHEnabled,
                                metal4BindlessEnabled: $metal4BindlessEnabled,
                                showDebugAABB: $showDebugAABB,
                                frustumCullingEnabled: $frustumCullingEnabled,
                                meshShaderEnabled: $meshShaderEnabled,
                                ditheredTransparencyEnabled: $ditheredTransparencyEnabled,
                                metal4SortingEnabled: $metal4SortingEnabled,
                                use2DGSMode: $use2DGSMode,
                                splatAnimationEnabled: $splatAnimationEnabled,
                                splatAnimationEffectRawValue: $splatAnimationEffectRawValue,
                                splatAnimationPlaying: $splatAnimationPlaying,
                                splatAnimationSpeed: $splatAnimationSpeed,
                                splatAnimationIntensity: $splatAnimationIntensity,
                                splatAnimationResetCounter: $splatAnimationResetCounter,
                                renderScale: $renderScale,
                                adaptiveRenderScaleEnabled: $adaptiveRenderScaleEnabled,
                                onDismiss: { showSettings = false }
                            )
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
        .fullScreenCover(isPresented: $navigateToAR) {
            NavigationStack {
                ARContentView(model: modelIdentifier)
                    .toolbar {
                        ToolbarItem(placement: .topBarLeading) {
                            Button("Done") {
                                navigateToAR = false
                            }
                        }
                    }
            }
        }
#endif
        .alert("AR Not Available", isPresented: $showARUnavailableAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text("AR features are not available on this device or require iOS 17.0+")
        }
        #if os(iOS)
        .sheet(item: $editingController.shareSheetItem) { item in
            ShareSheet(activityItems: [item.url])
        }
        #endif
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

private enum SplatSettingsCategory: String, CaseIterable, Identifiable {
    case rendering
    case performance
    case animation
    case display

    var id: String { rawValue }

    var title: String {
        switch self {
        case .rendering: return "Rendering"
        case .performance: return "Performance"
        case .animation: return "Animation"
        case .display: return "Display"
        }
    }

    var systemImage: String {
        switch self {
        case .rendering: return "paintpalette"
        case .performance: return "speedometer"
        case .animation: return "sparkles"
        case .display: return "display"
        }
    }
}

private struct SplatSettingsPanel: View {
    @Binding var fastSHEnabled: Bool
    @Binding var metal4BindlessEnabled: Bool
    @Binding var showDebugAABB: Bool
    @Binding var frustumCullingEnabled: Bool
    @Binding var meshShaderEnabled: Bool
    @Binding var ditheredTransparencyEnabled: Bool
    @Binding var metal4SortingEnabled: Bool
    @Binding var use2DGSMode: Bool
    @Binding var splatAnimationEnabled: Bool
    @Binding var splatAnimationEffectRawValue: UInt32
    @Binding var splatAnimationPlaying: Bool
    @Binding var splatAnimationSpeed: Double
    @Binding var splatAnimationIntensity: Double
    @Binding var splatAnimationResetCounter: Int
    @Binding var renderScale: CGFloat
    @Binding var adaptiveRenderScaleEnabled: Bool
    let onDismiss: () -> Void

    @State private var selectedCategory: SplatSettingsCategory = .rendering

    private var availableCategories: [SplatSettingsCategory] {
#if os(iOS)
        SplatSettingsCategory.allCases
#else
        [.rendering, .performance, .animation]
#endif
    }

    private var selectedAnimationEffect: SplatAnimationEffect {
        SplatAnimationEffect(rawValue: splatAnimationEffectRawValue) ?? .spread
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            header

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(availableCategories) { category in
                        Button {
                            selectedCategory = category
                        } label: {
                            Label(category.title, systemImage: category.systemImage)
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(selectedCategory == category ? Color.white : Color.primary)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .background(selectedCategory == category ? Color.accentColor : Color.secondary.opacity(0.12))
                                .clipShape(Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    content
                }
                .padding(.vertical, 2)
            }
            .frame(maxHeight: 420)
        }
        .padding(16)
#if os(iOS)
        .background(Color(UIColor.systemBackground).opacity(0.96))
#else
        .background(Color(NSColor.controlBackgroundColor).opacity(0.96))
#endif
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .shadow(radius: 12)
        .frame(maxWidth: 440)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Render Settings")
                    .font(.headline)
                Text("Grouped controls keep the renderer options manageable as the sample app grows.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Button(action: onDismiss) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
    }

    @ViewBuilder
    private var content: some View {
        switch selectedCategory {
        case .rendering:
            SplatSettingsSection("Shading") {
                settingsToggle(
                    title: "Fast Spherical Harmonics",
                    subtitle: "Optimized SH evaluation for better performance",
                    isOn: $fastSHEnabled
                )
                settingsToggle(
                    title: "Dithered Transparency",
                    subtitle: "Order-independent transparency without depth sorting",
                    isOn: $ditheredTransparencyEnabled
                )
                settingsToggle(
                    title: "2DGS Rendering Mode",
                    subtitle: "Flat oriented splats with normal extraction",
                    isOn: $use2DGSMode
                )
            }

        case .performance:
            SplatSettingsSection("GPU Pipeline") {
                settingsToggle(
                    title: "Frustum Culling",
                    subtitle: "GPU pre-filters splats outside camera view",
                    isOn: $frustumCullingEnabled
                )
                settingsToggle(
                    title: "Mesh Shaders",
                    subtitle: "Generate geometry on GPU and cut per-splat vertex work",
                    isOn: $meshShaderEnabled,
                    badge: "Metal 3+",
                    badgeTint: .blue
                )
                settingsToggle(
                    title: "Metal 4 Bindless Resources",
                    subtitle: "Lower CPU overhead with argument tables and residency sets",
                    isOn: $metal4BindlessEnabled,
                    badge: "Metal 4",
                    badgeTint: .green
                )
                settingsToggle(
                    title: "Metal 4 GPU Sorting",
                    subtitle: "Stable radix sort for very large splat counts",
                    isOn: $metal4SortingEnabled,
                    badge: "iOS 26+",
                    badgeTint: .purple
                )
            }

            SplatSettingsSection("Diagnostics") {
                settingsToggle(
                    title: "Show Bounding Box",
                    subtitle: "Visualize the GPU-computed scene bounds",
                    isOn: $showDebugAABB
                )

                if metal4BindlessEnabled {
                    Label("Argument tables enabled, residency sets active, bindless path selected.", systemImage: "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.green)
                        .padding(.top, 2)
                }
            }

        case .animation:
            SplatSettingsSection("Playback") {
                settingsToggle(
                    title: "Spark-Style Splat Animation",
                    subtitle: "Drives reveal and transition effects on both standard and Fast SH renderers",
                    isOn: $splatAnimationEnabled
                )

                if splatAnimationEnabled {
                    VStack(alignment: .leading, spacing: 10) {
                        Picker("Effect", selection: $splatAnimationEffectRawValue) {
                            ForEach(SplatAnimationEffect.allCases, id: \.rawValue) { effect in
                                Text(effect.displayName).tag(effect.rawValue)
                            }
                        }
                        .pickerStyle(.menu)

                        HStack(spacing: 10) {
                            Button(splatAnimationPlaying ? "Pause" : "Play") {
                                splatAnimationPlaying.toggle()
                            }
                            .buttonStyle(.bordered)

                            Button("Reset") {
                                splatAnimationPlaying = true
                                splatAnimationResetCounter += 1
                            }
                            .buttonStyle(.bordered)

                            Spacer()

                            Text(selectedAnimationEffect.displayName)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }

                        settingsSlider(
                            label: "Speed",
                            value: $splatAnimationSpeed,
                            range: 0.1...3.0
                        )
                        settingsSlider(
                            label: "Intensity",
                            value: $splatAnimationIntensity,
                            range: 0.1...2.0
                        )
                    }
                    .padding(.top, 4)
                }
            }

        case .display:
#if os(iOS)
            SplatSettingsSection("Resolution") {
                settingsToggle(
                    title: "Adaptive Render Scale",
                    subtitle: "Dynamically adjusts internal resolution to hold frame time",
                    isOn: $adaptiveRenderScaleEnabled
                )

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Render Scale")
                            .font(.subheadline)
                        Spacer()
                        Text("\(Int(renderScale * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    Slider(value: $renderScale, in: 0.55...1.0, step: 0.05)

                    Text("Lower values reduce fragment cost and thermal throttling.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
#endif
        }
    }

    private func settingsToggle(title: String,
                                subtitle: String,
                                isOn: Binding<Bool>,
                                badge: String? = nil,
                                badgeTint: Color = .blue) -> some View {
        Toggle(isOn: isOn) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(title)
                        .font(.subheadline)
                    if let badge {
                        Text(badge)
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(badgeTint)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(badgeTint.opacity(0.12))
                            .clipShape(Capsule())
                    }
                }
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func settingsSlider(label: String,
                                value: Binding<Double>,
                                range: ClosedRange<Double>) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .frame(width: 58, alignment: .leading)
            Slider(value: value, in: range)
            Text(String(format: "%.2f", value.wrappedValue))
                .font(.caption.monospacedDigit())
                .frame(width: 42)
        }
    }
}

private struct SplatSettingsSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)

            VStack(alignment: .leading, spacing: 12) {
                content
            }
            .padding(14)
            .background(Color.secondary.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
    }
}

private extension SplatAnimationEffect {
    var displayName: String {
        switch self {
        case .magic: return "Magic"
        case .spread: return "Spread"
        case .unroll: return "Unroll"
        case .twister: return "Twister"
        case .rain: return "Rain"
        case .spherical: return "Spherical"
        case .explosion: return "Explosion"
        case .flow: return "Flow"
        case .morph: return "Morph"
        }
    }
}

@MainActor
final class SceneEditingController: ObservableObject {
    enum Tool: String, CaseIterable, Identifiable {
        case select
        case rect
        case brush
        case lasso
        case flood
        case eyedropper
        case sphere
        case box
        case planeCut
        case align
        case polygon
        case measure
        case move
        case rotate
        case scale

        var id: String { rawValue }

        var label: String {
            switch self {
            case .select: "Select"
            case .rect: "Rect"
            case .brush: "Brush"
            case .lasso: "Lasso"
            case .flood: "Flood"
            case .eyedropper: "Color"
            case .sphere: "Sphere"
            case .box: "Box"
            case .planeCut: "Cut"
            case .align: "Align"
            case .polygon: "Polygon"
            case .measure: "Measure"
            case .move: "Move"
            case .rotate: "Rotate"
            case .scale: "Scale"
            }
        }

        var systemImage: String {
            switch self {
            case .select: "cursorarrow.click"
            case .rect: "rectangle.dashed"
            case .brush: "paintbrush.pointed"
            case .lasso: "lasso"
            case .flood: "drop.fill"
            case .eyedropper: "eyedropper"
            case .sphere: "circle.dashed"
            case .box: "cube.transparent"
            case .planeCut: "scissors"
            case .align: "angle"
            case .polygon: "point.3.connected.trianglepath.dotted"
            case .measure: "ruler"
            case .move: "arrow.up.left.and.arrow.down.right"
            case .rotate: "rotate.3d"
            case .scale: "arrow.up.left.and.down.right.and.arrow.up.right.and.down.left"
            }
        }

        var usesOverlaySelection: Bool {
            switch self {
            case .rect, .brush, .lasso, .flood, .eyedropper, .polygon, .measure:
                true
            default:
                false
            }
        }

        var usesDragOverlay: Bool {
            switch self {
            case .rect, .brush, .lasso:
                true
            default:
                false
            }
        }

        var usesTapOverlay: Bool {
            switch self {
            case .flood, .eyedropper, .polygon, .measure:
                true
            default:
                false
            }
        }

        var isTransformTool: Bool {
            switch self {
            case .move, .rotate, .scale:
                true
            default:
                false
            }
        }

        var showsSceneGuide: Bool {
            switch self {
            case .box, .planeCut:
                true
            default:
                false
            }
        }
    }

    struct ShareSheetPayload: Identifiable {
        let id = UUID()
        let url: URL
    }

    struct MeasuredPoint: Identifiable {
        let id = UUID()
        let screenPoint: CGPoint
        let worldPoint: SIMD3<Float>
    }

    enum PlaneAxis: String, CaseIterable, Identifiable {
        case x
        case y
        case z

        var id: String { rawValue }

        var label: String {
            switch self {
            case .x: "X"
            case .y: "Y"
            case .z: "Z"
            }
        }

        var negativeLabel: String {
            switch self {
            case .x: "Left"
            case .y: "Below"
            case .z: "Front"
            }
        }

        var positiveLabel: String {
            switch self {
            case .x: "Right"
            case .y: "Above"
            case .z: "Back"
            }
        }

        var unitNormal: SIMD3<Float> {
            switch self {
            case .x: SIMD3<Float>(1, 0, 0)
            case .y: SIMD3<Float>(0, 1, 0)
            case .z: SIMD3<Float>(0, 0, 1)
            }
        }

        func interpolatedCoordinate(in bounds: SplatSelectionBounds, offset: Float) -> Float {
            let clampedOffset = max(0, min(offset, 1))
            switch self {
            case .x:
                return bounds.min.x + ((bounds.max.x - bounds.min.x) * clampedOffset)
            case .y:
                return bounds.min.y + ((bounds.max.y - bounds.min.y) * clampedOffset)
            case .z:
                return bounds.min.z + ((bounds.max.z - bounds.min.z) * clampedOffset)
            }
        }

        func planePoint(in bounds: SplatSelectionBounds, offset: Float) -> SIMD3<Float> {
            var point = bounds.center
            let coordinate = interpolatedCoordinate(in: bounds, offset: offset)
            switch self {
            case .x:
                point.x = coordinate
            case .y:
                point.y = coordinate
            case .z:
                point.z = coordinate
            }
            return point
        }
    }

    enum AlignmentAxis: String, CaseIterable, Identifiable {
        case x
        case y
        case z

        var id: String { rawValue }

        var label: String { rawValue.uppercased() }

        var unitNormal: SIMD3<Float> {
            switch self {
            case .x: SIMD3<Float>(1, 0, 0)
            case .y: SIMD3<Float>(0, 1, 0)
            case .z: SIMD3<Float>(0, 0, 1)
            }
        }
    }

    enum OverlayPreview {
        case rect(CGRect)
        case brush([CGPoint])
        case lasso([CGPoint])
    }

    @Published var tool: Tool = .select {
        didSet {
            if !tool.usesOverlaySelection {
                overlayPreview = nil
            }
            if tool != .polygon {
                polygonPoints.removeAll(keepingCapacity: true)
            }
            if tool != .measure {
                measurePoints.removeAll(keepingCapacity: true)
            }
            if tool == .sphere || tool == .box {
                resetVolumeSelectionParameters()
            }
            if tool == .planeCut {
                resetCutPlaneParameters()
            }
            statusMessage = nil
        }
    }
    @Published var combineMode: SelectionCombineMode = .replace
    @Published private(set) var snapshot: SplatEditorSnapshot?
    @Published private(set) var isEditorAvailable = false
    @Published fileprivate var overlayPreview: OverlayPreview?
    @Published fileprivate var shareSheetItem: ShareSheetPayload?
    @Published private(set) var statusMessage: String?
    @Published var floodThreshold: Float = 0.2
    @Published var eyedropperThreshold: Float = 0.2
    @Published var sphereRadius: Float = 0.5
    @Published var boxExtentX: Float = 0.25
    @Published var boxExtentY: Float = 0.25
    @Published var boxExtentZ: Float = 0.25
    @Published var cutPlaneAxis: PlaneAxis = .y
    @Published var cutPlaneSide: SplatCutPlaneSide = .negative
    @Published var cutPlaneOffset: Float = 0.5
    @Published var alignmentAxis: AlignmentAxis = .y
    @Published private(set) var polygonPoints: [CGPoint] = []
    @Published private(set) var measurePoints: [MeasuredPoint] = []

    weak var renderer: MetalKitSceneRenderer?
    private var overlayCanvasSize: CGSize = .zero

    var isOverlayInteractionEnabled: Bool {
        isEditorAvailable && tool.usesOverlaySelection
    }

    var isOverlayVisible: Bool {
        isEditorAvailable && (tool.usesOverlaySelection || tool.showsSceneGuide)
    }

    var hasSelection: Bool {
        (snapshot?.selectedCount ?? 0) > 0
    }

    var canCommitPolygonSelection: Bool {
        polygonPoints.count >= 3
    }

    var measurementSummary: String {
        guard measurePoints.count >= 2 else {
            return "Tap splats to add measure points"
        }

        let segments = zip(measurePoints, measurePoints.dropFirst()).map { lhs, rhs in
            simd_length(rhs.worldPoint - lhs.worldPoint)
        }
        let total = segments.reduce(0, +)
        let last = segments.last ?? total
        return String(format: "Last %.3f m  Total %.3f m", last, total)
    }

    var selectionReferenceLabel: String {
        if snapshot?.selectionBounds != nil {
            return "Centered on current selection"
        }
        if snapshot?.visibleBounds != nil {
            return "Centered on visible scene bounds"
        }
        return "No visible splats"
    }

    var cutPlaneReferenceLabel: String {
        if snapshot?.visibleBounds != nil {
            return "Plane uses visible scene bounds"
        }
        return "No visible splats"
    }

    var cutPlaneSideLabel: String {
        switch cutPlaneSide {
        case .negative:
            return cutPlaneAxis.negativeLabel
        case .positive:
            return cutPlaneAxis.positiveLabel
        }
    }

    var cutPlaneCoordinateLabel: String {
        guard let bounds = cutPlaneBounds else { return "--" }
        let coordinate = cutPlaneAxis.interpolatedCoordinate(in: bounds, offset: cutPlaneOffset)
        return String(format: "%.3f", coordinate)
    }

    var alignmentReferenceLabel: String {
        hasSelection
            ? "Uses the current selection"
            : "Uses all visible editable splats when nothing is selected"
    }

    private var referenceBounds: SplatSelectionBounds? {
        snapshot?.selectionBounds ?? snapshot?.visibleBounds
    }

    private var cutPlaneBounds: SplatSelectionBounds? {
        snapshot?.visibleBounds
    }

    private static func defaultHalfExtents(for bounds: SplatSelectionBounds) -> SIMD3<Float> {
        let halfExtents = simd.max((bounds.max - bounds.min) * 0.5, SIMD3<Float>(repeating: 0.05))
        return halfExtents
    }

    func attach(renderer: MetalKitSceneRenderer?) {
        self.renderer = renderer
        Task {
            await refreshSnapshot()
        }
    }

    func refreshSnapshot() async {
        guard let renderer else {
            snapshot = nil
            isEditorAvailable = false
            return
        }
        snapshot = await renderer.currentEditorSnapshot()
        isEditorAvailable = snapshot != nil
    }

    func applySelectionSnapshot(_ snapshot: SplatEditorSnapshot?) {
        self.snapshot = snapshot
        isEditorAvailable = snapshot != nil
        if tool == .sphere || tool == .box {
            resetVolumeSelectionParameters()
        }
        statusMessage = nil
    }

    func setError(_ error: Error) {
        statusMessage = error.localizedDescription
    }

    func selectPoint(at point: CGPoint, in size: CGSize) {
        runSelection(query: .point(normalized: normalizedPoint(point, in: size), radius: 0.04), in: size)
    }

    func applySphereSelection() {
        guard let bounds = referenceBounds else {
            statusMessage = "No visible splats"
            return
        }

        runSelection(
            query: .sphere(center: bounds.center, radius: max(sphereRadius, 0.05))
        )
    }

    func applyBoxSelection() {
        guard let bounds = referenceBounds else {
            statusMessage = "No visible splats"
            return
        }

        runSelection(
            query: .box(
                center: bounds.center,
                extents: SIMD3<Float>(
                    max(boxExtentX, 0.05),
                    max(boxExtentY, 0.05),
                    max(boxExtentZ, 0.05)
                )
            )
        )
    }

    func resetVolumeSelectionParameters() {
        let halfExtents = referenceBounds.map(Self.defaultHalfExtents(for:)) ?? SIMD3<Float>(repeating: 0.25)
        sphereRadius = max(0.05, max(halfExtents.x, max(halfExtents.y, halfExtents.z)))
        boxExtentX = halfExtents.x
        boxExtentY = halfExtents.y
        boxExtentZ = halfExtents.z
    }

    func resetCutPlaneParameters() {
        cutPlaneOffset = 0.5
    }

    func selectCutPlaneSide() {
        guard let plane = currentCutPlane() else {
            statusMessage = "No visible splats"
            return
        }

        runEditorAction { renderer in
            try await renderer.selectEditableSplats(plane: plane, side: self.cutPlaneSide, mode: self.combineMode)
        }
    }

    func applyCutPlaneDeletion() {
        guard let plane = currentCutPlane() else {
            statusMessage = "No visible splats"
            return
        }

        runEditorAction { renderer in
            try await renderer.cutEditableSplats(plane: plane, side: self.cutPlaneSide)
        }
    }

    func centerSelectionXYZ() {
        runAlignmentAction { bounds in
            (
                transform: SplatEditTransform(translation: -bounds.center),
                pivot: bounds.center
            )
        }
    }

    func centerSelectionXZAndFloor() {
        runAlignmentAction { bounds in
            let translation = SIMD3<Float>(-bounds.center.x, -bounds.min.y, -bounds.center.z)
            return (
                transform: SplatEditTransform(translation: translation),
                pivot: bounds.center
            )
        }
    }

    func floorSelection() {
        runAlignmentAction { bounds in
            (
                transform: SplatEditTransform(translation: SIMD3<Float>(0, -bounds.min.y, 0)),
                pivot: bounds.center
            )
        }
    }

    func rotateSelectionQuarterTurn(clockwise: Bool) {
        runAlignmentAction { bounds in
            let angle: Float = clockwise ? (.pi / 2) : (-.pi / 2)
            return (
                transform: SplatEditTransform(rotation: simd_quatf(angle: angle, axis: self.alignmentAxis.unitNormal)),
                pivot: bounds.center
            )
        }
    }

    func rotateSelectionHalfTurn() {
        runAlignmentAction { bounds in
            (
                transform: SplatEditTransform(rotation: simd_quatf(angle: .pi, axis: self.alignmentAxis.unitNormal)),
                pivot: bounds.center
            )
        }
    }

    func clearPolygon() {
        polygonPoints.removeAll(keepingCapacity: true)
        overlayPreview = nil
        statusMessage = nil
    }

    func commitPolygonSelection(in size: CGSize) {
        let points = polygonPoints
        guard size != .zero else {
            statusMessage = "Polygon overlay is not ready"
            return
        }
        guard let mask = makeLassoMask(points: points, size: size) else {
            statusMessage = "Polygon needs at least three points"
            return
        }
        polygonPoints.removeAll(keepingCapacity: true)
        runSelection(query: .mask(alphaMask: mask, size: maskDimensions(for: size)), in: size)
    }

    func commitPolygonSelection() {
        commitPolygonSelection(in: overlayCanvasSize)
    }

    func clearMeasure() {
        measurePoints.removeAll(keepingCapacity: true)
        statusMessage = nil
    }

    func hideSelection() {
        runEditorAction { renderer in
            try await renderer.hideSelectedEditableSplats()
        }
    }

    func lockSelection() {
        runEditorAction { renderer in
            try await renderer.lockSelectedEditableSplats()
        }
    }

    func selectAll() {
        runEditorAction { renderer in
            try await renderer.selectAllEditableSplats()
        }
    }

    func clearSelection() {
        runEditorAction { renderer in
            try await renderer.clearEditableSelection()
        }
    }

    func invertSelection() {
        runEditorAction { renderer in
            try await renderer.invertEditableSelection()
        }
    }

    func unhideAll() {
        runEditorAction { renderer in
            try await renderer.unhideAllEditableSplats()
        }
    }

    func unlockAll() {
        runEditorAction { renderer in
            try await renderer.unlockAllEditableSplats()
        }
    }

    func deleteSelection() {
        runEditorAction { renderer in
            try await renderer.deleteSelectedEditableSplats()
        }
    }

    func duplicateSelection() {
        runEditorAction { renderer in
            try await renderer.duplicateSelectedEditableSplats()
        }
    }

    func separateSelection() {
        runEditorAction { renderer in
            try await renderer.separateSelectedEditableSplats()
        }
    }

    func undo() {
        runEditorAction { renderer in
            try await renderer.undoEditableChange()
        }
    }

    func redo() {
        runEditorAction { renderer in
            try await renderer.redoEditableChange()
        }
    }

    func exportScene() {
        guard let renderer else { return }
        Task {
            do {
                if let url = try await renderer.exportEditedScene() {
                    shareSheetItem = ShareSheetPayload(url: url)
                    statusMessage = nil
                }
            } catch {
                setError(error)
            }
        }
    }

    fileprivate func updateOverlayGesture(_ value: DragGesture.Value, in size: CGSize) {
        overlayCanvasSize = size
        switch tool {
        case .rect:
            overlayPreview = .rect(CGRect(
                x: min(value.startLocation.x, value.location.x),
                y: min(value.startLocation.y, value.location.y),
                width: abs(value.location.x - value.startLocation.x),
                height: abs(value.location.y - value.startLocation.y)
            ))
        case .brush:
            var points = overlayPreviewPoints
            points.append(value.location)
            overlayPreview = .brush(points)
        case .lasso:
            var points = overlayPreviewPoints
            points.append(value.location)
            overlayPreview = .lasso(points)
        default:
            break
        }
    }

    fileprivate func handleOverlayTap(at point: CGPoint, in size: CGSize) {
        overlayCanvasSize = size
        switch tool {
        case .flood:
            guard let renderer else { return }
            Task {
                do {
                    let snapshot = try await renderer.selectEditableFlood(
                        screenPoint: point,
                        threshold: floodThreshold,
                        mode: combineMode,
                        renderSize: size
                    )
                    applySelectionSnapshot(snapshot)
                } catch {
                    setError(error)
                }
            }
        case .eyedropper:
            guard let renderer else { return }
            Task {
                do {
                    let snapshot = try await renderer.selectEditableColorMatch(
                        screenPoint: point,
                        threshold: eyedropperThreshold,
                        mode: combineMode,
                        renderSize: size
                    )
                    applySelectionSnapshot(snapshot)
                } catch {
                    setError(error)
                }
            }
        case .polygon:
            if let first = polygonPoints.first,
               polygonPoints.count >= 3,
               distance(first, point) <= 24 {
                commitPolygonSelection(in: size)
                return
            }
            polygonPoints.append(point)
        case .measure:
            guard let renderer else { return }
            Task {
                do {
                    guard let pickedPoint = try await renderer.pickEditablePoint(screenPoint: point, renderSize: size) else {
                        statusMessage = "No splat at tap location"
                        return
                    }
                    measurePoints.append(MeasuredPoint(screenPoint: point, worldPoint: pickedPoint.position))
                    statusMessage = nil
                } catch {
                    setError(error)
                }
            }
        default:
            break
        }
    }

    fileprivate func finishOverlayGesture(_ value: DragGesture.Value, in size: CGSize) {
        overlayCanvasSize = size
        defer { overlayPreview = nil }

        switch tool {
        case .rect:
            let minPoint = CGPoint(x: min(value.startLocation.x, value.location.x), y: min(value.startLocation.y, value.location.y))
            let maxPoint = CGPoint(x: max(value.startLocation.x, value.location.x), y: max(value.startLocation.y, value.location.y))
            runSelection(
                query: .rect(
                    normalizedMin: normalizedPoint(minPoint, in: size),
                    normalizedMax: normalizedPoint(maxPoint, in: size)
                ),
                in: size
            )
        case .brush:
            let points = overlayPreviewPoints
            guard let mask = makeBrushMask(points: points, size: size) else { return }
            runSelection(query: .mask(alphaMask: mask, size: maskDimensions(for: size)), in: size)
        case .lasso:
            let points = overlayPreviewPoints
            guard let mask = makeLassoMask(points: points, size: size) else { return }
            runSelection(query: .mask(alphaMask: mask, size: maskDimensions(for: size)), in: size)
        default:
            break
        }
    }

    private var overlayPreviewPoints: [CGPoint] {
        switch overlayPreview {
        case let .brush(points), let .lasso(points):
            return points
        default:
            return []
        }
    }

    private func runSelection(query: SplatSelectionQuery, in size: CGSize) {
        runEditorAction { renderer in
            try await renderer.selectEditableSplats(query: query, mode: self.combineMode, renderSize: size)
        }
    }

    private func runSelection(query: SplatSelectionQuery) {
        runEditorAction { renderer in
            try await renderer.selectEditableSplats(query: query, mode: self.combineMode)
        }
    }

    private func runEditorAction(_ operation: @escaping (MetalKitSceneRenderer) async throws -> SplatEditorSnapshot?) {
        guard let renderer else { return }
        Task {
            do {
                let snapshot = try await operation(renderer)
                applySelectionSnapshot(snapshot)
            } catch {
                setError(error)
            }
        }
    }

    private func runAlignmentAction(_ build: @escaping (SplatSelectionBounds) -> (transform: SplatEditTransform, pivot: SIMD3<Float>)) {
        guard let renderer else { return }
        Task {
            do {
                guard let bounds = await renderer.currentEditableAlignmentBounds() else {
                    statusMessage = "No editable splats"
                    return
                }

                let plan = build(bounds)
                let snapshot = try await renderer.applyEditableAlignmentTransform(plan.transform, pivot: plan.pivot)
                applySelectionSnapshot(snapshot)
            } catch {
                setError(error)
            }
        }
    }

    private func normalizedPoint(_ point: CGPoint, in size: CGSize) -> SIMD2<Float> {
        let safeWidth = max(size.width, 1)
        let safeHeight = max(size.height, 1)
        return SIMD2<Float>(
            Float(min(max(point.x / safeWidth, 0), 1)),
            Float(min(max(point.y / safeHeight, 0), 1))
        )
    }

    private func maskDimensions(for size: CGSize) -> SIMD2<Int> {
        SIMD2<Int>(max(1, Int(size.width.rounded(.up))), max(1, Int(size.height.rounded(.up))))
    }

    private func makeBrushMask(points: [CGPoint], size: CGSize) -> Data? {
        guard points.count >= 1 else { return nil }
        return makeMaskData(size: size) { context in
            context.setLineWidth(28)
            context.setLineCap(.round)
            context.setLineJoin(.round)
            context.setStrokeColor(gray: 1, alpha: 1)
            if points.count == 1 {
                let point = points[0]
                let radius: CGFloat = 14
                context.fillEllipse(in: CGRect(x: point.x - radius, y: point.y - radius, width: radius * 2, height: radius * 2))
                return
            }
            context.beginPath()
            context.addLines(between: points)
            context.strokePath()
        }
    }

    private func makeLassoMask(points: [CGPoint], size: CGSize) -> Data? {
        guard points.count >= 3 else { return nil }
        return makeMaskData(size: size) { context in
            context.setFillColor(gray: 1, alpha: 1)
            context.beginPath()
            context.addLines(between: points)
            context.closePath()
            context.fillPath()
        }
    }

    private func makeMaskData(size: CGSize, draw: (CGContext) -> Void) -> Data? {
        let dimensions = maskDimensions(for: size)
        var bytes = [UInt8](repeating: 0, count: dimensions.x * dimensions.y)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let createdContext = bytes.withUnsafeMutableBytes { rawBuffer in
            CGContext(
                data: rawBuffer.baseAddress,
                width: dimensions.x,
                height: dimensions.y,
                bitsPerComponent: 8,
                bytesPerRow: dimensions.x,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.none.rawValue
            )
        }
        guard let context = createdContext else { return nil }
        context.translateBy(x: 0, y: CGFloat(dimensions.y))
        context.scaleBy(x: 1, y: -1)
        draw(context)
        return Data(bytes)
    }

    private func distance(_ lhs: CGPoint, _ rhs: CGPoint) -> CGFloat {
        hypot(lhs.x - rhs.x, lhs.y - rhs.y)
    }

    private func currentCutPlane() -> SplatCutPlane? {
        guard let bounds = cutPlaneBounds else { return nil }
        return SplatCutPlane(
            point: cutPlaneAxis.planePoint(in: bounds, offset: cutPlaneOffset),
            normal: cutPlaneAxis.unitNormal
        )
    }

    fileprivate func projectedPlaneGuide(in size: CGSize) -> [CGPoint]? {
        guard tool == .planeCut,
              let renderer,
              let bounds = cutPlaneBounds else {
            return nil
        }

        let coordinate = cutPlaneAxis.interpolatedCoordinate(in: bounds, offset: cutPlaneOffset)
        let corners: [SIMD3<Float>]
        switch cutPlaneAxis {
        case .x:
            corners = [
                SIMD3<Float>(coordinate, bounds.min.y, bounds.min.z),
                SIMD3<Float>(coordinate, bounds.min.y, bounds.max.z),
                SIMD3<Float>(coordinate, bounds.max.y, bounds.max.z),
                SIMD3<Float>(coordinate, bounds.max.y, bounds.min.z)
            ]
        case .y:
            corners = [
                SIMD3<Float>(bounds.min.x, coordinate, bounds.min.z),
                SIMD3<Float>(bounds.max.x, coordinate, bounds.min.z),
                SIMD3<Float>(bounds.max.x, coordinate, bounds.max.z),
                SIMD3<Float>(bounds.min.x, coordinate, bounds.max.z)
            ]
        case .z:
            corners = [
                SIMD3<Float>(bounds.min.x, bounds.min.y, coordinate),
                SIMD3<Float>(bounds.max.x, bounds.min.y, coordinate),
                SIMD3<Float>(bounds.max.x, bounds.max.y, coordinate),
                SIMD3<Float>(bounds.min.x, bounds.max.y, coordinate)
            ]
        }

        let projected = corners.compactMap { renderer.projectEditableGuidePoint($0, renderSize: size) }
        return projected.count == corners.count ? projected : nil
    }

    fileprivate func projectedPlaneNormalGuide(in size: CGSize) -> (start: CGPoint, end: CGPoint)? {
        guard tool == .planeCut,
              let renderer,
              let bounds = cutPlaneBounds else {
            return nil
        }

        let start3D = cutPlaneAxis.planePoint(in: bounds, offset: cutPlaneOffset)
        let span = bounds.max - bounds.min
        let axisSpan: Float
        switch cutPlaneAxis {
        case .x:
            axisSpan = max(span.y, span.z)
        case .y:
            axisSpan = max(span.x, span.z)
        case .z:
            axisSpan = max(span.x, span.y)
        }
        let direction: Float = cutPlaneSide == .positive ? 1 : -1
        let end3D = start3D + (cutPlaneAxis.unitNormal * max(axisSpan * 0.2, 0.1) * direction)

        guard let start = renderer.projectEditableGuidePoint(start3D, renderSize: size),
              let end = renderer.projectEditableGuidePoint(end3D, renderSize: size) else {
            return nil
        }

        return (start, end)
    }

    fileprivate func projectedBoxGuideSegments(in size: CGSize) -> [(CGPoint, CGPoint)] {
        guard tool == .box,
              let renderer,
              let bounds = referenceBounds else {
            return []
        }

        let center = bounds.center
        let extents = SIMD3<Float>(
            max(boxExtentX, 0.05),
            max(boxExtentY, 0.05),
            max(boxExtentZ, 0.05)
        )
        let corners = [
            SIMD3<Float>(center.x - extents.x, center.y - extents.y, center.z - extents.z),
            SIMD3<Float>(center.x + extents.x, center.y - extents.y, center.z - extents.z),
            SIMD3<Float>(center.x + extents.x, center.y + extents.y, center.z - extents.z),
            SIMD3<Float>(center.x - extents.x, center.y + extents.y, center.z - extents.z),
            SIMD3<Float>(center.x - extents.x, center.y - extents.y, center.z + extents.z),
            SIMD3<Float>(center.x + extents.x, center.y - extents.y, center.z + extents.z),
            SIMD3<Float>(center.x + extents.x, center.y + extents.y, center.z + extents.z),
            SIMD3<Float>(center.x - extents.x, center.y + extents.y, center.z + extents.z)
        ]
        let projected = corners.map { renderer.projectEditableGuidePoint($0, renderSize: size) }
        let edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        return edges.compactMap { startIndex, endIndex in
            guard let start = projected[startIndex], let end = projected[endIndex] else { return nil }
            return (start, end)
        }
    }
}

private struct SplatEditingToolbar: View {
    @ObservedObject var controller: SceneEditingController

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Picker("Combine Mode", selection: $controller.combineMode) {
                Text("Replace").tag(SelectionCombineMode.replace)
                Text("Add").tag(SelectionCombineMode.add)
                Text("Subtract").tag(SelectionCombineMode.subtract)
            }
            .pickerStyle(.segmented)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(SceneEditingController.Tool.allCases) { tool in
                        Button {
                            controller.tool = tool
                        } label: {
                            VStack(spacing: 4) {
                                Image(systemName: tool.systemImage)
                                    .font(.system(size: 16, weight: .semibold))
                                Text(tool.label)
                                    .font(.caption2)
                            }
                            .frame(width: 64, height: 54)
                            .background(controller.tool == tool ? Color.blue.opacity(0.85) : Color.black.opacity(0.45))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        }
                    }

                    actionButton("All", systemImage: "checklist", disabled: !controller.isEditorAvailable, action: controller.selectAll)
                    actionButton("None", systemImage: "xmark.circle", disabled: !controller.isEditorAvailable, action: controller.clearSelection)
                    actionButton("Invert", systemImage: "arrow.left.arrow.right.square", disabled: !controller.isEditorAvailable, action: controller.invertSelection)
                    actionButton("Hide", systemImage: "eye.slash", disabled: !controller.hasSelection, action: controller.hideSelection)
                    actionButton("Show", systemImage: "eye", disabled: !controller.isEditorAvailable, action: controller.unhideAll)
                    actionButton("Lock", systemImage: "lock", disabled: !controller.hasSelection, action: controller.lockSelection)
                    actionButton("Unlock", systemImage: "lock.open", disabled: !controller.isEditorAvailable, action: controller.unlockAll)
                    actionButton("Duplicate", systemImage: "plus.square.on.square", disabled: !controller.hasSelection, action: controller.duplicateSelection)
                    actionButton("Separate", systemImage: "square.split.2x1", disabled: !controller.hasSelection, action: controller.separateSelection)
                    actionButton("Delete", systemImage: "trash", disabled: !controller.hasSelection, action: controller.deleteSelection)
                    actionButton("Undo", systemImage: "arrow.uturn.backward", disabled: !controller.isEditorAvailable, action: controller.undo)
                    actionButton("Redo", systemImage: "arrow.uturn.forward", disabled: !controller.isEditorAvailable, action: controller.redo)
                    actionButton("Export", systemImage: "square.and.arrow.up", disabled: !controller.isEditorAvailable, action: controller.exportScene)
                }
                .padding(.horizontal, 2)
            }

            toolConfigurationView
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    @ViewBuilder
    private var toolConfigurationView: some View {
        switch controller.tool {
        case .flood:
            VStack(alignment: .leading, spacing: 8) {
                Text("Tap a splat to grow a connected selection using opacity similarity.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                thresholdSlider(
                    label: "Opacity",
                    value: Binding(
                        get: { Double(controller.floodThreshold) },
                        set: { controller.floodThreshold = Float($0) }
                    )
                )
            }
        case .eyedropper:
            VStack(alignment: .leading, spacing: 8) {
                Text("Tap a splat to select visible splats with similar base color.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                thresholdSlider(
                    label: "Color",
                    value: Binding(
                        get: { Double(controller.eyedropperThreshold) },
                        set: { controller.eyedropperThreshold = Float($0) }
                    )
                )
            }
        case .sphere:
            VStack(alignment: .leading, spacing: 8) {
                Text(controller.selectionReferenceLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack {
                    Text("Radius")
                        .font(.subheadline)
                    Slider(value: Binding(
                        get: { Double(controller.sphereRadius) },
                        set: { controller.sphereRadius = Float($0) }
                    ), in: 0.05...5.0)
                    Text(String(format: "%.2f", controller.sphereRadius))
                        .font(.caption.monospacedDigit())
                        .frame(width: 42)
                }
                HStack(spacing: 10) {
                    actionButton("Reset", systemImage: "arrow.counterclockwise", disabled: !controller.isEditorAvailable, action: controller.resetVolumeSelectionParameters)
                    actionButton("Apply", systemImage: "checkmark.circle", disabled: !controller.isEditorAvailable, action: controller.applySphereSelection)
                }
            }
        case .box:
            VStack(alignment: .leading, spacing: 8) {
                Text(controller.selectionReferenceLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                extentSlider(label: "X", value: Binding(
                    get: { Double(controller.boxExtentX) },
                    set: { controller.boxExtentX = Float($0) }
                ))
                extentSlider(label: "Y", value: Binding(
                    get: { Double(controller.boxExtentY) },
                    set: { controller.boxExtentY = Float($0) }
                ))
                extentSlider(label: "Z", value: Binding(
                    get: { Double(controller.boxExtentZ) },
                    set: { controller.boxExtentZ = Float($0) }
                ))
                HStack(spacing: 10) {
                    actionButton("Reset", systemImage: "arrow.counterclockwise", disabled: !controller.isEditorAvailable, action: controller.resetVolumeSelectionParameters)
                    actionButton("Apply", systemImage: "checkmark.circle", disabled: !controller.isEditorAvailable, action: controller.applyBoxSelection)
                }
            }
        case .planeCut:
            VStack(alignment: .leading, spacing: 8) {
                Text(controller.cutPlaneReferenceLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Picker("Axis", selection: $controller.cutPlaneAxis) {
                    ForEach(SceneEditingController.PlaneAxis.allCases) { axis in
                        Text(axis.label).tag(axis)
                    }
                }
                .pickerStyle(.segmented)
                Picker("Side", selection: $controller.cutPlaneSide) {
                    Text(controller.cutPlaneAxis.negativeLabel).tag(SplatCutPlaneSide.negative)
                    Text(controller.cutPlaneAxis.positiveLabel).tag(SplatCutPlaneSide.positive)
                }
                .pickerStyle(.segmented)
                HStack {
                    Text("Plane")
                        .font(.subheadline)
                        .frame(width: 42, alignment: .leading)
                    Slider(value: Binding(
                        get: { Double(controller.cutPlaneOffset) },
                        set: { controller.cutPlaneOffset = Float($0) }
                    ), in: 0...1)
                    Text(controller.cutPlaneCoordinateLabel)
                        .font(.caption.monospacedDigit())
                        .frame(width: 54)
                }
                Text("Targets the \(controller.cutPlaneSideLabel.lowercased()) side of the plane.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 10) {
                    actionButton("Reset", systemImage: "arrow.counterclockwise", disabled: !controller.isEditorAvailable, action: controller.resetCutPlaneParameters)
                    actionButton("Select", systemImage: "line.3.horizontal.decrease.circle", disabled: !controller.isEditorAvailable, action: controller.selectCutPlaneSide)
                    actionButton("Cut", systemImage: "scissors", disabled: !controller.isEditorAvailable, action: controller.applyCutPlaneDeletion)
                }
            }
        case .align:
            VStack(alignment: .leading, spacing: 10) {
                Text(controller.alignmentReferenceLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 10) {
                    actionButton("Center", systemImage: "scope", disabled: !controller.isEditorAvailable, action: controller.centerSelectionXYZ)
                    actionButton("Center+Floor", systemImage: "move.3d", disabled: !controller.isEditorAvailable, action: controller.centerSelectionXZAndFloor)
                    actionButton("Floor", systemImage: "arrow.down.to.line", disabled: !controller.isEditorAvailable, action: controller.floorSelection)
                }
                VStack(alignment: .leading, spacing: 8) {
                    Text("Rotation Axis")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Picker("Rotation Axis", selection: $controller.alignmentAxis) {
                        ForEach(SceneEditingController.AlignmentAxis.allCases) { axis in
                            Text(axis.label).tag(axis)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                HStack(spacing: 10) {
                    actionButton("-90°", systemImage: "rotate.left", disabled: !controller.isEditorAvailable, action: {
                        controller.rotateSelectionQuarterTurn(clockwise: false)
                    })
                    actionButton("+90°", systemImage: "rotate.right", disabled: !controller.isEditorAvailable, action: {
                        controller.rotateSelectionQuarterTurn(clockwise: true)
                    })
                    actionButton("180°", systemImage: "rotate.3d", disabled: !controller.isEditorAvailable, action: controller.rotateSelectionHalfTurn)
                }
            }
        case .polygon:
            VStack(alignment: .leading, spacing: 8) {
                Text("Tap to add vertices. Tap the first point again or press Close to select.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 10) {
                    actionButton("Close", systemImage: "checkmark.circle", disabled: !controller.canCommitPolygonSelection, action: controller.commitPolygonSelection)
                    actionButton("Clear", systemImage: "xmark.circle", disabled: controller.polygonPoints.isEmpty, action: controller.clearPolygon)
                }
            }
        case .measure:
            VStack(alignment: .leading, spacing: 8) {
                Text(controller.measurementSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 10) {
                    actionButton("Clear", systemImage: "xmark.circle", disabled: controller.measurePoints.isEmpty, action: controller.clearMeasure)
                }
            }
        default:
            EmptyView()
        }
    }

    private func actionButton(_ title: String,
                              systemImage: String,
                              disabled: Bool,
                              action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: systemImage)
                    .font(.system(size: 16, weight: .semibold))
                Text(title)
                    .font(.caption2)
            }
            .frame(width: 64, height: 54)
            .background(disabled ? Color.gray.opacity(0.25) : Color.black.opacity(0.45))
            .foregroundColor(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        }
        .disabled(disabled)
    }

    private func extentSlider(label: String, value: Binding<Double>) -> some View {
        HStack {
            Text(label)
                .font(.subheadline.monospaced())
                .frame(width: 18)
            Slider(value: value, in: 0.05...5.0)
            Text(String(format: "%.2f", value.wrappedValue))
                .font(.caption.monospacedDigit())
                .frame(width: 42)
        }
    }

    private func thresholdSlider(label: String, value: Binding<Double>) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .frame(width: 52, alignment: .leading)
            Slider(value: value, in: 0.01...1.0)
            Text(String(format: "%.2f", value.wrappedValue))
                .font(.caption.monospacedDigit())
                .frame(width: 42)
        }
    }
}

private struct SplatEditingStatusChip: View {
    @ObservedObject var controller: SceneEditingController

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Selected \(controller.snapshot?.selectedCount ?? 0) / Visible \(controller.snapshot?.visibleCount ?? 0)")
                .font(.caption.weight(.semibold))
            if let message = controller.statusMessage {
                Text(message)
                    .font(.caption2)
                    .lineLimit(2)
            } else if controller.tool == .measure {
                Text(controller.measurementSummary)
                    .font(.caption2)
                    .lineLimit(2)
            } else {
                Text(controller.tool.label)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: Capsule())
    }
}

private struct SplatEditingOverlay: View {
    @ObservedObject var controller: SceneEditingController

    var body: some View {
        GeometryReader { geometry in
            Canvas { context, _ in
                if let planeGuide = controller.projectedPlaneGuide(in: geometry.size) {
                    var planePath = Path()
                    planePath.addLines(planeGuide)
                    planePath.closeSubpath()
                    context.fill(planePath, with: .color(.pink.opacity(0.10)))
                    context.stroke(planePath, with: .color(.pink), style: StrokeStyle(lineWidth: 2, dash: [8, 4]))
                }

                if let normalGuide = controller.projectedPlaneNormalGuide(in: geometry.size) {
                    var normalPath = Path()
                    normalPath.move(to: normalGuide.start)
                    normalPath.addLine(to: normalGuide.end)
                    context.stroke(normalPath, with: .color(.pink), style: StrokeStyle(lineWidth: 2))

                    let dx = normalGuide.end.x - normalGuide.start.x
                    let dy = normalGuide.end.y - normalGuide.start.y
                    let length = max(hypot(dx, dy), 0.001)
                    let ux = dx / length
                    let uy = dy / length
                    let arrowLength: CGFloat = 12
                    let arrowWidth: CGFloat = 6
                    let arrowPointA = CGPoint(
                        x: normalGuide.end.x - (ux * arrowLength) + (uy * arrowWidth),
                        y: normalGuide.end.y - (uy * arrowLength) - (ux * arrowWidth)
                    )
                    let arrowPointB = CGPoint(
                        x: normalGuide.end.x - (ux * arrowLength) - (uy * arrowWidth),
                        y: normalGuide.end.y - (uy * arrowLength) + (ux * arrowWidth)
                    )
                    var arrowPath = Path()
                    arrowPath.move(to: normalGuide.end)
                    arrowPath.addLine(to: arrowPointA)
                    arrowPath.addLine(to: arrowPointB)
                    arrowPath.closeSubpath()
                    context.fill(arrowPath, with: .color(.pink))
                }

                for (start, end) in controller.projectedBoxGuideSegments(in: geometry.size) {
                    var edgePath = Path()
                    edgePath.move(to: start)
                    edgePath.addLine(to: end)
                    context.stroke(edgePath, with: .color(.yellow), style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                }

                switch controller.overlayPreview {
                case let .rect(rect):
                    context.stroke(Path(rect), with: .color(.cyan), style: StrokeStyle(lineWidth: 2, dash: [8, 4]))
                    context.fill(Path(rect), with: .color(.cyan.opacity(0.12)))
                case let .brush(points):
                    var path = Path()
                    if let first = points.first {
                        path.move(to: first)
                        for point in points.dropFirst() {
                            path.addLine(to: point)
                        }
                    }
                    context.stroke(path, with: .color(.cyan), style: StrokeStyle(lineWidth: 24, lineCap: .round, lineJoin: .round))
                case let .lasso(points):
                    var path = Path()
                    if let first = points.first {
                        path.move(to: first)
                        for point in points.dropFirst() {
                            path.addLine(to: point)
                        }
                    }
                    context.stroke(path, with: .color(.cyan), style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                case .none:
                    break
                }

                if controller.tool == .polygon, !controller.polygonPoints.isEmpty {
                    var polygonPath = Path()
                    if let first = controller.polygonPoints.first {
                        polygonPath.move(to: first)
                        for point in controller.polygonPoints.dropFirst() {
                            polygonPath.addLine(to: point)
                        }
                    }
                    context.stroke(polygonPath, with: .color(.orange), style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                    for point in controller.polygonPoints {
                        let marker = Path(ellipseIn: CGRect(x: point.x - 5, y: point.y - 5, width: 10, height: 10))
                        context.fill(marker, with: .color(.orange))
                    }
                }

                if controller.tool == .measure, !controller.measurePoints.isEmpty {
                    var measurePath = Path()
                    if let first = controller.measurePoints.first {
                        measurePath.move(to: first.screenPoint)
                        for point in controller.measurePoints.dropFirst() {
                            measurePath.addLine(to: point.screenPoint)
                        }
                    }
                    context.stroke(measurePath, with: .color(.green), style: StrokeStyle(lineWidth: 2))
                    for point in controller.measurePoints {
                        let marker = Path(ellipseIn: CGRect(x: point.screenPoint.x - 6, y: point.screenPoint.y - 6, width: 12, height: 12))
                        context.fill(marker, with: .color(.green))
                    }
                }
            }
            .contentShape(Rectangle())
            .allowsHitTesting(controller.tool.usesOverlaySelection)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        guard controller.tool.usesDragOverlay else { return }
                        controller.updateOverlayGesture(value, in: geometry.size)
                    }
                    .onEnded { value in
                        if controller.tool.usesDragOverlay {
                            controller.finishOverlayGesture(value, in: geometry.size)
                        } else if controller.tool.usesTapOverlay {
                            controller.handleOverlayTap(at: value.location, in: geometry.size)
                        }
                    }
            )
        }
    }
}

#if os(iOS)
private struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
#endif

struct MetalKitRendererView: ViewRepresentable {
    var modelIdentifier: ModelIdentifier?
    @ObservedObject var editingController: SceneEditingController
    @Binding var fastSHEnabled: Bool
    @Binding var metal4BindlessEnabled: Bool
    @Binding var showDebugAABB: Bool
    @Binding var frustumCullingEnabled: Bool
    @Binding var meshShaderEnabled: Bool
    @Binding var ditheredTransparencyEnabled: Bool
    @Binding var metal4SortingEnabled: Bool
    @Binding var use2DGSMode: Bool
    @Binding var splatAnimationEnabled: Bool
    @Binding var splatAnimationEffectRawValue: UInt32
    @Binding var splatAnimationPlaying: Bool
    @Binding var splatAnimationSpeed: Double
    @Binding var splatAnimationIntensity: Double
    @Binding var splatAnimationResetCounter: Int
    @Binding var renderScale: CGFloat
    @Binding var adaptiveRenderScaleEnabled: Bool

    class Coordinator: NSObject {
        var renderer: MetalKitSceneRenderer?
        var parent: MetalKitRendererView
        // Store camera interaction state
        var lastPanLocation: CGPoint?
        var lastRotation: Angle = .zero
        var lastRollRotation: Float = 0.0
        var zoom: Float = 1.0
        private var pendingLoadTask: Task<Void, Never>?
        private var lastRequestedModel: ModelIdentifier?
        private var editTransformActive = false
        private var lastAppliedAnimationResetCounter = 0
#if os(iOS)
        private var frameTimeEMA: TimeInterval = 1.0 / 60.0
        private var hasFrameTimeSample = false
        private var lastAdaptiveScaleAdjustmentTime: TimeInterval = 0
        private let targetFrameTime: TimeInterval = 1.0 / 60.0
        private let minRenderScale: CGFloat = 0.55
        private let maxRenderScale: CGFloat = 1.0
        private let interactionRenderScaleFloor: CGFloat = 0.55
        private let interactionRenderScaleRestoreDelay: TimeInterval = 0.20
        private var activeCameraInteractionCount = 0
        private var temporaryInteractionRenderScale: CGFloat?
        private var restoreInteractionRenderScaleWorkItem: DispatchWorkItem?
#endif
        
        init(_ parent: MetalKitRendererView) {
            self.parent = parent
        }
        
        deinit {
            pendingLoadTask?.cancel()
        }
        
        @MainActor
        func updateSettings() {
            // Update renderer settings silently
            renderer?.fastSHSettings.enabled = parent.fastSHEnabled
            renderer?.syncFastSHSettings()
            renderer?.setMetal4Bindless(parent.metal4BindlessEnabled)
            renderer?.setDebugAABB(parent.showDebugAABB)
            renderer?.setFrustumCulling(parent.frustumCullingEnabled)
            renderer?.setMeshShader(parent.meshShaderEnabled)
            renderer?.setDitheredTransparency(parent.ditheredTransparencyEnabled)
            renderer?.setMetal4Sorting(parent.metal4SortingEnabled)
            renderer?.set2DGSMode(parent.use2DGSMode)
            renderer?.setSplatAnimation(
                effect: parent.selectedAnimationEffect,
                isPlaying: parent.splatAnimationEnabled && parent.splatAnimationPlaying,
                speed: Float(parent.splatAnimationSpeed),
                intensity: Float(parent.splatAnimationIntensity)
            )
            if lastAppliedAnimationResetCounter != parent.splatAnimationResetCounter {
                lastAppliedAnimationResetCounter = parent.splatAnimationResetCounter
                renderer?.resetSplatAnimation()
            }

#if os(iOS)
            renderer?.setInternalRenderScale(effectiveRenderScale)
#endif
            parent.editingController.attach(renderer: renderer)
        }
        
        @MainActor
        func loadModelIfNeeded() {
            guard let renderer else { return }
            guard lastRequestedModel != parent.modelIdentifier else { return }
            
            let requestedModel = parent.modelIdentifier
            lastRequestedModel = requestedModel
            pendingLoadTask?.cancel()
            pendingLoadTask = Task { [weak self, weak renderer] in
                do {
                    try await renderer?.load(requestedModel)
                    await MainActor.run {
                        self?.updateSettings()
                    }
                    await self?.parent.editingController.refreshSnapshot()
                } catch is CancellationError {
                    // Newer model request superseded this load.
                } catch {
                    await MainActor.run {
                        guard let self else { return }
                        if self.lastRequestedModel == requestedModel {
                            self.lastRequestedModel = nil
                        }
                    }
                    print("Error loading model: \(error.localizedDescription)")
                }
            }
        }
        
#if os(iOS)
        fileprivate var effectiveRenderScale: CGFloat {
            temporaryInteractionRenderScale ?? parent.renderScale
        }

        @MainActor
        private func beginCameraInteraction() {
            restoreInteractionRenderScaleWorkItem?.cancel()
            restoreInteractionRenderScaleWorkItem = nil
            activeCameraInteractionCount += 1

            guard activeCameraInteractionCount == 1 else { return }

            let baseScale = max(minRenderScale, min(parent.renderScale, maxRenderScale))
            let reducedScale = max(
                interactionRenderScaleFloor,
                min(baseScale * 0.85, baseScale - 0.08)
            )
            guard reducedScale < baseScale - 0.005 else { return }

            temporaryInteractionRenderScale = reducedScale
            renderer?.setInternalRenderScale(reducedScale)
        }

        @MainActor
        private func endCameraInteraction() {
            activeCameraInteractionCount = max(0, activeCameraInteractionCount - 1)
            guard activeCameraInteractionCount == 0 else { return }

            restoreInteractionRenderScaleWorkItem?.cancel()

            let workItem = DispatchWorkItem { [weak self] in
                guard let self else { return }
                guard self.activeCameraInteractionCount == 0 else { return }
                self.temporaryInteractionRenderScale = nil
                self.renderer?.setInternalRenderScale(self.parent.renderScale)
            }
            restoreInteractionRenderScaleWorkItem = workItem
            DispatchQueue.main.asyncAfter(deadline: .now() + interactionRenderScaleRestoreDelay,
                                          execute: workItem)
        }

        @MainActor
        func configureAdaptiveRenderScale() {
            guard parent.adaptiveRenderScaleEnabled else {
                renderer?.onFrameTimeUpdate = nil
                hasFrameTimeSample = false
                return
            }

            renderer?.onFrameTimeUpdate = { [weak self] frameTime in
                self?.handleFrameTimeSample(frameTime)
            }
        }
        
        @MainActor
        private func handleFrameTimeSample(_ frameTime: TimeInterval) {
            guard frameTime.isFinite, frameTime > 0 else { return }
            
            if !hasFrameTimeSample {
                frameTimeEMA = frameTime
                hasFrameTimeSample = true
                return
            }
            
            let emaAlpha: TimeInterval = 0.12
            frameTimeEMA += (frameTime - frameTimeEMA) * emaAlpha
            
            let now = Date.timeIntervalSinceReferenceDate
            let currentScale = max(minRenderScale, min(parent.renderScale, maxRenderScale))
            let overloadThreshold = targetFrameTime * 1.08
            let recoveryThreshold = targetFrameTime * 0.82
            
            var adjustedScale = currentScale
            var cooldown: TimeInterval?
            
            if frameTimeEMA > overloadThreshold {
                let pressure = min((frameTimeEMA / targetFrameTime) - 1.0, 1.0)
                let step = CGFloat(0.03 + (0.06 * pressure))
                adjustedScale = max(minRenderScale, currentScale - step)
                cooldown = 0.35
            } else if frameTimeEMA < recoveryThreshold {
                let headroom = min((targetFrameTime - frameTimeEMA) / targetFrameTime, 1.0)
                let step = CGFloat(0.01 + (0.02 * headroom))
                adjustedScale = min(maxRenderScale, currentScale + step)
                cooldown = 1.20
            }
            
            guard let cooldown else { return }
            guard now - lastAdaptiveScaleAdjustmentTime >= cooldown else { return }
            adjustedScale = (adjustedScale * 20).rounded() / 20
            guard abs(adjustedScale - currentScale) >= 0.01 else { return }
            
            lastAdaptiveScaleAdjustmentTime = now
            parent.renderScale = adjustedScale
        }
#endif
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    private var selectedAnimationEffect: SplatAnimationEffect? {
        guard splatAnimationEnabled else { return nil }
        return SplatAnimationEffect(rawValue: splatAnimationEffectRawValue) ?? .spread
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
        applyPreferredFrameRate(to: view)
        updateView(context.coordinator)
    }
#endif

    private func makeView(_ coordinator: Coordinator) -> MTKView {
        let metalKitView = MTKView()

        if let metalDevice = MTLCreateSystemDefaultDevice() {
            metalKitView.device = metalDevice
        }
#if os(iOS)
        // MetalFX upscaler can write to drawable textures; allow non-framebuffer access.
        metalKitView.framebufferOnly = false
        applyPreferredFrameRate(to: metalKitView)
        DispatchQueue.main.async {
            self.applyPreferredFrameRate(to: metalKitView)
        }
#endif

        let renderer = MetalKitSceneRenderer(metalKitView)
        coordinator.renderer = renderer
        editingController.attach(renderer: renderer)
        metalKitView.delegate = renderer
        
        // Apply initial settings
        renderer?.fastSHSettings.enabled = fastSHEnabled
        renderer?.setMetal4Bindless(metal4BindlessEnabled)
        renderer?.setDebugAABB(showDebugAABB)
        renderer?.setFrustumCulling(frustumCullingEnabled)
        renderer?.setMeshShader(meshShaderEnabled)
        renderer?.setDitheredTransparency(ditheredTransparencyEnabled)
        renderer?.setMetal4Sorting(metal4SortingEnabled)
        renderer?.set2DGSMode(use2DGSMode)
        renderer?.setSplatAnimation(
            effect: selectedAnimationEffect,
            isPlaying: splatAnimationEnabled && splatAnimationPlaying,
            speed: Float(splatAnimationSpeed),
            intensity: Float(splatAnimationIntensity)
        )

#if os(iOS)
        renderer?.setInternalRenderScale(coordinator.effectiveRenderScale)
        coordinator.configureAdaptiveRenderScale()
#endif

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
        let singleTapGesture = UITapGestureRecognizer(target: coordinator, action: #selector(Coordinator.handleSingleTap(_:)))
        singleTapGesture.numberOfTapsRequired = 1
        singleTapGesture.require(toFail: doubleTapGesture)
        singleTapGesture.delegate = coordinator
        metalKitView.addGestureRecognizer(singleTapGesture)
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

        coordinator.loadModelIfNeeded()

        return metalKitView
    }

    private func updateView(_ coordinator: Coordinator) {
        guard coordinator.renderer != nil else { return }
        
        // Update coordinator's parent reference to get latest state
        coordinator.parent = self
        editingController.attach(renderer: coordinator.renderer)
        
        // Update settings when the view updates
        coordinator.updateSettings()
#if os(iOS)
        coordinator.configureAdaptiveRenderScale()
#endif
        coordinator.loadModelIfNeeded()
    }

#if os(iOS)
    private func applyPreferredFrameRate(to view: MTKView) {
        struct LogState {
            static var didLog = false
        }
        let desiredMax = 60
        let targetFPS: Int
        if let screen = view.window?.windowScene?.screen {
            targetFPS = min(screen.maximumFramesPerSecond, desiredMax)
            if !LogState.didLog {
                LogState.didLog = true
                print("MetalKitSceneView: screen max \(screen.maximumFramesPerSecond)fps, preferred \(targetFPS)fps")
            }
        } else {
            targetFPS = desiredMax
        }
        
        if view.preferredFramesPerSecond != targetFPS {
            view.preferredFramesPerSecond = targetFPS
        }
    }
#endif
}

#if os(iOS)
    // MARK: - Coordinator Gesture Handling
    extension MetalKitRendererView.Coordinator: UIGestureRecognizerDelegate {
        // Static keys for associated objects (stable addresses, no force unwrap)
        private static nonisolated(unsafe) var verticalRotationKey: UInt8 = 0
        private static nonisolated(unsafe) var pan2TranslationKey: UInt8 = 0

        private var shouldHandleMoveTransform: Bool {
            parent.editingController.tool == .move && parent.editingController.hasSelection
        }

        private var shouldHandleScaleTransform: Bool {
            parent.editingController.tool == .scale && parent.editingController.hasSelection
        }

        private var shouldHandleRotationTransform: Bool {
            parent.editingController.tool == .rotate && parent.editingController.hasSelection
        }

        @MainActor
        private func beginEditTransformIfNeeded() {
            guard !editTransformActive, let renderer else { return }
            Task { [weak self] in
                guard let self else { return }
                let started = await renderer.beginEditableTransformIfPossible()
                await MainActor.run {
                    self.editTransformActive = started
                }
            }
        }

        @MainActor
        private func commitEditTransformIfNeeded() {
            guard editTransformActive, let renderer else { return }
            editTransformActive = false
            Task { [weak self] in
                guard let self else { return }
                do {
                    let snapshot = try await renderer.commitEditableTransform()
                    await MainActor.run {
                        self.parent.editingController.applySelectionSnapshot(snapshot)
                    }
                } catch {
                    await MainActor.run {
                        self.parent.editingController.setError(error)
                    }
                }
            }
        }

        @MainActor
        private func cancelEditTransformIfNeeded() {
            guard editTransformActive, let renderer else { return }
            editTransformActive = false
            Task { [weak self] in
                await renderer.cancelEditableTransform()
                await self?.parent.editingController.refreshSnapshot()
            }
        }

        @MainActor @objc func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let renderer = renderer else { return }
            let location = gesture.location(in: gesture.view)

            if shouldHandleMoveTransform, gesture.numberOfTouches <= 1 {
                switch gesture.state {
                case .began:
                    beginEditTransformIfNeeded()
                case .changed:
                    let translation = gesture.translation(in: gesture.view)
                    let renderSize = gesture.view?.bounds.size ?? .zero
                    Task { [weak self, weak renderer] in
                        guard let self, let renderer else { return }
                        do {
                            try await renderer.updateEditableTranslation(screenDelta: translation, renderSize: renderSize)
                        } catch {
                            await MainActor.run {
                                self.parent.editingController.setError(error)
                            }
                        }
                    }
                case .ended:
                    commitEditTransformIfNeeded()
                case .cancelled, .failed:
                    cancelEditTransformIfNeeded()
                default:
                    break
                }
                return
            }

            if parent.editingController.tool == .measure, gesture.state == .began {
                parent.editingController.clearMeasure()
            }

            // --- Call endUserInteraction on gesture end ---
            if gesture.state == .ended || gesture.state == .cancelled || gesture.state == .failed {
                renderer.endUserInteraction()
                endCameraInteraction()
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
                    beginCameraInteraction()
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
                    beginCameraInteraction()
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

            if shouldHandleScaleTransform {
                switch gesture.state {
                case .began:
                    beginEditTransformIfNeeded()
                case .changed:
                    Task { [weak self, weak renderer] in
                        guard let self, let renderer else { return }
                        do {
                            try await renderer.updateEditableScale(Float(gesture.scale))
                        } catch {
                            await MainActor.run {
                                self.parent.editingController.setError(error)
                            }
                        }
                    }
                case .ended:
                    commitEditTransformIfNeeded()
                case .cancelled, .failed:
                    cancelEditTransformIfNeeded()
                default:
                    break
                }
                return
            }

            if parent.editingController.tool == .measure, gesture.state == .began {
                parent.editingController.clearMeasure()
            }

            // --- Call endUserInteraction on gesture end ---
            if gesture.state == .ended || gesture.state == .cancelled || gesture.state == .failed {
                renderer.endUserInteraction()
                endCameraInteraction()
                return // Don't process further if ended/cancelled
            }
            // --- End change ---

            switch gesture.state {
            case .began:
                beginCameraInteraction()
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

            if shouldHandleRotationTransform {
                switch gesture.state {
                case .began:
                    beginEditTransformIfNeeded()
                case .changed:
                    let renderSize = gesture.view?.bounds.size ?? .zero
                    Task { [weak self, weak renderer] in
                        guard let self, let renderer else { return }
                        do {
                            try await renderer.updateEditableRotation(angle: Float(gesture.rotation), renderSize: renderSize)
                        } catch {
                            await MainActor.run {
                                self.parent.editingController.setError(error)
                            }
                        }
                    }
                case .ended:
                    commitEditTransformIfNeeded()
                case .cancelled, .failed:
                    cancelEditTransformIfNeeded()
                default:
                    break
                }
                return
            }

            if parent.editingController.tool == .measure, gesture.state == .began {
                parent.editingController.clearMeasure()
            }

            // --- Call endUserInteraction on gesture end ---
            if gesture.state == .ended || gesture.state == .cancelled || gesture.state == .failed {
                renderer.endUserInteraction()
                endCameraInteraction()
                return // Don't process further if ended/cancelled
            }
            // --- End change ---

            switch gesture.state {
            case .began:
                beginCameraInteraction()
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

        @MainActor @objc func handleSingleTap(_ gesture: UITapGestureRecognizer) {
            guard parent.editingController.tool == .select,
                  let view = gesture.view else { return }
            parent.editingController.selectPoint(at: gesture.location(in: view), in: view.bounds.size)
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
