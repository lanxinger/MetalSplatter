import Foundation
import simd
import SplatIO

public struct EditableSplatState: OptionSet, Sendable {
    public let rawValue: UInt32

    public init(rawValue: UInt32) {
        self.rawValue = rawValue
    }

    public static let selected = EditableSplatState(rawValue: 1 << 0)
    public static let hidden = EditableSplatState(rawValue: 1 << 1)
    public static let locked = EditableSplatState(rawValue: 1 << 2)
    public static let deleted = EditableSplatState(rawValue: 1 << 3)
}

public enum SelectionCombineMode: UInt32, Sendable {
    case replace = 0
    case add = 1
    case subtract = 2
}

public enum SplatCutPlaneSide: String, Sendable {
    case negative
    case positive
}

public struct SplatCutPlane: Sendable {
    public var point: SIMD3<Float>
    public var normal: SIMD3<Float>

    public init(point: SIMD3<Float>, normal: SIMD3<Float>) {
        self.point = point

        let length = simd_length(normal)
        if length > .ulpOfOne {
            self.normal = normal / length
        } else {
            self.normal = SIMD3<Float>(0, 1, 0)
        }
    }
}

public enum SplatSelectionQuery: Sendable {
    case point(normalized: SIMD2<Float>, radius: Float = 0.04)
    case rect(normalizedMin: SIMD2<Float>, normalizedMax: SIMD2<Float>)
    case mask(alphaMask: Data, size: SIMD2<Int>)
    case sphere(center: SIMD3<Float>, radius: Float)
    case box(center: SIMD3<Float>, extents: SIMD3<Float>)
}

public struct SplatEditTransform: Sendable {
    public var translation: SIMD3<Float>
    public var rotation: simd_quatf
    public var scale: SIMD3<Float>

    public init(
        translation: SIMD3<Float> = .zero,
        rotation: simd_quatf = simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0)),
        scale: SIMD3<Float> = SIMD3<Float>(repeating: 1)
    ) {
        self.translation = translation
        self.rotation = rotation.normalized
        self.scale = scale
    }

    public static let identity = SplatEditTransform()
}

public struct SplatSelectionBounds: Sendable {
    public var min: SIMD3<Float>
    public var max: SIMD3<Float>

    public init(min: SIMD3<Float>, max: SIMD3<Float>) {
        self.min = min
        self.max = max
    }

    public var center: SIMD3<Float> {
        (min + max) * 0.5
    }
}

public struct SplatEditorSnapshot: Sendable {
    public var totalCount: Int
    public var visibleCount: Int
    public var selectedCount: Int
    public var hiddenCount: Int
    public var deletedCount: Int
    public var lockedCount: Int
    public var visibleBounds: SplatSelectionBounds?
    public var selectionBounds: SplatSelectionBounds?
}

public enum SplatEditorError: LocalizedError, Sendable {
    case rendererPointCountMismatch(renderer: Int, editor: Int)
    case previewTransformNotActive
    case invalidMaskSize
    case selectionEngineUnavailable

    public var errorDescription: String? {
        switch self {
        case let .rendererPointCountMismatch(renderer, editor):
            return "Renderer/editor point count mismatch: renderer has \(renderer), editor has \(editor)"
        case .previewTransformNotActive:
            return "No preview transform is active"
        case .invalidMaskSize:
            return "Mask data does not match the declared dimensions"
        case .selectionEngineUnavailable:
            return "The Metal selection engine is unavailable"
        }
    }
}

struct EditableSplatStateChange: Sendable {
    var index: Int
    var oldState: EditableSplatState
    var newState: EditableSplatState
}

struct EditableSplatPointChange: Sendable {
    var index: Int
    var oldPoint: SplatScenePoint
    var newPoint: SplatScenePoint
}

struct EditableSplatStoreSnapshot: Sendable {
    var points: [SplatScenePoint]
    var states: [EditableSplatState]
    var sceneIndices: [UInt32]
}

enum SplatEditHistoryEntry: Sendable {
    case states([EditableSplatStateChange])
    case points([EditableSplatPointChange])
    case snapshot(EditableSplatStoreSnapshot)
}

struct EditableSplatStore: Sendable {
    struct ProjectedCandidate: Sendable {
        var index: Int
        var normalized: SIMD2<Float>
        var opacity: Float
        var radius: Float
    }

    var points: [SplatScenePoint]
    var states: [EditableSplatState]
    var sceneIndices: [UInt32]

    init(points: [SplatScenePoint], sceneIndices: [UInt32]? = nil) {
        self.points = points
        self.states = Array(repeating: [], count: points.count)
        if let sceneIndices, sceneIndices.count == points.count {
            self.sceneIndices = sceneIndices
        } else {
            self.sceneIndices = Array(repeating: 0, count: points.count)
        }
    }

    var snapshot: EditableSplatStoreSnapshot {
        EditableSplatStoreSnapshot(points: points, states: states, sceneIndices: sceneIndices)
    }

    mutating func restore(_ snapshot: EditableSplatStoreSnapshot) {
        points = snapshot.points
        states = snapshot.states
        sceneIndices = snapshot.sceneIndices
    }

    var selectedIndices: [Int] {
        states.indices.filter { states[$0].contains(.selected) }
    }

    var selectableIndices: [Int] {
        states.indices.filter(isSelectable)
    }

    var visiblePoints: [SplatScenePoint] {
        points.enumerated().compactMap { index, point in
            let state = states[index]
            guard !state.contains(.hidden), !state.contains(.deleted) else {
                return nil
            }
            return point
        }
    }

    func snapshotSummary() -> SplatEditorSnapshot {
        var visibleCount = 0
        var selectedCount = 0
        var hiddenCount = 0
        var deletedCount = 0
        var lockedCount = 0
        var visibleMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var visibleMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var hasVisibleBounds = false
        var selectionMin = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var selectionMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var hasSelectionBounds = false

        for (index, point) in points.enumerated() {
            let state = states[index]
            if state.contains(.selected) {
                selectedCount += 1
                selectionMin = simd_min(selectionMin, point.position)
                selectionMax = simd_max(selectionMax, point.position)
                hasSelectionBounds = true
            }
            if state.contains(.hidden) {
                hiddenCount += 1
            }
            if state.contains(.deleted) {
                deletedCount += 1
            }
            if state.contains(.locked) {
                lockedCount += 1
            }
            if !state.contains(.hidden), !state.contains(.deleted) {
                visibleCount += 1
                visibleMin = simd_min(visibleMin, point.position)
                visibleMax = simd_max(visibleMax, point.position)
                hasVisibleBounds = true
            }
        }

        return SplatEditorSnapshot(
            totalCount: points.count,
            visibleCount: visibleCount,
            selectedCount: selectedCount,
            hiddenCount: hiddenCount,
            deletedCount: deletedCount,
            lockedCount: lockedCount,
            visibleBounds: hasVisibleBounds ? SplatSelectionBounds(min: visibleMin, max: visibleMax) : nil,
            selectionBounds: hasSelectionBounds ? SplatSelectionBounds(min: selectionMin, max: selectionMax) : nil
        )
    }

    mutating func applySelection(indices: [Int], mode: SelectionCombineMode) -> [EditableSplatStateChange] {
        let selectable = Set(indices.filter { index in
            isSelectable(index)
        })

        var changes: [EditableSplatStateChange] = []
        switch mode {
        case .replace:
            for index in states.indices {
                recordStateChange(at: index, into: &changes) { state in
                    if selectable.contains(index) {
                        state.insert(.selected)
                    } else {
                        state.remove(.selected)
                    }
                }
            }
        case .add:
            for index in selectable {
                recordStateChange(at: index, into: &changes) { state in
                    state.insert(.selected)
                }
            }
        case .subtract:
            for index in selectable {
                recordStateChange(at: index, into: &changes) { state in
                    state.remove(.selected)
                }
            }
        }

        return changes
    }

    mutating func hideSelection() -> [EditableSplatStateChange] {
        var changes: [EditableSplatStateChange] = []
        for index in selectedIndices {
            recordStateChange(at: index, into: &changes) { state in
                state.insert(.hidden)
                state.remove(.selected)
            }
        }
        return changes
    }

    mutating func lockSelection() -> [EditableSplatStateChange] {
        var changes: [EditableSplatStateChange] = []
        for index in selectedIndices {
            recordStateChange(at: index, into: &changes) { state in
                state.insert(.locked)
                state.remove(.selected)
            }
        }
        return changes
    }

    mutating func selectAllVisible() -> [EditableSplatStateChange] {
        var changes: [EditableSplatStateChange] = []
        for index in states.indices where isSelectable(index) {
            recordStateChange(at: index, into: &changes) { state in
                state.insert(.selected)
            }
        }
        return changes
    }

    mutating func clearSelection() -> [EditableSplatStateChange] {
        var changes: [EditableSplatStateChange] = []
        for index in states.indices {
            recordStateChange(at: index, into: &changes) { state in
                state.remove(.selected)
            }
        }
        return changes
    }

    mutating func invertSelection() -> [EditableSplatStateChange] {
        var changes: [EditableSplatStateChange] = []
        for index in states.indices where isSelectable(index) {
            recordStateChange(at: index, into: &changes) { state in
                if state.contains(.selected) {
                    state.remove(.selected)
                } else {
                    state.insert(.selected)
                }
            }
        }
        return changes
    }

    mutating func unhideAll() -> [EditableSplatStateChange] {
        var changes: [EditableSplatStateChange] = []
        for index in states.indices {
            recordStateChange(at: index, into: &changes) { state in
                state.remove(.hidden)
            }
        }
        return changes
    }

    mutating func unlockAll() -> [EditableSplatStateChange] {
        var changes: [EditableSplatStateChange] = []
        for index in states.indices {
            recordStateChange(at: index, into: &changes) { state in
                state.remove(.locked)
            }
        }
        return changes
    }

    mutating func deleteSelection() -> [EditableSplatStateChange] {
        delete(indices: selectedIndices)
    }

    mutating func delete(indices: [Int]) -> [EditableSplatStateChange] {
        var changes: [EditableSplatStateChange] = []
        for index in indices {
            guard states.indices.contains(index) else { continue }
            recordStateChange(at: index, into: &changes) { state in
                state.insert(.deleted)
                state.remove(.selected)
            }
        }
        return changes
    }

    mutating func applyCommittedTransform(_ transform: SplatEditTransform,
                                          pivot: SIMD3<Float>) -> [EditableSplatPointChange] {
        applyCommittedTransform(transform, pivot: pivot, indices: selectedIndices)
    }

    mutating func applyCommittedTransform(_ transform: SplatEditTransform,
                                          pivot: SIMD3<Float>,
                                          indices: [Int]) -> [EditableSplatPointChange] {
        guard !indices.isEmpty else { return [] }

        var changes: [EditableSplatPointChange] = []
        for index in indices {
            let oldPoint = points[index]
            let newPoint = oldPoint.applying(transform: transform, around: pivot)
            points[index] = newPoint
            changes.append(EditableSplatPointChange(index: index, oldPoint: oldPoint, newPoint: newPoint))
        }

        return changes
    }

    mutating func duplicateSelection() -> [Int] {
        let sourceIndices = selectedIndices
        guard !sourceIndices.isEmpty else { return [] }

        let insertionStart = points.count
        let duplicates = sourceIndices.map { points[$0] }
        points.append(contentsOf: duplicates)
        sceneIndices.append(contentsOf: sourceIndices.map { sceneIndices[$0] })

        for index in sourceIndices {
            states[index].remove(.selected)
        }

        let duplicateStates = sourceIndices.map { index in
            var state = states[index]
            state.remove([.hidden, .deleted, .locked, .selected])
            state.insert(.selected)
            return state
        }
        states.append(contentsOf: duplicateStates)

        return Array(insertionStart..<(insertionStart + duplicateStates.count))
    }

    mutating func separateSelection() -> Bool {
        let sourceIndices = selectedIndices
        guard !sourceIndices.isEmpty else { return false }

        points = sourceIndices.map { points[$0] }
        sceneIndices = sourceIndices.map { sceneIndices[$0] }
        states = sourceIndices.map { index in
            var state = states[index]
            state.remove([.hidden, .deleted, .locked])
            state.insert(.selected)
            return state
        }
        return true
    }

    mutating func applyStateChanges(_ changes: [EditableSplatStateChange], useNewValues: Bool) {
        for change in changes {
            states[change.index] = useNewValues ? change.newState : change.oldState
        }
    }

    mutating func applyPointChanges(_ changes: [EditableSplatPointChange], useNewValues: Bool) {
        for change in changes {
            points[change.index] = useNewValues ? change.newPoint : change.oldPoint
        }
    }

    func colorMatchIndices(referenceIndex: Int, threshold: Float) -> [Int] {
        guard points.indices.contains(referenceIndex), isSelectable(referenceIndex) else { return [] }

        let clampedThreshold = max(0, min(threshold, 1))
        let reference = points[referenceIndex].color.asLinearFloat

        return points.indices.filter { index in
            guard isSelectable(index) else { return false }
            let color = points[index].color.asLinearFloat
            let delta = simd_abs(color - reference)
            return delta.x <= clampedThreshold && delta.y <= clampedThreshold && delta.z <= clampedThreshold
        }
    }

    func floodFillIndices(seedIndex: Int,
                          threshold: Float,
                          viewport: SplatRenderer.ViewportDescriptor) -> [Int] {
        let candidates = projectedCandidates(viewport: viewport)
        guard let seed = candidates.first(where: { $0.index == seedIndex }) else { return [] }

        let clampedThreshold = max(0.001, min(threshold, 1))
        let cellSize = max(seed.radius * 1.5, 0.01)
        let candidateLookup = Dictionary(uniqueKeysWithValues: candidates.indices.map { (candidates[$0].index, $0) })

        var cells: [SIMD2<Int32>: [Int]] = [:]
        cells.reserveCapacity(candidates.count)

        for candidateIndex in candidates.indices {
            let cell = Self.gridCell(for: candidates[candidateIndex].normalized, cellSize: cellSize)
            cells[cell, default: []].append(candidateIndex)
        }

        var visited: Set<Int> = [seed.index]
        var pending: [Int] = [seed.index]
        var result: [Int] = []

        while let currentPointIndex = pending.popLast(),
              let currentCandidateIndex = candidateLookup[currentPointIndex] {
            let current = candidates[currentCandidateIndex]
            result.append(current.index)

            let searchRadius = max(current.radius * 2.5, cellSize)
            let cellRange = max(1, Int(ceil(Double(searchRadius / cellSize))))
            let currentCell = Self.gridCell(for: current.normalized, cellSize: cellSize)

            for deltaY in -cellRange...cellRange {
                for deltaX in -cellRange...cellRange {
                    let neighborCell = SIMD2<Int32>(
                        currentCell.x + Int32(deltaX),
                        currentCell.y + Int32(deltaY)
                    )
                    guard let neighborIndices = cells[neighborCell] else { continue }

                    for neighborCandidateIndex in neighborIndices {
                        let neighbor = candidates[neighborCandidateIndex]
                        guard !visited.contains(neighbor.index) else { continue }
                        guard abs(neighbor.opacity - current.opacity) <= clampedThreshold else { continue }

                        let combinedRadius = max(searchRadius, neighbor.radius * 2.5)
                        guard simd_distance(current.normalized, neighbor.normalized) <= combinedRadius else { continue }

                        visited.insert(neighbor.index)
                        pending.append(neighbor.index)
                    }
                }
            }
        }

        return result
    }

    func planeSelectionIndices(plane: SplatCutPlane, side: SplatCutPlaneSide) -> [Int] {
        points.indices.filter { index in
            guard isSelectable(index) else { return false }
            let signedDistance = simd_dot(points[index].position - plane.point, plane.normal)
            return side.contains(signedDistance: signedDistance)
        }
    }

    func bounds(for indices: [Int]) -> SplatSelectionBounds? {
        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
        var hasBounds = false

        for index in indices where points.indices.contains(index) {
            minimum = simd_min(minimum, points[index].position)
            maximum = simd_max(maximum, points[index].position)
            hasBounds = true
        }

        guard hasBounds else { return nil }
        return SplatSelectionBounds(min: minimum, max: maximum)
    }

    private func isSelectable(_ index: Int) -> Bool {
        let state = states[index]
        return !state.contains(.hidden) && !state.contains(.deleted) && !state.contains(.locked)
    }

    private mutating func recordStateChange(at index: Int,
                                            into changes: inout [EditableSplatStateChange],
                                            update: (inout EditableSplatState) -> Void) {
        let oldState = states[index]
        var newState = oldState
        update(&newState)
        guard newState != oldState else { return }
        states[index] = newState
        changes.append(EditableSplatStateChange(index: index, oldState: oldState, newState: newState))
    }

    private func projectedCandidates(viewport: SplatRenderer.ViewportDescriptor) -> [ProjectedCandidate] {
        let cameraRight = Self.cameraRightAxis(viewMatrix: viewport.viewMatrix)

        return points.indices.compactMap { index in
            guard isSelectable(index),
                  let normalized = Self.project(points[index].position, viewport: viewport) else {
                return nil
            }

            let scale = max(
                points[index].scale.asLinearFloat.x,
                max(points[index].scale.asLinearFloat.y, points[index].scale.asLinearFloat.z)
            )
            let offsetPosition = points[index].position + cameraRight * max(scale, 0.001)
            let offsetNormalized = Self.project(offsetPosition, viewport: viewport) ?? normalized
            let radius = max(simd_distance(normalized, offsetNormalized), 0.008)

            return ProjectedCandidate(
                index: index,
                normalized: normalized,
                opacity: points[index].opacity.asLinearFloat,
                radius: radius
            )
        }
    }

    private static func project(_ position: SIMD3<Float>,
                                viewport: SplatRenderer.ViewportDescriptor) -> SIMD2<Float>? {
        let clip = viewport.projectionMatrix * viewport.viewMatrix * SIMD4<Float>(position, 1)
        guard abs(clip.w) > .ulpOfOne else { return nil }

        let ndc = clip / clip.w
        return SIMD2<Float>(
            (ndc.x + 1) * 0.5,
            1 - ((ndc.y + 1) * 0.5)
        )
    }

    private static func cameraRightAxis(viewMatrix: simd_float4x4) -> SIMD3<Float> {
        let inverseView = simd_inverse(viewMatrix)
        let axis = SIMD3<Float>(inverseView.columns.0.x, inverseView.columns.0.y, inverseView.columns.0.z)
        let length = simd_length(axis)
        return length > .ulpOfOne ? axis / length : SIMD3<Float>(1, 0, 0)
    }

    private static func gridCell(for normalized: SIMD2<Float>, cellSize: Float) -> SIMD2<Int32> {
        SIMD2<Int32>(
            Int32(floor(normalized.x / cellSize)),
            Int32(floor(normalized.y / cellSize))
        )
    }
}

struct PreviewTransformState: Sendable {
    var pivot: SIMD3<Float>
    var transform: SplatEditTransform
}

public actor SplatEditor {
    private let renderer: SplatRenderer
    private let selectionEngine: SplatSelectionEngine
    private var store: EditableSplatStore
    private var previewTransform: PreviewTransformState?
    private var previewTransformIndices: [UInt32]
    private var previewTransformTouchedIndices: [Int]
    private var transformPalette: [simd_float4x4]
    private var undoStack: [SplatEditHistoryEntry] = []
    private var redoStack: [SplatEditHistoryEntry] = []

    public init(points: [SplatScenePoint], renderer: SplatRenderer) async throws {
        self.renderer = renderer
        self.selectionEngine = try SplatSelectionEngine(device: renderer.device)
        let initialSceneIndices = renderer.animationSceneIndices.count == points.count
            ? renderer.animationSceneIndices
            : Array(repeating: 0, count: points.count)
        self.store = EditableSplatStore(points: points, sceneIndices: initialSceneIndices)
        self.previewTransformIndices = Array(repeating: 0, count: points.count)
        self.previewTransformTouchedIndices = []
        self.transformPalette = [matrix_identity_float4x4, matrix_identity_float4x4]

        if renderer.splatCount == 0 {
            try renderer.add(points)
        } else if renderer.splatCount != points.count {
            throw SplatEditorError.rendererPointCountMismatch(renderer: renderer.splatCount, editor: points.count)
        }

        try renderer.ensureEditingResources(pointCount: points.count)
        try replaceRendererState()
    }

    public func select(_ query: SplatSelectionQuery,
                       mode: SelectionCombineMode,
                       viewport: SplatRenderer.ViewportDescriptor) async throws {
        let selected = try await selectionEngine.select(
            query: query,
            viewport: viewport,
            renderer: renderer
        )

        let changes = store.applySelection(indices: selected, mode: mode)
        try applyStateHistory(changes)
    }

    public func select(plane: SplatCutPlane,
                       side: SplatCutPlaneSide,
                       mode: SelectionCombineMode) async throws {
        let selected = store.planeSelectionIndices(plane: plane, side: side)
        let changes = store.applySelection(indices: selected, mode: mode)
        try applyStateHistory(changes)
    }

    public func beginPreviewTransform(pivot: SIMD3<Float>) async {
        previewTransform = PreviewTransformState(pivot: pivot, transform: .identity)
        previewTransformTouchedIndices = store.selectedIndices
        for index in previewTransformTouchedIndices {
            previewTransformIndices[index] = 1
        }
        transformPalette[1] = matrix_identity_float4x4

        let values = Array(repeating: UInt32(1), count: previewTransformTouchedIndices.count)
        try? renderer.setPreviewTransformActive(!previewTransformTouchedIndices.isEmpty)
        renderer.beginInteraction()
        try? renderer.updateTransformIndices(at: previewTransformTouchedIndices, values: values)
        try? renderer.updateTransformPalette(transformPalette)
    }

    public func updatePreviewTransform(_ transform: SplatEditTransform) async throws {
        guard let current = previewTransform else {
            throw SplatEditorError.previewTransformNotActive
        }

        previewTransform = PreviewTransformState(pivot: current.pivot, transform: transform)
        transformPalette[1] = float4x4(transform: transform, pivot: current.pivot)
        try renderer.updateTransformPalette(transformPalette)
    }

    public func commitPreviewTransform() async throws {
        guard let current = previewTransform else {
            throw SplatEditorError.previewTransformNotActive
        }

        let changes = store.applyCommittedTransform(current.transform, pivot: current.pivot)
        try clearPreviewTransformState()
        try applyPointHistory(changes)
    }

    public func applyTransform(_ transform: SplatEditTransform,
                               pivot: SIMD3<Float>) async throws {
        try clearPreviewTransformState()
        let changes = store.applyCommittedTransform(transform, pivot: pivot)
        try applyPointHistory(changes)
    }

    public func alignmentBounds() async -> SplatSelectionBounds? {
        let indices = store.selectedIndices.isEmpty ? store.selectableIndices : store.selectedIndices
        return store.bounds(for: indices)
    }

    public func applyAlignmentTransform(_ transform: SplatEditTransform,
                                        pivot: SIMD3<Float>) async throws {
        try clearPreviewTransformState()
        let indices = store.selectedIndices.isEmpty ? store.selectableIndices : store.selectedIndices
        let changes = store.applyCommittedTransform(transform, pivot: pivot, indices: indices)
        try applyPointHistory(changes)
    }

    public func cancelPreviewTransform() async {
        try? clearPreviewTransformState()
    }

    public func hideSelection() async throws {
        try applyStateHistory(store.hideSelection())
    }

    public func lockSelection() async throws {
        try applyStateHistory(store.lockSelection())
    }

    public func selectAll() async throws {
        try applyStateHistory(store.selectAllVisible())
    }

    public func clearSelection() async throws {
        try applyStateHistory(store.clearSelection())
    }

    public func invertSelection() async throws {
        try applyStateHistory(store.invertSelection())
    }

    public func unhideAll() async throws {
        try applyStateHistory(store.unhideAll())
    }

    public func unlockAll() async throws {
        try applyStateHistory(store.unlockAll())
    }

    public func deleteSelection() async throws {
        try applyStateHistory(store.deleteSelection())
    }

    public func cut(plane: SplatCutPlane,
                    side: SplatCutPlaneSide) async throws {
        let changes = store.delete(indices: store.planeSelectionIndices(plane: plane, side: side))
        try applyStateHistory(changes)
    }

    public func duplicateSelection() async throws {
        let before = store.snapshot
        let changedIndices = store.duplicateSelection()
        guard !changedIndices.isEmpty else { return }

        pushHistory(.snapshot(before))
        previewTransform = nil
        try renderer.replaceAllSplats(with: store.points, sceneIndices: store.sceneIndices)
        previewTransformIndices = Array(repeating: 0, count: store.points.count)
        previewTransformTouchedIndices = []
        transformPalette[1] = matrix_identity_float4x4
        try replaceRendererState()
    }

    public func separateSelection() async throws {
        let before = store.snapshot
        guard store.separateSelection() else { return }

        pushHistory(.snapshot(before))
        previewTransform = nil
        try renderer.replaceAllSplats(with: store.points, sceneIndices: store.sceneIndices)
        previewTransformIndices = Array(repeating: 0, count: store.points.count)
        previewTransformTouchedIndices = []
        transformPalette[1] = matrix_identity_float4x4
        try replaceRendererState()
    }

    public func undo() async throws {
        guard let entry = undoStack.popLast() else { return }
        try clearPreviewTransformState()
        try applyHistoryEntry(entry, useNewValues: false)
        redoStack.append(entry)
    }

    public func redo() async throws {
        guard let entry = redoStack.popLast() else { return }
        try clearPreviewTransformState()
        try applyHistoryEntry(entry, useNewValues: true)
        undoStack.append(entry)
    }

    public func exportVisiblePoints() async throws -> [SplatScenePoint] {
        store.visiblePoints
    }

    public func snapshot() async -> SplatEditorSnapshot {
        store.snapshotSummary()
    }

    public func selectFloodFill(normalized: SIMD2<Float>,
                                threshold: Float,
                                mode: SelectionCombineMode,
                                viewport: SplatRenderer.ViewportDescriptor) async throws {
        guard let seedIndex = try await pickNearestIndex(
            normalized: normalized,
            radius: 0.04,
            viewport: viewport
        ) else { return }

        let selected = store.floodFillIndices(
            seedIndex: seedIndex,
            threshold: threshold,
            viewport: viewport
        )
        let changes = store.applySelection(indices: selected, mode: mode)
        try applyStateHistory(changes)
    }

    public func selectColorMatch(normalized: SIMD2<Float>,
                                 threshold: Float,
                                 mode: SelectionCombineMode,
                                 viewport: SplatRenderer.ViewportDescriptor) async throws {
        guard let referenceIndex = try await pickNearestIndex(
            normalized: normalized,
            radius: 0.04,
            viewport: viewport
        ) else { return }

        let selected = store.colorMatchIndices(referenceIndex: referenceIndex, threshold: threshold)
        let changes = store.applySelection(indices: selected, mode: mode)
        try applyStateHistory(changes)
    }

    public func pickPoint(normalized: SIMD2<Float>,
                          radius: Float,
                          viewport: SplatRenderer.ViewportDescriptor) async throws -> SplatScenePoint? {
        let nearestIndex = try await pickNearestIndex(
            normalized: normalized,
            radius: radius,
            viewport: viewport
        )
        guard let nearestIndex else { return nil }
        return store.points[nearestIndex]
    }

    private func replaceRendererState() throws {
        let rawStates = store.states.map(\.rawValue)
        try renderer.replaceEditingState(
            rawStates,
            transformIndices: previewTransformIndices,
            transformPalette: transformPalette
        )
    }

    private func pushHistory(_ entry: SplatEditHistoryEntry) {
        undoStack.append(entry)
        redoStack.removeAll(keepingCapacity: true)
    }

    private func applyStateHistory(_ changes: [EditableSplatStateChange]) throws {
        guard !changes.isEmpty else { return }
        pushHistory(.states(changes))
        try renderer.updateEditStates(at: changes.map(\.index), values: changes.map(\.newState.rawValue))
    }

    private func applyPointHistory(_ changes: [EditableSplatPointChange]) throws {
        guard !changes.isEmpty else { return }
        pushHistory(.points(changes))
        try renderer.updateSplats(store.points, at: changes.map(\.index))
    }

    private func applyHistoryEntry(_ entry: SplatEditHistoryEntry, useNewValues: Bool) throws {
        switch entry {
        case .states(let changes):
            store.applyStateChanges(changes, useNewValues: useNewValues)
            let values = useNewValues ? changes.map(\.newState.rawValue) : changes.map(\.oldState.rawValue)
            try renderer.updateEditStates(at: changes.map(\.index), values: values)
        case .points(let changes):
            store.applyPointChanges(changes, useNewValues: useNewValues)
            try renderer.updateSplats(store.points, at: changes.map(\.index))
        case .snapshot(let snapshot):
            store.restore(snapshot)
            previewTransform = nil
            previewTransformIndices = Array(repeating: 0, count: store.points.count)
            previewTransformTouchedIndices = []
            transformPalette[1] = matrix_identity_float4x4
            try renderer.replaceAllSplats(with: store.points, sceneIndices: store.sceneIndices)
            try replaceRendererState()
        }
    }

    private func clearPreviewTransformState() throws {
        guard previewTransform != nil || !previewTransformTouchedIndices.isEmpty else { return }

        previewTransform = nil
        transformPalette[1] = matrix_identity_float4x4

        if !previewTransformTouchedIndices.isEmpty {
            for index in previewTransformTouchedIndices {
                previewTransformIndices[index] = 0
            }
            let values = Array(repeating: UInt32(0), count: previewTransformTouchedIndices.count)
            try renderer.updateTransformIndices(at: previewTransformTouchedIndices, values: values)
        }

        previewTransformTouchedIndices.removeAll(keepingCapacity: true)
        try renderer.updateTransformPalette(transformPalette)
        try renderer.setPreviewTransformActive(false)
        renderer.endInteraction()
    }

    private func pickNearestIndex(normalized: SIMD2<Float>,
                                  radius: Float,
                                  viewport: SplatRenderer.ViewportDescriptor) async throws -> Int? {
        try await selectionEngine.pickNearest(
            normalized: normalized,
            radius: radius,
            viewport: viewport,
            renderer: renderer
        )
    }
}

private extension SplatScenePoint {
    func applying(transform: SplatEditTransform, around pivot: SIMD3<Float>) -> SplatScenePoint {
        let scaledOffset = (position - pivot) * transform.scale
        let rotatedOffset = transform.rotation.act(scaledOffset)
        let transformedPosition = pivot + rotatedOffset + transform.translation

        let transformedScale = scale.asLinearFloat * transform.scale
        let transformedRotation = (transform.rotation * rotation).normalized

        return SplatScenePoint(
            position: transformedPosition,
            color: color.rotated(by: transform.rotation),
            opacity: opacity,
            scale: .linearFloat(transformedScale),
            rotation: transformedRotation
        )
    }
}

private extension SplatScenePoint.Color {
    func rotated(by rotation: simd_quatf) -> SplatScenePoint.Color {
        guard case let .sphericalHarmonic(coefficients) = self, coefficients.count > 1 else {
            return self
        }

        var rotated = coefficients
        let firstOrderEnd = min(rotated.count, 4)
        if firstOrderEnd > 1 {
            for index in 1..<firstOrderEnd {
                rotated[index] = rotation.act(rotated[index])
            }
        }
        return .sphericalHarmonic(rotated)
    }
}

private extension SplatCutPlaneSide {
    func contains(signedDistance: Float) -> Bool {
        switch self {
        case .negative:
            return signedDistance <= 0
        case .positive:
            return signedDistance >= 0
        }
    }
}

private extension float4x4 {
    init(transform: SplatEditTransform, pivot: SIMD3<Float>) {
        let translateToOrigin = float4x4(translation: -pivot)
        let scale = float4x4(scale: transform.scale)
        let rotate = matrix_float4x4(transform.rotation)
        let translateBack = float4x4(translation: pivot + transform.translation)
        self = translateBack * rotate * scale * translateToOrigin
    }

    init(translation: SIMD3<Float>) {
        self = matrix_identity_float4x4
        columns.3 = SIMD4<Float>(translation.x, translation.y, translation.z, 1)
    }

    init(scale: SIMD3<Float>) {
        self = matrix_identity_float4x4
        columns.0.x = scale.x
        columns.1.y = scale.y
        columns.2.z = scale.z
    }
}
