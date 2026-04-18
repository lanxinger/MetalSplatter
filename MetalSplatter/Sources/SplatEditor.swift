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

actor SplatEditHistory {
    struct Snapshot: Sendable {
        var points: [SplatScenePoint]
        var states: [EditableSplatState]
    }

    private var undoStack: [Snapshot] = []
    private var redoStack: [Snapshot] = []

    func pushUndo(_ snapshot: Snapshot) {
        undoStack.append(snapshot)
        redoStack.removeAll(keepingCapacity: true)
    }

    func popUndo(current: Snapshot) -> Snapshot? {
        guard let previous = undoStack.popLast() else { return nil }
        redoStack.append(current)
        return previous
    }

    func popRedo(current: Snapshot) -> Snapshot? {
        guard let next = redoStack.popLast() else { return nil }
        undoStack.append(current)
        return next
    }
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

    init(points: [SplatScenePoint]) {
        self.points = points
        self.states = Array(repeating: [], count: points.count)
    }

    var snapshot: SplatEditHistory.Snapshot {
        SplatEditHistory.Snapshot(points: points, states: states)
    }

    mutating func restore(_ snapshot: SplatEditHistory.Snapshot) {
        points = snapshot.points
        states = snapshot.states
    }

    var selectedIndices: [Int] {
        states.indices.filter { states[$0].contains(.selected) }
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

    mutating func applySelection(indices: [Int], mode: SelectionCombineMode) {
        let selectable = Set(indices.filter { index in
            isSelectable(index)
        })

        switch mode {
        case .replace:
            for index in states.indices {
                if selectable.contains(index) {
                    states[index].insert(.selected)
                } else {
                    states[index].remove(.selected)
                }
            }
        case .add:
            for index in selectable {
                states[index].insert(.selected)
            }
        case .subtract:
            for index in selectable {
                states[index].remove(.selected)
            }
        }
    }

    mutating func hideSelection() {
        for index in selectedIndices {
            states[index].insert(.hidden)
            states[index].remove(.selected)
        }
    }

    mutating func unhideAll() {
        for index in states.indices {
            states[index].remove(.hidden)
        }
    }

    mutating func deleteSelection() {
        for index in selectedIndices {
            states[index].insert(.deleted)
            states[index].remove(.selected)
        }
    }

    mutating func applyCommittedTransform(_ transform: SplatEditTransform, pivot: SIMD3<Float>) -> [Int] {
        let selected = selectedIndices
        guard !selected.isEmpty else { return [] }

        for index in selected {
            var point = points[index]
            point = point.applying(transform: transform, around: pivot)
            points[index] = point
        }

        return selected
    }

    mutating func duplicateSelection() -> [Int] {
        let sourceIndices = selectedIndices
        guard !sourceIndices.isEmpty else { return [] }

        let insertionStart = points.count
        let duplicates = sourceIndices.map { points[$0] }
        points.append(contentsOf: duplicates)

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
        states = sourceIndices.map { index in
            var state = states[index]
            state.remove([.hidden, .deleted, .locked])
            state.insert(.selected)
            return state
        }
        return true
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

    private func isSelectable(_ index: Int) -> Bool {
        let state = states[index]
        return !state.contains(.hidden) && !state.contains(.deleted) && !state.contains(.locked)
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
    private let history = SplatEditHistory()
    private var store: EditableSplatStore
    private var previewTransform: PreviewTransformState?
    private var previewTransformIndices: [UInt32]
    private var transformPalette: [simd_float4x4]

    public init(points: [SplatScenePoint], renderer: SplatRenderer) async throws {
        self.renderer = renderer
        self.selectionEngine = try SplatSelectionEngine(device: renderer.device)
        self.store = EditableSplatStore(points: points)
        self.previewTransformIndices = Array(repeating: 0, count: points.count)
        self.transformPalette = [matrix_identity_float4x4, matrix_identity_float4x4]

        if renderer.splatCount == 0 {
            try renderer.add(points)
        } else if renderer.splatCount != points.count {
            throw SplatEditorError.rendererPointCountMismatch(renderer: renderer.splatCount, editor: points.count)
        }

        try renderer.ensureEditingResources(pointCount: points.count)
        try syncRendererState()
    }

    public func select(_ query: SplatSelectionQuery,
                       mode: SelectionCombineMode,
                       viewport: SplatRenderer.ViewportDescriptor) async throws {
        let before = store.snapshot
        let selected = try await selectionEngine.select(
            query: query,
            viewport: viewport,
            renderer: renderer
        )

        store.applySelection(indices: selected, mode: mode)
        await history.pushUndo(before)
        try syncRendererState()
    }

    public func beginPreviewTransform(pivot: SIMD3<Float>) async {
        previewTransform = PreviewTransformState(pivot: pivot, transform: .identity)

        previewTransformIndices = Array(repeating: 0, count: store.points.count)
        for index in store.selectedIndices {
            previewTransformIndices[index] = 1
        }
        transformPalette[1] = matrix_identity_float4x4

        try? syncRendererState()
    }

    public func updatePreviewTransform(_ transform: SplatEditTransform) async throws {
        guard let current = previewTransform else {
            throw SplatEditorError.previewTransformNotActive
        }

        previewTransform = PreviewTransformState(pivot: current.pivot, transform: transform)
        transformPalette[1] = float4x4(transform: transform, pivot: current.pivot)
        try syncRendererState()
    }

    public func commitPreviewTransform() async throws {
        guard let current = previewTransform else {
            throw SplatEditorError.previewTransformNotActive
        }

        let before = store.snapshot
        let changedIndices = store.applyCommittedTransform(current.transform, pivot: current.pivot)
        await history.pushUndo(before)

        previewTransform = nil
        previewTransformIndices = Array(repeating: 0, count: store.points.count)
        transformPalette[1] = matrix_identity_float4x4

        try renderer.updateSplats(store.points, at: changedIndices)
        try syncRendererState()
    }

    public func cancelPreviewTransform() async {
        previewTransform = nil
        previewTransformIndices = Array(repeating: 0, count: store.points.count)
        transformPalette[1] = matrix_identity_float4x4
        try? syncRendererState()
    }

    public func hideSelection() async throws {
        let before = store.snapshot
        store.hideSelection()
        await history.pushUndo(before)
        try syncRendererState()
    }

    public func unhideAll() async throws {
        let before = store.snapshot
        store.unhideAll()
        await history.pushUndo(before)
        try syncRendererState()
    }

    public func deleteSelection() async throws {
        let before = store.snapshot
        store.deleteSelection()
        await history.pushUndo(before)
        try syncRendererState()
    }

    public func duplicateSelection() async throws {
        let before = store.snapshot
        let changedIndices = store.duplicateSelection()
        guard !changedIndices.isEmpty else { return }

        await history.pushUndo(before)
        try renderer.replaceAllSplats(with: store.points)
        try syncRendererState()
    }

    public func separateSelection() async throws {
        let before = store.snapshot
        guard store.separateSelection() else { return }

        await history.pushUndo(before)
        try renderer.replaceAllSplats(with: store.points)
        try syncRendererState()
    }

    public func undo() async throws {
        guard let previous = await history.popUndo(current: store.snapshot) else { return }
        store.restore(previous)
        previewTransform = nil
        previewTransformIndices = Array(repeating: 0, count: store.points.count)
        transformPalette[1] = matrix_identity_float4x4
        try renderer.replaceAllSplats(with: store.points)
        try syncRendererState()
    }

    public func redo() async throws {
        guard let next = await history.popRedo(current: store.snapshot) else { return }
        store.restore(next)
        previewTransform = nil
        previewTransformIndices = Array(repeating: 0, count: store.points.count)
        transformPalette[1] = matrix_identity_float4x4
        try renderer.replaceAllSplats(with: store.points)
        try syncRendererState()
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

        let before = store.snapshot
        let selected = store.floodFillIndices(
            seedIndex: seedIndex,
            threshold: threshold,
            viewport: viewport
        )
        store.applySelection(indices: selected, mode: mode)
        await history.pushUndo(before)
        try syncRendererState()
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

        let before = store.snapshot
        let selected = store.colorMatchIndices(referenceIndex: referenceIndex, threshold: threshold)
        store.applySelection(indices: selected, mode: mode)
        await history.pushUndo(before)
        try syncRendererState()
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

    private func syncRendererState() throws {
        let rawStates = store.states.map(\.rawValue)
        try renderer.updateEditingState(
            rawStates,
            transformIndices: previewTransformIndices,
            transformPalette: transformPalette
        )
    }

    private func projectedDistanceSquared(point: SplatScenePoint,
                                          target: SIMD2<Float>,
                                          viewport: SplatRenderer.ViewportDescriptor) -> Float {
        let clip = viewport.projectionMatrix * viewport.viewMatrix * SIMD4<Float>(point.position, 1)
        guard abs(clip.w) > .ulpOfOne else { return .greatestFiniteMagnitude }

        let ndc = clip / clip.w
        let normalized = SIMD2<Float>(
            (ndc.x + 1) * 0.5,
            1 - ((ndc.y + 1) * 0.5)
        )
        let delta = normalized - target
        return simd_dot(delta, delta)
    }

    private func pickNearestIndex(normalized: SIMD2<Float>,
                                  radius: Float,
                                  viewport: SplatRenderer.ViewportDescriptor) async throws -> Int? {
        let indices = try await selectionEngine.select(
            query: .point(normalized: normalized, radius: radius),
            viewport: viewport,
            renderer: renderer
        )

        guard !indices.isEmpty else { return nil }

        return indices.min { lhs, rhs in
            let lhsDistance = projectedDistanceSquared(
                point: store.points[lhs],
                target: normalized,
                viewport: viewport
            )
            let rhsDistance = projectedDistanceSquared(
                point: store.points[rhs],
                target: normalized,
                viewport: viewport
            )
            return lhsDistance < rhsDistance
        }
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
