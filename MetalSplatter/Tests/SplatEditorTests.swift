import XCTest
import Metal
import simd
import SplatIO
@testable import MetalSplatter

final class SplatEditorTests: XCTestCase {
    private var device: MTLDevice?

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
    }

    override func tearDown() {
        device = nil
        super.tearDown()
    }

    func testEditableStoreSelectionModesAndVisibility() {
        var store = EditableSplatStore(points: makePoints())

        store.applySelection(indices: [1, 2], mode: .replace)
        XCTAssertEqual(store.selectedIndices, [1, 2])

        store.applySelection(indices: [3], mode: .add)
        XCTAssertEqual(store.selectedIndices, [1, 2, 3])

        store.applySelection(indices: [2], mode: .subtract)
        XCTAssertEqual(store.selectedIndices, [1, 3])

        store.hideSelection()
        let snapshot = store.snapshotSummary()
        XCTAssertEqual(snapshot.hiddenCount, 2)
        XCTAssertEqual(snapshot.visibleCount, 2)
        XCTAssertEqual(snapshot.selectedCount, 0)
    }

    func testEditableStoreCommittedTransformBakesPositionAndSelectionBounds() {
        var store = EditableSplatStore(points: makePoints())
        store.applySelection(indices: [1], mode: .replace)

        let changedIndices = store.applyCommittedTransform(
            SplatEditTransform(
                translation: SIMD3<Float>(0.5, 0.25, -0.5),
                rotation: simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 1, 0)),
                scale: SIMD3<Float>(2, 1, 1)
            ),
            pivot: SIMD3<Float>(0, 0, -2)
        )

        XCTAssertEqual(changedIndices, [1])
        XCTAssertEqual(store.points[1].position.x, 0.5, accuracy: 0.0001)
        XCTAssertEqual(store.points[1].position.y, 0.25, accuracy: 0.0001)
        XCTAssertEqual(store.points[1].position.z, -2.5, accuracy: 0.0001)

        let snapshot = store.snapshotSummary()
        XCTAssertEqual(snapshot.selectedCount, 1)
        XCTAssertEqual(snapshot.selectionBounds?.center.x ?? 0, 0.5, accuracy: 0.0001)
    }

    func testEditHistoryUndoRedoSnapshots() async {
        let points = makePoints()
        let history = SplatEditHistory()
        let original = SplatEditHistory.Snapshot(points: points, states: Array(repeating: [], count: points.count))
        var editedStates = Array(repeating: EditableSplatState(), count: points.count)
        editedStates[1] = [.selected]
        let edited = SplatEditHistory.Snapshot(points: points, states: editedStates)

        await history.pushUndo(original)
        let undone = await history.popUndo(current: edited)
        XCTAssertEqual(undone?.states[1], [])

        let redone = await history.popRedo(current: original)
        XCTAssertEqual(redone?.states[1], [.selected])
    }

    func testRectSelectionHideDeleteUndoRedoAndExportRoundTrip() async throws {
        let renderer = try makeRenderer()
        let editor = try await SplatEditor(points: makePoints(), renderer: renderer)

        try await editor.select(
            .rect(normalizedMin: SIMD2<Float>(0.40, 0.40), normalizedMax: SIMD2<Float>(0.60, 0.60)),
            mode: .replace,
            viewport: makeViewport()
        )

        var snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 1)
        XCTAssertEqual(snapshot.visibleCount, 4)

        try await editor.hideSelection()
        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.visibleCount, 3)
        XCTAssertEqual(snapshot.hiddenCount, 1)

        try await editor.undo()
        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.visibleCount, 4)
        XCTAssertEqual(snapshot.hiddenCount, 0)

        try await editor.deleteSelection()
        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.visibleCount, 3)
        XCTAssertEqual(snapshot.deletedCount, 1)

        let exported = try await editor.exportVisiblePoints()
        XCTAssertEqual(exported.count, 3)

        let roundTripURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("ply")
        defer { try? FileManager.default.removeItem(at: roundTripURL) }

        let writer = try SplatPLYSceneWriter(toFileAtPath: roundTripURL.path, append: false)
        try writer.start(binary: true, pointCount: exported.count)
        try writer.write(exported)
        try writer.close()

        let reader = try AutodetectSceneReader(roundTripURL)
        let reread = try reader.readScene()
        XCTAssertEqual(reread.count, 3)
    }

    func testPreviewTransformCommitUndoRedoBakesSelection() async throws {
        let renderer = try makeRenderer()
        let editor = try await SplatEditor(points: makePoints(), renderer: renderer)

        try await editor.select(
            .sphere(center: SIMD3<Float>(0, 0, -2), radius: 0.2),
            mode: .replace,
            viewport: makeViewport()
        )

        await editor.beginPreviewTransform(pivot: SIMD3<Float>(0, 0, -2))
        try await editor.updatePreviewTransform(
            SplatEditTransform(
                translation: SIMD3<Float>(1, 0, 0),
                rotation: simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 1, 0)),
                scale: SIMD3<Float>(2, 2, 2)
            )
        )
        try await editor.commitPreviewTransform()

        var exported = try await editor.exportVisiblePoints()
        XCTAssertEqual(exported[1].position.x, 1.0, accuracy: 0.0001)

        try await editor.undo()
        exported = try await editor.exportVisiblePoints()
        XCTAssertEqual(exported[1].position.x, 0.0, accuracy: 0.0001)

        try await editor.redo()
        exported = try await editor.exportVisiblePoints()
        XCTAssertEqual(exported[1].position.x, 1.0, accuracy: 0.0001)
    }

    func testMaskSelectionAndSelectionBounds() async throws {
        let renderer = try makeRenderer()
        let editor = try await SplatEditor(points: makePoints(), renderer: renderer)

        let mask = Data([
            0, 0, 0, 0,
            0, 255, 255, 0,
            0, 255, 255, 0,
            0, 0, 0, 0
        ])

        try await editor.select(
            .mask(alphaMask: mask, size: SIMD2<Int>(4, 4)),
            mode: .replace,
            viewport: makeViewport()
        )

        let snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 2)
        XCTAssertNotNil(snapshot.selectionBounds)
    }

    func testBoxSelectionAndVisibleBounds() async throws {
        let renderer = try makeRenderer()
        let editor = try await SplatEditor(points: makePoints(), renderer: renderer)

        let initialSnapshot = await editor.snapshot()
        let initialVisibleBounds = try XCTUnwrap(initialSnapshot.visibleBounds)
        XCTAssertEqual(initialVisibleBounds.min.x, -0.75, accuracy: 0.0001)
        XCTAssertEqual(initialVisibleBounds.min.y, -0.75, accuracy: 0.0001)
        XCTAssertEqual(initialVisibleBounds.max.x, 0.75, accuracy: 0.0001)
        XCTAssertEqual(initialVisibleBounds.max.y, 0.75, accuracy: 0.0001)

        try await editor.select(
            .box(center: SIMD3<Float>(0.275, 0, -2), extents: SIMD3<Float>(0.3, 0.2, 0.2)),
            mode: .replace,
            viewport: makeViewport()
        )

        let snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 2)
        let selectionBounds = try XCTUnwrap(snapshot.selectionBounds)
        XCTAssertEqual(selectionBounds.min.x, 0.0, accuracy: 0.0001)
        XCTAssertEqual(selectionBounds.max.x, 0.55, accuracy: 0.0001)
    }

    func testPointPickingReturnsNearestVisibleSplatWithoutChangingSelection() async throws {
        let renderer = try makeRenderer()
        let editor = try await SplatEditor(points: makePoints(), renderer: renderer)

        let picked = try await editor.pickPoint(
            normalized: SIMD2<Float>(0.5, 0.5),
            radius: 0.05,
            viewport: makeViewport()
        )

        let pickedPoint = try XCTUnwrap(picked)
        XCTAssertEqual(pickedPoint.position.x, 0.0, accuracy: 0.0001)
        XCTAssertEqual(pickedPoint.position.y, 0.0, accuracy: 0.0001)
        XCTAssertEqual(pickedPoint.position.z, -2.0, accuracy: 0.0001)

        let snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 0)
    }

    private func makeRenderer() throws -> SplatRenderer {
        guard let device else {
            throw XCTSkip("Metal device unavailable")
        }

        do {
            return try SplatRenderer(
                device: device,
                colorFormat: .bgra8Unorm,
                depthFormat: .depth32Float,
                sampleCount: 1,
                maxViewCount: 1,
                maxSimultaneousRenders: 3
            )
        } catch {
            throw XCTSkip("Renderer unavailable in swift test environment: \(error.localizedDescription)")
        }
    }

    private func makeViewport() -> SplatRenderer.ViewportDescriptor {
        SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: 256, height: 256, znear: 0, zfar: 1),
            projectionMatrix: matrix_identity_float4x4,
            viewMatrix: matrix_identity_float4x4,
            screenSize: SIMD2<Int>(256, 256)
        )
    }

    private func makePoints() -> [SplatScenePoint] {
        [
            makePoint(position: SIMD3<Float>(-0.75, -0.75, -2.0), color: SIMD3<Float>(1.0, 0.0, 0.0)),
            makePoint(position: SIMD3<Float>(0.0, 0.0, -2.0), color: SIMD3<Float>(0.0, 1.0, 0.0)),
            makePoint(position: SIMD3<Float>(0.55, 0.0, -2.0), color: SIMD3<Float>(0.0, 0.0, 1.0)),
            makePoint(position: SIMD3<Float>(0.75, 0.75, -2.0), color: SIMD3<Float>(1.0, 1.0, 0.0))
        ]
    }

    private func makePoint(position: SIMD3<Float>, color: SIMD3<Float>) -> SplatScenePoint {
        SplatScenePoint(
            position: position,
            color: .linearFloat(color),
            opacity: .linearFloat(1.0),
            scale: .linearFloat(SIMD3<Float>(repeating: 0.05)),
            rotation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        )
    }
}
