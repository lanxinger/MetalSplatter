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

        _ = store.applySelection(indices: [1, 2], mode: .replace)
        XCTAssertEqual(store.selectedIndices, [1, 2])

        _ = store.applySelection(indices: [3], mode: .add)
        XCTAssertEqual(store.selectedIndices, [1, 2, 3])

        _ = store.applySelection(indices: [2], mode: .subtract)
        XCTAssertEqual(store.selectedIndices, [1, 3])

        _ = store.hideSelection()
        let snapshot = store.snapshotSummary()
        XCTAssertEqual(snapshot.hiddenCount, 2)
        XCTAssertEqual(snapshot.visibleCount, 2)
        XCTAssertEqual(snapshot.selectedCount, 0)
    }

    func testEditableStoreCommittedTransformBakesPositionAndSelectionBounds() {
        var store = EditableSplatStore(points: makePoints())
        _ = store.applySelection(indices: [1], mode: .replace)

        let pointChanges = store.applyCommittedTransform(
            SplatEditTransform(
                translation: SIMD3<Float>(0.5, 0.25, -0.5),
                rotation: simd_quatf(angle: .pi / 2, axis: SIMD3<Float>(0, 1, 0)),
                scale: SIMD3<Float>(2, 1, 1)
            ),
            pivot: SIMD3<Float>(0, 0, -2)
        )

        XCTAssertEqual(pointChanges.map(\.index), [1])
        XCTAssertEqual(store.points[1].position.x, 0.5, accuracy: 0.0001)
        XCTAssertEqual(store.points[1].position.y, 0.25, accuracy: 0.0001)
        XCTAssertEqual(store.points[1].position.z, -2.5, accuracy: 0.0001)

        let snapshot = store.snapshotSummary()
        XCTAssertEqual(snapshot.selectedCount, 1)
        XCTAssertEqual(snapshot.selectionBounds?.center.x ?? 0, 0.5, accuracy: 0.0001)
    }

    func testEditableStoreStateChangesApplyAndRestoreIncrementally() {
        var store = EditableSplatStore(points: makePoints())
        let changes = store.applySelection(indices: [1, 2], mode: .replace)

        XCTAssertEqual(store.selectedIndices, [1, 2])

        store.applyStateChanges(changes, useNewValues: false)
        XCTAssertEqual(store.selectedIndices, [])

        store.applyStateChanges(changes, useNewValues: true)
        XCTAssertEqual(store.selectedIndices, [1, 2])
    }

    func testEditableStoreColorMatchUsesPerChannelThresholdAndVisibility() {
        var store = EditableSplatStore(points: [
            makePoint(position: SIMD3<Float>(0.0, 0.0, -2.0), color: SIMD3<Float>(0.20, 0.40, 0.60)),
            makePoint(position: SIMD3<Float>(0.1, 0.0, -2.0), color: SIMD3<Float>(0.22, 0.38, 0.61)),
            makePoint(position: SIMD3<Float>(0.2, 0.0, -2.0), color: SIMD3<Float>(0.20, 0.52, 0.60)),
            makePoint(position: SIMD3<Float>(0.3, 0.0, -2.0), color: SIMD3<Float>(0.19, 0.39, 0.58))
        ])
        store.states[3].insert(.hidden)

        let matched = store.colorMatchIndices(referenceIndex: 0, threshold: 0.03)
        XCTAssertEqual(matched, [0, 1])
    }

    func testEditableStoreFloodFillUsesProjectedConnectivityAndOpacityThreshold() {
        let points = [
            makePoint(position: SIMD3<Float>(0.00, 0.00, -2.0), color: SIMD3<Float>(1.0, 0.0, 0.0), opacity: 0.20),
            makePoint(position: SIMD3<Float>(0.03, 0.00, -2.0), color: SIMD3<Float>(0.9, 0.1, 0.0), opacity: 0.24),
            makePoint(position: SIMD3<Float>(0.06, 0.00, -2.0), color: SIMD3<Float>(0.8, 0.2, 0.0), opacity: 0.22),
            makePoint(position: SIMD3<Float>(0.45, 0.00, -2.0), color: SIMD3<Float>(0.7, 0.3, 0.0), opacity: 0.21),
            makePoint(position: SIMD3<Float>(0.02, 0.02, -2.0), color: SIMD3<Float>(0.6, 0.4, 0.0), opacity: 0.80)
        ]
        let store = EditableSplatStore(points: points)

        let selected = store.floodFillIndices(seedIndex: 0, threshold: 0.08, viewport: makeViewport())
        XCTAssertEqual(selected.sorted(), [0, 1, 2])
    }

    func testEditableStoreDuplicateSelectionAppendsCopiesAndSelectsDuplicates() {
        var store = EditableSplatStore(points: makePoints())
        _ = store.applySelection(indices: [1, 2], mode: .replace)

        let duplicateIndices = store.duplicateSelection()

        XCTAssertEqual(duplicateIndices, [4, 5])
        XCTAssertEqual(store.points.count, 6)
        XCTAssertEqual(store.selectedIndices, [4, 5])
        XCTAssertEqual(store.points[4].position, store.points[1].position)
        XCTAssertEqual(store.points[5].position, store.points[2].position)
    }

    func testEditableStorePreservesSceneIndicesAcrossSnapshotEdits() {
        var store = EditableSplatStore(points: makePoints(), sceneIndices: [0, 0, 1, 1])
        _ = store.applySelection(indices: [1, 2], mode: .replace)

        let snapshot = store.snapshot
        _ = store.duplicateSelection()
        XCTAssertEqual(store.sceneIndices, [0, 0, 1, 1, 0, 1])

        store.restore(snapshot)
        XCTAssertEqual(store.sceneIndices, [0, 0, 1, 1])

        _ = store.applySelection(indices: [1, 3], mode: .replace)
        XCTAssertTrue(store.separateSelection())
        XCTAssertEqual(store.sceneIndices, [0, 1])
    }

    func testEditableStoreSeparateSelectionKeepsOnlySelectedPoints() {
        var store = EditableSplatStore(points: makePoints())
        _ = store.applySelection(indices: [1, 3], mode: .replace)

        let didSeparate = store.separateSelection()

        XCTAssertTrue(didSeparate)
        XCTAssertEqual(store.points.count, 2)
        XCTAssertEqual(store.selectedIndices, [0, 1])
        XCTAssertEqual(store.points[0].position, SIMD3<Float>(0.0, 0.0, -2.0))
        XCTAssertEqual(store.points[1].position, SIMD3<Float>(0.75, 0.75, -2.0))
    }

    func testRendererReplaceAllSplatsPreservesExplicitSceneIndices() throws {
        let renderer = try makeRenderer()
        let layers = [
            SplatSceneLayer(points: Array(makePoints().prefix(2))),
            SplatSceneLayer(points: Array(makePoints().suffix(2)))
        ]
        try renderer.replaceSceneLayers(layers)

        let duplicatedPoints = makePoints() + [makePoints()[0], makePoints()[2]]
        try renderer.replaceAllSplats(with: duplicatedPoints, sceneIndices: [0, 0, 1, 1, 0, 1])

        XCTAssertEqual(renderer.animationSceneIndices, [0, 0, 1, 1, 0, 1])
        XCTAssertEqual(renderer.animationSceneCounts, [3, 3])
        XCTAssertEqual(renderer.animationSceneMetrics.count, 2)
    }

    func testRendererUpdateEditingStateClampsTransformIndicesToPointCount() throws {
        let renderer = try makeRenderer()

        try renderer.replaceEditingState(
            [0, 0],
            transformIndices: [7, 9, 11, 13],
            transformPalette: [matrix_identity_float4x4]
        )

        guard let buffer = renderer.editTransformIndexBuffer else {
            XCTFail("Missing edit transform index buffer")
            return
        }

        let values = buffer.contents().bindMemory(to: UInt32.self, capacity: 2)
        XCTAssertEqual(values[0], 7)
        XCTAssertEqual(values[1], 9)
    }

    func testRendererIncrementalEditStateTrackingUpdatesRenderableCount() throws {
        let renderer = try makeRenderer()
        try renderer.add(makePoints().prefix(3).map { $0 })
        try renderer.replaceEditingState(
            [0, 0, 0],
            transformIndices: [0, 0, 0],
            transformPalette: [matrix_identity_float4x4]
        )

        XCTAssertEqual(renderer.renderableSplatCountForCurrentEditState, 3)

        try renderer.updateEditStates(
            at: [0, 1, 2],
            values: [0, EditableSplatState.hidden.rawValue, EditableSplatState.selected.rawValue]
        )

        XCTAssertEqual(renderer.renderableSplatCountForCurrentEditState, 2)
        XCTAssertTrue(renderer.editingEnabled)
    }

    func testRendererPreviewTransformRestoresOptimizedSettings() throws {
        let renderer = try makeRenderer()
        renderer.meshShaderEnabled = true
        renderer.batchPrecomputeEnabled = true

        try renderer.setPreviewTransformActive(true)
        XCTAssertFalse(renderer.meshShaderEnabled)
        XCTAssertFalse(renderer.batchPrecomputeEnabled)

        try renderer.setPreviewTransformActive(false)
        XCTAssertTrue(renderer.meshShaderEnabled)
        XCTAssertTrue(renderer.batchPrecomputeEnabled)
    }

    func testEditableStoreSelectAllClearAndInvertRespectVisibility() {
        var store = EditableSplatStore(points: makePoints())
        store.states[2].insert(.hidden)

        _ = store.selectAllVisible()
        XCTAssertEqual(store.selectedIndices, [0, 1, 3])

        _ = store.invertSelection()
        XCTAssertEqual(store.selectedIndices, [])

        _ = store.selectAllVisible()
        _ = store.clearSelection()
        XCTAssertEqual(store.selectedIndices, [])
    }

    func testEditableStoreLockAndUnlockAffectSelectionEligibility() {
        var store = EditableSplatStore(points: makePoints())
        _ = store.applySelection(indices: [1, 2], mode: .replace)

        _ = store.lockSelection()
        XCTAssertEqual(store.selectedIndices, [])
        XCTAssertTrue(store.states[1].contains(.locked))
        XCTAssertTrue(store.states[2].contains(.locked))

        _ = store.applySelection(indices: [0, 1, 2, 3], mode: .replace)
        XCTAssertEqual(store.selectedIndices, [0, 3])

        _ = store.unlockAll()
        _ = store.applySelection(indices: [0, 1, 2, 3], mode: .replace)
        XCTAssertEqual(store.selectedIndices, [0, 1, 2, 3])
    }

    func testEditableStorePlaneSelectionRespectsAxisSideAndVisibility() {
        var store = EditableSplatStore(points: makePoints())
        store.states[3].insert(.locked)

        let plane = SplatCutPlane(
            point: SIMD3<Float>(0.2, 0.0, -2.0),
            normal: SIMD3<Float>(1.0, 0.0, 0.0)
        )

        XCTAssertEqual(store.planeSelectionIndices(plane: plane, side: .negative), [0, 1])
        XCTAssertEqual(store.planeSelectionIndices(plane: plane, side: .positive), [2])
    }

    func testEditableStoreBoundsForSelectableIndicesExcludeLockedPoints() throws {
        var store = EditableSplatStore(points: makePoints())
        store.states[3].insert(.locked)

        let bounds = try XCTUnwrap(store.bounds(for: store.selectableIndices))
        XCTAssertEqual(bounds.min.x, -0.75, accuracy: 0.0001)
        XCTAssertEqual(bounds.max.x, 0.55, accuracy: 0.0001)
        XCTAssertEqual(bounds.max.y, 0.0, accuracy: 0.0001)
    }

    func testEditableStoreOutlierIndicesSelectDetachedSparseVoxels() {
        let store = EditableSplatStore(points: makeOutlierPoints())

        let outliers = store.outlierIndices(
            config: OutlierSelectionConfig(
                scope: .visibleOnly,
                voxelFractionOfBounds: 0.05,
                minimumVoxelSize: 0.04,
                maximumVoxelSize: 0.2,
                scaleInfluence: 1.0,
                coreDensityThreshold: 0.3,
                annexDensityThreshold: 0.05
            )
        )

        XCTAssertEqual(outliers.sorted(), [6, 7])
    }

    func testEditableStoreOutlierIndicesRespectSelectionScope() {
        var store = EditableSplatStore(points: makeOutlierPoints())
        _ = store.applySelection(indices: [0, 1, 2, 6], mode: .replace)

        let outliers = store.outlierIndices(
            config: OutlierSelectionConfig(
                scope: .selectionOnly,
                voxelFractionOfBounds: 0.05,
                minimumVoxelSize: 0.04,
                maximumVoxelSize: 0.2,
                scaleInfluence: 1.0,
                coreDensityThreshold: 0.3,
                annexDensityThreshold: 0.05
            )
        )

        XCTAssertEqual(outliers, [6])
    }

    func testEditableStoreOutlierIndicesSkipHiddenLockedAndDeletedSplats() {
        var store = EditableSplatStore(points: makeOutlierPoints())
        store.states[6].insert(.hidden)
        store.states[7].insert(.locked)
        store.states[5].insert(.deleted)

        let outliers = store.outlierIndices(
            config: OutlierSelectionConfig(
                scope: .visibleOnly,
                voxelFractionOfBounds: 0.05,
                minimumVoxelSize: 0.04,
                maximumVoxelSize: 0.2,
                scaleInfluence: 1.0,
                coreDensityThreshold: 0.3,
                annexDensityThreshold: 0.05
            )
        )

        XCTAssertEqual(outliers, [])
    }

    func testEditableStoreOutlierIndicesDownWeightOversizedLooseSplats() {
        let points = [
            makePoint(position: SIMD3<Float>(0.00, 0.00, -2.0), color: SIMD3<Float>(1.0, 0.2, 0.2), opacity: 0.25),
            makePoint(position: SIMD3<Float>(0.06, 0.00, -2.0), color: SIMD3<Float>(0.9, 0.2, 0.2), opacity: 0.25),
            makePoint(position: SIMD3<Float>(0.00, 0.06, -2.0), color: SIMD3<Float>(0.8, 0.2, 0.2), opacity: 0.25),
            makePoint(
                position: SIMD3<Float>(1.8, 1.8, -2.0),
                color: SIMD3<Float>(0.2, 0.2, 1.0),
                opacity: 1.0,
                scale: SIMD3<Float>(repeating: 1.5)
            )
        ]
        let store = EditableSplatStore(points: points)

        let outliers = store.outlierIndices(
            config: OutlierSelectionConfig(
                scope: .visibleOnly,
                voxelFractionOfBounds: 0.05,
                minimumVoxelSize: 0.04,
                maximumVoxelSize: 0.2,
                scaleInfluence: 1.0,
                coreDensityThreshold: 0.3,
                annexDensityThreshold: 0.05,
                largeSplatPenalty: 1.0
            )
        )

        XCTAssertEqual(outliers, [3])
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

    func testOutlierSelectionReplaceModeParticipatesInUndoRedoHistory() async throws {
        let renderer = try makeRenderer()
        let editor = try await SplatEditor(points: makeOutlierPoints(), renderer: renderer)

        try await editor.selectOutliers(
            config: makeOutlierSelectionConfig(),
            mode: .replace
        )

        var snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 2)

        try await editor.undo()
        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 0)

        try await editor.redo()
        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 2)
    }

    func testOutlierSelectionRespectsAddAndSubtractModes() async throws {
        let renderer = try makeRenderer()
        let editor = try await SplatEditor(points: makeOutlierPoints(), renderer: renderer)

        try await editor.select(
            .sphere(center: SIMD3<Float>(0.0, 0.0, -2.0), radius: 0.15),
            mode: .replace,
            viewport: makeViewport()
        )

        var snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 6)

        try await editor.selectOutliers(
            config: makeOutlierSelectionConfig(),
            mode: .add
        )

        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 8)

        try await editor.selectOutliers(
            config: makeOutlierSelectionConfig(),
            mode: .subtract
        )

        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 6)
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

    func testPlaneSelectionCutAndUndoRoundTrip() async throws {
        let renderer = try makeRenderer()
        let editor = try await SplatEditor(points: makePoints(), renderer: renderer)
        let plane = SplatCutPlane(
            point: SIMD3<Float>(0.2, 0.0, -2.0),
            normal: SIMD3<Float>(1.0, 0.0, 0.0)
        )

        try await editor.select(plane: plane, side: .positive, mode: .replace)

        var snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 2)

        try await editor.cut(plane: plane, side: .positive)
        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.deletedCount, 2)
        XCTAssertEqual(snapshot.visibleCount, 2)
        XCTAssertEqual(snapshot.selectedCount, 0)

        try await editor.undo()
        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.deletedCount, 0)
        XCTAssertEqual(snapshot.visibleCount, 4)
        XCTAssertEqual(snapshot.selectedCount, 2)
    }

    func testAlignmentTransformWithoutSelectionUsesVisibleEditableSplatsAndKeepsSelectionEmpty() async throws {
        let renderer = try makeRenderer()
        let editor = try await SplatEditor(points: makePoints(), renderer: renderer)

        let maybeBounds = await editor.alignmentBounds()
        let bounds = try XCTUnwrap(maybeBounds)
        XCTAssertEqual(bounds.min.x, -0.75, accuracy: 0.0001)
        XCTAssertEqual(bounds.max.x, 0.75, accuracy: 0.0001)

        try await editor.applyAlignmentTransform(
            SplatEditTransform(translation: SIMD3<Float>(1, 0, 0)),
            pivot: bounds.center
        )

        var snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 0)

        var exported = try await editor.exportVisiblePoints()
        XCTAssertEqual(exported[0].position.x, 0.25, accuracy: 0.0001)
        XCTAssertEqual(exported[3].position.x, 1.75, accuracy: 0.0001)

        try await editor.undo()
        snapshot = await editor.snapshot()
        XCTAssertEqual(snapshot.selectedCount, 0)

        exported = try await editor.exportVisiblePoints()
        XCTAssertEqual(exported[0].position.x, -0.75, accuracy: 0.0001)
        XCTAssertEqual(exported[3].position.x, 0.75, accuracy: 0.0001)
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

    private func makeOutlierPoints() -> [SplatScenePoint] {
        [
            makePoint(position: SIMD3<Float>(0.00, 0.00, -2.0), color: SIMD3<Float>(0.9, 0.1, 0.1)),
            makePoint(position: SIMD3<Float>(0.04, 0.00, -2.0), color: SIMD3<Float>(0.85, 0.1, 0.1)),
            makePoint(position: SIMD3<Float>(0.00, 0.04, -2.0), color: SIMD3<Float>(0.8, 0.15, 0.1)),
            makePoint(position: SIMD3<Float>(-0.04, 0.00, -2.0), color: SIMD3<Float>(0.75, 0.15, 0.1)),
            makePoint(position: SIMD3<Float>(0.00, -0.04, -2.0), color: SIMD3<Float>(0.7, 0.2, 0.1)),
            makePoint(position: SIMD3<Float>(0.08, 0.02, -2.0), color: SIMD3<Float>(0.65, 0.25, 0.1)),
            makePoint(position: SIMD3<Float>(1.60, 1.55, -2.0), color: SIMD3<Float>(0.1, 0.7, 1.0), opacity: 0.5),
            makePoint(position: SIMD3<Float>(1.85, 1.70, -2.0), color: SIMD3<Float>(0.1, 0.6, 1.0), opacity: 0.45)
        ]
    }

    private func makeOutlierSelectionConfig() -> OutlierSelectionConfig {
        OutlierSelectionConfig(
            scope: .visibleOnly,
            voxelFractionOfBounds: 0.05,
            minimumVoxelSize: 0.04,
            maximumVoxelSize: 0.2,
            scaleInfluence: 1.0,
            coreDensityThreshold: 0.3,
            annexDensityThreshold: 0.05
        )
    }

    private func makePoint(position: SIMD3<Float>,
                           color: SIMD3<Float>,
                           opacity: Float = 1.0,
                           scale: SIMD3<Float> = SIMD3<Float>(repeating: 0.05)) -> SplatScenePoint {
        SplatScenePoint(
            position: position,
            color: .linearFloat(color),
            opacity: .linearFloat(opacity),
            scale: .linearFloat(scale),
            rotation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0))
        )
    }
}
