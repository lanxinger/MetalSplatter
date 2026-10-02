import XCTest
import Metal
import SplatIO
import simd
@testable import MetalSplatter

/// Actual Metal render regressions. A missing device may skip these tests;
/// missing shaders, failed commands, and incorrect pixels must fail them.
@MainActor
final class FastSHRendererRegressionTests: XCTestCase {
    private let outgoing = simd_normalize(SIMD3<Float>(0.3, 0.4, 0.8660254))

    func testRepeatedSortsPreserveIndicesRetainedByEarlierGPUWork() async throws {
        for useGPU in [false, true] {
            try await assertEarlierIndicesSurviveSorts(useGPU: useGPU)
        }
    }

    func testRepeatedSortsWithoutRenderingKeepPoolBounded() async throws {
        let gpu = try GPU()
        try await gpu.load([point([.zero])])
        for useGPU in [false, true] {
            for _ in 0..<40 { try await gpu.sort(useGPU: useGPU) }
            let statistics = gpu.renderer.sortedIndexPoolStatistics
            XCTAssertEqual(statistics.leasedBuffers, 2)
            XCTAssertLessThanOrEqual(statistics.availableBuffers, 1)
        }
    }

    func testRetiredIndicesWaitForTheirOwnRenderReaders() async throws {
        // Ordinary vertex, ordinary mesh, and FastSH bindings share retirement.
        for variant in 0..<3 {
            let gpu = try GPU()
            gpu.renderer.fastSHConfig.enabled = variant == 2
            gpu.renderer.meshShaderEnabled = variant == 1
            var near = point([.zero])
            near.position = SIMD3(0, 0, -1)
            var far = near
            far.position.z = -2
            try await gpu.load([near, far])
            let viewport = camera(eye: SIMD3(0, 0, 5))
            _ = try await gpu.render(viewport, frameCount: 1)
            try await gpu.sort()
            let initial = try XCTUnwrap(gpu.renderer.getCurrentSortedIndicesBuffer())
            let expected = Array(UnsafeBufferPointer(start: initial.values, count: 2))
            let firstReader = try gpu.encodeRender(viewport)
            let secondReader = try gpu.encodeRender(viewport)
            let readback = try XCTUnwrap(initial.device.makeBuffer(length: 8, options: .storageModeShared))
            let blit = try XCTUnwrap(secondReader.makeBlitCommandEncoder())
            blit.copy(from: initial.buffer, sourceOffset: 0, to: readback, destinationOffset: 0, size: 8)
            blit.endEncoding()

            try await gpu.sort()
            try await gpu.sort()
            XCTAssertEqual(gpu.renderer.sortedIndexPoolStatistics.leasedBuffers, 3)
            // A later frame can finish before either earlier reader is submitted.
            try await gpu.complete(try gpu.encodeRender(viewport))
            XCTAssertEqual(initial.count, 2, "An unrelated frame must not release this buffer")
            try await gpu.complete(firstReader)
            XCTAssertEqual(initial.count, 2, "The second reader still owns the buffer")
            try await gpu.complete(secondReader)
            let values = readback.contents().bindMemory(to: Int32.self, capacity: 2)
            XCTAssertEqual(Array(UnsafeBufferPointer(start: values, count: 2)), expected)
            XCTAssertEqual(initial.count, 0, "The last reader must return the retired lease")
            XCTAssertEqual(gpu.renderer.sortedIndexPoolStatistics.leasedBuffers, 2)
        }
    }

    func testVisibilityCompactionPreservesIndicesHeldByRenderReaders() async throws {
        for variant in 0..<3 {
            let gpu = try GPU()
            gpu.renderer.fastSHConfig.enabled = variant == 2
            gpu.renderer.meshShaderEnabled = variant == 1
            var near = point([.zero])
            near.position = SIMD3(0, 0, -1)
            var far = near
            far.position.z = -2
            try await gpu.load([near, far])
            let viewport = camera(eye: SIMD3(0, 0, 5))
            _ = try await gpu.render(viewport, frameCount: 1)
            try await gpu.sort()
            let initial = try XCTUnwrap(gpu.renderer.getCurrentSortedIndicesBuffer())
            let expected = Array(UnsafeBufferPointer(start: initial.values, count: 2))
            let reader = try gpu.encodeRender(viewport)
            let readback = try XCTUnwrap(initial.device.makeBuffer(length: 8, options: .storageModeShared))
            let blit = try XCTUnwrap(reader.makeBlitCommandEncoder())
            blit.copy(from: initial.buffer, sourceOffset: 0, to: readback, destinationOffset: 0, size: 8)
            blit.endEncoding()

            try gpu.renderer.updateEditStates(at: [1], values: [EditableSplatState.hidden.rawValue])
            let compacted = try XCTUnwrap(gpu.renderer.getCurrentSortedIndicesBuffer())
            XCTAssertFalse(compacted === initial)
            XCTAssertEqual(compacted.count, 1)
            XCTAssertEqual(compacted.values[0], 0)
            // A second reduction retires the slot held by the encoded reader.
            try gpu.renderer.updateEditStates(at: [0], values: [EditableSplatState.hidden.rawValue])
            XCTAssertEqual(gpu.renderer.getCurrentSortedIndicesBuffer()?.count, 0)
            XCTAssertEqual(initial.count, 2)
            XCTAssertEqual(gpu.renderer.sortedIndexPoolStatistics.leasedBuffers, 3)
            try await gpu.complete(reader)
            let values = readback.contents().bindMemory(to: Int32.self, capacity: 2)
            XCTAssertEqual(Array(UnsafeBufferPointer(start: values, count: 2)), expected)
            XCTAssertEqual(initial.count, 0)
            XCTAssertEqual(gpu.renderer.sortedIndexPoolStatistics.leasedBuffers, 2)
        }
    }

    func testVisibilityCompactionDuringSortDefersToNextSort() async throws {
        let gpu = try GPU()
        var near = point([.zero])
        near.position = SIMD3(0, 0, -1)
        var far = near
        far.position.z = -2
        try await gpu.load([near, far])
        let initial = try XCTUnwrap(gpu.renderer.getCurrentSortedIndicesBuffer())
        var editError: Error?
        var observedIndices: MetalBuffer<Int32>?
        var observedCount: Int?
        // This callback runs after the sort gate is acquired and before its
        // worker starts, giving the edit a deterministic busy-sort interleaving.
        gpu.renderer.onSortStart = {
            do {
                try gpu.renderer.updateEditStates(at: [1], values: [EditableSplatState.hidden.rawValue])
                observedIndices = gpu.renderer.getCurrentSortedIndicesBuffer()
                observedCount = observedIndices?.count
            } catch {
                editError = error
            }
        }
        try await gpu.sort()
        gpu.renderer.onSortStart = nil
        XCTAssertNil(editError)
        XCTAssertTrue(observedIndices === initial, "A busy sort must leave the published input intact")
        XCTAssertEqual(observedCount, 2, "The edit must not compact a snapshot while a sort can retire it")
        try await gpu.sort(useGPU: true)
        let filtered = try XCTUnwrap(gpu.renderer.getCurrentSortedIndicesBuffer())
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered.values[0], 0)
    }

    private func assertEarlierIndicesSurviveSorts(useGPU: Bool) async throws {
        let gpu = try GPU()
        let renderer = gpu.renderer
        renderer.mortonOrderingEnabled = false
        var near = point([.zero])
        near.position = SIMD3(0, 0, -1)
        var far = near
        far.position.z = -2
        try renderer.add([near, far])
        let byteCount = 2 * MemoryLayout<Int32>.stride
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let initialIndices = try XCTUnwrap(renderer.getCurrentSortedIndicesBuffer(for: command))
        let readback = try XCTUnwrap(device.makeBuffer(length: byteCount, options: .storageModeShared))
        let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
        blit.copy(from: initialIndices.buffer, sourceOffset: 0,
                  to: readback, destinationOffset: 0, size: byteCount)
        blit.endEncoding()

        // Hold an earlier GPU reader while two sorts rotate the published slots.
        // Cover both writers; the held reader makes premature reuse deterministic.
        try await gpu.sort(useGPU: useGPU)
        try await gpu.sort(useGPU: useGPU)
        let currentIndices = try XCTUnwrap(renderer.getCurrentSortedIndicesBuffer())
        XCTAssertEqual(Array(UnsafeBufferPointer(start: currentIndices.values, count: 2)), [1, 0])
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            command.addCompletedHandler { _ in continuation.resume() }
            command.commit()
        }
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        let values = readback.contents().bindMemory(to: Int32.self, capacity: 2)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: values, count: 2)), [0, 1], "GPU sort: \(useGPU)")
    }

    func testAllSHDegreesBasisFunctionsAndRGBChannelsThroughBaseDispatch() async throws {
        let gpu = try GPU()
        let viewport = camera(eye: outgoing * 5)
        for degree in 0...3 {
            let count = (degree + 1) * (degree + 1)
            for coefficient in 0..<count {
                for channel in 0..<3 {
                    var coefficients = Array(repeating: SIMD3<Float>.zero, count: count)
                    coefficients[coefficient][channel] = 0.6
                    try await gpu.load([point(coefficients)])
                    let rgba = try await gpu.render(viewport)
                    var expected = SIMD3<Float>(repeating: 0.5)
                    expected[channel] += 0.6 * graphdecoBasis(-outgoing)[coefficient]
                    assertColor(rgba, x: 32, y: 32, expected: expected,
                                context: "degree \(degree), coefficient \(coefficient), channel \(channel)")
                }
            }
        }
    }

    func testNonGrayDCMatchesExistingGamma22ColorContract() async throws {
        let gpu = try GPU()
        let color = SIMD3<Float>(0.18, 0.65, 0.8)
        let dc = (color - SIMD3(repeating: 0.5)) / 0.2820947918
        for count in [1, 16] {
            var coefficients = Array(repeating: SIMD3<Float>.zero, count: count)
            coefficients[0] = dc
            try await gpu.load([point(coefficients)])
            assertColor(try await gpu.render(camera(eye: outgoing * 5)), x: 32, y: 32,
                        expected: color, context: "non-gray DC with \(count) coefficients", accuracy: 0.0006)
        }
    }

    func testCovarianceQuaternionDoesNotRotateWorldSpaceSH() async throws {
        let gpu = try GPU()
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 16)
        coefficients[1] = SIMD3(0.6, 0, 0)
        coefficients[6] = SIMD3(0, 0.6, 0)
        coefficients[12] = SIMD3(0, 0, 0.6)
        let expected = evaluate(coefficients, direction: -outgoing)
        for rotation in [simd_quatf(real: 1, imag: .zero),
                         simd_quatf(angle: 1.1, axis: simd_normalize(SIMD3(0.2, 0.8, 0.4))),
                         simd_quatf(angle: -.pi / 2, axis: SIMD3(0, 0, 1))] {
            var value = point(coefficients)
            value.rotation = rotation
            value.scale = .linearFloat(SIMD3(0.5, 0.3, 0.15))
            try await gpu.load([value])
            assertColor(try await gpu.render(camera(eye: outgoing * 5)), x: 32, y: 32,
                        expected: expected, context: "covariance rotation \(rotation.vector)")
        }
    }

    func testBaseDispatchRespondsToCameraAndKeepsSHOnStationaryFrames() async throws {
        let gpu = try GPU()
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 4)
        coefficients[3] = SIMD3(0.8, 0, 0)
        try await gpu.load([point(coefficients)])
        var colors: [SIMD3<Float>] = []
        for eye in [SIMD3<Float>(3, 0, 4), SIMD3<Float>(-3, 0, 4)] {
            let viewport = camera(eye: eye)
            let first = try await gpu.render(viewport)
            let repeated = try await gpu.render(viewport)
            let expected = evaluate(coefficients, direction: -simd_normalize(eye))
            assertColor(first, x: 32, y: 32, expected: expected, context: "camera move")
            assertColor(repeated, x: 32, y: 32, expected: expected, context: "stationary repeat")
            colors.append(unpremultiplied(repeated, x: 32, y: 32))
        }
        XCTAssertGreaterThan(abs(colors[0].x - colors[1].x), 0.1)
        coefficients[3].x = -0.8
        try await gpu.load([point(coefficients)])
        assertColor(try await gpu.render(camera(eye: SIMD3(-3, 0, 4))), x: 32, y: 32,
                    expected: evaluate(coefficients, direction: simd_normalize(SIMD3(3, 0, -4))),
                    context: "data changes at stationary camera")
    }

    func testDisabledFastSHAndDisabledSHRenderingUseDCTerm() async throws {
        let gpu = try GPU()
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 16)
        coefficients[3] = SIMD3(0.8, 0.4, -0.6)
        try await gpu.load([point(coefficients)])
        gpu.renderer.fastSHConfig.enabled = false
        assertColor(try await gpu.render(camera(eye: outgoing * 5)), x: 32, y: 32,
                    expected: SIMD3(repeating: 0.5), context: "FastSH configuration disabled")
        gpu.renderer.fastSHConfig.enabled = true
        gpu.renderer.shRenderingEnabled = false
        assertColor(try await gpu.render(camera(eye: outgoing * 5)), x: 32, y: 32,
                    expected: SIMD3(repeating: 0.5), context: "SH rendering disabled")
        gpu.renderer.shRenderingEnabled = true
        assertColor(try await gpu.render(camera(eye: outgoing * 5)), x: 32, y: 32,
                    expected: evaluate(coefficients, direction: -outgoing), context: "SH reenabled")
    }

    func testOrthographicSHUsesCameraForwardForEveryPosition() async throws {
        let gpu = try GPU()
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 4)
        coefficients[3] = SIMD3(0.8, 0.2, -0.6)
        let positions = [SIMD3<Float>(-0.8, 0, 0), SIMD3<Float>(0.8, 0, 0)]
        try await gpu.load(positions.map { position in
            var value = point(coefficients); value.position = position; return value
        })
        let viewport = camera(eye: outgoing * 5, orthographic: true)
        let image = try await gpu.render(viewport)
        for position in positions {
            let clip = viewport.projectionMatrix * viewport.viewMatrix * SIMD4(position, 1)
            let x = Int((clip.x / clip.w + 1) * 32.5)
            let y = Int((1 - clip.y / clip.w) * 32.5)
            assertColor(image, x: x, y: y, expected: evaluate(coefficients, direction: -outgoing),
                        context: "orthographic position \(position)")
        }
    }

    func testEditingSelectionVisibilityAndUndoPreserveSH() async throws {
        let gpu = try GPU()
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 16)
        coefficients[6] = SIMD3(0.6, -0.4, 0.2)
        let values = [-0.7 as Float, 0.7].map { x in
            var value = point(coefficients)
            value.position.x = x
            value.scale = .linearFloat(SIMD3(0.15, 0.15, 0.03))
            return value
        }
        try await gpu.load(values)
        let editor = try await SplatEditor(points: values, renderer: gpu.renderer)
        try await gpu.sort()
        let viewport = camera(eye: SIMD3(0, 0, 5), orthographic: true)
        let before = try await gpu.render(viewport)
        let expected = evaluate(coefficients, direction: SIMD3(0, 0, -1))
        assertColor(before, x: 21, y: 32, expected: expected, context: "editable SH")
        try await editor.select(.sphere(center: SIMD3(-0.7, 0, 0), radius: 0.3), mode: .replace, viewport: viewport)
        let selected = try await gpu.render(viewport)
        XCTAssertGreaterThan(abs(unpremultiplied(before, x: 21, y: 32).z - unpremultiplied(selected, x: 21, y: 32).z), 0.05)
        XCTAssertTrue(zip(before, selected).contains { $1.w > $0.w + 0.01 }, "Selection outline must add visible ring coverage")
        assertColor(selected, x: 43, y: 32, expected: expected, context: "unselected neighbor")
        try await editor.hideSelection()
        let hidden = try await gpu.render(viewport)
        XCTAssertLessThan(hidden[32 * 65 + 21].w, 0.01)
        assertColor(hidden, x: 43, y: 32, expected: expected, context: "visible neighbor after hiding")
        try await editor.undo()
        try await gpu.sort() // Undo expands the visible index list asynchronously.
        let restored = try await gpu.render(viewport)
        XCTAssertGreaterThan(restored[32 * 65 + 21].w, 0.9)
        try await editor.clearSelection()
        assertColor(try await gpu.render(viewport), x: 21, y: 32,
                    expected: expected, context: "selection cleared after undo")
    }

    func testNonzeroViewportAndLoadActionPreserveExistingBackground() async throws {
        let gpu = try GPU()
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 4)
        coefficients[1] = SIMD3(0.6, -0.4, 0.2)
        try await gpu.load([point(coefficients)])
        var viewport = camera(eye: outgoing * 5)
        viewport.viewport = MTLViewport(originX: 7, originY: 9, width: 41, height: 41, znear: 0, zfar: 1)
        viewport.screenSize = SIMD2(41, 41)
        gpu.renderer.clearColor = MTLClearColorMake(0.8, 0.7, 0.6, 1)
        let image = try await gpu.render(viewport, loadAction: .load,
                                         initialClear: MTLClearColorMake(0.1, 0.2, 0.3, 1))
        XCTAssertEqual(image[0].x, 0.1, accuracy: 0.0003)
        XCTAssertEqual(image[0].y, 0.2, accuracy: 0.0003)
        XCTAssertEqual(image[0].z, 0.3, accuracy: 0.0003)
        assertColor(image, x: 27, y: 29, expected: evaluate(coefficients, direction: -outgoing), context: "offset viewport center")
    }

    func testColorOnlyUpdateReplacesSHWithoutApplyingGammaTwice() async throws {
        let gpu = try GPU()
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 16)
        coefficients[6] = SIMD3(0.6, -0.4, 0.2)
        try await gpu.load([point(coefficients)])
        let viewport = camera(eye: outgoing * 5)
        _ = try await gpu.render(viewport)
        let linear = SIMD3<Float>(0.15, 0.35, 0.65)
        gpu.renderer.updateColorsOnly([SIMD4(linear, 1)])
        for _ in 0..<2 {
            let actual = unpremultiplied(try await gpu.render(viewport), x: 32, y: 32)
            for channel in 0..<3 {
                XCTAssertEqual(actual[channel], (linear[channel] * 255).rounded() / 255, accuracy: 0.0006)
            }
        }
    }

    func testMixedDegreeDCFirstRetainsLaterSH3AndPartialColorUpdates() async throws {
        let gpu = try GPU()
        var dc = point([.zero]); dc.position.x = -0.7
        dc.scale = .linearFloat(SIMD3(0.15, 0.15, 0.03))
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 16)
        coefficients[12] = SIMD3(0.4, -0.3, 0.2)
        var sh = point(coefficients); sh.position.x = 0.7; sh.scale = dc.scale
        try await gpu.load([dc, sh])
        let viewport = camera(eye: SIMD3(0, 0, 5), orthographic: true)
        let expected = evaluate(coefficients, direction: SIMD3(0, 0, -1))
        let initial = try await gpu.render(viewport)
        XCTAssertEqual(gpu.renderer.shDegree, 3)
        assertColor(initial, x: 21, y: 32, expected: SIMD3(repeating: 0.5), context: "DC first in mixed scene")
        assertColor(initial, x: 43, y: 32, expected: expected, context: "SH3 after DC")
        let replacement = SIMD4<Float>(0.15, 0.35, 0.65, 0.6)
        gpu.renderer.updateColorOnly(replacement, at: 0)
        let updated = try await gpu.render(viewport)
        let actual = unpremultiplied(updated, x: 21, y: 32)
        for channel in 0..<3 {
            XCTAssertEqual(actual[channel], (replacement[channel] * 255).rounded() / 255, accuracy: 0.001)
        }
        XCTAssertLessThan(updated[32 * 65 + 21].w, 0.65)
        XCTAssertGreaterThan(updated[32 * 65 + 21].w, 0.5)
        assertColor(updated, x: 43, y: 32, expected: expected, context: "unchanged SH neighbor after partial color update")
    }

    func testEmptyAndFullyHiddenScenesHonorClearAndLoadOnReusedTarget() async throws {
        let gpu = try GPU()
        let viewport = camera(eye: outgoing * 5)
        for hidePoint in [false, true] {
            if hidePoint {
                let value = point(Array(repeating: .zero, count: 16))
                try await gpu.load([value])
                let editor = try await SplatEditor(points: [value], renderer: gpu.renderer)
                try await editor.selectAll()
                try await editor.hideSelection()
            }
            gpu.renderer.clearColor = MTLClearColorMake(0.8, 0.7, 0.6, 1)
            let sentinel = MTLClearColorMake(0.1, 0.2, 0.3, 1)
            _ = try await gpu.render(viewport, loadAction: .load, initialClear: sentinel, reuseTexture: true)
            let loaded = try await gpu.render(viewport, loadAction: .load, preclear: false, reuseTexture: true)
            XCTAssertEqual(loaded[0].x, 0.1, accuracy: 0.0003)
            XCTAssertEqual(loaded[32 * 65 + 32].z, 0.3, accuracy: 0.0003)
            let cleared = try await gpu.render(viewport, preclear: false, reuseTexture: true)
            XCTAssertEqual(cleared[0].x, 0.8, accuracy: 0.0003)
            XCTAssertEqual(cleared[32 * 65 + 32].z, 0.6, accuracy: 0.0003)
        }
    }

    func testPreviewRotationAndCommittedSHHaveMatchingAppearance() async throws {
        let gpu = try GPU()
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 16)
        coefficients[1] = SIMD3(0.6, -0.2, 0.1)
        coefficients[6] = SIMD3(-0.1, 0.5, -0.2)
        coefficients[12] = SIMD3(0.1, -0.1, 0.4)
        var value = point(coefficients)
        value.scale = .linearFloat(SIMD3(0.5, 0.3, 0.2))
        try await gpu.load([value])
        let editor = try await SplatEditor(points: [value], renderer: gpu.renderer)
        let viewport = camera(eye: outgoing * 5, orthographic: true)
        let rotation = simd_quatf(angle: 1.1, axis: simd_normalize(SIMD3<Float>(0.2, 0.8, 0.4)))
        let originalColor = evaluate(coefficients, direction: -outgoing)
        let rotatedColor = evaluate(coefficients, direction: rotation.inverse.act(-outgoing))
        let initial = try await gpu.render(viewport)
        assertColor(initial, x: 32, y: 32, expected: originalColor, context: "before preview")
        await editor.beginPreviewTransform(pivot: .zero, scope: .selectionOrVisible)
        try await editor.updatePreviewTransform(SplatEditTransform(rotation: rotation))
        let preview = try await gpu.render(viewport)
        assertColor(preview, x: 32, y: 32, expected: rotatedColor, context: "preview SH rotation")
        try await editor.commitPreviewTransform()
        try await gpu.sort()
        let committed = try await gpu.render(viewport)
        assertColor(committed, x: 32, y: 32, expected: rotatedColor, context: "committed SH rotation")
        let previewRGB = unpremultiplied(preview, x: 32, y: 32)
        let committedRGB = unpremultiplied(committed, x: 32, y: 32)
        for channel in 0..<3 { XCTAssertEqual(previewRGB[channel], committedRGB[channel], accuracy: 0.004) }
        XCTAssertGreaterThan(simd_length(unpremultiplied(initial, x: 32, y: 32) - previewRGB), 0.03)
        try await editor.undo()
        try await gpu.sort()
        assertColor(try await gpu.render(viewport), x: 32, y: 32, expected: originalColor, context: "rotation undo")
    }

    func testColorQueuedAfterFrameDrainIsAppliedOnNextFrame() async throws {
        let gpu = try GPU(injectColorAfterDrain: true)
        let interleaved = try XCTUnwrap(gpu.renderer as? InterleavedColorRenderer)
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 16)
        coefficients[6] = SIMD3(0.4, -0.2, 0.3)
        let values = [-0.7 as Float, 0.7].map { x in
            var value = point(coefficients); value.position.x = x
            value.scale = .linearFloat(SIMD3(0.15, 0.15, 0.03)); return value
        }
        try await gpu.load(values)
        let viewport = camera(eye: SIMD3(0, 0, 5), orthographic: true)
        _ = try await gpu.render(viewport)
        let first = SIMD4<Float>(0.1, 0.3, 0.6, 1)
        let second = SIMD4<Float>(0.8, 0.2, 0.1, 1)
        let revision = gpu.renderer.appliedColorRevision
        gpu.renderer.updateColorOnly(first, at: 0)
        XCTAssertEqual(gpu.renderer.appliedColorRevision, revision, "Enqueue is not application")
        interleaved.colorAfterDrain = second
        let firstFrame = try await gpu.render(viewport, frameCount: 1)
        XCTAssertEqual(gpu.renderer.appliedColorRevision, revision + 1)
        let nextFrame = try await gpu.render(viewport, frameCount: 1)
        XCTAssertEqual(gpu.renderer.appliedColorRevision, revision + 2)
        for (frame, expected) in [(firstFrame, first), (nextFrame, second)] {
            let actual = unpremultiplied(frame, x: 21, y: 32)
            for channel in 0..<3 {
                XCTAssertEqual(actual[channel], (expected[channel] * 255).rounded() / 255, accuracy: 0.001)
            }
            assertColor(frame, x: 43, y: 32, expected: evaluate(coefficients, direction: SIMD3(0, 0, -1)),
                        context: "Interleaved color update preserves SH neighbor")
        }
    }

    func testSceneReplacementDiscardsQueuedColorsForPreviousScene() async throws {
        let gpu = try GPU()
        var coefficients = Array(repeating: SIMD3<Float>.zero, count: 16)
        coefficients[6] = SIMD3(0.4, -0.2, 0.3)
        try await gpu.load([point(coefficients)])
        let viewport = camera(eye: outgoing * 5)
        _ = try await gpu.render(viewport)
        let revision = gpu.renderer.appliedColorRevision
        gpu.renderer.updateColorOnly(SIMD4(0.9, 0.1, 0.2, 1), at: 0)
        coefficients[6] = SIMD3(-0.4, 0.3, -0.2)
        try await gpu.load([point(coefficients)])
        assertColor(try await gpu.render(viewport), x: 32, y: 32,
                    expected: evaluate(coefficients, direction: -outgoing), context: "Replacement scene keeps its own SH")
        XCTAssertEqual(gpu.renderer.appliedColorRevision, revision, "Discarded old-scene update must not apply")
    }

    private func point(_ coefficients: [SIMD3<Float>]) -> SplatScenePoint {
        SplatScenePoint(position: .zero, color: .sphericalHarmonic(coefficients),
                        opacity: .linearFloat(0.999), scale: .linearFloat(SIMD3(0.5, 0.5, 0.03)),
                        rotation: simd_quatf(real: 1, imag: .zero))
    }

    private func camera(eye: SIMD3<Float>, orthographic: Bool = false) -> SplatRenderer.ViewportDescriptor {
        let back = simd_normalize(eye)
        let right = simd_normalize(simd_cross(SIMD3<Float>(0, 1, 0), back))
        let up = simd_cross(back, right)
        let view = simd_float4x4(columns: (
            SIMD4(right.x, up.x, back.x, 0), SIMD4(right.y, up.y, back.y, 0),
            SIMD4(right.z, up.z, back.z, 0),
            SIMD4(-simd_dot(right, eye), -simd_dot(up, eye), -simd_dot(back, eye), 1)))
        let z: Float = 30 / (0.05 - 30), focal: Float = 1 / tan(40 * .pi / 360)
        let projection = orthographic ? simd_float4x4(columns: (
            SIMD4(0.5, 0, 0, 0), SIMD4(0, 0.5, 0, 0),
            SIMD4(0, 0, 1 / (0.05 - 30), 0), SIMD4(0, 0, 0.05 / (0.05 - 30), 1))) :
            simd_float4x4(columns: (SIMD4(focal, 0, 0, 0), SIMD4(0, focal, 0, 0),
                                   SIMD4(0, 0, z, -1), SIMD4(0, 0, z * 0.05, 0)))
        return .init(viewport: MTLViewport(originX: 0, originY: 0, width: 65, height: 65, znear: 0, zfar: 1),
                     projectionMatrix: projection, viewMatrix: view, screenSize: SIMD2(65, 65), isOrthographic: orthographic)
    }

    private func unpremultiplied(_ image: [SIMD4<Float>], x: Int, y: Int) -> SIMD3<Float> {
        let p = image[y * 65 + x]
        return SIMD3(p.x, p.y, p.z) / max(p.w, 1e-8)
    }

    private func assertColor(_ image: [SIMD4<Float>], x: Int, y: Int, expected: SIMD3<Float>, context: String,
                             accuracy: Float = 0.004,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertGreaterThan(image[y * 65 + x].w, 0.1, context, file: file, line: line)
        let actual = unpremultiplied(image, x: x, y: y)
        for channel in 0..<3 {
            let c = min(1, max(0, expected[channel]))
            // Preserve the renderer's existing gamma-2.2 contract. This is
            // deliberately not the IEC piecewise sRGB EOTF.
            let linear = pow(c, 2.2)
            let packed = (linear * 255).rounded() / 255
            XCTAssertEqual(actual[channel], packed, accuracy: accuracy, "\(context), RGB \(channel)", file: file, line: line)
        }
    }

    private func evaluate(_ coefficients: [SIMD3<Float>], direction: SIMD3<Float>) -> SIMD3<Float> {
        let basis = graphdecoBasis(direction)
        return coefficients.enumerated().reduce(SIMD3<Float>(repeating: 0.5)) { $0 + $1.element * basis[$1.offset] }
    }

    // Independent CPU reference for the public Graphdeco SH0-SH3 convention.
    private func graphdecoBasis(_ d: SIMD3<Float>) -> [Float] {
        let x = d.x, y = d.y, z = d.z, xx = x * x, yy = y * y, zz = z * z
        return [0.2820947918, -0.4886025119 * y, 0.4886025119 * z, -0.4886025119 * x,
                1.0925484306 * x * y, -1.0925484306 * y * z, 0.3153915653 * (2 * zz - xx - yy),
                -1.0925484306 * x * z, 0.5462742153 * (xx - yy),
                -0.5900435899 * y * (3 * xx - yy), 2.8906114426 * x * y * z,
                -0.4570457995 * y * (4 * zz - xx - yy), 0.3731763326 * z * (2 * zz - 3 * xx - 3 * yy),
                -0.4570457995 * x * (4 * zz - xx - yy), 1.4453057213 * z * (xx - yy),
                -0.5900435899 * x * (xx - 3 * yy)]
    }
}

@MainActor
private final class GPU {
    let renderer: FastSHSplatRenderer
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let completedSorts = SortCounter()
    private var reusableTexture: MTLTexture?

    init(injectColorAfterDrain: Bool = false) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal device unavailable") }
        self.device = device
        queue = try XCTUnwrap(device.makeCommandQueue())
        if injectColorAfterDrain {
            renderer = try InterleavedColorRenderer(device: device, colorFormat: .rgba16Float, depthFormat: .invalid,
                                                    sampleCount: 1, maxViewCount: 1, maxSimultaneousRenders: 1)
        } else {
            renderer = try FastSHSplatRenderer(device: device, colorFormat: .rgba16Float, depthFormat: .invalid,
                                              sampleCount: 1, maxViewCount: 1, maxSimultaneousRenders: 1)
        }
        renderer.frustumCullingEnabled = false
        renderer.minimumSHUpdateInterval = 0
        renderer.onSortComplete = { [completedSorts] _ in completedSorts.increment() }
    }

    func load(_ points: [SplatScenePoint]) async throws {
        try await renderer.loadSplatsWithSH(points)
        try await sort()
    }

    func sort(useGPU: Bool = false) async throws {
        let start = completedSorts.value
        renderer.resort(useGPU: useGPU)
        for _ in 0..<500 {
            if completedSorts.value > start { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Sort timed out (GPU: \(useGPU))")
        throw RegressionError.sortTimeout
    }

    func encodeRender(_ viewport: SplatRenderer.ViewportDescriptor) throws -> MTLCommandBuffer {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 65, height: 65, mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]
        descriptor.storageMode = .shared
        let texture = try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        try renderer.render(viewports: [viewport], colorTexture: texture, colorStoreAction: .store,
                            depthTexture: nil, rasterizationRateMap: nil, renderTargetArrayLength: 1, to: command)
        return command
    }

    func complete(_ command: MTLCommandBuffer) async throws {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            command.addCompletedHandler { _ in continuation.resume() }
            command.commit()
        }
        XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
        if command.status != .completed { throw RegressionError.commandFailed }
    }

    func render(_ viewport: SplatRenderer.ViewportDescriptor, loadAction: MTLLoadAction = .clear,
                initialClear: MTLClearColor = MTLClearColorMake(0, 0, 0, 0),
                preclear: Bool = true, reuseTexture: Bool = false, frameCount: Int = 3) async throws -> [SIMD4<Float>] {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 65, height: 65, mipmapped: false)
        descriptor.usage = [.renderTarget, .shaderRead]; descriptor.storageMode = .shared
        let texture = try XCTUnwrap(reuseTexture ? (reusableTexture ?? device.makeTexture(descriptor: descriptor)) : device.makeTexture(descriptor: descriptor))
        if reuseTexture { reusableTexture = texture }
        // Intentionally base-typed: this is the signature used by app previews.
        let baseRenderer: SplatRenderer = renderer
        for _ in 0..<frameCount {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            if preclear {
                let clear = MTLRenderPassDescriptor()
                clear.colorAttachments[0].texture = texture
                clear.colorAttachments[0].loadAction = .clear
                clear.colorAttachments[0].storeAction = .store
                clear.colorAttachments[0].clearColor = initialClear
                try XCTUnwrap(command.makeRenderCommandEncoder(descriptor: clear)).endEncoding()
            }
            try baseRenderer.render(viewports: [viewport], colorTexture: texture, colorLoadAction: loadAction, colorStoreAction: .store,
                                    depthTexture: nil, rasterizationRateMap: nil, renderTargetArrayLength: 1, to: command)
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                command.addCompletedHandler { _ in continuation.resume() }; command.commit()
            }
            XCTAssertEqual(command.status, .completed, "\(String(describing: command.error))")
            if command.status != .completed { throw RegressionError.commandFailed }
        }
        var values = [Float16](repeating: 0, count: 65 * 65 * 4)
        values.withUnsafeMutableBytes { texture.getBytes($0.baseAddress!, bytesPerRow: 65 * 8,
            from: MTLRegionMake2D(0, 0, 65, 65), mipmapLevel: 0) }
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        return stride(from: 0, to: values.count, by: 4).map { SIMD4(Float(values[$0]), Float(values[$0 + 1]), Float(values[$0 + 2]), Float(values[$0 + 3])) }
    }
}

/// Deterministic interleaving at the seam after base color-queue draining and
/// before FastSH observes the applied revision. No concurrent timing assumptions.
private final class InterleavedColorRenderer: FastSHSplatRenderer, @unchecked Sendable {
    var colorAfterDrain: SIMD4<Float>?

    override func renderSpecialized(viewports: [ViewportDescriptor], colorTexture: MTLTexture,
                                    colorLoadAction: MTLLoadAction, colorStoreAction: MTLStoreAction,
                                    depthTexture: MTLTexture?, depthStoreAction: MTLStoreAction,
                                    rasterizationRateMap: MTLRasterizationRateMap?, renderTargetArrayLength: Int,
                                    animatedSplatsUpdated: Bool, to commandBuffer: MTLCommandBuffer) throws -> Bool {
        if let color = colorAfterDrain {
            colorAfterDrain = nil
            updateColorOnly(color, at: 0)
        }
        return try super.renderSpecialized(viewports: viewports, colorTexture: colorTexture,
                                           colorLoadAction: colorLoadAction, colorStoreAction: colorStoreAction,
                                           depthTexture: depthTexture, depthStoreAction: depthStoreAction,
                                           rasterizationRateMap: rasterizationRateMap,
                                           renderTargetArrayLength: renderTargetArrayLength,
                                           animatedSplatsUpdated: animatedSplatsUpdated, to: commandBuffer)
    }
}

private enum RegressionError: Error { case sortTimeout, commandFailed }
private final class SortCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func increment() { lock.withLock { count += 1 } }
    var value: Int { lock.withLock { count } }
}
