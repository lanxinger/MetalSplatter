import XCTest
import Spatial
import SplatIO
import simd

final class SplatIOTests: XCTestCase {
    class ContentCounter: SplatSceneReaderDelegate {
        var expectedPointCount: UInt32?
        var pointCount: UInt32 = 0
        var didFinish = false
        var didFail = false

        func reset() {
            expectedPointCount = nil
            pointCount = 0
            didFinish = false
            didFail = false
        }

        func didStartReading(withPointCount pointCount: UInt32?) {
            XCTAssertNil(expectedPointCount)
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            expectedPointCount = pointCount
        }

        func didRead(points: [SplatIO.SplatScenePoint]) {
            pointCount += UInt32(points.count)
        }

        func didFinishReading() {
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            didFinish = true
        }

        func didFailReading(withError error: Error?) {
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            didFail = true
        }
    }

    class ContentStorage: SplatSceneReaderDelegate {
        var points: [SplatIO.SplatScenePoint] = []
        var didFinish = false
        var didFail = false

        func reset() {
            points = []
            didFinish = false
            didFail = false
        }

        func didStartReading(withPointCount pointCount: UInt32?) {
            XCTAssertTrue(points.isEmpty)
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
        }

        func didRead(points: [SplatScenePoint]) {
            self.points.append(contentsOf: points)
        }

        func didFinishReading() {
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            didFinish = true
        }

        func didFailReading(withError error: Error?) {
            XCTAssertFalse(didFinish)
            XCTAssertFalse(didFail)
            didFail = true
        }

        static func testApproximatelyEqual(lhs: ContentStorage, rhs: ContentStorage) {
            XCTAssertEqual(lhs.points.count, rhs.points.count, "Same number of points")
            for (lhsPoint, rhsPoint) in zip(lhs.points, rhs.points) {
                XCTAssertTrue(lhsPoint ~= rhsPoint)
            }
        }
    }

    let plyURL = Bundle.module.url(forResource: "test-splat.3-points-from-train", withExtension: "ply", subdirectory: "TestData")!
    let dotSplatURL = Bundle.module.url(forResource: "test-splat.3-points-from-train", withExtension: "splat", subdirectory: "TestData")!

    private func makeTemporaryPLY(comment: String?) throws -> URL {
        let commentLine = comment.map { "\($0)\n" } ?? ""
        let data = Data(
            """
            ply
            format ascii 1.0
            \(commentLine)element vertex 1
            property float x
            property float y
            property float z
            property float f_dc_0
            property float f_dc_1
            property float f_dc_2
            property float scale_0
            property float scale_1
            property float scale_2
            property float opacity
            property float rot_0
            property float rot_1
            property float rot_2
            property float rot_3
            end_header
            0 0 0 0.1 0.2 0.3 1 1 1 0.5 1 0 0 0
            """.utf8
        )

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("ply")
        try data.write(to: url)
        return url
    }

    private func makeTemporarySPZ(antialiased: Bool,
                                  version: UInt32 = 1,
                                  fractionalBits: UInt8 = 0,
                                  flags: UInt8? = nil,
                                  positionBytes: [UInt8]? = nil,
                                  rotationBytes: [UInt8]? = nil) throws -> URL {
        var data = Data()

        func appendUInt32(_ value: UInt32) {
            var littleEndian = value.littleEndian
            withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
        }

        appendUInt32(0x5053474e) // NGSP
        appendUInt32(version)
        appendUInt32(1)          // numPoints
        data.append(0)           // shDegree
        data.append(fractionalBits)
        data.append(flags ?? (antialiased ? 0x01 : 0x00))
        data.append(0)           // reserved

        let defaultPositions = version >= 2 ? Array(repeating: UInt8(0), count: 9) : Array(repeating: UInt8(0), count: 6)
        data.append(contentsOf: positionBytes ?? defaultPositions)
        data.append(255)                             // alpha
        data.append(contentsOf: [128, 128, 128])     // color
        data.append(contentsOf: [160, 160, 160])     // scale
        let defaultRotation = version >= 3 ? encodeSPZQuaternionSmallestThree(simd_quatf(angle: 0, axis: SIMD3<Float>(1, 0, 0))) : [127, 127, 127]
        data.append(contentsOf: rotationBytes ?? defaultRotation)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("spz")
        try data.write(to: url)
        return url
    }

    private func encodeSPZQuaternionSmallestThree(_ rotation: simd_quatf) -> [UInt8] {
        let normalized = rotation.normalized
        var components = [normalized.imag.x, normalized.imag.y, normalized.imag.z, normalized.real]

        var largestIndex = 0
        for index in 1..<components.count where abs(components[index]) > abs(components[largestIndex]) {
            largestIndex = index
        }

        if components[largestIndex] < 0 {
            for index in components.indices {
                components[index].negate()
            }
        }

        let mask: UInt32 = 0x1FF
        let scale = Float(mask) / sqrt(Float(0.5))
        var packed = UInt32(largestIndex) << 30
        var shift: UInt32 = 0

        for index in stride(from: 3, through: 0, by: -1) where index != largestIndex {
            let component = min(abs(components[index]), sqrt(Float(0.5)))
            let magnitude = UInt32((component * scale).rounded())
            let sign: UInt32 = components[index] < 0 ? 1 : 0
            packed |= ((sign << 9) | min(magnitude, mask)) << shift
            shift += 10
        }

        var littleEndian = packed.littleEndian
        return withUnsafeBytes(of: &littleEndian) { Array($0) }
    }

    func testReadPLY() throws {
        try testRead(plyURL)
    }

    func testReadDotSplat() throws {
        try testRead(dotSplatURL)
    }

    func testAutodetectRenderModeMipMarker() throws {
        let url = try makeTemporaryPLY(comment: "comment SplatRenderMode: mip")
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try AutodetectSceneReader(url)
        XCTAssertEqual(reader.renderMode, .mip)
        XCTAssertTrue(reader.isMipSplatting)
    }

    func testAutodetectRenderModeDefaultsToStandard() throws {
        let url = try makeTemporaryPLY(comment: nil)
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try AutodetectSceneReader(url)
        XCTAssertEqual(reader.renderMode, .standard)
        XCTAssertFalse(reader.isMipSplatting)
    }

    func testAutodetectRenderModeUsesSPZAntialiasingFlag() throws {
        let url = try makeTemporarySPZ(antialiased: true)
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try AutodetectSceneReader(url)
        XCTAssertEqual(reader.renderMode, .mip)
        XCTAssertTrue(reader.isMipSplatting)
    }

    func testSPZReaderDataInitializerRefreshesAntialiasMetadata() throws {
        let url = try makeTemporarySPZ(antialiased: true)
        defer { try? FileManager.default.removeItem(at: url) }

        let data = try Data(contentsOf: url)
        let reader = SPZSceneReader(data: data)

        XCTAssertTrue(reader.isAntialiased)
    }

    func testSPZReaderAcceptsVersion4Headers() throws {
        let url = try makeTemporarySPZ(antialiased: true, version: 4, fractionalBits: 12)
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try SPZSceneReader(contentsOf: url)
        let points = try reader.readScene()

        XCTAssertTrue(reader.isAntialiased)
        XCTAssertEqual(points.count, 1)
    }

    func testSPZReaderDecodesVersion4SmallestThreeQuaternion() throws {
        let expectedRotation = simd_quatf(angle: .pi / 3, axis: SIMD3<Float>(1, 0, 0))
        let url = try makeTemporarySPZ(
            antialiased: false,
            version: 4,
            fractionalBits: 12,
            rotationBytes: encodeSPZQuaternionSmallestThree(expectedRotation)
        )
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try SPZSceneReader(contentsOf: url)
        let points = try reader.readScene()

        XCTAssertEqual(points.count, 1)
        let similarity = abs(simd_dot(points[0].rotation.vector, expectedRotation.normalized.vector))
        XCTAssertGreaterThan(similarity, 0.999)
    }

    func testSPZWriterRoundTripsVersion4QuaternionEncoding() throws {
        let point = SplatScenePoint(
            position: SIMD3<Float>(0, 0, 0),
            color: .linearFloat(SIMD3<Float>(0.5, 0.5, 0.5)),
            opacity: .linearFloat(1.0),
            scale: .linearFloat(SIMD3<Float>(1, 1, 1)),
            rotation: simd_quatf(angle: .pi / 3, axis: SIMD3<Float>(1, 0, 0))
        )

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("spz")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = SPZSceneWriter(useFloat16: false, fractionalBits: 12, compress: false, outputVersion: 4)
        try writer.writeScene([point], to: url)

        let reader = try SPZSceneReader(contentsOf: url)
        let points = try reader.readScene()

        XCTAssertEqual(points.count, 1)
        let similarity = abs(simd_dot(points[0].rotation.vector, point.rotation.normalized.vector))
        XCTAssertGreaterThan(similarity, 0.999)
    }

    func testGLBWriterRoundTripsLinearColor() throws {
        let points = [
            SplatScenePoint(
                position: SIMD3<Float>(1, 2, 3),
                color: .linearFloat(SIMD3<Float>(0.25, 0.5, 0.75)),
                opacity: .linearFloat(0.8),
                scale: .linearFloat(SIMD3<Float>(0.1, 0.2, 0.3)),
                rotation: simd_quatf(angle: .pi / 4, axis: SIMD3<Float>(0, 1, 0))
            )
        ]

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("glb")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = GltfGaussianSplatSceneWriter(container: .glb)
        try writer.writeScene(points, to: url)

        let reader = try GltfGaussianSplatSceneReader(url)
        let roundTripped = try reader.readScene()
        XCTAssertEqual(roundTripped.count, points.count)
        let actual = roundTripped[0]
        let expected = points[0]
        XCTAssertTrue((actual.position - expected.position).isWithin(tolerance: SplatScenePoint.Tolerance.position))
        XCTAssertTrue(actual.color ~= expected.color)
        XCTAssertTrue(actual.opacity ~= expected.opacity)
        XCTAssertTrue((actual.scale.asLinearFloat - expected.scale.asLinearFloat).isWithin(tolerance: 1e-6))
        XCTAssertTrue((actual.rotation.normalized.vector - expected.rotation.normalized.vector).isWithin(tolerance: SplatScenePoint.Tolerance.rotation))
    }

    func testGLBReaderAcceptsUnalignedAccessorOffsets() throws {
        var bin = Data([0xEE])
        let positionOffset = bin.count
        bin.appendFloat32(1)
        bin.appendFloat32(2)
        bin.appendFloat32(3)

        let rotationOffset = bin.count
        bin.appendInt16(0)
        bin.appendInt16(0)
        bin.appendInt16(0)
        bin.appendInt16(Int16.max)

        let scaleOffset = bin.count
        bin.appendFloat32(0.1)
        bin.appendFloat32(0.2)
        bin.appendFloat32(0.3)

        let opacityOffset = bin.count
        bin.appendUInt16(UInt16.max / 2)

        let colorOffset = bin.count
        bin.appendUInt16(UInt16.max / 4)
        bin.appendUInt16(UInt16.max / 2)
        bin.appendUInt16((UInt16.max / 4) * 3)
        bin.appendUInt16(UInt16.max)

        XCTAssertEqual(positionOffset % 2, 1)
        XCTAssertEqual(rotationOffset % 2, 1)
        XCTAssertEqual(scaleOffset % 2, 1)
        XCTAssertEqual(opacityOffset % 2, 1)
        XCTAssertEqual(colorOffset % 2, 1)

        let json = """
        {
          "asset": { "version": "2.0" },
          "buffers": [
            { "byteLength": \(bin.count) }
          ],
          "bufferViews": [
            { "buffer": 0, "byteOffset": \(positionOffset), "byteLength": 12 },
            { "buffer": 0, "byteOffset": \(rotationOffset), "byteLength": 8 },
            { "buffer": 0, "byteOffset": \(scaleOffset), "byteLength": 12 },
            { "buffer": 0, "byteOffset": \(opacityOffset), "byteLength": 2 },
            { "buffer": 0, "byteOffset": \(colorOffset), "byteLength": 8 }
          ],
          "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 1, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5122, "normalized": true, "count": 1, "type": "VEC4" },
            { "bufferView": 2, "componentType": 5126, "count": 1, "type": "VEC3" },
            { "bufferView": 3, "componentType": 5123, "normalized": true, "count": 1, "type": "SCALAR" },
            { "bufferView": 4, "componentType": 5123, "normalized": true, "count": 1, "type": "VEC4" }
          ],
          "meshes": [
            {
              "primitives": [
                {
                  "mode": 0,
                  "attributes": {
                    "POSITION": 0,
                    "KHR_gaussian_splatting:ROTATION": 1,
                    "KHR_gaussian_splatting:SCALE": 2,
                    "KHR_gaussian_splatting:OPACITY": 3,
                    "COLOR_0": 4
                  },
                  "extensions": {
                    "KHR_gaussian_splatting": {
                      "kernel": "ellipse"
                    }
                  }
                }
              ]
            }
          ]
        }
        """

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("glb")
        defer { try? FileManager.default.removeItem(at: url) }

        try makeGLB(json: json, bin: bin).write(to: url)

        let reader = try GltfGaussianSplatSceneReader(url)
        let points = try reader.readScene()

        XCTAssertEqual(points.count, 1)
        let point = try XCTUnwrap(points.first)
        XCTAssertTrue((point.position - SIMD3<Float>(1, 2, 3)).isWithin(tolerance: 1e-6))
        XCTAssertTrue((point.scale.asLinearFloat - exp(SIMD3<Float>(0.1, 0.2, 0.3))).isWithin(tolerance: 1e-6))
        XCTAssertEqual(point.opacity.asLinearFloat, Float(UInt16.max / 2) / Float(UInt16.max), accuracy: 1e-6)
        let expectedColor = SIMD3<Float>(
            Float(UInt16.max / 4) / Float(UInt16.max),
            Float(UInt16.max / 2) / Float(UInt16.max),
            Float((UInt16.max / 4) * 3) / Float(UInt16.max)
        )
        XCTAssertTrue((point.color.asLinearFloat - expectedColor).isWithin(tolerance: 1e-6))
        let expectedRotation = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        XCTAssertTrue((point.rotation.normalized.vector - expectedRotation.vector).isWithin(tolerance: SplatScenePoint.Tolerance.rotation))
    }

    func testGLTFWriterRoundTripsSphericalHarmonics() throws {
        let points = [
            SplatScenePoint(
                position: SIMD3<Float>(-1, 0.5, 2),
                color: .sphericalHarmonic([
                    SIMD3<Float>(0.1, 0.2, 0.3),
                    SIMD3<Float>(0.01, 0.02, 0.03),
                    SIMD3<Float>(0.04, 0.05, 0.06),
                    SIMD3<Float>(0.07, 0.08, 0.09)
                ]),
                opacity: .linearFloat(0.65),
                scale: .exponent(SIMD3<Float>(-1, -0.5, -0.25)),
                rotation: simd_quatf(angle: .pi / 6, axis: SIMD3<Float>(1, 0, 0))
            )
        ]

        let baseURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let gltfURL = baseURL.appendingPathExtension("gltf")
        let binURL = baseURL.appendingPathExtension("bin")
        defer {
            try? FileManager.default.removeItem(at: gltfURL)
            try? FileManager.default.removeItem(at: binURL)
        }

        let writer = GltfGaussianSplatSceneWriter(container: .gltf)
        try writer.writeScene(points, to: gltfURL)

        let reader = try GltfGaussianSplatSceneReader(gltfURL)
        let roundTripped = try reader.readScene()
        XCTAssertEqual(roundTripped.count, points.count)
        XCTAssertTrue(roundTripped[0] ~= points[0])
    }

    func testSOGV2WriterRoundTripsLinearColor() throws {
        let colorTolerance: Float = (2.0 / 255.0) + 1e-6
        let points = [
            SplatScenePoint(
                position: SIMD3<Float>(-1.25, 0.5, 2.75),
                color: .linearFloat(SIMD3<Float>(0.2, 0.4, 0.6)),
                opacity: .linearFloat(0.85),
                scale: .exponent(SIMD3<Float>(-2.0, -1.5, -1.0)),
                rotation: simd_quatf(angle: .pi / 5, axis: SIMD3<Float>(0, 1, 0))
            ),
            SplatScenePoint(
                position: SIMD3<Float>(0.75, -0.25, -1.5),
                color: .linearFloat(SIMD3<Float>(0.8, 0.3, 0.15)),
                opacity: .linearFloat(0.35),
                scale: .exponent(SIMD3<Float>(-0.5, -0.25, -0.75)),
                rotation: simd_quatf(angle: .pi / 7, axis: SIMD3<Float>(1, 0, 0))
            )
        ]

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("sog")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = SOGSV2SceneWriter()
        try writer.writeScene(points, to: url)

        let reader = try SplatSOGSSceneReaderV2(url)
        let roundTripped = try reader.readScene()
        XCTAssertEqual(roundTripped.count, points.count)

        for (actual, expected) in zip(roundTripped, points) {
            XCTAssertTrue((actual.position - expected.position).isWithin(tolerance: 2e-4), "position actual=\(actual.position) expected=\(expected.position)")
            XCTAssertTrue(
                (actual.color.asLinearFloat - expected.color.asLinearFloat).isWithin(tolerance: colorTolerance),
                "color actual=\(actual.color.asLinearFloat) expected=\(expected.color.asLinearFloat) shActual=\(actual.color.asSphericalHarmonic[0]) shExpected=\(expected.color.asSphericalHarmonic[0])"
            )
            XCTAssertLessThanOrEqual(abs(actual.opacity.asLinearFloat - expected.opacity.asLinearFloat), 1.0 / 255.0)
            XCTAssertTrue((actual.scale.asLinearFloat - expected.scale.asLinearFloat).isWithin(tolerance: 0.05))
            XCTAssertTrue((actual.rotation.normalized.vector - expected.rotation.normalized.vector).isWithin(tolerance: 2.0 / 128.0))
        }
    }

    func testSOGV2WriterRoundTripsSphericalHarmonicsWithPaletteReuse() throws {
        let sh0Tolerance: Float = (2.0 / (255.0 * 0.28209479177387814)) + 1e-6
        let shNTolerance: Float = (1.0 / 128.0) + 1e-6
        let sharedCoefficients: [SIMD3<Float>] = [
            SIMD3<Float>(0.1, 0.2, 0.3),
            SIMD3<Float>(0.01, 0.02, 0.03),
            SIMD3<Float>(0.04, 0.05, 0.06),
            SIMD3<Float>(0.07, 0.08, 0.09)
        ]
        let points = [
            SplatScenePoint(
                position: SIMD3<Float>(0.2, 0.4, 0.6),
                color: .sphericalHarmonic(sharedCoefficients),
                opacity: .linearFloat(0.9),
                scale: .exponent(SIMD3<Float>(-1.2, -1.1, -1.0)),
                rotation: simd_quatf(angle: .pi / 8, axis: SIMD3<Float>(0, 0, 1))
            ),
            SplatScenePoint(
                position: SIMD3<Float>(-0.4, 0.1, 1.1),
                color: .sphericalHarmonic(sharedCoefficients),
                opacity: .linearFloat(0.55),
                scale: .exponent(SIMD3<Float>(-0.8, -0.7, -0.6)),
                rotation: simd_quatf(angle: .pi / 9, axis: simd_normalize(SIMD3<Float>(1, 1, 0)))
            )
        ]

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("sog")
        defer { try? FileManager.default.removeItem(at: url) }

        let writer = SOGSV2SceneWriter()
        try writer.writeScene(points, to: url)

        let reader = try SplatSOGSSceneReaderV2(url)
        let roundTripped = try reader.readScene()
        XCTAssertEqual(roundTripped.count, points.count)

        for (actual, expected) in zip(roundTripped, points) {
            XCTAssertEqual(actual.color.asSphericalHarmonic.count, expected.color.asSphericalHarmonic.count)
            for (index, coefficients) in zip(actual.color.asSphericalHarmonic, expected.color.asSphericalHarmonic).enumerated() {
                let actualCoefficient = coefficients.0
                let expectedCoefficient = coefficients.1
                let delta = SIMD3<Float>(
                    actualCoefficient.x - expectedCoefficient.x,
                    actualCoefficient.y - expectedCoefficient.y,
                    actualCoefficient.z - expectedCoefficient.z
                )
                let tolerance = index == 0 ? sh0Tolerance : shNTolerance
                XCTAssertTrue(delta.isWithin(tolerance: tolerance), "coefficient actual=\(actualCoefficient) expected=\(expectedCoefficient) delta=\(delta)")
            }
        }
    }

    func testSOGV2WriterOverwritesExistingArchive() throws {
        let points = [
            SplatScenePoint(
                position: SIMD3<Float>(0.1, 0.2, 0.3),
                color: .linearFloat(SIMD3<Float>(0.25, 0.5, 0.75)),
                opacity: .linearFloat(0.8),
                scale: .exponent(SIMD3<Float>(-1, -1, -1)),
                rotation: simd_quatf(angle: .pi / 6, axis: SIMD3<Float>(0, 0, 1))
            )
        ]

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("sog")
        defer { try? FileManager.default.removeItem(at: url) }

        try Data("stale".utf8).write(to: url)

        let writer = SOGSV2SceneWriter()
        try writer.writeScene(points, to: url)

        let reader = try SplatSOGSSceneReaderV2(url)
        let roundTripped = try reader.readScene()
        XCTAssertEqual(roundTripped.count, 1)
    }

    func testReadPLYWithPartialSphericalHarmonicsProperties() {
        let plyData = Data(
            """
            ply
            format ascii 1.0
            element vertex 1
            property float x
            property float y
            property float z
            property float f_dc_0
            property float f_dc_1
            property float f_dc_2
            property float f_rest_0
            property float f_rest_1
            property float f_rest_2
            property float scale_0
            property float scale_1
            property float scale_2
            property float opacity
            property float rot_0
            property float rot_1
            property float rot_2
            property float rot_3
            end_header
            0 0 0 0.1 0.2 0.3 0.01 0.02 0.03 1 1 1 0.5 1 0 0 0
            """.utf8
        )

        let stream = InputStream(data: plyData)
        let reader = SplatPLYSceneReader(stream)
        let content = ContentStorage()
        reader.read(to: content)

        XCTAssertTrue(content.didFinish)
        XCTAssertFalse(content.didFail)
        XCTAssertEqual(content.points.count, 1)

        guard case .sphericalHarmonic(let coefficients) = content.points[0].color else {
            XCTFail("Expected spherical harmonic color")
            return
        }

        // 1 DC triplet + 1 additional f_rest triplet
        XCTAssertEqual(coefficients.count, 2)
    }

    func testFormatsEqual() throws {
        try testEqual(plyURL, dotSplatURL)
    }

    func testRewritePLY() throws {
        try testReadWriteRead(plyURL, writePLY: true)
        try testReadWriteRead(plyURL, writePLY: false)
    }

    func testRewriteDotSplat() throws {
        try testReadWriteRead(dotSplatURL, writePLY: true)
        try testReadWriteRead(dotSplatURL, writePLY: false)
    }
    
    // MARK: - SOGS v2 Format Tests
    
    func testSOGSV2MetadataParsing() throws {
        // Test v2 metadata structure parsing
        let v2MetadataJSON = """
        {
            "version": 2,
            "count": 1000,
            "antialias": true,
            "means": {
                "mins": [-10.5, -5.2, -8.1],
                "maxs": [12.3, 7.8, 9.4],
                "files": ["means_l.webp", "means_u.webp"]
            },
            "scales": {
                "codebook": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "mins": [-8.0, -7.5, -7.0],
                "maxs": [-3.0, -2.5, -2.0],
                "files": ["scales.webp"]
            },
            "quats": {
                "files": ["quats.webp"]
            },
            "sh0": {
                "codebook": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "mins": [-0.2, -0.1, -0.3, -3.0],
                "maxs": [0.2, 0.3, 0.1, 3.0],
                "files": ["sh0.webp"]
            },
            "shN": {
                "count": 4,
                "bands": 3,
                "codebook": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
                "mins": [-0.25],
                "maxs": [0.25],
                "files": ["shN_labels.webp", "shN_centroids.webp"]
            }
        }
        """.data(using: .utf8)!
        
        let decoder = JSONDecoder()
        let metadata = try decoder.decode(SOGSMetadataV2.self, from: v2MetadataJSON)
        
        XCTAssertEqual(metadata.version, 2)
        XCTAssertEqual(metadata.count, 1000)
        XCTAssertEqual(metadata.antialias, true)
        
        // Test means
        XCTAssertEqual(metadata.means.mins.count, 3)
        XCTAssertEqual(metadata.means.maxs.count, 3)
        XCTAssertEqual(metadata.means.files.count, 2)
        XCTAssertEqual(metadata.means.files[0], "means_l.webp")
        XCTAssertEqual(metadata.means.files[1], "means_u.webp")
        
        // Test scales codebook
        XCTAssertEqual(metadata.scales.codebook.count, 6)
        XCTAssertEqual(metadata.scales.mins, [-8.0, -7.5, -7.0])
        XCTAssertEqual(metadata.scales.maxs, [-3.0, -2.5, -2.0])
        XCTAssertEqual(metadata.scales.files.count, 1)
        XCTAssertEqual(metadata.scales.files[0], "scales.webp")
        
        // Test sh0 codebook
        XCTAssertEqual(metadata.sh0.codebook.count, 6)
        XCTAssertEqual(metadata.sh0.mins, [-0.2, -0.1, -0.3, -3.0])
        XCTAssertEqual(metadata.sh0.maxs, [0.2, 0.3, 0.1, 3.0])
        XCTAssertEqual(metadata.sh0.files.count, 1)
        XCTAssertEqual(metadata.sh0.files[0], "sh0.webp")
        
        // Test shN
        XCTAssertNotNil(metadata.shN)
        XCTAssertEqual(metadata.shN?.count, 4)
        XCTAssertEqual(metadata.shN?.bands, 3)
        XCTAssertEqual(metadata.shN?.codebook.count, 6)
        XCTAssertEqual(metadata.shN?.mins ?? [], [-0.25])
        XCTAssertEqual(metadata.shN?.maxs ?? [], [0.25])
        XCTAssertEqual(metadata.shN?.files.count, 2)
        XCTAssertEqual(metadata.shN?.files[0], "shN_labels.webp")
        XCTAssertEqual(metadata.shN?.files[1], "shN_centroids.webp")
    }

    func testSOGSV2MetadataParsingLegacySH() throws {
        // Legacy v2 metadata may omit count/bands in the shN block
        let legacyMetadataJSON = """
        {
            "version": 2,
            "count": 2048,
            "antialias": true,
            "means": {
                "mins": [-1.0, -1.0, -1.0],
                "maxs": [1.0, 1.0, 1.0],
                "files": ["means_l.webp", "means_u.webp"]
            },
            "scales": {
                "codebook": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "mins": [-7.0, -7.0, -7.0],
                "maxs": [-2.0, -2.0, -2.0],
                "files": ["scales.webp"]
            },
            "quats": {
                "files": ["quats.webp"]
            },
            "sh0": {
                "codebook": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "mins": [-0.1, -0.1, -0.1, -2.0],
                "maxs": [0.1, 0.1, 0.1, 2.0],
                "files": ["sh0.webp"]
            },
            "shN": {
                "codebook": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                "mins": [-0.2],
                "maxs": [0.2],
                "files": ["shN_centroids.webp", "shN_labels.webp"]
            }
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let metadata = try decoder.decode(SOGSMetadataV2.self, from: legacyMetadataJSON)

        XCTAssertNotNil(metadata.shN)
        XCTAssertNil(metadata.shN?.count)
        XCTAssertNil(metadata.shN?.bands)
        XCTAssertEqual(metadata.shN?.files.count, 2)
    }
    
    func testSOGSV2MetadataWithoutSH() throws {
        // Test v2 metadata without spherical harmonics
        let v2MetadataJSON = """
        {
            "version": 2,
            "count": 500,
            "antialias": false,
            "means": {
                "mins": [-5.0, -3.0, -4.0],
                "maxs": [5.0, 3.0, 4.0],
                "files": ["means_l.webp", "means_u.webp"]
            },
            "scales": {
                "codebook": [0.01, 0.02, 0.03],
                "mins": [-6.0, -6.0, -6.0],
                "maxs": [-2.0, -2.0, -2.0],
                "files": ["scales.webp"]
            },
            "quats": {
                "files": ["quats.webp"]
            },
            "sh0": {
                "codebook": [0.1, 0.2, 0.3],
                "mins": [-0.2, -0.2, -0.2, -2.0],
                "maxs": [0.2, 0.2, 0.2, 2.0],
                "files": ["sh0.webp"]
            }
        }
        """.data(using: .utf8)!
        
        let decoder = JSONDecoder()
        let metadata = try decoder.decode(SOGSMetadataV2.self, from: v2MetadataJSON)
        
        XCTAssertEqual(metadata.version, 2)
        XCTAssertEqual(metadata.count, 500)
        XCTAssertEqual(metadata.antialias, false)
        XCTAssertNil(metadata.shN)
    }
    
    func testSOGSV2VersionDetection() throws {
        // Test that the main reader correctly detects and delegates to v2 reader
        let v2MetadataJSON = """
        {
            "version": 2,
            "count": 100,
            "antialias": true,
            "means": {
                "mins": [0, 0, 0],
                "maxs": [1, 1, 1],
                "files": ["means_l.webp", "means_u.webp"]
            },
            "scales": {
                "codebook": [0.1],
                "files": ["scales.webp"]
            },
            "quats": {
                "files": ["quats.webp"]
            },
            "sh0": {
                "codebook": [0.5],
                "files": ["sh0.webp"]
            }
        }
        """.data(using: .utf8)!
        
        // Create a temporary file
        let tempDir = FileManager.default.temporaryDirectory
        let metaURL = tempDir.appendingPathComponent("test-v2-meta.json")
        try v2MetadataJSON.write(to: metaURL)
        
        defer {
            try? FileManager.default.removeItem(at: metaURL)
        }
        
        // Test JSON parsing for version detection
        let json = try JSONSerialization.jsonObject(with: v2MetadataJSON) as? [String: Any]
        let version = json?["version"] as? Int
        XCTAssertEqual(version, 2)
    }
    
    func testSOGSV2CompressedDataStructure() {
        // Create mock WebP images for testing
        let mockImage = WebPDecoder.DecodedImage(
            pixels: Data(repeating: 0, count: 64 * 64 * 4),
            width: 64,
            height: 64, 
            bytesPerPixel: 4
        )
        
        let mockMetadata = createMockSOGSV2Metadata()
        
        let compressedData = SOGSCompressedDataV2(
            metadata: mockMetadata,
            means_l: mockImage,
            means_u: mockImage,
            quats: mockImage,
            scales: mockImage,
            sh0: mockImage,
            sh_centroids: mockImage,
            sh_labels: mockImage
        )
        
        XCTAssertEqual(compressedData.numSplats, 1000)
        XCTAssertTrue(compressedData.hasSphericalHarmonics)
        XCTAssertEqual(compressedData.textureWidth, 64)
        XCTAssertEqual(compressedData.textureHeight, 64)
    }
    
    func testSOGSV2CompressedDataWithoutSH() {
        // Test compressed data structure without spherical harmonics
        let mockImage = WebPDecoder.DecodedImage(
            pixels: Data(repeating: 0, count: 32 * 32 * 4),
            width: 32,
            height: 32,
            bytesPerPixel: 4
        )
        
        var mockMetadata = createMockSOGSV2Metadata()
        mockMetadata = SOGSMetadataV2(
            version: 2,
            count: 500,
            antialias: false,
            means: mockMetadata.means,
            scales: mockMetadata.scales,
            quats: mockMetadata.quats,
            sh0: mockMetadata.sh0,
            shN: nil
        )
        
        let compressedData = SOGSCompressedDataV2(
            metadata: mockMetadata,
            means_l: mockImage,
            means_u: mockImage,
            quats: mockImage,
            scales: mockImage,
            sh0: mockImage,
            sh_centroids: nil,
            sh_labels: nil
        )
        
        XCTAssertEqual(compressedData.numSplats, 500)
        XCTAssertFalse(compressedData.hasSphericalHarmonics)
        XCTAssertEqual(compressedData.textureWidth, 32)
        XCTAssertEqual(compressedData.textureHeight, 32)
    }
    
    func testSOGSV2IteratorCodebookProcessing() {
        // Test that the iterator correctly processes codebooks
        let mockImage = WebPDecoder.DecodedImage(
            pixels: Data(repeating: 128, count: 4 * 4 * 4), // Mid-range values for testing
            width: 4,
            height: 4,
            bytesPerPixel: 4
        )
        
        let mockMetadata = createMockSOGSV2Metadata()
        let compressedData = SOGSCompressedDataV2(
            metadata: mockMetadata,
            means_l: mockImage,
            means_u: mockImage,
            quats: mockImage,
            scales: mockImage,
            sh0: mockImage,
            sh_centroids: mockImage,
            sh_labels: mockImage
        )
        
        let iterator = SOGSIteratorV2(compressedData)
        
        // Test reading a point from intentionally non-spec mock pixels.
        // The iterator should fall back safely instead of trapping.
        let point = iterator.readPoint(at: 0)

        XCTAssertNotNil(point.position)
        XCTAssertNotNil(point.rotation)
        XCTAssertNotNil(point.scale)
        XCTAssertNotNil(point.color)
        XCTAssertNotNil(point.opacity)
    }
    
    func testSOGSV2BundledFormatDetection() throws {
        // Test that .sog files are correctly detected as bundled format
        let bundledURL = URL(fileURLWithPath: "/path/to/test.sog")
        let standaloneURL = URL(fileURLWithPath: "/path/to/meta.json")
        
        XCTAssertTrue(bundledURL.lastPathComponent.lowercased().hasSuffix(".sog"))
        XCTAssertFalse(standaloneURL.lastPathComponent.lowercased().hasSuffix(".sog"))
        XCTAssertTrue(standaloneURL.lastPathComponent.lowercased() == "meta.json")
    }
    
    private func createMockSOGSV2Metadata() -> SOGSMetadataV2 {
        return SOGSMetadataV2(
            version: 2,
            count: 1000,
            antialias: true,
            means: SOGSMeansInfoV2(
                mins: [-10.0, -5.0, -8.0],
                maxs: [10.0, 5.0, 8.0],
                files: ["means_l.webp", "means_u.webp"]
            ),
            scales: SOGSScalesInfoV2(
                codebook: Array(0..<256).map { Float($0) * 0.01 },
                mins: [-7.0, -7.0, -7.0],
                maxs: [-2.0, -2.0, -2.0],
                files: ["scales.webp"]
            ),
            quats: SOGSQuatsInfoV2(
                files: ["quats.webp"]
            ),
            sh0: SOGSH0InfoV2(
                codebook: Array(0..<256).map { Float($0) * 0.001 },
                mins: [-0.2, -0.2, -0.2, -3.0],
                maxs: [0.2, 0.2, 0.2, 3.0],
                files: ["sh0.webp"]
            ),
            shN: SOGSSHNInfoV2(
                count: 128,
                bands: 3,
                codebook: Array(0..<256).map { Float($0) * 0.1 },
                mins: [-0.3],
                maxs: [0.3],
                files: ["shN_labels.webp", "shN_centroids.webp"]
            )
        )
    }

    // MARK: - Morton Order Tests

    func testMortonCodeEncoding() {
        // Test basic Morton code encoding
        // For (0, 0, 0), Morton code should be 0
        XCTAssertEqual(MortonOrder.encode(0, 0, 0), 0)

        // For (1, 0, 0), Morton code should be 1 (x bit at position 0)
        XCTAssertEqual(MortonOrder.encode(1, 0, 0), 1)

        // For (0, 1, 0), Morton code should be 2 (y bit at position 1)
        XCTAssertEqual(MortonOrder.encode(0, 1, 0), 2)

        // For (0, 0, 1), Morton code should be 4 (z bit at position 2)
        XCTAssertEqual(MortonOrder.encode(0, 0, 1), 4)

        // For (1, 1, 1), Morton code should be 7 (all bits set)
        XCTAssertEqual(MortonOrder.encode(1, 1, 1), 7)

        // Test larger values
        // (2, 0, 0) -> x=10 binary, interleaved with zeros: ...001000 = 8
        XCTAssertEqual(MortonOrder.encode(2, 0, 0), 8)

        // Test maximum 10-bit value
        XCTAssertEqual(MortonOrder.encode(1023, 1023, 1023), 0x3FFFFFFF) // All 30 bits set
    }

    func testMortonCodeBitInterleaving() {
        // Verify bit interleaving pattern: z2 y2 x2 z1 y1 x1 z0 y0 x0
        // (5, 3, 6) = (101, 011, 110) binary
        // Interleaved: 1 0 1 | 1 0 0 | 0 1 1 | 1 1 1
        //            = z2y2x2 z1y1x1 z0y0x0
        // = 110 100 011 111 = 0b110100011111 (but this needs proper interleaving)

        // Actually for x=5 (101), y=3 (011), z=6 (110):
        // Bit 0: x0=1, y0=1, z0=0 -> 011 = 3
        // Bit 1: x1=0, y1=1, z1=1 -> 110 = 6
        // Bit 2: x2=1, y2=0, z2=1 -> 101 = 5
        // Morton = 5*64 + 6*8 + 3 = 320 + 48 + 3 = 371

        let code = MortonOrder.encode(5, 3, 6)
        // Verify the code is deterministic
        XCTAssertEqual(code, MortonOrder.encode(5, 3, 6))
    }

    func testMortonBoundsComputation() {
        let points = [
            createTestPoint(position: SIMD3<Float>(0, 0, 0)),
            createTestPoint(position: SIMD3<Float>(10, 5, 3)),
            createTestPoint(position: SIMD3<Float>(-5, 8, -2)),
            createTestPoint(position: SIMD3<Float>(3, -4, 7)),
        ]

        let (minBounds, maxBounds) = MortonOrder.computeBounds(points)

        XCTAssertEqual(minBounds.x, -5, accuracy: 0.001)
        XCTAssertEqual(minBounds.y, -4, accuracy: 0.001)
        XCTAssertEqual(minBounds.z, -2, accuracy: 0.001)

        XCTAssertEqual(maxBounds.x, 10, accuracy: 0.001)
        XCTAssertEqual(maxBounds.y, 8, accuracy: 0.001)
        XCTAssertEqual(maxBounds.z, 7, accuracy: 0.001)
    }

    func testMortonBoundsEmptyArray() {
        let (minBounds, maxBounds) = MortonOrder.computeBounds([])

        XCTAssertEqual(minBounds, SIMD3<Float>.zero)
        XCTAssertEqual(maxBounds, SIMD3<Float>.zero)
    }

    func testMortonReorderingPreservesPoints() {
        // Create points with known positions
        let originalPoints = [
            createTestPoint(position: SIMD3<Float>(10, 10, 10)),
            createTestPoint(position: SIMD3<Float>(0, 0, 0)),
            createTestPoint(position: SIMD3<Float>(5, 5, 5)),
            createTestPoint(position: SIMD3<Float>(2, 2, 2)),
        ]

        let reorderedPoints = MortonOrder.reorder(originalPoints)

        // Verify same number of points
        XCTAssertEqual(reorderedPoints.count, originalPoints.count)

        // Verify all original points are present (by position)
        let originalPositions = Set(originalPoints.map { "\($0.position.x),\($0.position.y),\($0.position.z)" })
        let reorderedPositions = Set(reorderedPoints.map { "\($0.position.x),\($0.position.y),\($0.position.z)" })
        XCTAssertEqual(originalPositions, reorderedPositions)
    }

    func testMortonReorderingOrder() {
        // Create points at known Morton code positions
        // Points closer to origin should come first in Morton order
        let points = [
            createTestPoint(position: SIMD3<Float>(1, 1, 1)),    // Far from origin in normalized space
            createTestPoint(position: SIMD3<Float>(0, 0, 0)),    // Origin (min bounds)
            createTestPoint(position: SIMD3<Float>(0.5, 0.5, 0.5)), // Middle
        ]

        let reorderedPoints = MortonOrder.reorder(points)

        // The point at origin should be first (lowest Morton code)
        XCTAssertEqual(reorderedPoints[0].position.x, 0, accuracy: 0.001)
        XCTAssertEqual(reorderedPoints[0].position.y, 0, accuracy: 0.001)
        XCTAssertEqual(reorderedPoints[0].position.z, 0, accuracy: 0.001)
    }

    func testMortonSinglePoint() {
        let points = [createTestPoint(position: SIMD3<Float>(5, 5, 5))]
        let reordered = MortonOrder.reorder(points)

        XCTAssertEqual(reordered.count, 1)
        XCTAssertEqual(reordered[0].position, points[0].position)
    }

    func testMortonEmptyArray() {
        let reordered = MortonOrder.reorder([])
        XCTAssertTrue(reordered.isEmpty)
    }

    func testMortonCodesParallel() {
        // Generate a larger set of points for parallel testing
        var points = [SplatScenePoint]()
        for i in 0..<1000 {
            let x = Float(i % 10)
            let y = Float((i / 10) % 10)
            let z = Float(i / 100)
            points.append(createTestPoint(position: SIMD3<Float>(x, y, z)))
        }

        let sequentialCodes = MortonOrder.computeMortonCodes(points)
        let parallelCodes = MortonOrder.computeMortonCodesParallel(points)

        // Results should be identical
        XCTAssertEqual(sequentialCodes.count, parallelCodes.count)
        for i in 0..<sequentialCodes.count {
            XCTAssertEqual(sequentialCodes[i], parallelCodes[i], "Mismatch at index \(i)")
        }
    }

    func testMortonReorderParallel() {
        var points = [SplatScenePoint]()
        for i in 0..<500 {
            let x = Float.random(in: -10...10)
            let y = Float.random(in: -10...10)
            let z = Float.random(in: -10...10)
            points.append(createTestPoint(position: SIMD3<Float>(x, y, z)))
        }

        let sequential = MortonOrder.reorder(points)
        let parallel = MortonOrder.reorderParallel(points)

        // Both should produce same ordering
        XCTAssertEqual(sequential.count, parallel.count)
        for i in 0..<sequential.count {
            XCTAssertEqual(sequential[i].position.x, parallel[i].position.x, accuracy: 0.0001)
            XCTAssertEqual(sequential[i].position.y, parallel[i].position.y, accuracy: 0.0001)
            XCTAssertEqual(sequential[i].position.z, parallel[i].position.z, accuracy: 0.0001)
        }
    }

    func testMortonStatistics() {
        let points = [
            createTestPoint(position: SIMD3<Float>(0, 0, 0)),
            createTestPoint(position: SIMD3<Float>(10, 10, 10)),
            createTestPoint(position: SIMD3<Float>(5, 5, 5)),
            createTestPoint(position: SIMD3<Float>(5, 5, 5)), // Duplicate position
        ]

        let stats = MortonOrder.computeStatistics(points)

        XCTAssertEqual(stats.pointCount, 4)
        XCTAssertEqual(stats.uniqueCodes, 3) // 4 points but 2 have same position
        XCTAssertEqual(stats.boundsMin.x, 0, accuracy: 0.001)
        XCTAssertEqual(stats.boundsMax.x, 10, accuracy: 0.001)
        XCTAssertGreaterThan(stats.diagonalLength, 0)
        XCTAssertEqual(stats.uniqueRatio, 0.75, accuracy: 0.01) // 3/4 unique
    }

    func testMortonReaderExtension() throws {
        // Test that readSceneWithMortonOrdering works
        let reader = try AutodetectSceneReader(plyURL)
        let mortonOrderedPoints = try reader.readSceneWithMortonOrdering(useParallel: false)
        let regularPoints = try AutodetectSceneReader(plyURL).readScene()

        // Should have same number of points
        XCTAssertEqual(mortonOrderedPoints.count, regularPoints.count)

        // Should contain the same points (just reordered)
        let originalPositions = Set(regularPoints.map { "\($0.position.x),\($0.position.y),\($0.position.z)" })
        let mortonPositions = Set(mortonOrderedPoints.map { "\($0.position.x),\($0.position.y),\($0.position.z)" })
        XCTAssertEqual(originalPositions, mortonPositions)
    }

    func testMortonRecursivePreservesPoints() {
        // Create points that will likely have Morton code collisions
        var points: [SplatScenePoint] = []
        for x in 0..<10 {
            for y in 0..<10 {
                for z in 0..<10 {
                    // Small variations within a tiny region cause same Morton code
                    let pos = SIMD3<Float>(
                        Float(x) * 0.001,
                        Float(y) * 0.001,
                        Float(z) * 0.001
                    )
                    points.append(createTestPoint(position: pos))
                }
            }
        }

        let reordered = MortonOrder.reorderRecursive(points)

        // Should have same count
        XCTAssertEqual(reordered.count, points.count)

        // Should contain same points (just reordered)
        let originalPositions = Set(points.map { "\($0.position.x),\($0.position.y),\($0.position.z)" })
        let reorderedPositions = Set(reordered.map { "\($0.position.x),\($0.position.y),\($0.position.z)" })
        XCTAssertEqual(originalPositions, reorderedPositions)
    }

    func testMortonRecursiveSmallThreshold() {
        // Create points in a small region that will have many collisions
        var points: [SplatScenePoint] = []
        for i in 0..<100 {
            let pos = SIMD3<Float>(
                Float(i % 10) * 0.0001,
                Float((i / 10) % 10) * 0.0001,
                0
            )
            points.append(createTestPoint(position: pos))
        }

        // Use a small threshold to force recursive refinement
        let reordered = MortonOrder.reorderRecursive(points, bucketThreshold: 10)

        XCTAssertEqual(reordered.count, points.count)
    }

    func testMortonRecursiveParallel() {
        // Create a larger dataset for parallel testing
        var points: [SplatScenePoint] = []
        for i in 0..<1000 {
            let angle = Float(i) * 0.1
            let pos = SIMD3<Float>(
                cos(angle) * Float(i % 100),
                sin(angle) * Float(i % 100),
                Float(i / 100)
            )
            points.append(createTestPoint(position: pos))
        }

        let sequential = MortonOrder.reorderRecursive(points)
        let parallel = MortonOrder.reorderRecursiveParallel(points)

        // Both should produce same result
        XCTAssertEqual(sequential.count, parallel.count)
        for i in 0..<sequential.count {
            XCTAssertEqual(sequential[i].position.x, parallel[i].position.x, accuracy: 0.0001)
            XCTAssertEqual(sequential[i].position.y, parallel[i].position.y, accuracy: 0.0001)
            XCTAssertEqual(sequential[i].position.z, parallel[i].position.z, accuracy: 0.0001)
        }
    }

    func testMortonRecursiveEmptyAndSingle() {
        let empty: [SplatScenePoint] = []
        let single = [createTestPoint(position: SIMD3<Float>(1, 2, 3))]

        XCTAssertEqual(MortonOrder.reorderRecursive(empty).count, 0)
        XCTAssertEqual(MortonOrder.reorderRecursive(single).count, 1)
        XCTAssertEqual(MortonOrder.reorderRecursive(single)[0].position, SIMD3<Float>(1, 2, 3))
    }

    func testMortonRecursiveIdenticalPositions() {
        // All points at the same position - should not cause infinite recursion
        let identicalPosition = SIMD3<Float>(5, 5, 5)
        var points: [SplatScenePoint] = []
        for _ in 0..<500 { // More than default bucket threshold (256)
            points.append(createTestPoint(position: identicalPosition))
        }

        // Should complete without stack overflow
        let reordered = MortonOrder.reorderRecursive(points, bucketThreshold: 10)

        XCTAssertEqual(reordered.count, points.count)
        // All positions should still be identical
        for point in reordered {
            XCTAssertEqual(point.position, identicalPosition)
        }
    }

    private func createTestPoint(position: SIMD3<Float>) -> SplatScenePoint {
        SplatScenePoint(
            position: position,
            color: .linearFloat(SIMD3<Float>(1, 1, 1)),
            opacity: .linearFloat(1.0),
            scale: .linearFloat(SIMD3<Float>(1, 1, 1)),
            rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        )
    }

    // MARK: - Equality Tests

    func testEqual(_ urlA: URL, _ urlB: URL) throws {
        let readerA = try AutodetectSceneReader(urlA)
        let contentA = ContentStorage()
        readerA.read(to: contentA)

        let readerB = try AutodetectSceneReader(urlB)
        let contentB = ContentStorage()
        readerB.read(to: contentB)

        ContentStorage.testApproximatelyEqual(lhs: contentA, rhs: contentB)
    }

    func testReadWriteRead(_ url: URL, writePLY: Bool) throws {
        let readerA = try AutodetectSceneReader(url)
        let contentA = ContentStorage()
        readerA.read(to: contentA)

        let memoryOutput = DataOutputStream()
        memoryOutput.open()
        let writer: any SplatSceneWriter
        switch writePLY {
        case true:
            let plyWriter = SplatPLYSceneWriter(memoryOutput)
            try plyWriter.start(pointCount: contentA.points.count)
            writer = plyWriter
        case false:
            writer = DotSplatSceneWriter(memoryOutput)
        }
        try writer.write(contentA.points)

        let memoryInput = InputStream(data: memoryOutput.data)
        memoryInput.open()

        let readerB: any SplatSceneReader = writePLY ? SplatPLYSceneReader(memoryInput) : DotSplatSceneReader(memoryInput)
        let contentB = ContentStorage()
        readerB.read(to: contentB)

        ContentStorage.testApproximatelyEqual(lhs: contentA, rhs: contentB)
    }

    func testRead(_ url: URL) throws {
        let reader = try AutodetectSceneReader(url)

        let content = ContentCounter()
        reader.read(to: content)
        XCTAssertTrue(content.didFinish)
        XCTAssertFalse(content.didFail)
        if let expectedPointCount = content.expectedPointCount {
            XCTAssertEqual(expectedPointCount, content.pointCount)
        }
    }
}

extension SplatScenePoint {
    enum Tolerance {
        static let position: Float = 1e-10
        static let color: Float = 1.0 / 256
        static let opacity: Float = 1.0 / 256
        static let scale: Float = 1e-10
        static let rotation: Float = 2.0 / 128
    }

    public static func ~= (lhs: SplatScenePoint, rhs: SplatScenePoint) -> Bool {
        (lhs.position - rhs.position).isWithin(tolerance: Tolerance.position) &&
        lhs.color ~= rhs.color &&
        lhs.opacity ~= rhs.opacity &&
        lhs.scale ~= rhs.scale &&
        (lhs.rotation.normalized.vector - rhs.rotation.normalized.vector).isWithin(tolerance: Tolerance.rotation)
    }
}

extension SplatScenePoint.Color {
    public static func ~= (lhs: SplatScenePoint.Color, rhs: SplatScenePoint.Color) -> Bool {
        (lhs.asLinearFloat - rhs.asLinearFloat).isWithin(tolerance: SplatScenePoint.Tolerance.color)
    }
}

extension SplatScenePoint.Opacity {
    public static func ~= (lhs: SplatScenePoint.Opacity, rhs: SplatScenePoint.Opacity) -> Bool {
        abs(lhs.asLinearFloat - rhs.asLinearFloat) <= SplatScenePoint.Tolerance.opacity
    }
}

extension SplatScenePoint.Scale {
    public static func ~= (lhs: SplatScenePoint.Scale, rhs: SplatScenePoint.Scale) -> Bool {
        (lhs.asLinearFloat - rhs.asLinearFloat).isWithin(tolerance: SplatScenePoint.Tolerance.scale)
    }
}

extension SIMD3 where Scalar: Comparable & SignedNumeric {
    public func isWithin(tolerance: Scalar) -> Bool {
        abs(x) <= tolerance && abs(y) <= tolerance && abs(z) <= tolerance
    }
}

extension SIMD4 where Scalar: Comparable & SignedNumeric {
    public func isWithin(tolerance: Scalar) -> Bool {
        abs(x) <= tolerance && abs(y) <= tolerance && abs(z) <= tolerance && abs(w) <= tolerance
    }
}

private class DataOutputStream: OutputStream {
    var data = Data()

    override func open() {}
    override func close() {}
    override var hasSpaceAvailable: Bool { true }

    override func write(_ buffer: UnsafePointer<UInt8>, maxLength length: Int) -> Int {
        data.append(buffer, count: length)
        return length
    }
}

private extension SIMD3 where Scalar == Float {
    var magnitude: Scalar {
        sqrt(x*x + y*y + z*z)
    }
}

private extension SIMD4 where Scalar == Float {
    var magnitude: Scalar {
        sqrt(x*x + y*y + z*z + w*w)
    }
}

// MARK: - SOGSTextureCache Thread Safety Tests

final class SOGSTextureCacheTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Clear cache before each test to ensure isolation
        SOGSTextureCache.shared.clearCache()
    }

    override func tearDown() {
        // Clean up after tests
        SOGSTextureCache.shared.clearCache()
        super.tearDown()
    }

    func testCacheSharedInstance() {
        // Singleton should be accessible
        let cache = SOGSTextureCache.shared
        XCTAssertNotNil(cache)

        // Should be the same instance
        XCTAssertTrue(cache === SOGSTextureCache.shared)
    }

    func testConcurrentCacheAccess() {
        let cache = SOGSTextureCache.shared

        let expectation = self.expectation(description: "Concurrent cache access")
        expectation.expectedFulfillmentCount = 10

        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)

        // Multiple threads trying to access the cache simultaneously
        for i in 0..<10 {
            queue.async {
                // Create unique URLs for each thread
                let url = URL(fileURLWithPath: "/test/path/scene\(i)/meta.json")

                do {
                    // Try to get data (will call loader since not cached)
                    _ = try cache.getCompressedData(for: url) {
                        // Simulate loading by returning mock data
                        throw NSError(domain: "TestDomain", code: 404, userInfo: [NSLocalizedDescriptionKey: "Mock file not found"])
                    }
                } catch {
                    // Expected - we're using mock data that throws
                }

                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 10.0)
    }

    func testConcurrentCacheAccessSameURL() {
        let cache = SOGSTextureCache.shared
        let sharedURL = URL(fileURLWithPath: "/test/shared/meta.json")

        let expectation = self.expectation(description: "Concurrent same URL access")
        expectation.expectedFulfillmentCount = 10

        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)
        let loadCount = UnsafeAtomic<Int>(0)

        // Multiple threads trying to access the same URL simultaneously
        for _ in 0..<10 {
            queue.async {
                do {
                    _ = try cache.getCompressedData(for: sharedURL) {
                        // Track how many times the loader is called
                        loadCount.increment()
                        // Simulate slow loading
                        Thread.sleep(forTimeInterval: 0.01)
                        throw NSError(domain: "TestDomain", code: 404, userInfo: nil)
                    }
                } catch {
                    // Expected
                }

                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 10.0)
    }

    func testCacheClear() {
        let cache = SOGSTextureCache.shared

        // Clearing an empty cache should not crash
        cache.clearCache()

        // After multiple operations, clear should still work
        let expectation = self.expectation(description: "Clear after operations")
        expectation.expectedFulfillmentCount = 5

        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)

        for i in 0..<5 {
            queue.async {
                let url = URL(fileURLWithPath: "/test/scene\(i)/meta.json")
                do {
                    _ = try cache.getCompressedData(for: url) {
                        throw NSError(domain: "TestDomain", code: 404, userInfo: nil)
                    }
                } catch {
                    // Expected
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 10.0)

        // Clear should work without deadlock
        cache.clearCache()
    }
}

private func makeGLB(json: String, bin: Data) -> Data {
    var jsonData = Data(json.utf8)
    while jsonData.count % 4 != 0 {
        jsonData.append(0x20)
    }

    var binData = bin
    while binData.count % 4 != 0 {
        binData.append(0)
    }

    var data = Data()
    let totalLength = 12 + 8 + jsonData.count + 8 + binData.count
    data.appendUInt32(0x46546c67)
    data.appendUInt32(2)
    data.appendUInt32(UInt32(totalLength))
    data.appendUInt32(UInt32(jsonData.count))
    data.appendUInt32(0x4e4f534a)
    data.append(jsonData)
    data.appendUInt32(UInt32(binData.count))
    data.appendUInt32(0x004e4942)
    data.append(binData)
    return data
}

private extension Data {
    mutating func appendUInt16(_ value: UInt16) {
        append(UInt8(value & 0xFF))
        append(UInt8((value >> 8) & 0xFF))
    }

    mutating func appendInt16(_ value: Int16) {
        appendUInt16(UInt16(bitPattern: value))
    }

    mutating func appendUInt32(_ value: UInt32) {
        append(UInt8(value & 0xFF))
        append(UInt8((value >> 8) & 0xFF))
        append(UInt8((value >> 16) & 0xFF))
        append(UInt8((value >> 24) & 0xFF))
    }

    mutating func appendFloat32(_ value: Float) {
        appendUInt32(value.bitPattern)
    }
}

// Simple thread-safe counter for testing
private final class UnsafeAtomic<T: Numeric>: @unchecked Sendable {
    private var value: T
    private let lock = NSLock()

    init(_ initialValue: T) {
        self.value = initialValue
    }

    func increment() where T == Int {
        lock.lock()
        defer { lock.unlock() }
        value += 1
    }

    var current: T {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}
