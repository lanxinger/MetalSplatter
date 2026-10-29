import Foundation
import XCTest
@testable import SplatIO
import simd

/// Regressions for files written by other tools (SplatKit/Plinth, gsbox, splat-transform,
/// Brush, Postshot, nianticlabs/spz) and for malformed input.
final class SplatFormatInteropTests: XCTestCase {
    // MARK: - GLB value encoding and render mode

    func testGLBReadsRatifiedLinearScaleAndOpacityWithMipRenderMode() throws {
        var glb = GLBBuilder()
        glb.addSplats([splat(scale: SIMD3(0.0101, 0.02, 0.03), opacity: 0.78)])
        glb.rootExtras = #"{ "SplatRenderMode": "mip" }"#
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try GltfGaussianSplatSceneReader(url)
        XCTAssertTrue(reader.isAntialiased)
        let point = try XCTUnwrap(reader.readScene().first)
        XCTAssertTrue((point.scale.asLinearFloat - SIMD3(0.0101, 0.02, 0.03)).isWithin(tolerance: 1e-6))
        XCTAssertEqual(point.opacity.asLinearFloat, 0.78, accuracy: 1e-6)

        let autodetect = try AutodetectSceneReader(url)
        XCTAssertEqual(autodetect.renderMode, .mip)
    }

    func testGLBReadsGsboxLogScaleAndLogitOpacity() throws {
        var glb = GLBBuilder()
        glb.generator = "gsbox v4.8.4"
        // log(0.01) and logit(0.78): values a ratified file could also hold.
        glb.addSplats([splat(scale: SIMD3(repeating: 0.5), opacity: 0.5)],
                      rawScales: [SIMD3(repeating: log(0.01))], rawOpacities: [log(0.78 / 0.22)])
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        let reader = try GltfGaussianSplatSceneReader(url)
        XCTAssertFalse(reader.isAntialiased)
        let point = try XCTUnwrap(reader.readScene().first)
        XCTAssertTrue((point.scale.asLinearFloat - SIMD3(repeating: 0.01)).isWithin(tolerance: 1e-6))
        XCTAssertEqual(point.opacity.asLinearFloat, 0.78, accuracy: 1e-5)
    }

    func testGLBDetectsUnmarkedLegacyEncodingPerAttribute() throws {
        // Older MetalSplatter and Plinth writers: log-space scale (negative here), linear opacity.
        var glb = GLBBuilder()
        glb.addSplats([splat(), splat()],
                      rawScales: [SIMD3(repeating: -2), SIMD3(repeating: -3)], rawOpacities: [0.25, 0.9])
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        let points = try GltfGaussianSplatSceneReader(url).readScene()
        XCTAssertTrue((points[0].scale.asLinearFloat - SIMD3(repeating: exp(-2))).isWithin(tolerance: 1e-6))
        XCTAssertEqual(points[0].opacity.asLinearFloat, 0.25, accuracy: 1e-6)
        XCTAssertEqual(points[1].opacity.asLinearFloat, 0.9, accuracy: 1e-6)
    }

    func testGLBEncodingMarkerOverridesValues() throws {
        var glb = GLBBuilder()
        glb.assetExtras = #"{ "splatkitEncoding": "gsbox" }"#
        glb.addSplats([splat()], rawScales: [SIMD3(repeating: 0.5)], rawOpacities: [0.5])
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        let point = try XCTUnwrap(GltfGaussianSplatSceneReader(url).readScene().first)
        XCTAssertTrue((point.scale.asLinearFloat - SIMD3(repeating: exp(0.5))).isWithin(tolerance: 1e-5))
        XCTAssertEqual(point.opacity.asLinearFloat, 1 / (1 + exp(-0.5)), accuracy: 1e-6)
    }

    func testGLBQuantizedScaleAndOpacityAreLinearEvenFromGsbox() throws {
        var glb = GLBBuilder()
        glb.generator = "gsbox v4.8.4"
        glb.addSplats([splat()], quantizedScaleAndOpacity: true,
                      rawScales: [SIMD3(0.25, 0.5, 0.75)], rawOpacities: [0.5])
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        let point = try XCTUnwrap(GltfGaussianSplatSceneReader(url).readScene().first)
        XCTAssertTrue((point.scale.asLinearFloat - SIMD3(0.25, 0.5, 0.75)).isWithin(tolerance: 1e-4))
        XCTAssertEqual(point.opacity.asLinearFloat, 0.5, accuracy: 1e-4)
    }

    func testGLBWriterWritesLinearScaleThatReadsBack() throws {
        let points = [splat(scale: SIMD3(0.05, 0.1, 2)), splat(scale: SIMD3(1.5, 3, 0.2))]
        let url = temporaryURL("glb")
        defer { try? FileManager.default.removeItem(at: url) }
        try GltfGaussianSplatSceneWriter(container: .glb).writeScene(points, to: url)

        let read = try GltfGaussianSplatSceneReader(url).readScene()
        for (actual, expected) in zip(read, points) {
            XCTAssertTrue((actual.scale.asLinearFloat - expected.scale.asLinearFloat).isWithin(tolerance: 1e-5))
        }
    }

    // MARK: - GLB nodes

    func testGLBNodeCycleDoesNotRecurseForever() throws {
        var glb = GLBBuilder()
        glb.addSplats([splat(position: SIMD3(1, 2, 3))])
        glb.nodes = #"[ { "mesh": 0, "children": [1] }, { "children": [0, 1] } ]"#
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        let points = try GltfGaussianSplatSceneReader(url).readScene()
        XCTAssertEqual(points.count, 1)
        XCTAssertTrue((points[0].position - SIMD3(1, 2, 3)).isWithin(tolerance: 1e-6))
    }

    func testGLBNonUniformNodeScaleTransformsCovariance() throws {
        let source = splat(position: SIMD3(1, 1, 1), scale: SIMD3(0.1, 0.2, 0.3),
                           rotation: simd_quatf(angle: 0.7, axis: simd_normalize(SIMD3(1, 2, 3))))
        let nodeRotation = simd_quatf(angle: .pi / 3, axis: SIMD3(0, 0, 1))
        var glb = GLBBuilder()
        glb.addSplats([source])
        glb.nodes = """
        [ { "mesh": 0, "translation": [0, 0, 5], "scale": [3, 1, 0.5],
            "rotation": [\(nodeRotation.imag.x), \(nodeRotation.imag.y), \(nodeRotation.imag.z), \(nodeRotation.real)] } ]
        """
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        let point = try XCTUnwrap(GltfGaussianSplatSceneReader(url).readScene().first)
        let linear = simd_float3x3(nodeRotation) * simd_float3x3(diagonal: SIMD3(3, 1, 0.5))
        XCTAssertTrue((point.position - (linear * SIMD3(1, 1, 1) + SIMD3(0, 0, 5))).isWithin(tolerance: 1e-5))
        let expected = linear * covariance(source) * linear.transpose
        XCTAssertTrue(matricesClose(covariance(point), expected, tolerance: 1e-5),
                      "covariance \(covariance(point)) expected \(expected)")
    }

    func testGLBMatrixNodeRotatesSphericalHarmonics() throws {
        let source = splat(sh: randomSH(seed: 1))
        let rotation = simd_quatf(angle: 1.1, axis: simd_normalize(SIMD3(0.3, -1, 0.2)))
        let m = simd_float3x3(rotation) * 2
        var glb = GLBBuilder()
        glb.addSplats([source])
        glb.nodes = """
        [ { "mesh": 0, "matrix": [\(m[0][0]), \(m[0][1]), \(m[0][2]), 0, \(m[1][0]), \(m[1][1]), \(m[1][2]), 0,
                                  \(m[2][0]), \(m[2][1]), \(m[2][2]), 0, 4, 5, 6, 1] } ]
        """
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        let point = try XCTUnwrap(GltfGaussianSplatSceneReader(url).readScene().first)
        XCTAssertTrue((point.scale.asLinearFloat - source.scale.asLinearFloat * 2).isWithin(tolerance: 1e-5))
        // A splat seen from rotation·d after the transform looks as it did from d before it.
        for direction in testDirections {
            let before = evaluateSH(source.color.asSphericalHarmonic, direction)
            let after = evaluateSH(point.color.asSphericalHarmonic, rotation.act(direction))
            XCTAssertTrue((before - after).isWithin(tolerance: 1e-4), "\(before) vs \(after) at \(direction)")
        }
    }

    func testGLBMirroredNodeMirrorsSphericalHarmonics() throws {
        let source = splat(position: SIMD3(1, 2, 3), sh: randomSH(seed: 2))
        var glb = GLBBuilder()
        glb.addSplats([source])
        glb.nodes = #"[ { "mesh": 0, "scale": [-1, 1, 1] } ]"#
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        let point = try XCTUnwrap(GltfGaussianSplatSceneReader(url).readScene().first)
        let mirror = SIMD3<Float>(-1, 1, 1)
        XCTAssertTrue((point.position - SIMD3(-1, 2, 3)).isWithin(tolerance: 1e-6))
        for direction in testDirections {
            let before = evaluateSH(source.color.asSphericalHarmonic, direction)
            let after = evaluateSH(point.color.asSphericalHarmonic, direction * mirror)
            XCTAssertTrue((before - after).isWithin(tolerance: 1e-4), "\(before) vs \(after) at \(direction)")
        }
        let diagonal = simd_float3x3(diagonal: mirror)
        XCTAssertTrue(matricesClose(covariance(point), diagonal * covariance(source) * diagonal, tolerance: 1e-5))
    }

    func testGLBZeroScaleNodeHidesItsMesh() throws {
        var glb = GLBBuilder()
        glb.addSplats([splat()])
        glb.nodes = #"[ { "mesh": 0, "scale": [0, 0, 0] } ]"#
        let url = try glb.write()
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertEqual(try GltfGaussianSplatSceneReader(url).readScene().count, 0)
    }

    func testSymmetricEigendecompositionReconstructsMatrix() {
        let axes = simd_double3x3(rows: [SIMD3(1, 0.3, -0.2), SIMD3(0.1, 2, 0.4), SIMD3(-0.5, 0.2, 0.7)])
        let matrix = axes * axes.transpose
        let (values, vectors) = symmetricEigendecomposition(matrix)
        let rebuilt = vectors * simd_double3x3(diagonal: values) * vectors.transpose
        XCTAssertEqual(vectors.determinant, 1, accuracy: 1e-12)
        for column in 0..<3 {
            XCTAssertTrue(simd_distance(rebuilt[column], matrix[column]) < 1e-12)
        }
    }

    // MARK: - GLB with SPZ-compressed primitives

    func testGLBReadsSPZCompressedPrimitiveWithNodeTransform() throws {
        let points = [splat(position: SIMD3(1, 2, 3)), splat(position: SIMD3(-1, 0.5, 2))]
        let spz = try spzData(points, writer: SPZSceneWriter(antialiased: true))
        let expected = try SPZSceneReader(data: spz).readScene()

        let json = """
        {
          "asset": { "version": "2.0" },
          "extensionsUsed": ["KHR_gaussian_splatting", "KHR_gaussian_splatting_compression_spz_2"],
          "buffers": [ { "byteLength": \(spz.count) } ],
          "bufferViews": [ { "buffer": 0, "byteLength": \(spz.count) } ],
          "meshes": [ { "primitives": [ { "mode": 0, "attributes": {},
            "extensions": { "KHR_gaussian_splatting": { "kernel": "ellipse",
              "extensions": { "KHR_gaussian_splatting_compression_spz_2": { "bufferView": 0 } } } } } ] } ],
          "nodes": [ { "mesh": 0, "translation": [10, 0, 0] } ],
          "scenes": [ { "nodes": [0] } ]
        }
        """
        let url = temporaryURL("glb")
        defer { try? FileManager.default.removeItem(at: url) }
        try glbContainer(json: json, bin: spz).write(to: url)

        let reader = try GltfGaussianSplatSceneReader(url)
        XCTAssertTrue(reader.isAntialiased)
        let read = try reader.readScene()
        XCTAssertEqual(read.count, 2)
        for (actual, base) in zip(read, expected) {
            XCTAssertTrue((actual.position - (base.position + SIMD3(10, 0, 0))).isWithin(tolerance: 1e-5))
        }
    }

    // MARK: - SPZ

    func testSPZVersion3WithExtensionRecordsIsNotReadAsFloat16() throws {
        let source = splat(position: SIMD3(1.5, -2.25, 3), rotation: simd_quatf(angle: 0.5, axis: SIMD3(0, 1, 0)))
        var data = try spzData([source], writer: SPZSceneWriter(compress: false, outputVersion: 3))
        data[14] |= 0x02
        data.append(coordinateSystemRecord(4)) // RUB

        let point = try XCTUnwrap(SPZSceneReader(data: data).readScene().first)
        // RUB → RDF flips y and z.
        XCTAssertTrue((point.position - SIMD3(1.5, 2.25, -3)).isWithin(tolerance: 1e-3), "\(point.position)")
        let flipped = simd_quatf(ix: source.rotation.imag.x, iy: -source.rotation.imag.y, iz: -source.rotation.imag.z, r: source.rotation.real)
        XCTAssertGreaterThan(abs(simd_dot(point.rotation.vector, flipped.vector)), 0.999)
    }

    func testSPZUntaggedDataKeepsStoredAxes() throws {
        let source = splat(position: SIMD3(1.5, -2.25, 3))
        let point = try XCTUnwrap(SPZSceneReader(data: try spzData([source], writer: SPZSceneWriter())).readScene().first)
        XCTAssertTrue((point.position - source.position).isWithin(tolerance: 1e-3))
    }

    func testSPZVersion4ZUpCoordinateSystemConvertsToRDF() throws {
        var data = try spzData([splat(position: SIMD3(1, 2, 3))], writer: SPZSceneWriter())
        let record = coordinateSystemRecord(12) // RFU: x right, y forward, z up
        data.insert(contentsOf: record, at: 32)
        data[14] |= 0x02
        data.replaceSubrange(16..<20, with: withUnsafeBytes(of: UInt32(32 + record.count).littleEndian) { Data($0) })

        let point = try XCTUnwrap(SPZSceneReader(data: data).readScene().first)
        // RDF = (right, down, forward) = (x, −z, y).
        XCTAssertTrue((point.position - SIMD3(1, -3, 2)).isWithin(tolerance: 1e-3), "\(point.position)")
    }

    func testSPZZUpConversionRotatesSphericalHarmonics() {
        // nianticlabs/spz order: flips into RDF's family first, then R_x(−π/2).
        let conversion = SPZCoordinateConversion(from: .rfu)
        let source = randomSH(seed: 3)
        let rfuToRBD = SIMD3<Float>(1, -1, -1)
        var flippedSH = source
        for index in 1..<flippedSH.count {
            flippedSH[index] *= conversion.flips.flipSh[index - 1]
        }
        var point = SplatScenePoint(position: SIMD3(1, 2, 3) * rfuToRBD, color: .sphericalHarmonic(flippedSH),
                                    opacity: .linearFloat(1), scale: .linearFloat(SIMD3(repeating: 0.1)),
                                    rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
        conversion.rotate(&point)
        XCTAssertTrue((point.position - SIMD3(1, -3, 2)).isWithin(tolerance: 1e-5))

        // A direction (x, y, z) in RFU is (x, −z, y) in RDF.
        for direction in testDirections {
            let before = evaluateSH(source, direction)
            let after = evaluateSH(point.color.asSphericalHarmonic, SIMD3(direction.x, -direction.z, direction.y))
            XCTAssertTrue((before - after).isWithin(tolerance: 1e-4), "\(before) vs \(after) at \(direction)")
        }
    }

    func testSPZAcceptsMoreThanSixteenFractionalBits() throws {
        let source = splat(position: SIMD3(1.5, -2.25, 3))
        let data = try spzData([source], writer: SPZSceneWriter(fractionalBits: 20, compress: false, outputVersion: 3))
        let point = try XCTUnwrap(SPZSceneReader(data: data).readScene().first)
        XCTAssertTrue((point.position - source.position).isWithin(tolerance: 1e-5))
    }

    func testSPZVersion4RejectsPointCountTheFileCannotHold() throws {
        var data = try spzData([splat()], writer: SPZSceneWriter())
        data.replaceSubrange(8..<12, with: [0xFF, 0xFF, 0xFF, 0x7F])
        XCTAssertThrowsError(try SPZSceneReader(data: data).readScene())
    }

    func testZstdFrameSizeIsCheckedBeforeAllocating() throws {
        let compressed = try zstdCompressed(Data(repeating: 7, count: 100), compressionLevel: 3)
        XCTAssertThrowsError(try zstdDecompressed(Data(compressed), expectedSize: 1 << 40))
        XCTAssertEqual(try zstdDecompressed(Data(compressed), expectedSize: 100), [UInt8](repeating: 7, count: 100))
    }

    func testSPZGzipWithUnknownCompressionMethodFallsBackToCappedRawDeflate() throws {
        let points = [splat(position: SIMD3(1, 2, 3)), splat(position: SIMD3(4, 5, 6))]
        let raw = try spzData(points, writer: SPZSceneWriter(compress: false, outputVersion: 2))
        let gzip = gzipContainer(rawDeflate: try (raw as NSData).compressed(using: .zlib) as Data, compressionMethod: 9)

        let read = try SPZSceneReader(data: gzip).readScene()
        XCTAssertEqual(read.count, 2)
        XCTAssertTrue((read[1].position - SIMD3(4, 5, 6)).isWithin(tolerance: 1e-2))
    }

    func testSPZGzipCannotInflatePastWhatItsHeaderDescribes() throws {
        // One point claimed, 64 MB of payload: the inflater stops at the header's size.
        var bomb = Data()
        bomb.append(contentsOf: [0x4E, 0x47, 0x53, 0x50, 2, 0, 0, 0, 1, 0, 0, 0, 0, 12, 0, 0])
        bomb.append(Data(count: 64 * 1024 * 1024))
        let deflated = try (bomb as NSData).compressed(using: .zlib) as Data
        let limit = SPZSceneReader.legacyDecompressedSizeLimit(header: bomb.prefix(16), compressedSize: deflated.count + 18)
        XCTAssertEqual(limit, 16 + 19)
        XCTAssertThrowsError(try deflated.inflated(windowBits: Data.rawDeflateWindowBits, maxOutputSize: limit)) { error in
            guard case SplatFileFormatError.decompressionOutputTooLarge = error else {
                return XCTFail("Unexpected error \(error)")
            }
        }
        for method: UInt8 in [8, 9] {
            XCTAssertThrowsError(try SPZSceneReader(data: gzipContainer(rawDeflate: deflated, compressionMethod: method)).readScene())
        }
    }

    // MARK: - Render mode

    func testPLYRenderModeComments() throws {
        let cases: [([String], AutodetectSceneReader.RenderMode)] = [
            (["SplatRenderMode: MIP"], .mip),
            (["antialiased 1"], .mip),
            (["SplatRenderMode: mip", "SplatRenderMode: default"], .standard),
            (["SplatRenderMode: default", "antialiased 1"], .mip),
            (["antialiased 0"], .standard),
        ]
        for (comments, expected) in cases {
            let url = try writePLY(makePLY(comments: comments))
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertEqual(try AutodetectSceneReader(url).renderMode, expected, "\(comments)")
        }
    }

    func testSOGRenderModeFromMetaJSON() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let meta = directory.appendingPathComponent("meta.json")
        try SOGSV2SceneWriter(antialias: true).writeScene([splat(), splat(position: SIMD3(1, 1, 1))], to: meta)
        XCTAssertEqual(try AutodetectSceneReader(meta).renderMode, .mip)

        // splat-transform's `model` field.
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: meta)) as? [String: Any])
        json["antialias"] = nil
        json["model"] = "antialiased"
        try JSONSerialization.data(withJSONObject: json).write(to: meta)
        XCTAssertEqual(try AutodetectSceneReader(meta).renderMode, .mip)

        json["model"] = "3dgs"
        try JSONSerialization.data(withJSONObject: json).write(to: meta)
        XCTAssertEqual(try AutodetectSceneReader(meta).renderMode, .standard)

        let bundle = temporaryURL("sog")
        defer { try? FileManager.default.removeItem(at: bundle) }
        try SOGSV2SceneWriter(antialias: true).writeScene([splat(), splat(position: SIMD3(1, 1, 1))], to: bundle)
        XCTAssertEqual(try AutodetectSceneReader(bundle).renderMode, .mip)
    }

    // MARK: - SOG

    func testSOGNullScaleCodebookEntryMeansZeroScale() throws {
        let json = #"{ "codebook": [null, -1.5, 0.5], "files": ["scales.webp"] }"#
        let scales = try JSONDecoder().decode(SOGSScalesInfoV2.self, from: Data(json.utf8))
        XCTAssertEqual(scales.codebook[0], -Float.greatestFiniteMagnitude)
        XCTAssertEqual(SplatScenePoint.Scale.exponent(SIMD3(repeating: scales.codebook[0])).asLinearFloat, .zero)
        // Other codebooks keep null as 0.
        let sh0 = try JSONDecoder().decode(SOGSH0InfoV2.self, from: Data(json.utf8))
        XCTAssertEqual(sh0.codebook[0], 0)
    }

    func testSOGNonNumericCodebookEntryFailsInsteadOfHanging() {
        let json = #"{ "codebook": [1, "x", 2], "mins": [0, "y"], "files": ["scales.webp"] }"#
        XCTAssertThrowsError(try JSONDecoder().decode(SOGSScalesInfoV2.self, from: Data(json.utf8)))
        XCTAssertThrowsError(try JSONDecoder().decode(SOGSH0InfoV2.self, from: Data(json.utf8)))
    }

    func testSOGInvalidQuaternionTagReadsAsIdentity() {
        let quats = image(width: 1, height: 1, pixels: [200, 40, 90, 17])
        let point = SOGSIteratorV2(sogData(quats: quats)).readPoint(at: 0)
        XCTAssertEqual(point.rotation.vector, simd_quatf(ix: 0, iy: 0, iz: 0, r: 1).vector)
    }

    func testSOGNarrowDegreeOnePaletteIsAddressedLinearly() throws {
        // gsbox writes degree-1 palettes 96 pixels wide: entry n starts at pixel 3n. Entry 70 is
        // row 2 there, but would be row 1 with splat-transform's 64 entries per row.
        let label = 70
        var centroids = [UInt8](repeating: 0, count: 96 * 3 * 4)
        for coefficient in 0..<3 {
            let pixel = label * 3 + coefficient
            centroids.replaceSubrange((pixel * 4)..<(pixel * 4 + 4), with: [UInt8(10 + coefficient), 20, 30, 255])
        }
        let data = sogData(shBands: 1, centroids: image(width: 96, height: 3, pixels: centroids),
                           labels: image(width: 1, height: 1, pixels: [UInt8(label), 0, 0, 255]))
        XCTAssertEqual(SOGSIteratorV2.shNCoefficientsPerEntry(centroidsWidth: 96, bandsHint: 3), 3)
        XCTAssertEqual(SOGSIteratorV2.shNCoefficientsPerEntry(centroidsWidth: 192, bandsHint: 8), 3)

        let coefficients = SOGSIteratorV2(data).readPoint(at: 0).color.asSphericalHarmonic
        XCTAssertEqual(coefficients.count, 4)
        for coefficient in 0..<3 {
            // The test codebook maps byte b to b / 100.
            XCTAssertEqual(coefficients[1 + coefficient], SIMD3(Float(10 + coefficient), 20, 30) / 100, "\(coefficient)")
        }
    }

    // MARK: - PLY

    func testPLYWithLatin1CommentAndTrailingSpaceAfterEndHeader() throws {
        let url = try writePLY(makePLY(comments: ["caf\u{E9}"], latin1: true, endHeader: "end_header \r"))
        defer { try? FileManager.default.removeItem(at: url) }
        let points = try SplatPLYSceneReader(url).readScene()
        XCTAssertEqual(points.count, 1)
        XCTAssertTrue((points[0].position - SIMD3(1, 2, 3)).isWithin(tolerance: 1e-6))
    }

    func testPLY2DGSWithoutThirdScaleIsFlat() throws {
        let url = try writePLY(makePLY(omitScale2: true))
        defer { try? FileManager.default.removeItem(at: url) }
        let point = try XCTUnwrap(SplatPLYSceneReader(url).readScene().first)
        XCTAssertTrue((point.scale.asLinearFloat - SIMD3(exp(-1), exp(-2), 0)).isWithin(tolerance: 1e-6))
    }

    func testPLYRGBPointCloudReadsAsSmallOpaqueSpheres() throws {
        var body = Data()
        for value: Float in [1, 2, 3] { body.appendLittleEndian(value.bitPattern) }
        body.append(contentsOf: [255, 128, 0])
        let header = """
        ply
        format binary_little_endian 1.0
        element vertex 1
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header

        """
        let url = try writePLY(Data(header.utf8) + body)
        defer { try? FileManager.default.removeItem(at: url) }

        let point = try XCTUnwrap(SplatPLYSceneReader(url).readScene().first)
        XCTAssertTrue((point.position - SIMD3(1, 2, 3)).isWithin(tolerance: 1e-6))
        XCTAssertEqual(point.color.asLinearUInt8, SIMD3(255, 128, 0))
        XCTAssertEqual(point.opacity.asLinearFloat, 1)
        XCTAssertTrue((point.scale.asLinearFloat - SIMD3(repeating: 0.01)).isWithin(tolerance: 1e-7))
    }

    func testPLYHeaderCountDoesNotDriveAllocation() throws {
        // A header claiming ~4 billion vertices over one vertex of data fails cleanly.
        var data = makePLY()
        let header = String(decoding: data, as: UTF8.self)
        let range = try XCTUnwrap(header.range(of: "element vertex 1\n"))
        let prefix = header[..<range.lowerBound].utf8.count
        data.replaceSubrange(prefix..<(prefix + "element vertex 1\n".utf8.count), with: Data("element vertex 4294967295\n".utf8))
        let url = try writePLY(data)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try SplatPLYSceneReader(url).readScene())
    }

    // MARK: - Morton order

    func testMortonCodesToleratesNaNPositions() {
        let points = [splat(position: SIMD3(.nan, 0, 1)), splat(position: SIMD3(1, 2, 3)), splat(position: SIMD3(-1, .nan, .infinity))]
        XCTAssertEqual(MortonOrder.computeMortonCodes(points).count, 3)
    }
}

// MARK: - Helpers

private let testDirections: [SIMD3<Float>] = [
    SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1),
    simd_normalize(SIMD3(1, 2, 3)), simd_normalize(SIMD3(-0.4, 0.9, -0.2)), simd_normalize(SIMD3(0.3, -0.3, -1)),
]

private func splat(position: SIMD3<Float> = .zero,
                   scale: SIMD3<Float> = SIMD3(0.1, 0.2, 0.3),
                   opacity: Float = 0.8,
                   rotation: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1),
                   sh: [SIMD3<Float>]? = nil) -> SplatScenePoint {
    SplatScenePoint(position: position,
                    color: .sphericalHarmonic(sh ?? [SIMD3(0.5, -0.25, 1)]),
                    opacity: .linearFloat(opacity),
                    scale: .linearFloat(scale),
                    rotation: rotation)
}

private func randomSH(seed: UInt64) -> [SIMD3<Float>] {
    var state = seed
    func next() -> Float {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Float(state >> 40) / Float(1 << 24) - 0.5
    }
    return (0..<16).map { _ in SIMD3(next(), next(), next()) }
}

/// The renderer's SH basis (spherical_harmonics_evaluate.metal), degrees 0–3.
private func evaluateSH(_ coefficients: [SIMD3<Float>], _ d: SIMD3<Float>) -> SIMD3<Float> {
    let c1: Float = 0.4886025119029199
    let c2: [Float] = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
    let c3: [Float] = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                       -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
    let (x, y, z) = (d.x, d.y, d.z)
    let (xx, yy, zz) = (x * x, y * y, z * z)
    let basis: [Float] = [
        0.28209479177387814,
        -c1 * y, c1 * z, -c1 * x,
        c2[0] * x * y, c2[1] * y * z, c2[2] * (2 * zz - xx - yy), c2[3] * x * z, c2[4] * (xx - yy),
        c3[0] * y * (3 * xx - yy), c3[1] * x * y * z, c3[2] * y * (4 * zz - xx - yy),
        c3[3] * z * (2 * zz - 3 * xx - 3 * yy), c3[4] * x * (4 * zz - xx - yy), c3[5] * z * (xx - yy),
        c3[6] * x * (xx - 3 * yy),
    ]
    return zip(coefficients, basis).reduce(SIMD3<Float>.zero) { $0 + $1.0 * $1.1 }
}

private func covariance(_ point: SplatScenePoint) -> simd_float3x3 {
    let axes = simd_float3x3(point.rotation.normalized) * simd_float3x3(diagonal: point.scale.asLinearFloat)
    return axes * axes.transpose
}

private func matricesClose(_ a: simd_float3x3, _ b: simd_float3x3, tolerance: Float) -> Bool {
    (0..<3).allSatisfy { simd_distance(a[$0], b[$0]) <= tolerance }
}

private func temporaryURL(_ pathExtension: String) -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension(pathExtension)
}

private func spzData(_ points: [SplatScenePoint], writer: SPZSceneWriter) throws -> Data {
    let url = temporaryURL("spz")
    defer { try? FileManager.default.removeItem(at: url) }
    try writer.writeScene(points, to: url)
    return try Data(contentsOf: url)
}

private func coordinateSystemRecord(_ system: UInt32) -> Data {
    var record = Data()
    for value in [UInt32(0xADBE_0003), 4, system] { record.appendLittleEndian(value) }
    return record
}

/// A gzip member around a raw deflate stream. The trailer's CRC is left zero.
private func gzipContainer(rawDeflate: Data, compressionMethod: UInt8) -> Data {
    var data = Data([0x1F, 0x8B, compressionMethod, 0, 0, 0, 0, 0, 0, 3])
    data.append(rawDeflate)
    data.append(Data(count: 8))
    return data
}

private func image(width: Int, height: Int, pixels: [UInt8]) -> WebPDecoder.DecodedImage {
    WebPDecoder.DecodedImage(pixels: Data(pixels), width: width, height: height, bytesPerPixel: 4)
}

/// One splat whose textures are all mid-gray, apart from those passed in.
private func sogData(quats: WebPDecoder.DecodedImage? = nil,
                     shBands: Int? = nil,
                     centroids: WebPDecoder.DecodedImage? = nil,
                     labels: WebPDecoder.DecodedImage? = nil) -> SOGSCompressedDataV2 {
    let gray = image(width: 1, height: 1, pixels: [128, 128, 128, 255])
    let codebook = (0..<256).map { Float($0) / 100 }
    let metadata = SOGSMetadataV2(
        version: 2, count: 1, antialias: nil,
        means: SOGSMeansInfoV2(mins: [0, 0, 0], maxs: [1, 1, 1], files: ["means_l.webp", "means_u.webp"]),
        scales: SOGSScalesInfoV2(codebook: codebook, mins: nil, maxs: nil, files: ["scales.webp"]),
        quats: SOGSQuatsInfoV2(files: ["quats.webp"]),
        sh0: SOGSH0InfoV2(codebook: codebook, mins: nil, maxs: nil, files: ["sh0.webp"]),
        shN: shBands.map { SOGSSHNInfoV2(count: nil, bands: $0, codebook: codebook, mins: nil, maxs: nil,
                                         files: ["shN_labels.webp", "shN_centroids.webp"]) })
    return SOGSCompressedDataV2(metadata: metadata, means_l: gray, means_u: gray, quats: quats ?? gray,
                                scales: gray, sh0: gray, sh_centroids: centroids, sh_labels: labels)
}

private func makePLY(comments: [String] = [], latin1: Bool = false, endHeader: String = "end_header",
                     omitScale2: Bool = false) -> Data {
    var properties = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                      "rot_0", "rot_1", "rot_2", "rot_3"]
    var values: [Float] = [1, 2, 3, 0.1, 0.2, 0.3, 2, -1, -2, -3, 1, 0, 0, 0]
    if omitScale2 {
        let index = properties.firstIndex(of: "scale_2")!
        properties.remove(at: index)
        values.remove(at: index)
    }
    var header = "ply\nformat binary_little_endian 1.0\n"
    header += comments.map { "comment \($0)\n" }.joined()
    header += "element vertex 1\n"
    header += properties.map { "property float \($0)\n" }.joined()
    header += "\(endHeader)\n"
    var data = latin1 ? header.data(using: .isoLatin1)! : Data(header.utf8)
    for value in values { data.appendLittleEndian(value.bitPattern) }
    return data
}

private func writePLY(_ data: Data) throws -> URL {
    let url = temporaryURL("ply")
    try data.write(to: url)
    return url
}

private func glbContainer(json: String, bin: Data) -> Data {
    var jsonData = Data(json.utf8)
    while jsonData.count % 4 != 0 { jsonData.append(0x20) }
    var binData = bin
    while binData.count % 4 != 0 { binData.append(0) }
    var data = Data()
    for value in [UInt32(0x46546C67), 2, UInt32(28 + jsonData.count + binData.count), UInt32(jsonData.count), 0x4E4F534A] {
        data.appendLittleEndian(value)
    }
    data.append(jsonData)
    data.appendLittleEndian(UInt32(binData.count))
    data.appendLittleEndian(UInt32(0x004E4942))
    data.append(binData)
    return data
}

/// A single-primitive KHR_gaussian_splatting GLB with float attributes.
private struct GLBBuilder {
    var generator: String?
    var assetExtras: String?
    var rootExtras: String?
    var nodes = #"[ { "mesh": 0 } ]"#
    private var bin = Data()
    private var bufferViews: [String] = []
    private var accessors: [String] = []
    private var attributes: [String] = []

    mutating func addSplats(_ points: [SplatScenePoint],
                            quantizedScaleAndOpacity: Bool = false,
                            rawScales: [SIMD3<Float>]? = nil,
                            rawOpacities: [Float]? = nil) {
        add("POSITION", points.flatMap { [$0.position.x, $0.position.y, $0.position.z] }, type: "VEC3", count: points.count)
        add("KHR_gaussian_splatting:ROTATION", points.flatMap { [$0.rotation.imag.x, $0.rotation.imag.y, $0.rotation.imag.z, $0.rotation.real] },
            type: "VEC4", count: points.count)
        let scales = rawScales ?? points.map(\.scale.asLinearFloat)
        let opacities = rawOpacities ?? points.map(\.opacity.asLinearFloat)
        add("KHR_gaussian_splatting:SCALE", scales.flatMap { [$0.x, $0.y, $0.z] }, type: "VEC3", count: points.count,
            unorm16: quantizedScaleAndOpacity)
        add("KHR_gaussian_splatting:OPACITY", opacities, type: "SCALAR", count: points.count, unorm16: quantizedScaleAndOpacity)
        let sh = points.map(\.color.asSphericalHarmonic)
        let names = ["SH_DEGREE_0_COEF_0"] + (0..<3).map { "SH_DEGREE_1_COEF_\($0)" }
            + (0..<5).map { "SH_DEGREE_2_COEF_\($0)" } + (0..<7).map { "SH_DEGREE_3_COEF_\($0)" }
        for (index, name) in names.enumerated() where sh.allSatisfy({ $0.count > index }) {
            add("KHR_gaussian_splatting:\(name)", sh.flatMap { [$0[index].x, $0[index].y, $0[index].z] }, type: "VEC3", count: points.count)
        }
    }

    private mutating func add(_ name: String, _ values: [Float], type: String, count: Int, unorm16: Bool = false) {
        let offset = bin.count
        for value in values {
            if unorm16 {
                bin.appendLittleEndian(UInt16((value * 65535).rounded()))
            } else {
                bin.appendLittleEndian(value.bitPattern)
            }
        }
        while bin.count % 4 != 0 { bin.append(0) }
        bufferViews.append(#"{ "buffer": 0, "byteOffset": \#(offset), "byteLength": \#(values.count * (unorm16 ? 2 : 4)) }"#)
        let component = unorm16 ? #""componentType": 5123, "normalized": true"# : #""componentType": 5126"#
        accessors.append(#"{ "bufferView": \#(bufferViews.count - 1), \#(component), "count": \#(count), "type": "\#(type)" }"#)
        attributes.append(#""\#(name)": \#(accessors.count - 1)"#)
    }

    func write() throws -> URL {
        var asset = #""version": "2.0""#
        if let generator { asset += #", "generator": "\#(generator)""# }
        if let assetExtras { asset += #", "extras": \#(assetExtras)"# }
        let json = """
        {
          "asset": { \(asset) },
          \(rootExtras.map { #""extras": \#($0),"# } ?? "")
          "buffers": [ { "byteLength": \(bin.count) } ],
          "bufferViews": [ \(bufferViews.joined(separator: ", ")) ],
          "accessors": [ \(accessors.joined(separator: ", ")) ],
          "meshes": [ { "primitives": [ { "mode": 0, "attributes": { \(attributes.joined(separator: ", ")) },
            "extensions": { "KHR_gaussian_splatting": { "kernel": "ellipse" } } } ] } ],
          "nodes": \(nodes),
          "scenes": [ { "nodes": [0] } ],
          "scene": 0
        }
        """
        let url = temporaryURL("glb")
        try glbContainer(json: json, bin: bin).write(to: url)
        return url
    }
}

private extension Data {
    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
    }
}
