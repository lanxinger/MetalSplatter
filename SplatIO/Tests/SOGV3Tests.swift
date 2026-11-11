import Foundation
import XCTest
import simd
@testable import SplatIO

final class SOGV3Tests: XCTestCase {
    private func fixture(_ degree: Int) throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: "independent-v3-sh\(degree)", withExtension: "sog", subdirectory: "TestData"))
    }
    func testV3RotationsOpacityAndRawRGB() throws {
        let reader = try AutodetectSceneReader(fixture(0))
        let points = try reader.readScene()
        XCTAssertTrue(reader.isMipSplatting)
        XCTAssertEqual(points.count, 4)
        XCTAssertEqual(points[0].opacity.asLinearFloat, 0)
        XCTAssertEqual(points[1].opacity.asLinearFloat, 1)
        XCTAssertEqual(points[2].opacity.asLinearFloat, 1.5625)
        XCTAssertEqual(points[2].opacity.asLogitFloat, .infinity)
        XCTAssertEqual(points[3].opacity.asLinearFloat, pow(64.0 / 204, 2), accuracy: 1e-6)
        XCTAssertEqual(points[0].color.asSphericalHarmonic[0].x, -118.0 / 128)
        XCTAssertEqual(points[0].rotation.real, 1, accuracy: 1e-6)
        XCTAssertEqual(abs(points[1].rotation.imag.z), 1, accuracy: 1e-6)
        XCTAssertEqual(abs(points[3].rotation.imag.x), 1, accuracy: 1e-6)
    }
    func testV3HighLabelByteAndFourthSHBand() throws {
        for degree in [3, 4] {
            let points = try AutodetectSceneReader(fixture(degree)).readScene()
            XCTAssertEqual(points[2].color.asSphericalHarmonic.count, (degree + 1) * (degree + 1))
            XCTAssertEqual(points[0].color.asSphericalHarmonic[1].x, 1.0 / 128)
            XCTAssertEqual(points[2].color.asSphericalHarmonic[1].x, 2.0 / 128)
            XCTAssertTrue(points[3].color.asSphericalHarmonic.dropFirst().allSatisfy { $0 == .zero })
        }
    }
    func testV3CompatibilityExportStillWritesV2() throws {
        let points = try AutodetectSceneReader(fixture(3)).readScene()
        let output = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathExtension("sog")
        defer { try? FileManager.default.removeItem(at: output) }
        try SOGSV2SceneWriter(antialias: true).writeScene(points, to: output)
        let decoded = try AutodetectSceneReader(output).readScene()
        XCTAssertEqual(decoded.count, points.count)
        XCTAssertEqual(decoded[2].opacity.asLinearFloat, 1)
        XCTAssertEqual(decoded[2].color.asSphericalHarmonic.count, 16)
    }
    func testUnsupportedVersionAndMalformedV3AreRejected() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("meta.json")
        for json in [#"{"version":4,"count":1}"#, #"{"version":3,"count":2147483647}"#] {
            try Data(json.utf8).write(to: file)
            XCTAssertThrowsError(try SplatSOGSSceneReaderV2(file).readScene())
        }
    }
}
