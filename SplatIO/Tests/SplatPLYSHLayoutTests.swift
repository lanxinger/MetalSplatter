import Foundation
import simd
import XCTest
import SplatIO

final class SplatPLYSHLayoutTests: XCTestCase {
    private final class Contents: SplatSceneReaderDelegate {
        var points: [SplatScenePoint] = []
        var finished = false
        var failed = false
        var error: Error?

        func didStartReading(withPointCount pointCount: UInt32?) {}
        func didRead(points: [SplatScenePoint]) { self.points.append(contentsOf: points) }
        func didFinishReading() { finished = true }
        func didFailReading(withError error: Error?) { failed = true; self.error = error }
    }

    private func coefficients(degree: Int, point: Int) -> [SIMD3<Float>] {
        (0..<((degree + 1) * (degree + 1))).map { index in
            let value = Float(point * 100 + index)
            return SIMD3(value + 0.125, value + 10.25, value + 20.5)
        }
    }

    private func propertyNames(degree: Int) -> [String] {
        let restCount = ((degree + 1) * (degree + 1) - 1) * 3
        return ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
            + (0..<restCount).map { "f_rest_\($0)" }
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    }

    private func propertyValues(degree: Int, point: Int) -> [String: Float] {
        var values: [String: Float] = [
            "x": Float(point) + 1.25, "y": -2.5, "z": 3.75,
            "nx": 0, "ny": 0, "nz": 1, "opacity": -1.75,
            "scale_0": -2.25, "scale_1": -0.0, "scale_2": 0.75,
            "rot_0": 0.5, "rot_1": -0.5, "rot_2": 0.5, "rot_3": -0.5,
        ]
        let sh = coefficients(degree: degree, point: point)
        for channel in 0..<3 {
            values["f_dc_\(channel)"] = sh[0][channel]
            // Independent external PLY contract: all R, then all G, then all B.
            for coefficient in 1..<sh.count {
                values["f_rest_\(channel * (sh.count - 1) + coefficient - 1)"] = sh[coefficient][channel]
            }
        }
        return values
    }

    private func fixture(degree: Int, binary: Bool, names: [String]? = nil) -> Data {
        let names = names ?? propertyNames(degree: degree)
        let format = binary ? "binary_little_endian" : "ascii"
        let header = ["ply", "format \(format) 1.0", "element vertex 2"]
            + names.map { "property float \($0)" } + ["end_header"]
        var data = Data((header.joined(separator: "\n") + "\n").utf8)
        for point in 0..<2 {
            let values = propertyValues(degree: degree, point: point)
            if binary {
                for name in names {
                    var bits = (values[name] ?? 0).bitPattern.littleEndian
                    withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
                }
            } else {
                data.append(Data((names.map { String(values[$0] ?? 0) }.joined(separator: " ") + "\n").utf8))
            }
        }
        return data
    }

    private func read(_ data: Data) -> Contents {
        let result = Contents()
        SplatPLYSceneReader(InputStream(data: data)).read(to: result)
        return result
    }

    private func write(_ points: [SplatScenePoint], degree: Int, binary: Bool) throws -> Data {
        let stream = OutputStream.toMemory()
        let writer = SplatPLYSceneWriter(stream)
        try writer.start(sphericalHarmonicDegree: UInt(degree), binary: binary, pointCount: points.count)
        try writer.write(points)
        try writer.close()
        return try XCTUnwrap(stream.property(forKey: .dataWrittenToMemoryStreamKey) as? Data)
    }

    // Decode the writer bytes independently of SplatPLYSceneReader, so matching
    // reader/writer layout bugs cannot cancel out in a roundtrip assertion.
    private func binaryProperties(_ data: Data) throws -> [[String: UInt32]] {
        let boundary = try XCTUnwrap(data.range(of: Data("end_header\n".utf8))).upperBound
        let header = try XCTUnwrap(String(data: data[..<boundary], encoding: .utf8))
        let names = header.split(separator: "\n").filter { $0.hasPrefix("property float ") }
            .map { String($0.split(separator: " ")[2]) }
        let payload = Array(data[boundary...])
        XCTAssertEqual(payload.count, names.count * 4 * 2)
        return (0..<2).map { point in
            var values: [String: UInt32] = [:]
            for (index, name) in names.enumerated() {
                let offset = (point * names.count + index) * 4
                values[name] = UInt32(payload[offset]) | UInt32(payload[offset + 1]) << 8
                    | UInt32(payload[offset + 2]) << 16 | UInt32(payload[offset + 3]) << 24
            }
            return values
        }
    }

    func testExternalChannelMajorSH0Through3InASCIIAndBinary() {
        for degree in 0...3 {
            for binary in [false, true] {
                // Reverse header/payload order to require property-name mapping.
                let names = Array(propertyNames(degree: degree).reversed())
                let result = read(fixture(degree: degree, binary: binary, names: names))
                XCTAssertTrue(result.finished, "degree \(degree), binary \(binary)")
                XCTAssertFalse(result.failed)
                XCTAssertEqual(result.points.count, 2)
                for (index, point) in result.points.enumerated() {
                    XCTAssertEqual(point.color.asSphericalHarmonic, coefficients(degree: degree, point: index))
                    let values = propertyValues(degree: degree, point: index)
                    let actual: [String: Float] = [
                        "x": point.position.x, "y": point.position.y, "z": point.position.z,
                        "opacity": point.opacity.asLogitFloat,
                        "scale_0": point.scale.asExponent.x, "scale_1": point.scale.asExponent.y,
                        "scale_2": point.scale.asExponent.z, "rot_0": point.rotation.real,
                        "rot_1": point.rotation.imag.x, "rot_2": point.rotation.imag.y,
                        "rot_3": point.rotation.imag.z,
                    ]
                    for (name, value) in actual {
                        XCTAssertEqual(value.bitPattern, values[name]?.bitPattern, name)
                    }
                }
            }
        }
    }

    func testWriterEmitsExternalChannelMajorLayoutAndPreservesGeometryBits() throws {
        for degree in 0...3 {
            let external = fixture(degree: degree, binary: true)
            let input = read(external)
            XCTAssertFalse(input.failed)
            let encoded = try write(input.points, degree: degree, binary: true)
            let records = try binaryProperties(encoded)
            for (index, record) in records.enumerated() {
                for (name, expected) in propertyValues(degree: degree, point: index) {
                    XCTAssertEqual(record[name], expected.bitPattern, "degree \(degree), \(name)")
                }
            }
            let roundtrip = read(encoded)
            XCTAssertTrue(roundtrip.finished)
            XCTAssertFalse(roundtrip.failed)
            for (index, point) in roundtrip.points.enumerated() {
                XCTAssertEqual(point.color.asSphericalHarmonic, coefficients(degree: degree, point: index))
            }
        }
    }

    func testASCIIWriterUsesChannelMajorLayout() throws {
        let input = read(fixture(degree: 2, binary: true))
        let output = try write(input.points, degree: 2, binary: false)
        let text = try XCTUnwrap(String(data: output, encoding: .utf8))
        let boundary = try XCTUnwrap(text.range(of: "end_header\n")).upperBound
        let rows = text[boundary...].split(separator: "\n")
        let names = propertyNames(degree: 2)
        XCTAssertEqual(rows.count, 2)
        for (index, row) in rows.enumerated() {
            let values = row.split(separator: " ").compactMap { Float($0) }
            XCTAssertEqual(values.count, names.count)
            let expected = propertyValues(degree: 2, point: index)
            for (name, value) in zip(names, values) {
                XCTAssertEqual(value.bitPattern, expected[name]?.bitPattern, name)
            }
        }
    }

    func testWriterPadsMissingBandsWithinEachChannel() throws {
        let input = read(fixture(degree: 1, binary: true))
        let encoded = try write(input.points, degree: 3, binary: true)
        let records = try binaryProperties(encoded)
        for (point, record) in records.enumerated() {
            let sh = coefficients(degree: 1, point: point)
            for channel in 0..<3 {
                for coefficient in 1..<16 {
                    let expected: Float = coefficient < sh.count ? sh[coefficient][channel] : 0
                    XCTAssertEqual(record["f_rest_\(channel * 15 + coefficient - 1)"], expected.bitPattern)
                }
            }
        }
    }

    func testMalformedRestPropertiesFailInsteadOfSilentlyDroppingData() {
        let names = propertyNames(degree: 1)
        let malformed = [
            names.filter { $0 != "f_rest_8" }, // Incomplete RGB channels.
            names.map { $0 == "f_rest_4" ? "f_rest_9" : $0 }, // Gap with divisible count.
            names.map { $0 == "f_rest_0" ? "f_rest_9" : $0 }, // Missing first index.
            names.map { $0 == "f_rest_8" ? "f_rest_7" : $0 }, // Duplicate property.
        ]
        for propertyNames in malformed {
            let result = read(fixture(degree: 1, binary: true, names: propertyNames))
            XCTAssertTrue(result.failed)
            XCTAssertFalse(result.finished)
            XCTAssertTrue(result.points.isEmpty)
            XCTAssertNotNil(result.error)
        }
    }
}
