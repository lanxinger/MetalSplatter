import XCTest
import simd
import SplatIO
@testable import MetalSplatter

final class SphericalHarmonicRotationTests: XCTestCase {
    // Independent Cartesian Graphdeco basis, separate from the production
    // associated-Legendre recurrence and its quadrature construction.
    private func basis(_ direction: SIMD3<Double>) -> [Double] {
        let x = direction.x, y = direction.y, z = direction.z
        let xx = x*x, yy = y*y, zz = z*z
        return [
            0.28209479177387814,
            -0.4886025119029199*y, 0.4886025119029199*z, -0.4886025119029199*x,
            1.0925484305920792*x*y, -1.0925484305920792*y*z,
            0.31539156525252005*(2*zz-xx-yy), -1.0925484305920792*x*z,
            0.5462742152960396*(xx-yy),
            -0.5900435899266435*y*(3*xx-yy), 2.890611442640554*x*y*z,
            -0.4570457994644658*y*(4*zz-xx-yy), 0.3731763325901154*z*(2*zz-3*xx-3*yy),
            -0.4570457994644658*x*(4*zz-xx-yy), 1.445305721320277*z*(xx-yy),
            -0.5900435899266435*x*(xx-3*yy),
            2.5033429417967046*x*y*(xx-yy), -1.7701307697799304*y*z*(3*xx-yy),
            0.9461746957575601*x*y*(7*zz-1), -0.6690465435572892*y*z*(7*zz-3),
            0.10578554691520431*(zz*(35*zz-30)+3), -0.6690465435572892*x*z*(7*zz-3),
            0.47308734787878004*(xx-yy)*(7*zz-1), -1.7701307697799304*x*z*(xx-3*yy),
            0.6258357354491761*(xx*(xx-3*yy)-yy*(3*xx-yy)),
        ]
    }

    private func evaluate(_ coefficients: [SIMD3<Float>], _ direction: SIMD3<Double>) -> SIMD3<Double> {
        var result = SIMD3<Double>.zero
        for (value, coefficient) in zip(basis(direction), coefficients) {
            result += value * SIMD3(Double(coefficient.x), Double(coefficient.y), Double(coefficient.z))
        }
        return result
    }

    private func assertClose(_ actual: SIMD3<Double>, _ expected: SIMD3<Double>, accuracy: Double = 2e-6,
                             file: StaticString = #filePath, line: UInt = #line) {
        for channel in 0..<3 { XCTAssertEqual(actual[channel], expected[channel], accuracy: accuracy, file: file, line: line) }
    }

    private func assertSameBits(_ actual: [SIMD3<Float>], _ expected: [SIMD3<Float>],
                                file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (a, b) in zip(actual, expected) {
            for channel in 0..<3 { XCTAssertEqual(a[channel].bitPattern, b[channel].bitPattern, file: file, line: line) }
        }
    }

    private func makePoint(_ coefficients: [SIMD3<Float>]) -> SplatScenePoint {
        SplatScenePoint(position: SIMD3(1, 2, -3), color: .sphericalHarmonic(coefficients),
                        opacity: .logitFloat(2), scale: .exponent(SIMD3(-1, -2, -3)),
                        rotation: simd_quatf(angle: 0, axis: SIMD3(0, 1, 0)))
    }

    func testEveryBandThroughDegreeFourIsRotationallyEquivariant() {
        let rotations = [
            simd_quatf(angle: .pi/2, axis: SIMD3(1, 0, 0)),
            simd_quatf(angle: .pi/2, axis: SIMD3(0, 1, 0)),
            simd_quatf(angle: .pi/2, axis: SIMD3(0, 0, 1)),
            simd_quatf(angle: 0.73, axis: simd_normalize(SIMD3(0.3, -0.4, 0.8))),
        ]
        for rotation in rotations {
            let transform = SphericalHarmonicRotation(rotation: rotation)
            let q = rotation.vector
            let inverse = simd_quatd(vector: SIMD4(Double(q.x), Double(q.y), Double(q.z), Double(q.w))).normalized.inverse
            for degree in 0...4 {
                var source = Array(repeating: SIMD3<Float>.zero, count: (degree+1)*(degree+1))
                for index in degree*degree..<source.count {
                    let value = Float(index+1) * 0.03
                    source[index] = SIMD3(value, -value*0.7, value*0.4)
                }
                let rotated = transform.applying(to: .sphericalHarmonic(source)).asSphericalHarmonic
                for sample in 0..<31 {
                    let z = -1 + 2*(Double(sample)+0.37)/31
                    let phi = Double(sample)*2.399963229728653 + 0.29
                    let radius = sqrt(1-z*z)
                    let direction = SIMD3(radius*cos(phi), radius*sin(phi), z)
                    assertClose(evaluate(rotated, direction), evaluate(source, inverse.act(direction)))
                }
            }
        }
    }

    func testRotationNeverMixesRGBChannelsAndPreservesDCBits() {
        let transform = SphericalHarmonicRotation(rotation: simd_quatf(angle: 1.17, axis: simd_normalize(SIMD3(1, 2, 3))))
        for channel in 0..<3 {
            var coefficients = Array(repeating: SIMD3<Float>.zero, count: 25)
            coefficients[0] = SIMD3(-0.0, 0.125, -0.375)
            for index in 1..<25 { coefficients[index][channel] = Float(index)*0.025 }
            let rotated = transform.applying(to: .sphericalHarmonic(coefficients)).asSphericalHarmonic
            assertSameBits([rotated[0]], [coefficients[0]])
            for index in 1..<25 {
                for other in 0..<3 where other != channel { XCTAssertEqual(rotated[index][other], 0) }
            }
            let before = coefficients.dropFirst().reduce(0.0) { $0 + Double($1[channel]) * Double($1[channel]) }
            let after = rotated.dropFirst().reduce(0.0) { $0 + Double($1[channel]) * Double($1[channel]) }
            XCTAssertEqual(after, before, accuracy: 1e-6)
        }
    }

    func testIdentityAndDCRemainBitExact() {
        let coefficients = (0..<25).map { SIMD3<Float>(Float($0)*0.125, -0.0, -Float($0)*0.25) }
        for real: Float in [1, -1] {
            let identity = SphericalHarmonicRotation(rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: real))
            XCTAssertTrue(identity.isIdentity)
            assertSameBits(identity.applying(to: .sphericalHarmonic(coefficients)).asSphericalHarmonic, coefficients)
        }
        let rotation = SphericalHarmonicRotation(rotation: simd_quatf(angle: 0.8, axis: SIMD3(0, 1, 0)))
        assertSameBits(rotation.applying(to: .sphericalHarmonic([coefficients[0]])).asSphericalHarmonic, [coefficients[0]])
        guard case let .linearUInt8(value) = rotation.applying(to: .linearUInt8(SIMD3(12, 34, 56))) else {
            return XCTFail("Non-SH color representation changed")
        }
        XCTAssertEqual(value, SIMD3(12, 34, 56))
    }

    func testPartialBandIsPaddedBeforeRotation() {
        let source = [SIMD3<Float>(0.1, 0.2, 0.3), SIMD3<Float>(0.4, 0.5, 0.6)]
        let rotation = simd_quatf(angle: .pi/2, axis: SIMD3(0, 0, 1))
        let result = SphericalHarmonicRotation(rotation: rotation).applying(to: .sphericalHarmonic(source)).asSphericalHarmonic
        XCTAssertEqual(result.count, 4)
        let direction = SIMD3<Double>(1, 0, 0)
        assertClose(evaluate(result, direction), evaluate(source, SIMD3(0, -1, 0)))
    }

    func testCommittedRotationAndHistoryRestoreEveryCoefficientExactly() throws {
        let source = (0..<25).map { SIMD3<Float>(Float($0)*0.01, -0.0, -Float($0)*0.02) }
        var store = EditableSplatStore(points: [makePoint(source), makePoint(source)])
        _ = store.applySelection(indices: [0], mode: .replace)
        let changes = try store.applyCommittedTransform(
            SplatEditTransform(rotation: simd_quatf(angle: 0.63, axis: simd_normalize(SIMD3(1, -2, 3)))), pivot: .zero)
        XCTAssertEqual(changes.map(\.index), [0])
        let committed = store.points[0].color.asSphericalHarmonic
        XCTAssertNotEqual(committed, source)
        assertSameBits(store.points[1].color.asSphericalHarmonic, source)
        assertSameBits([committed[0]], [source[0]])
        store.applyPointChanges(changes, useNewValues: false)
        assertSameBits(store.points[0].color.asSphericalHarmonic, source)
        store.applyPointChanges(changes, useNewValues: true)
        assertSameBits(store.points[0].color.asSphericalHarmonic, committed)
    }

    func testUnsupportedHigherBandRejectsWholeEditBeforeMutation() {
        let supported = Array(repeating: SIMD3<Float>(0.1, 0.2, 0.3), count: 16)
        let unsupported = Array(repeating: SIMD3<Float>(0.1, 0.2, 0.3), count: 36)
        var store = EditableSplatStore(points: [makePoint(supported), makePoint(unsupported)])
        _ = store.applySelection(indices: [0, 1], mode: .replace)
        XCTAssertThrowsError(try store.applyCommittedTransform(
            SplatEditTransform(rotation: simd_quatf(angle: 0.5, axis: SIMD3(1, 0, 0))), pivot: .zero))
        XCTAssertEqual(store.points[0].position, SIMD3(1, 2, -3))
        assertSameBits(store.points[0].color.asSphericalHarmonic, supported)
        assertSameBits(store.points[1].color.asSphericalHarmonic, unsupported)
    }
}
