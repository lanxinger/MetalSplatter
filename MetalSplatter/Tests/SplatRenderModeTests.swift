import XCTest
import Metal
import simd
@testable import MetalSplatter

final class SplatRenderModeTests: XCTestCase {
    func testStandardModeOpacityCompensationIsOne() {
        let covariance = SIMD3<Float>(4.0, 0.5, 3.0)
        let scale = SplatProjectionMath.opacityCompensation(for: .standard,
                                                            covariance: covariance,
                                                            blur: 0.3)
        XCTAssertEqual(scale, 1.0, accuracy: 0.0001)
    }

    func testMipModeOpacityCompensationMatchesBrushFormula() {
        let covariance = SIMD3<Float>(4.0, 0.5, 3.0)
        let blur: Float = 0.1
        let expected = sqrt(max(0.0, (4.0 * 3.0) - (0.5 * 0.5)) /
                            (((4.0 + blur) * (3.0 + blur)) - (0.5 * 0.5)))

        let scale = SplatProjectionMath.opacityCompensation(for: .mip,
                                                            covariance: covariance,
                                                            blur: blur)
        XCTAssertEqual(scale, expected, accuracy: 0.0001)
    }

    func testMipModeOpacityCompensationClampsDegenerateCovariance() {
        let covariance = SIMD3<Float>(0.0, 1.0, 0.0)
        let scale = SplatProjectionMath.opacityCompensation(for: .mip,
                                                            covariance: covariance,
                                                            blur: 0.1)
        XCTAssertEqual(scale, 0.0, accuracy: 0.0001)
        XCTAssertFalse(scale.isNaN)
        XCTAssertTrue(scale.isFinite)
    }

    func testRendererMipModeBuildsMipUniforms() {
        let uniforms = SplatRenderer.makeUniforms(for: makeViewport(),
                                                  splatCount: 4,
                                                  indexedSplatCount: 4,
                                                  debugFlags: 0,
                                                  renderMode: .mip,
                                                  covarianceBlur: SplatRenderer.SplatRenderMode.mip.defaultCovarianceBlur,
                                                  lodThresholds: SIMD3<Float>(10, 25, 50))
        XCTAssertEqual(uniforms.renderMode, SplatRenderer.SplatRenderMode.mip.rawValue)
        XCTAssertEqual(uniforms.covarianceBlur, 0.1, accuracy: 0.0001)
    }

    func testFastSHRendererMipModeBuildsMipUniforms() {
        let uniforms = SplatRenderer.makeUniforms(for: makeViewport(),
                                                  splatCount: 4,
                                                  indexedSplatCount: 4,
                                                  debugFlags: 7,
                                                  renderMode: .mip,
                                                  covarianceBlur: 0.12,
                                                  lodThresholds: SIMD3<Float>(10, 25, 50))
        XCTAssertEqual(uniforms.renderMode, SplatRenderer.SplatRenderMode.mip.rawValue)
        XCTAssertEqual(uniforms.covarianceBlur, 0.12, accuracy: 0.0001)
        XCTAssertEqual(uniforms.debugFlags, 7)
    }

    func testMeshShaderCapableRendererUsesSameMipUniformContract() {
        let uniforms = SplatRenderer.makeUniforms(for: makeViewport(),
                                                  splatCount: 8,
                                                  indexedSplatCount: 8,
                                                  debugFlags: 0,
                                                  renderMode: .mip,
                                                  covarianceBlur: 0.1,
                                                  lodThresholds: SIMD3<Float>(10, 25, 50))
        XCTAssertEqual(uniforms.renderMode, SplatRenderer.SplatRenderMode.mip.rawValue)
        XCTAssertEqual(uniforms.covarianceBlur, 0.1, accuracy: 0.0001)
    }

    func testOrthographicViewportBuildsOrthographicUniforms() {
        let uniforms = SplatRenderer.makeUniforms(for: makeViewport(isOrthographic: true),
                                                  splatCount: 4,
                                                  indexedSplatCount: 4,
                                                  debugFlags: 0,
                                                  renderMode: .standard,
                                                  covarianceBlur: 0.3,
                                                  lodThresholds: SIMD3<Float>(10, 25, 50))
        XCTAssertEqual(uniforms.isOrthographic, 1)
    }

    func testDegree4CoefficientsRemainRenderableAndCanBeExplicitlyReduced() {
        let coefficients = (0..<25).map { SIMD3<Float>(repeating: Float($0)) }

        var bands = [SIMD3<Float16>](repeating: SIMD3(repeating: -1), count: 15)
        bands.withUnsafeMutableBufferPointer {
            FastSHSplatRenderer.writeRendererSupportedSHBands(coefficients, count: 15, to: $0.baseAddress!)
        }

        XCTAssertEqual(SphericalHarmonicsEvaluator.degreeFromCoefficientCount(coefficients.count), 4)
        XCTAssertEqual(bands[0].x, 1)
        XCTAssertEqual(bands[14].x, 15)
    }

    private func makeViewport(isOrthographic: Bool = false) -> SplatRenderer.ViewportDescriptor {
        SplatRenderer.ViewportDescriptor(
            viewport: MTLViewport(originX: 0, originY: 0, width: 128, height: 128, znear: 0, zfar: 1),
            projectionMatrix: matrix_identity_float4x4,
            viewMatrix: matrix_identity_float4x4,
            screenSize: SIMD2<Int>(128, 128),
            isOrthographic: isOrthographic
        )
    }
}
