import Foundation
import simd

/// Rotates a world-space Gaussian SH function independently in each RGB channel.
/// Construct once per edit, then reuse the small per-band matrices for every point.
public struct SphericalHarmonicRotation: Sendable {
    private let bands: [[Double]]
    public var isIdentity: Bool { bands.isEmpty }

    public init(rotation: simd_quatf) {
        guard rotation.imag != .zero else {
            bands = []
            return
        }
        let q = rotation.vector
        let inverse = simd_quatd(vector: SIMD4(Double(q.x), Double(q.y), Double(q.z), Double(q.w))).normalized.inverse
        var matrices = (1...4).map { Array(repeating: 0.0, count: (2 * $0 + 1) * (2 * $0 + 1)) }
        // Five Gauss–Legendre nodes and ten azimuths integrate products of SH
        // through degree four exactly (polynomial degree eight), up to roundoff.
        let nodes = [-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640]
        let weights = [0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]
        for (z, weight) in zip(nodes, weights) {
            let radius = sqrt(1 - z * z)
            for azimuth in 0..<10 {
                let phi = 2 * Double.pi * Double(azimuth) / 10
                let direction = SIMD3(radius * cos(phi), radius * sin(phi), z)
                let basis = Self.basis(direction)
                let inverseBasis = Self.basis(inverse.act(direction))
                let sampleWeight = weight * 2 * Double.pi / 10
                for degree in 1...4 {
                    let width = 2 * degree + 1
                    let start = degree * degree
                    for row in 0..<width {
                        for column in 0..<width {
                            matrices[degree - 1][row * width + column] += sampleWeight
                                * basis[start + row] * inverseBasis[start + column]
                        }
                    }
                }
            }
        }
        bands = matrices
    }

    public func applying(to color: SplatScenePoint.Color) -> SplatScenePoint.Color {
        guard !isIdentity, case let .sphericalHarmonic(coefficients) = color, coefficients.count > 1 else {
            return color
        }
        // Readers produce at most degree 4, and the editor validates this before mutating any
        // selected point.
        precondition(coefficients.count <= 25)
        var degree = 1
        while (degree + 1) * (degree + 1) < coefficients.count { degree += 1 }
        // Partial imported bands mean zero for missing terms. Rotation can fill
        // those terms, so retain the whole final band instead of truncating it.
        var result = Array(repeating: SIMD3<Float>.zero, count: (degree + 1) * (degree + 1))
        result[0] = coefficients[0] // Preserve DC bits exactly.
        for band in 1...degree {
            let width = 2 * band + 1
            let start = band * band
            for row in 0..<width {
                var value = SIMD3<Double>.zero
                for column in 0..<width where start + column < coefficients.count {
                    let source = coefficients[start + column]
                    value += bands[band - 1][row * width + column]
                        * SIMD3(Double(source.x), Double(source.y), Double(source.z))
                }
                result[start + row] = SIMD3(Float(value.x), Float(value.y), Float(value.z))
            }
        }
        return .sphericalHarmonic(result)
    }

    /// Normalized real SH, ordered by l then m=-l...l, including the
    /// Condon–Shortley phase. This is the standard Graphdeco PLY basis.
    private static func basis(_ direction: SIMD3<Double>) -> [Double] {
        let z = max(-1, min(1, direction.z))
        let phi = atan2(direction.y, direction.x)
        let radius = sqrt(max(0, 1 - z * z))
        var values = Array(repeating: 0.0, count: 25)
        var diagonal = 1.0
        for order in 0...4 {
            if order > 0 { diagonal *= -Double(2 * order - 1) * radius }
            var previous = 0.0
            var current = diagonal
            for degree in order...4 {
                if degree > order {
                    let next = (Double(2 * degree - 1) * z * current
                        - Double(degree + order - 1) * previous) / Double(degree - order)
                    previous = current
                    current = next
                }
                var factorialRatio = 1.0
                if order > 0 {
                    for factor in (degree - order + 1)...(degree + order) { factorialRatio /= Double(factor) }
                }
                let normalized = sqrt(Double(2 * degree + 1) / (4 * .pi) * factorialRatio) * current
                let center = degree * degree + degree
                if order == 0 {
                    values[center] = normalized
                } else {
                    values[center - order] = sqrt(2) * normalized * sin(Double(order) * phi)
                    values[center + order] = sqrt(2) * normalized * cos(Double(order) * phi)
                }
            }
        }
        return values
    }
}
