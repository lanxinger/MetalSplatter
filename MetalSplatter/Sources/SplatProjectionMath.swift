import simd

enum SplatProjectionMath {
    static func determinant(_ covariance: SIMD3<Float>) -> Float {
        covariance.x * covariance.z - covariance.y * covariance.y
    }

    static func applyingBlur(_ covariance: SIMD3<Float>, blur: Float) -> SIMD3<Float> {
        SIMD3(covariance.x + blur, covariance.y, covariance.z + blur)
    }

    static func opacityCompensation(for renderMode: SplatRenderer.SplatRenderMode,
                                    covariance: SIMD3<Float>,
                                    blur: Float) -> Float {
        guard renderMode == .mip else { return 1.0 }

        let detBefore = max(determinant(covariance), 0.0)
        let detAfter = determinant(applyingBlur(covariance, blur: blur))
        guard detAfter > 0.0 else { return 0.0 }
        return sqrt(detBefore / detAfter)
    }
}
