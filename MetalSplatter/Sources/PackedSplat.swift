import Foundation
import Metal
import simd
import SplatIO

struct PackedSplat {
    var data: SIMD4<UInt32>
}

struct PackedSplatChunk {
    var minPosition: MTLPackedFloat3
    var maxPosition: MTLPackedFloat3
    var minScale: MTLPackedFloat3
    var maxScale: MTLPackedFloat3
    var minColor: MTLPackedFloat3
    var maxColor: MTLPackedFloat3
}

struct PackedSplatData {
    var splats: [PackedSplat]
    var chunks: [PackedSplatChunk]
}

enum PackedSplatBuilder {
    static let chunkSize = 256

    static func build(from points: [SplatScenePoint]) -> PackedSplatData {
        guard !points.isEmpty else {
            return PackedSplatData(splats: [], chunks: [])
        }

        let chunkCount = (points.count + chunkSize - 1) / chunkSize
        var chunks: [PackedSplatChunk] = []
        chunks.reserveCapacity(chunkCount)

        var packedSplats: [PackedSplat] = []
        packedSplats.reserveCapacity(points.count)

        for chunkIndex in 0..<chunkCount {
            let start = chunkIndex * chunkSize
            let end = min(start + chunkSize, points.count)
            let chunkPoints = points[start..<end]

            var minPos = SIMD3<Float>(repeating: .infinity)
            var maxPos = SIMD3<Float>(repeating: -.infinity)
            var minScale = SIMD3<Float>(repeating: .infinity)
            var maxScale = SIMD3<Float>(repeating: -.infinity)
            var minColor = SIMD3<Float>(repeating: .infinity)
            var maxColor = SIMD3<Float>(repeating: -.infinity)

            for point in chunkPoints {
                let pos = point.position
                minPos = simd.min(minPos, pos)
                maxPos = simd.max(maxPos, pos)

                let scaleExp = point.scale.asExponent
                minScale = simd.min(minScale, scaleExp)
                maxScale = simd.max(maxScale, scaleExp)

                let color = toLinear(point.color.asLinearFloat)
                minColor = simd.min(minColor, color)
                maxColor = simd.max(maxColor, color)
            }

            let chunk = PackedSplatChunk(
                minPosition: MTLPackedFloat3Make(minPos.x, minPos.y, minPos.z),
                maxPosition: MTLPackedFloat3Make(maxPos.x, maxPos.y, maxPos.z),
                minScale: MTLPackedFloat3Make(minScale.x, minScale.y, minScale.z),
                maxScale: MTLPackedFloat3Make(maxScale.x, maxScale.y, maxScale.z),
                minColor: MTLPackedFloat3Make(minColor.x, minColor.y, minColor.z),
                maxColor: MTLPackedFloat3Make(maxColor.x, maxColor.y, maxColor.z)
            )
            chunks.append(chunk)

            let posRange = maxPos - minPos
            let scaleRange = maxScale - minScale
            let colorRange = maxColor - minColor

            let invPosRange = safeReciprocal(posRange)
            let invScaleRange = safeReciprocal(scaleRange)
            let invColorRange = safeReciprocal(colorRange)

            for point in chunkPoints {
                let posNorm = normalized(point.position, min: minPos, invRange: invPosRange)
                let scaleNorm = normalized(point.scale.asExponent, min: minScale, invRange: invScaleRange)
                let colorValue = toLinear(point.color.asLinearFloat)
                let colorNorm = normalized(colorValue, min: minColor, invRange: invColorRange)
                let alpha = point.opacity.asLinearFloat

                let packedPosition = pack111011(posNorm)
                let packedRotation = packRotation(point.rotation)
                let packedScale = pack111011(scaleNorm)
                let packedColor = packColor(SIMD4<Float>(colorNorm.x, colorNorm.y, colorNorm.z, alpha))

                packedSplats.append(PackedSplat(data: SIMD4(packedPosition, packedRotation, packedScale, packedColor)))
            }
        }

        return PackedSplatData(splats: packedSplats, chunks: chunks)
    }

    private static func toLinear(_ color: SIMD3<Float>) -> SIMD3<Float> {
        let gamma: Float = 2.2
        return SIMD3<Float>(x: powf(color.x, gamma),
                            y: powf(color.y, gamma),
                            z: powf(color.z, gamma))
    }

    private static func safeReciprocal(_ value: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(
            value.x > 1e-6 ? 1.0 / value.x : 0.0,
            value.y > 1e-6 ? 1.0 / value.y : 0.0,
            value.z > 1e-6 ? 1.0 / value.z : 0.0
        )
    }

    private static func normalized(_ value: SIMD3<Float>,
                                   min: SIMD3<Float>,
                                   invRange: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3(
            normalizeComponent(value.x, min: min.x, invRange: invRange.x),
            normalizeComponent(value.y, min: min.y, invRange: invRange.y),
            normalizeComponent(value.z, min: min.z, invRange: invRange.z)
        )
    }

    private static func normalizeComponent(_ value: Float, min: Float, invRange: Float) -> Float {
        guard invRange > 0 else { return 0.5 }
        return (value - min) * invRange
    }

    private static func packUnorm(_ value: Float, bits: Int) -> UInt32 {
        let clamped = min(max(value, 0.0), 1.0)
        let maxValue = Float((1 << bits) - 1)
        return UInt32((clamped * maxValue).rounded())
    }

    private static func pack111011(_ value: SIMD3<Float>) -> UInt32 {
        let x = packUnorm(value.x, bits: 11)
        let y = packUnorm(value.y, bits: 10)
        let z = packUnorm(value.z, bits: 11)
        return (x << 21) | (y << 11) | z
    }

    private static func packColor(_ color: SIMD4<Float>) -> UInt32 {
        let r = UInt32((min(max(color.x, 0.0), 1.0) * 255.0).rounded())
        let g = UInt32((min(max(color.y, 0.0), 1.0) * 255.0).rounded())
        let b = UInt32((min(max(color.z, 0.0), 1.0) * 255.0).rounded())
        let a = UInt32((min(max(color.w, 0.0), 1.0) * 255.0).rounded())
        return (r << 24) | (g << 16) | (b << 8) | a
    }

    private static func packRotation(_ rotation: simd_quatf) -> UInt32 {
        var q = rotation.normalized.vector
        let absQ = SIMD4<Float>(abs(q.x), abs(q.y), abs(q.z), abs(q.w))

        var maxIndex = 0
        var maxValue = absQ.x
        if absQ.y > maxValue { maxValue = absQ.y; maxIndex = 1 }
        if absQ.z > maxValue { maxValue = absQ.z; maxIndex = 2 }
        if absQ.w > maxValue { maxValue = absQ.w; maxIndex = 3 }

        if q[maxIndex] < 0 {
            q = -q
        }

        let invNorm: Float = 1.0 / sqrtf(2.0)
        let offset: Float = 0.5
        let a: Float
        let b: Float
        let c: Float
        let encodedIndex: UInt32

        switch maxIndex {
        case 3: // w is largest
            a = q.x
            b = q.y
            c = q.z
            encodedIndex = 0
        case 0: // x is largest
            a = q.w
            b = q.y
            c = q.z
            encodedIndex = 1
        case 1: // y is largest
            a = q.w
            b = q.x
            c = q.z
            encodedIndex = 2
        default: // z is largest
            a = q.w
            b = q.x
            c = q.y
            encodedIndex = 3
        }

        let packedA = packUnorm(a * invNorm + offset, bits: 10)
        let packedB = packUnorm(b * invNorm + offset, bits: 10)
        let packedC = packUnorm(c * invNorm + offset, bits: 10)

        return (encodedIndex << 30) | (packedA << 20) | (packedB << 10) | packedC
    }
}
