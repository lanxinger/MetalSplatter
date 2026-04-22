import Foundation
import Metal
import simd
import SplatIO

public enum SplatAnimationEffect: UInt32, CaseIterable, Sendable {
    case magic
    case spread
    case unroll
    case twister
    case rain
    case spherical
    case explosion
    case flow
    case morph
}

public struct SplatAnimationConfiguration: Equatable, Sendable {
    public var effect: SplatAnimationEffect
    public var time: Float
    public var speed: Float
    public var intensity: Float
    public var minimumScale: Float
    public var radius: Float
    public var height: Float
    public var duration: Float
    public var holdDuration: Float
    public var transitionDuration: Float
    public var randomRadius: Float
    public var explosionStrength: Float
    public var gravity: Float
    public var bounceDamping: Float
    public var floorLevel: Float
    public var waves: Float
    public var origin: SIMD3<Float>?

    public init(
        effect: SplatAnimationEffect,
        time: Float = 0,
        speed: Float = 1,
        intensity: Float = 1,
        minimumScale: Float = 0.002,
        radius: Float = 1,
        height: Float = 2,
        duration: Float = 2,
        holdDuration: Float = 1.5,
        transitionDuration: Float = 2,
        randomRadius: Float = 1.3,
        explosionStrength: Float = 4.5,
        gravity: Float = 9.8,
        bounceDamping: Float = 0.4,
        floorLevel: Float = 0,
        waves: Float = 0.5,
        origin: SIMD3<Float>? = nil
    ) {
        self.effect = effect
        self.time = time
        self.speed = speed
        self.intensity = intensity
        self.minimumScale = minimumScale
        self.radius = radius
        self.height = height
        self.duration = duration
        self.holdDuration = holdDuration
        self.transitionDuration = transitionDuration
        self.randomRadius = randomRadius
        self.explosionStrength = explosionStrength
        self.gravity = gravity
        self.bounceDamping = bounceDamping
        self.floorLevel = floorLevel
        self.waves = waves
        self.origin = origin
    }
}

public struct SplatSceneLayer: Sendable {
    public var name: String?
    public var points: [SplatScenePoint]

    public init(name: String? = nil, points: [SplatScenePoint]) {
        self.name = name
        self.points = points
    }
}

internal struct SplatAnimationSceneMetrics: Sendable {
    var sceneIndex: Int
    var range: Range<Int>
    var center: SIMD3<Float>
    var centerOfMass: SIMD3<Float>
    var boundsMin: SIMD3<Float>
    var boundsMax: SIMD3<Float>
    var radialExtent: Float
}

internal struct SplatAnimationSample {
    var point: SplatScenePoint
    var tint: SIMD3<Float> = SIMD3<Float>(repeating: 1)
}

internal enum SplatAnimationEngine {
    static func apply(
        point: SplatScenePoint,
        globalIndex: Int,
        sceneIndex: Int,
        sceneMetrics: [SplatAnimationSceneMetrics],
        configuration: SplatAnimationConfiguration
    ) -> SplatAnimationSample {
        guard !sceneMetrics.isEmpty else {
            return SplatAnimationSample(point: point)
        }

        let metrics = sceneMetrics[min(max(sceneIndex, 0), sceneMetrics.count - 1)]
        let sceneCount = max(sceneMetrics.count, 1)
        let time = configuration.time * configuration.speed
        let intensity = max(configuration.intensity, 0)
        let origin = configuration.origin ?? metrics.center

        switch configuration.effect {
        case .magic:
            return magic(point: point, time: time, intensity: intensity, minimumScale: configuration.minimumScale, origin: origin)
        case .spread:
            return spread(point: point, time: time, intensity: intensity, minimumScale: configuration.minimumScale, origin: origin)
        case .unroll:
            return unroll(point: point, time: time, intensity: intensity, minimumScale: configuration.minimumScale, origin: origin)
        case .twister:
            return twister(point: point, time: time, intensity: intensity, minimumScale: configuration.minimumScale, origin: origin)
        case .rain:
            return rain(point: point, time: time, intensity: intensity, minimumScale: configuration.minimumScale, origin: origin)
        case .spherical:
            return spherical(point: point,
                             time: time,
                             sceneIndex: sceneIndex,
                             sceneCount: sceneCount,
                             intensity: intensity,
                             radius: configuration.radius,
                             height: configuration.height,
                             minimumScale: configuration.minimumScale,
                             origin: origin)
        case .explosion:
            return explosion(point: point,
                             globalIndex: globalIndex,
                             time: time,
                             sceneIndex: sceneIndex,
                             sceneCount: sceneCount,
                             holdDuration: configuration.holdDuration,
                             intensity: intensity,
                             duration: configuration.duration,
                             explosionStrength: configuration.explosionStrength,
                             gravity: configuration.gravity,
                             bounceDamping: configuration.bounceDamping,
                             floorLevel: configuration.floorLevel,
                             minimumScale: configuration.minimumScale,
                             origin: configuration.origin ?? SIMD3<Float>(repeating: 0))
        case .flow:
            return flow(point: point,
                        globalIndex: globalIndex,
                        time: time,
                        sceneIndex: sceneIndex,
                        sceneMetrics: sceneMetrics,
                        holdDuration: configuration.holdDuration,
                        intensity: intensity,
                        minimumScale: configuration.minimumScale,
                        waves: configuration.waves)
        case .morph:
            return morph(point: point,
                         globalIndex: globalIndex,
                         time: time,
                         sceneIndex: sceneIndex,
                         sceneCount: sceneCount,
                         holdDuration: configuration.holdDuration,
                         transitionDuration: configuration.transitionDuration,
                         intensity: intensity,
                         minimumScale: configuration.minimumScale,
                         randomRadius: configuration.randomRadius,
                         origin: origin)
        }
    }

    private static func magic(
        point: SplatScenePoint,
        time: Float,
        intensity: Float,
        minimumScale: Float,
        origin: SIMD3<Float>
    ) -> SplatAnimationSample {
        var sample = SplatAnimationSample(point: point)
        var local = point.position - origin
        let radial = simd_length(SIMD2<Float>(local.x, local.z))
        let revealRadius = smoothStep(0, 10, time - 4.5) * 10
        let border = abs(revealRadius - radial - 0.5)
        local *= 1 - 0.2 * exp(-20 * border) * intensity

        let reveal = smoothStep(revealRadius - 0.5, revealRadius, radial + 0.5)
        local += 0.1 * noise3(local * 2 + SIMD3<Float>(repeating: time * 0.5)) * reveal * intensity

        let angle = atan2(local.x, local.z) / .pi
        let visible = step(angle, time - .pi)
        let glow = exp(-20 * border) + exp(-50 * abs(time - angle - .pi)) * 0.5

        sample.point.position = origin + local
        sample.point.scale = .linearFloat(scaleMix(minimumScale: minimumScale, original: point.scale.asLinearFloat, factor: reveal))
        sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * visible)
        sample.tint = SIMD3<Float>(repeating: 1 + glow * intensity)
        return sample
    }

    private static func spread(
        point: SplatScenePoint,
        time: Float,
        intensity: Float,
        minimumScale: Float,
        origin: SIMD3<Float>
    ) -> SplatAnimationSample {
        var sample = SplatAnimationSample(point: point)
        var local = point.position - origin
        let radial = simd_length(SIMD2<Float>(local.x, local.z))
        let tt = time * time * 0.4 + 0.5
        local.x *= min(1, 0.3 + max(0, tt * 0.05))
        local.z *= min(1, 0.3 + max(0, tt * 0.05))

        let largeReveal = clamp(tt - 7 - radial * 2.5, lower: 0, upper: 1)
        let smallReveal = clamp(tt - 1 - radial * 2, lower: 0, upper: 1)
        let scaleA = point.scale.asLinearFloat * largeReveal
        let scaleB = point.scale.asLinearFloat * 0.2 * smallReveal

        sample.point.position = origin + local
        sample.point.scale = .linearFloat(simd_max(scaleA, simd_max(scaleB, SIMD3<Float>(repeating: minimumScale))))
        sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * max(largeReveal, smallReveal))
        let colorReveal = clamp(tt - radial * 2.5 - 3, lower: 0, upper: 1)
        sample.tint = simd_mix(SIMD3<Float>(repeating: 0.3), SIMD3<Float>(repeating: 1), SIMD3<Float>(repeating: colorReveal * intensity))
        return sample
    }

    private static func unroll(
        point: SplatScenePoint,
        time: Float,
        intensity: Float,
        minimumScale: Float,
        origin: SIMD3<Float>
    ) -> SplatAnimationSample {
        var sample = SplatAnimationSample(point: point)
        var local = point.position - origin
        let angle = (local.y * 50 - 20) * exp(-time) * intensity
        local = SIMD3<Float>(rotateXZ(SIMD2<Float>(local.x, local.z), angle), local.y)
        local *= (1 - exp(-time) * 2)

        let reveal = smoothStep(0.3, 0.7, time + (point.position.y - origin.y) - 2)
        sample.point.position = origin + local
        sample.point.scale = .linearFloat(scaleMix(minimumScale: minimumScale, original: point.scale.asLinearFloat, factor: reveal))
        sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * step(0, time * 0.5 + (point.position.y - origin.y) - 0.5))
        return sample
    }

    private static func twister(
        point: SplatScenePoint,
        time: Float,
        intensity: Float,
        minimumScale: Float,
        origin: SIMD3<Float>
    ) -> SplatAnimationSample {
        var sample = SplatAnimationSample(point: point)
        var local = point.position - origin
        let h = hash3(point.position)
        let radial = simd_length(SIMD2<Float>(local.x, local.z))
        let s = smoothStep(0, 8, time * time * 0.1 - radial * 2 + 2)
        let scaleLength = simd_length(point.scale.asLinearFloat)

        if scaleLength < 0.05 {
            local.y = simd_mix(-10, local.y, pow(s, 2 * h.x))
        }

        let radialFactor = pow(s, 2 * h.x)
        let xz = simd_mix(SIMD2<Float>(local.x, local.z) * 0.5, SIMD2<Float>(local.x, local.z), SIMD2<Float>(repeating: radialFactor))
        let rotationTime = time * (1 - s) * 0.2 * max(intensity, 0.1)
        let swirl = rotationTime + local.y * 20 * (1 - s) * exp(-simd_length(xz))
        let spun = rotateXZ(xz, swirl)

        sample.point.position = origin + SIMD3<Float>(spun.x, local.y, spun.y)
        sample.point.scale = .linearFloat(scaleMix(minimumScale: minimumScale, original: point.scale.asLinearFloat, factor: pow(s, 12)))
        sample.point.rotation = simd_quatf(angle: -time * 0.3 * (1 - s), axis: SIMD3<Float>(0, 1, 0)) * point.rotation
        sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * pow(s, 4))
        return sample
    }

    private static func rain(
        point: SplatScenePoint,
        time: Float,
        intensity: Float,
        minimumScale: Float,
        origin: SIMD3<Float>
    ) -> SplatAnimationSample {
        var sample = SplatAnimationSample(point: point)
        let h = hash3(point.position)
        var local = point.position - origin
        let originalY = local.y
        let radial = simd_length(SIMD2<Float>(local.x, local.z))
        let exponent = 0.5 + h.x
        let s = pow(max(smoothStep(0, 5, time * time * 0.1 - radial * 2 + 1), 0), exponent)

        local.y = min(-10 + s * 15, local.y)
        let scaledXZ = simd_mix(SIMD2<Float>(local.x, local.z) * 0.3, SIMD2<Float>(local.x, local.z), SIMD2<Float>(repeating: s))
        let spun = rotateXZ(scaledXZ, time * 0.3 * max(intensity, 0.1))

        sample.point.position = origin + SIMD3<Float>(spun.x, local.y, spun.y)
        sample.point.scale = .linearFloat(scaleMix(minimumScale: max(minimumScale, 0.005), original: point.scale.asLinearFloat, factor: pow(s, 30)))
        sample.point.rotation = simd_quatf(angle: -time * 0.3, axis: SIMD3<Float>(0, 1, 0)) * point.rotation
        sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * smoothStep(-10, originalY, local.y))
        return sample
    }

    private static func spherical(
        point: SplatScenePoint,
        time: Float,
        sceneIndex: Int,
        sceneCount: Int,
        intensity: Float,
        radius: Float,
        height: Float,
        minimumScale: Float,
        origin: SIMD3<Float>
    ) -> SplatAnimationSample {
        let norm = transitionNormTime(time: time, sceneIndex: sceneIndex, sceneCount: sceneCount, stay: 0, transition: 1)
        guard norm.active else {
            var hidden = SplatAnimationSample(point: point)
            hidden.point.opacity = .linearFloat(0)
            return hidden
        }

        var sample = SplatAnimationSample(point: point)
        let local = point.position - origin
        let t = norm.local
        let targetCenter = SIMD3<Float>(0, (0.5 + 0.5 * pow(abs(1 - 2 * t), 0.2)) * height, 0)
        let dir = simd_normalize(local - targetCenter)
        let targetPoint = targetCenter + dir * radius

        var transformed = local
        if t >= 0.25 && t < 0.45 {
            transformed = mix(local, targetPoint, pow((t - 0.25) * 5, 4))
        } else if t >= 0.45 && t < 0.55 {
            let transitionT = (t - 0.45) * 10
            let churnAngle = transitionT * 2 * Float.pi
            let rotVec = SIMD3<Float>(sin(churnAngle), 0, cos(churnAngle))
            transformed = targetPoint + simd_cross(dir, rotVec) * 0.1 * sin(transitionT * Float.pi) * intensity
        } else if t >= 0.55 && t < 0.75 {
            transformed = mix(targetPoint, local, pow((t - 0.55) * 5, 4))
        }

        let scaleFactor: Float
        switch t {
        case ..<0.25, 0.75...:
            scaleFactor = 1
        case 0.25..<0.45:
            scaleFactor = mix(1, minimumScaleScale(point.scale.asLinearFloat, minimumScale), pow((t - 0.25) * 5, 2))
        case 0.45..<0.55:
            scaleFactor = minimumScaleScale(point.scale.asLinearFloat, minimumScale)
        default:
            scaleFactor = mix(minimumScaleScale(point.scale.asLinearFloat, minimumScale), 1, pow((t - 0.55) * 5, 2))
        }

        sample.point.position = origin + transformed
        sample.point.scale = .linearFloat(simd_max(point.scale.asLinearFloat * scaleFactor, SIMD3<Float>(repeating: minimumScale)))
        sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * sceneFadeAlpha(localTime: t, fadeIn: norm.fadeIn))
        return sample
    }

    private static func explosion(
        point: SplatScenePoint,
        globalIndex: Int,
        time: Float,
        sceneIndex: Int,
        sceneCount: Int,
        holdDuration: Float,
        intensity: Float,
        duration: Float,
        explosionStrength: Float,
        gravity: Float,
        bounceDamping: Float,
        floorLevel: Float,
        minimumScale: Float,
        origin: SIMD3<Float>
    ) -> SplatAnimationSample {
        let idleDuration = max(holdDuration, 0)
        let cycle = idleDuration + max(duration, 0.25)
        let total = max(Float(sceneCount), 1) * cycle
        let wrapped = positiveMod(time, total)
        let currentScene = Int(floor(wrapped / cycle)) % sceneCount
        let nextScene = (currentScene + 1) % sceneCount
        let local = positiveMod(wrapped, cycle)
        let inTransition = local >= idleDuration
        let transitionTime = max(local - idleDuration, 0)

        var sample = SplatAnimationSample(point: point)
        if sceneIndex == currentScene {
            guard inTransition else {
                return sample
            }

            let exploded = simulatedExplosion(point: point,
                                              index: globalIndex,
                                              dropTime: transitionTime,
                                              gravity: gravity,
                                              bounceDamping: bounceDamping,
                                              floorLevel: floorLevel,
                                              strength: explosionStrength * max(intensity, 0.25),
                                              friction: 0.98,
                                              shrinkSpeed: 2,
                                              minimumScale: minimumScale)
            sample.point.position = exploded.position
            sample.point.scale = .linearFloat(exploded.scale)
            return sample
        }

        if sceneIndex == nextScene && inTransition {
            let birthDuration: Float = 0.5
            guard transitionTime < birthDuration else {
                return sample
            }

            let birthed = birthedExplosion(point: point,
                                           birthTime: transitionTime,
                                           birthDuration: birthDuration,
                                           origin: origin,
                                           minimumScale: minimumScale)
            sample.point.position = birthed.position
            sample.point.scale = .linearFloat(birthed.scale)
            sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * birthed.alpha)
            return sample
        }

        sample.point.opacity = .linearFloat(0)
        return sample
    }

    private static func flow(
        point: SplatScenePoint,
        globalIndex: Int,
        time: Float,
        sceneIndex: Int,
        sceneMetrics: [SplatAnimationSceneMetrics],
        holdDuration: Float,
        intensity: Float,
        minimumScale: Float,
        waves: Float
    ) -> SplatAnimationSample {
        var sample = SplatAnimationSample(point: point)
        let cycle = max(1 + holdDuration, 1.1)
        let total = Float(sceneMetrics.count) * cycle
        let wrapped = positiveMod(time, total)
        let local = positiveMod(wrapped, cycle)
        let normT = local > 1 ? 1 : local
        let fade = abs(mix(-1, 1, normT))
        let nextScene = (sceneIndex + 1) % sceneMetrics.count
        let centerNext = sceneMetrics[nextScene].centerOfMass
        let centerOwn = sceneMetrics[sceneIndex].centerOfMass
        let blend = pow(fade, 0.5 + sparkHash11(Float(globalIndex)) * 2)
        var position = normT < 0.5 ? mix(centerNext, point.position, blend) : mix(centerOwn, point.position, blend)
        let waveSample = SIMD3<Float>(sin(position.x * 2.5), sin(position.y * 2.5), sin(position.z * 2.5))
        let waveStrength = simd_length(waveSample) * waves * (1 - fade) * smoothStep(0.5, 0, normT) * 2 * max(intensity, 0)
        position += SIMD3<Float>(repeating: waveStrength)

        sample.point.position = position
        let collapsedScale = simd_max(point.scale.asLinearFloat * 0.2, SIMD3<Float>(repeating: minimumScale))
        sample.point.scale = .linearFloat(simd_mix(collapsedScale, point.scale.asLinearFloat, SIMD3<Float>(repeating: pow(fade, 3))))
        let activeScene = Int(floor(positiveMod(time + holdDuration + 0.5, total) / cycle)) % sceneMetrics.count
        let alpha = activeScene == sceneIndex ? 0.1 + fade : 0
        sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * alpha)
        sample.tint = SIMD3<Float>(repeating: 0.5 + fade * 0.5)
        return sample
    }

    private static func morph(
        point: SplatScenePoint,
        globalIndex: Int,
        time: Float,
        sceneIndex: Int,
        sceneCount: Int,
        holdDuration: Float,
        transitionDuration: Float,
        intensity _: Float,
        minimumScale: Float,
        randomRadius: Float,
        origin: SIMD3<Float>
    ) -> SplatAnimationSample {
        let stay = max(holdDuration, 0.1)
        let trans = max(transitionDuration, 0.1)
        let cycle = stay + trans
        let total = Float(sceneCount) * cycle
        let wrapped = positiveMod(time, total)
        let current = Int(floor(wrapped / cycle)) % sceneCount
        let next = (current + 1) % sceneCount
        let local = positiveMod(wrapped, cycle)
        let inTransition = local > stay
        let phase = inTransition ? clamp((local - stay) / trans, lower: 0, upper: 1) : 0
        let scatterPhase = phase < 0.5 ? phase / 0.5 : (phase - 0.5) / 0.5
        let eased = ease(scatterPhase)

        var sample = SplatAnimationSample(point: point)
        let randomMid = randomMorphPosition(index: globalIndex, radius: randomRadius) + origin
        let midpoint = mix(point.position, randomMid, 0.7)
        let small = simd_max(point.scale.asLinearFloat * 0.2, SIMD3<Float>(repeating: minimumScale))

        if sceneIndex == current {
            if !inTransition {
                sample.point.opacity = point.opacity
            } else if phase < 0.5 {
                sample.point.position = mix(point.position, midpoint, eased)
                sample.point.scale = .linearFloat(simd_mix(point.scale.asLinearFloat, small, SIMD3<Float>(repeating: eased)))
                sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * (1 - eased * 0.5))
            } else {
                sample.point.position = midpoint
                sample.point.scale = .linearFloat(small)
                sample.point.opacity = .linearFloat(0)
            }
        } else if sceneIndex == next {
            if !inTransition || phase < 0.5 {
                sample.point.position = midpoint
                sample.point.scale = .linearFloat(small)
                sample.point.opacity = .linearFloat(0)
            } else {
                sample.point.position = mix(midpoint, point.position, eased)
                sample.point.scale = .linearFloat(simd_mix(small, point.scale.asLinearFloat, SIMD3<Float>(repeating: eased)))
                sample.point.opacity = .linearFloat(point.opacity.asLinearFloat * max(eased, 0.5))
            }
        } else {
            sample.point.opacity = .linearFloat(0)
        }
        return sample
    }

    private static func sceneFadeAlpha(localTime: Float, fadeIn: Bool) -> Float {
        if fadeIn {
            if localTime < 0.4 { return 0 }
            if localTime < 0.6 { return pow((localTime - 0.4) * 5, 2) }
            return 1
        }
        if localTime < 0.4 { return 1 }
        if localTime < 0.6 { return 1 - pow((localTime - 0.4) * 5, 2) }
        return 0
    }

    private static func transitionNormTime(
        time: Float,
        sceneIndex: Int,
        sceneCount: Int,
        stay: Float,
        transition: Float
    ) -> (active: Bool, fadeIn: Bool, local: Float) {
        let cycle = stay + transition
        let total = max(Float(sceneCount), 1) * cycle
        let wrapped = positiveMod(time, total)
        let fadeInStart = Float(sceneIndex) * cycle
        let fadeOutStart = Float((sceneIndex + 1) % sceneCount) * cycle
        let fadeIn = wrapped >= fadeInStart && wrapped < fadeInStart + transition
        let fadeOut = wrapped >= fadeOutStart && wrapped < fadeOutStart + transition
        let local = positiveMod(wrapped, cycle) / cycle
        let inStay = stay > 0 && wrapped >= fadeInStart + transition && wrapped < fadeInStart + cycle
        return (fadeIn || fadeOut || inStay, fadeIn || inStay, local)
    }

    private static func simulatedExplosion(
        point: SplatScenePoint,
        index _: Int,
        dropTime: Float,
        gravity: Float,
        bounceDamping: Float,
        floorLevel: Float,
        strength: Float,
        friction: Float,
        shrinkSpeed: Float,
        minimumScale: Float
    ) -> (position: SIMD3<Float>, scale: SIMD3<Float>) {
        guard dropTime > 0 else {
            return (point.position, point.scale.asLinearFloat)
        }

        let original = point.position
        let timeVariation = sparkHash(original + SIMD3<Float>(repeating: 42)) * 0.2 - 0.1
        let adjustedDropTime = max(0, dropTime + timeVariation)
        let velocity = SIMD3<Float>(
            (sparkHash(original + SIMD3<Float>(repeating: 1)) - 0.5) * strength * (0.3 + sparkHash(original + SIMD3<Float>(repeating: 10)) * 0.4),
            abs(sparkHash(original + SIMD3<Float>(repeating: 3))) * strength * (0.8 + sparkHash(original + SIMD3<Float>(repeating: 20)) * 0.4) + 0.5,
            (sparkHash(original + SIMD3<Float>(repeating: 2)) - 0.5) * strength * (0.3 + sparkHash(original + SIMD3<Float>(repeating: 30)) * 0.4)
        )
        let frictionDecay = pow(friction, adjustedDropTime * 60)

        var position = original
        let frictionDivisor = max(1 - friction, 0.0001)
        position.x += velocity.x * (1 - frictionDecay) / frictionDivisor / 60
        position.z += velocity.z * (1 - frictionDecay) / frictionDivisor / 60
        position.y += velocity.y * adjustedDropTime - 0.5 * gravity * adjustedDropTime * adjustedDropTime
        if position.y <= floorLevel {
            let bounceCount = floor(adjustedDropTime * 3)
            let timeSinceBounce = adjustedDropTime - bounceCount / 3
            let bounceHeight = velocity.y * pow(bounceDamping, bounceCount) * max(0, 1 - timeSinceBounce * 3)
            if bounceHeight > 0.1 {
                position.y = floorLevel + abs(sin(timeSinceBounce * Float.pi * 3)) * bounceHeight
            } else {
                position.y = floorLevel
                let scatterFactor = sparkHash(original + SIMD3<Float>(repeating: 50)) * 0.2
                position.x += (sparkHash(original + SIMD3<Float>(repeating: 60)) - 0.5) * scatterFactor
                position.z += (sparkHash(original + SIMD3<Float>(repeating: 70)) - 0.5) * scatterFactor
            }
        }

        let factor = exp(-dropTime * shrinkSpeed)
        let targetScale = SIMD3<Float>(repeating: max(minimumScale, 0.005))
        let scale = simd_mix(point.scale.asLinearFloat, targetScale, SIMD3<Float>(repeating: 1 - factor))
        return (position, scale)
    }

    private static func birthedExplosion(
        point: SplatScenePoint,
        birthTime: Float,
        birthDuration: Float,
        origin: SIMD3<Float>,
        minimumScale: Float
    ) -> (position: SIMD3<Float>, scale: SIMD3<Float>, alpha: Float) {
        let progress = clamp(birthTime / max(birthDuration, 0.0001), lower: 0, upper: 1)
        let birthOffset = sparkHash(point.position) * 0.1
        let adjusted = clamp((progress - birthOffset / birthDuration) / max(1 - birthOffset / birthDuration, 0.0001), lower: 0, upper: 1)
        let eased = pow(ease(adjusted), 0.6)
        let position = mix(origin, point.position, eased)
        let scale = simd_mix(SIMD3<Float>(repeating: minimumScale), point.scale.asLinearFloat, SIMD3<Float>(repeating: eased))
        return (position, scale, eased)
    }

    private static func randomMorphPosition(index: Int, radius: Float) -> SIMD3<Float> {
        let h = sparkHash3(index)
        let theta = 2 * Float.pi * h.x
        let r = radius * sqrt(h.y)
        return SIMD3<Float>(r * cos(theta), 0, r * sin(theta))
    }

    private static func noise3(_ p: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(
            x: fract(sin(simd_dot(p, SIMD3<Float>(12.9898, 78.233, 37.719))) * 43_758.5453),
            y: fract(sin(simd_dot(p, SIMD3<Float>(93.9898, 67.345, 54.123))) * 24_631.6345),
            z: fract(sin(simd_dot(p, SIMD3<Float>(45.332, 18.654, 91.122))) * 12_345.6789)
        ) * 2 - 1
    }

    private static func hash3(_ p: SIMD3<Float>) -> SIMD3<Float> {
        SIMD3<Float>(
            fract(sin(simd_dot(p, SIMD3<Float>(127.1, 311.7, 74.7))) * 43_758.5453),
            fract(sin(simd_dot(p, SIMD3<Float>(269.5, 183.3, 246.1))) * 43_758.5453),
            fract(sin(simd_dot(p, SIMD3<Float>(113.5, 271.9, 124.6))) * 43_758.5453)
        )
    }

    private static func hash1(_ value: Float) -> Float {
        fract(sin(value * 12.9898) * 43_758.5453)
    }

    private static func sparkHash(_ value: SIMD3<Float>) -> Float {
        fract(sin(simd_dot(value, SIMD3<Float>(127.1, 311.7, 74.7))) * 43_758.5453)
    }

    private static func sparkHash11(_ value: Float) -> Float {
        var x = fract(value * 0.1031)
        x += x * (x + 33.33)
        return fract(x * x)
    }

    private static func sparkHash3(_ value: Int) -> SIMD3<Float> {
        let x = Float(value)
        return SIMD3<Float>(
            fract(sin(x) * 43_758.5453123),
            fract(sin(x + 1) * 43_758.5453123),
            fract(sin(x + 2) * 43_758.5453123)
        )
    }

    private static func rotateXZ(_ value: SIMD2<Float>, _ angle: Float) -> SIMD2<Float> {
        let s = sin(angle)
        let c = cos(angle)
        return SIMD2<Float>(c * value.x - s * value.y, s * value.x + c * value.y)
    }

    private static func smoothStep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
        guard edge0 != edge1 else { return x >= edge1 ? 1 : 0 }
        let t = clamp((x - edge0) / (edge1 - edge0), lower: 0, upper: 1)
        return t * t * (3 - 2 * t)
    }

    private static func ease(_ x: Float) -> Float {
        x * x * (3 - 2 * x)
    }

    private static func fract(_ value: Float) -> Float {
        value - floor(value)
    }

    private static func positiveMod(_ value: Float, _ modulus: Float) -> Float {
        let result = fmod(value, modulus)
        return result < 0 ? result + modulus : result
    }

    private static func clamp(_ value: Float, lower: Float, upper: Float) -> Float {
        min(max(value, lower), upper)
    }

    private static func step(_ edge: Float, _ value: Float) -> Float {
        value >= edge ? 1 : 0
    }

    private static func mix(_ a: Float, _ b: Float, _ t: Float) -> Float {
        a + (b - a) * t
    }

    private static func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
        a + (b - a) * t
    }

    private static func minimumScaleScale(_ original: SIMD3<Float>, _ minimumScale: Float) -> Float {
        let target = SIMD3<Float>(repeating: minimumScale)
        let ratios = target / simd_max(original, SIMD3<Float>(repeating: 0.0001))
        return max(ratios.x, max(ratios.y, ratios.z))
    }

    private static func scaleMix(minimumScale: Float, original: SIMD3<Float>, factor: Float) -> SIMD3<Float> {
        simd_mix(SIMD3<Float>(repeating: minimumScale), original, SIMD3<Float>(repeating: clamp(factor, lower: 0, upper: 1)))
    }
}

internal extension SplatRenderer {
    var animationEnabled: Bool {
        animationConfiguration != nil && !sourceScenePoints.isEmpty
    }

    func setAnimationSourcePoints(_ points: [SplatScenePoint], sceneCounts: [Int]? = nil, sceneIndices: [UInt32]? = nil) {
        sourceScenePoints = points
        if let sceneIndices, sceneIndices.count == points.count {
            let normalized = Self.normalizeSceneIndices(sceneIndices)
            animationSceneIndices = normalized.indices
            animationSceneCounts = normalized.counts
            animationSceneMetrics = Self.makeSceneMetrics(points: points,
                                                          sceneIndices: animationSceneIndices,
                                                          sceneCounts: animationSceneCounts)
        } else {
            let counts = (sceneCounts?.isEmpty == false) ? sceneCounts! : [points.count]
            animationSceneCounts = counts
            animationSceneIndices = counts.enumerated().flatMap { scene, count in
                Array(repeating: UInt32(scene), count: count)
            }
            if animationSceneIndices.count != points.count {
                animationSceneIndices = Array(repeating: 0, count: points.count)
                animationSceneCounts = [points.count]
            }
            animationSceneMetrics = Self.makeSceneMetrics(points: points, sceneCounts: animationSceneCounts)
        }
        animationDirty = true
    }

    var activeSplatBufferForRendering: MetalBuffer<Splat> {
        animationEnabled ? (animatedSplatBuffer ?? splatBuffer) : splatBuffer
    }

    @discardableResult
    func updateAnimatedSplatsIfNeeded(to commandBuffer: MTLCommandBuffer) -> Bool {
        guard animationEnabled, let configuration = animationConfiguration else {
            if let animatedSplatBuffer {
                self.animatedSplatBuffer = nil
                lastAppliedAnimationTime = nil
                animationDirty = true
                markAnimationDependentDataDirty()
                releaseAnimatedSplatBuffer(animatedSplatBuffer, on: commandBuffer)
            }
            return false
        }

        if !animationDirty, lastAppliedAnimationTime == configuration.time {
            return false
        }

        let animatedSplatBuffer: MetalBuffer<Splat>
        do {
            animatedSplatBuffer = try acquireAnimatedSplatBuffer(minimumCapacity: max(sourceScenePoints.count, 1))
        } catch {
            Self.log.error("Failed to allocate animated splat buffer: \(error)")
            return false
        }
        animatedSplatBuffer.count = 0

        for (index, sourcePoint) in sourceScenePoints.enumerated() {
            let sceneIndex = index < animationSceneIndices.count ? Int(animationSceneIndices[index]) : 0
            var sample = SplatAnimationEngine.apply(
                point: sourcePoint,
                globalIndex: index,
                sceneIndex: sceneIndex,
                sceneMetrics: animationSceneMetrics,
                configuration: configuration
            )

            let tintedColor = simd_clamp(sample.point.color.asLinearFloat * sample.tint, SIMD3<Float>(repeating: 0), SIMD3<Float>(repeating: 4))
            sample.point.color = .linearFloat(tintedColor)
            sample.point.rotation = sample.point.rotation.normalized

            animatedSplatBuffer.append(Splat(sample.point))
        }

        let previousAnimatedBuffer = self.animatedSplatBuffer
        self.animatedSplatBuffer = animatedSplatBuffer
        if let previousAnimatedBuffer {
            releaseAnimatedSplatBuffer(previousAnimatedBuffer, on: commandBuffer)
        }

        markAnimationDependentDataDirty()
        lastAppliedAnimationTime = configuration.time
        animationDirty = false
        return true
    }

    static func makeSceneMetrics(points: [SplatScenePoint], sceneCounts: [Int]) -> [SplatAnimationSceneMetrics] {
        guard !points.isEmpty else { return [] }
        var metrics: [SplatAnimationSceneMetrics] = []
        var lowerBound = 0

        for (sceneIndex, rawCount) in sceneCounts.enumerated() {
            let count = max(rawCount, 0)
            let upperBound = min(lowerBound + count, points.count)
            guard lowerBound < upperBound else {
                lowerBound = upperBound
                continue
            }
            let slice = points[lowerBound..<upperBound]
            var minBounds = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var maxBounds = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
            var sum = SIMD3<Float>(repeating: 0)

            for point in slice {
                minBounds = simd_min(minBounds, point.position)
                maxBounds = simd_max(maxBounds, point.position)
                sum += point.position
            }

            let center = (minBounds + maxBounds) * 0.5
            let centerOfMass = sum / Float(slice.count)
            var radialExtent: Float = 0.001
            for point in slice {
                radialExtent = max(radialExtent, simd_length(point.position - center))
            }

            metrics.append(
                SplatAnimationSceneMetrics(
                    sceneIndex: sceneIndex,
                    range: lowerBound..<upperBound,
                    center: center,
                    centerOfMass: centerOfMass,
                    boundsMin: minBounds,
                    boundsMax: maxBounds,
                    radialExtent: radialExtent
                )
            )
            lowerBound = upperBound
        }

        if metrics.isEmpty {
            let fallbackCenter = points.map(\.position).reduce(SIMD3<Float>(repeating: 0), +) / Float(points.count)
            metrics.append(
                SplatAnimationSceneMetrics(
                    sceneIndex: 0,
                    range: 0..<points.count,
                    center: fallbackCenter,
                    centerOfMass: fallbackCenter,
                    boundsMin: fallbackCenter,
                    boundsMax: fallbackCenter,
                    radialExtent: 0.001
                )
            )
        }

        return metrics
    }

    static func makeSceneMetrics(points: [SplatScenePoint], sceneIndices: [UInt32], sceneCounts: [Int]) -> [SplatAnimationSceneMetrics] {
        guard !points.isEmpty, sceneIndices.count == points.count else {
            return makeSceneMetrics(points: points, sceneCounts: sceneCounts)
        }

        var metrics: [SplatAnimationSceneMetrics] = []
        var lowerBound = 0
        for sceneIndex in 0..<sceneCounts.count {
            let matchingIndices = sceneIndices.enumerated().compactMap { index, value in
                Int(value) == sceneIndex ? index : nil
            }
            guard !matchingIndices.isEmpty else {
                continue
            }

            var minBounds = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
            var maxBounds = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)
            var sum = SIMD3<Float>(repeating: 0)

            for index in matchingIndices {
                let point = points[index]
                minBounds = simd_min(minBounds, point.position)
                maxBounds = simd_max(maxBounds, point.position)
                sum += point.position
            }

            let center = (minBounds + maxBounds) * 0.5
            let centerOfMass = sum / Float(matchingIndices.count)
            var radialExtent: Float = 0.001
            for index in matchingIndices {
                radialExtent = max(radialExtent, simd_length(points[index].position - center))
            }

            metrics.append(
                SplatAnimationSceneMetrics(
                    sceneIndex: sceneIndex,
                    range: lowerBound..<(lowerBound + matchingIndices.count),
                    center: center,
                    centerOfMass: centerOfMass,
                    boundsMin: minBounds,
                    boundsMax: maxBounds,
                    radialExtent: radialExtent
                )
            )
            lowerBound += matchingIndices.count
        }

        return metrics.isEmpty ? makeSceneMetrics(points: points, sceneCounts: [points.count]) : metrics
    }

    private static func normalizeSceneIndices(_ sceneIndices: [UInt32]) -> (indices: [UInt32], counts: [Int]) {
        var remap: [UInt32: UInt32] = [:]
        var normalized: [UInt32] = []
        var counts: [Int] = []

        for sceneIndex in sceneIndices {
            let compactIndex: UInt32
            if let existing = remap[sceneIndex] {
                compactIndex = existing
            } else {
                compactIndex = UInt32(remap.count)
                remap[sceneIndex] = compactIndex
                counts.append(0)
            }

            normalized.append(compactIndex)
            counts[Int(compactIndex)] += 1
        }

        return (normalized, counts)
    }
}
