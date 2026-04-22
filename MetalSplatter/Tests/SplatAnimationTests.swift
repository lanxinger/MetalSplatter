import XCTest
import simd
@testable import MetalSplatter
import SplatIO

final class SplatAnimationTests: XCTestCase {
    func testSceneMetricsPreserveLayerRangesAndCenters() {
        let layerA = [
            makePoint(position: SIMD3<Float>(0, 0, 0)),
            makePoint(position: SIMD3<Float>(2, 0, 0))
        ]
        let layerB = [
            makePoint(position: SIMD3<Float>(10, 1, -2)),
            makePoint(position: SIMD3<Float>(14, 5, 2))
        ]

        let metrics = SplatRenderer.makeSceneMetrics(points: layerA + layerB,
                                                     sceneCounts: [layerA.count, layerB.count])

        XCTAssertEqual(metrics.count, 2)
        XCTAssertEqual(metrics[0].range, 0..<2)
        XCTAssertEqual(metrics[1].range, 2..<4)
        XCTAssertEqual(metrics[0].center.x, 1, accuracy: 0.0001)
        XCTAssertEqual(metrics[1].center.x, 12, accuracy: 0.0001)
        XCTAssertEqual(metrics[1].center.y, 3, accuracy: 0.0001)
        XCTAssertEqual(metrics[1].center.z, 0, accuracy: 0.0001)
    }

    func testSpreadEffectShrinksNearOriginEarlyInReveal() {
        let point = makePoint(position: SIMD3<Float>(0.2, 0, 0.2),
                              scale: SIMD3<Float>(0.4, 0.5, 0.6),
                              opacity: 1)
        let metrics = SplatRenderer.makeSceneMetrics(points: [point], sceneCounts: [1])
        let configuration = SplatAnimationConfiguration(effect: .spread,
                                                        time: 1.25,
                                                        intensity: 1,
                                                        minimumScale: 0.01)

        let sample = SplatAnimationEngine.apply(point: point,
                                                globalIndex: 0,
                                                sceneIndex: 0,
                                                sceneMetrics: metrics,
                                                configuration: configuration)

        XCTAssertLessThan(sample.point.scale.asLinearFloat.x, point.scale.asLinearFloat.x)
        XCTAssertLessThanOrEqual(sample.point.opacity.asLinearFloat, point.opacity.asLinearFloat)
        XCTAssertGreaterThanOrEqual(sample.point.opacity.asLinearFloat, 0)
    }

    func testSphericalOnlyActivatesCurrentAndNextScenes() {
        let layers = [
            makePoint(position: SIMD3<Float>(0, 0, 0)),
            makePoint(position: SIMD3<Float>(10, 0, 0)),
            makePoint(position: SIMD3<Float>(20, 0, 0))
        ]
        let metrics = SplatRenderer.makeSceneMetrics(points: layers, sceneCounts: [1, 1, 1])
        let configuration = SplatAnimationConfiguration(effect: .spherical,
                                                        time: 0.5,
                                                        minimumScale: 0.01)

        let first = SplatAnimationEngine.apply(point: layers[0],
                                               globalIndex: 0,
                                               sceneIndex: 0,
                                               sceneMetrics: metrics,
                                               configuration: configuration)
        let second = SplatAnimationEngine.apply(point: layers[1],
                                                globalIndex: 1,
                                                sceneIndex: 1,
                                                sceneMetrics: metrics,
                                                configuration: configuration)
        let third = SplatAnimationEngine.apply(point: layers[2],
                                               globalIndex: 2,
                                               sceneIndex: 2,
                                               sceneMetrics: metrics,
                                               configuration: configuration)

        XCTAssertGreaterThan(first.point.opacity.asLinearFloat, 0)
        XCTAssertEqual(second.point.opacity.asLinearFloat, 0, accuracy: 0.0001)
        XCTAssertGreaterThan(third.point.opacity.asLinearFloat, 0)
    }

    func testFlowMovesCurrentSceneTowardNextCenterAndKeepsOtherScenesHidden() {
        let layers = [
            makePoint(position: SIMD3<Float>(-10, 0, 0)),
            makePoint(position: SIMD3<Float>(0, 0, 0)),
            makePoint(position: SIMD3<Float>(10, 0, 0))
        ]
        let metrics = SplatRenderer.makeSceneMetrics(points: layers, sceneCounts: [1, 1, 1])
        let configuration = SplatAnimationConfiguration(effect: .flow,
                                                        time: 0.25,
                                                        minimumScale: 0.01,
                                                        holdDuration: 2,
                                                        waves: 0)

        let active = SplatAnimationEngine.apply(point: layers[0],
                                                globalIndex: 0,
                                                sceneIndex: 0,
                                                sceneMetrics: metrics,
                                                configuration: configuration)
        let inactive = SplatAnimationEngine.apply(point: layers[1],
                                                  globalIndex: 1,
                                                  sceneIndex: 1,
                                                  sceneMetrics: metrics,
                                                  configuration: configuration)

        XCTAssertGreaterThan(active.point.position.x, layers[0].position.x)
        XCTAssertLessThan(active.point.position.x, 0)
        XCTAssertEqual(active.point.opacity.asLinearFloat, 0.6, accuracy: 0.0001)
        XCTAssertEqual(inactive.point.opacity.asLinearFloat, 0, accuracy: 0.0001)
    }

    func testExplosionKeepsCurrentSceneVisibleAndBirthsNextSceneFromOrigin() {
        let layers = [
            makePoint(position: SIMD3<Float>(0, 1, 0),
                      scale: SIMD3<Float>(0.4, 0.4, 0.4)),
            makePoint(position: SIMD3<Float>(2, 1, 0),
                      scale: SIMD3<Float>(0.4, 0.4, 0.4))
        ]
        let metrics = SplatRenderer.makeSceneMetrics(points: layers, sceneCounts: [1, 1])
        let configuration = SplatAnimationConfiguration(effect: .explosion,
                                                        time: 1.25,
                                                        minimumScale: 0.01,
                                                        duration: 3,
                                                        holdDuration: 1,
                                                        origin: .zero)

        let dying = SplatAnimationEngine.apply(point: layers[0],
                                               globalIndex: 0,
                                               sceneIndex: 0,
                                               sceneMetrics: metrics,
                                               configuration: configuration)
        let birthing = SplatAnimationEngine.apply(point: layers[1],
                                                  globalIndex: 1,
                                                  sceneIndex: 1,
                                                  sceneMetrics: metrics,
                                                  configuration: configuration)

        XCTAssertEqual(dying.point.opacity.asLinearFloat, 1, accuracy: 0.0001)
        XCTAssertLessThan(dying.point.scale.asLinearFloat.x, layers[0].scale.asLinearFloat.x)
        XCTAssertGreaterThan(birthing.point.opacity.asLinearFloat, 0)
        XCTAssertLessThan(birthing.point.position.x, layers[1].position.x)
    }

    func testMorphUsesStableMidpointWithoutIntensityScaling() {
        let layers = [
            makePoint(position: SIMD3<Float>(0, 0, 0),
                      scale: SIMD3<Float>(0.5, 0.5, 0.5)),
            makePoint(position: SIMD3<Float>(5, 0, 0),
                      scale: SIMD3<Float>(0.5, 0.5, 0.5))
        ]
        let metrics = SplatRenderer.makeSceneMetrics(points: layers, sceneCounts: [1, 1])
        let configuration = SplatAnimationConfiguration(effect: .morph,
                                                        time: 2.5,
                                                        intensity: 5,
                                                        minimumScale: 0.01,
                                                        holdDuration: 1.5,
                                                        transitionDuration: 2,
                                                        randomRadius: 1.3)

        let current = SplatAnimationEngine.apply(point: layers[0],
                                                 globalIndex: 0,
                                                 sceneIndex: 0,
                                                 sceneMetrics: metrics,
                                                 configuration: configuration)
        let next = SplatAnimationEngine.apply(point: layers[1],
                                              globalIndex: 1,
                                              sceneIndex: 1,
                                              sceneMetrics: metrics,
                                              configuration: configuration)

        XCTAssertEqual(current.point.opacity.asLinearFloat, 0, accuracy: 0.0001)
        XCTAssertEqual(next.point.opacity.asLinearFloat, 0.5, accuracy: 0.0001)
        XCTAssertLessThan(next.point.scale.asLinearFloat.x, layers[1].scale.asLinearFloat.x)
    }

    private func makePoint(position: SIMD3<Float>,
                           scale: SIMD3<Float> = SIMD3<Float>(repeating: 0.2),
                           opacity: Float = 1) -> SplatScenePoint {
        SplatScenePoint(position: position,
                        color: .linearFloat(SIMD3<Float>(1, 0.5, 0.25)),
                        opacity: .linearFloat(opacity),
                        scale: .linearFloat(scale),
                        rotation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0)))
    }
}
