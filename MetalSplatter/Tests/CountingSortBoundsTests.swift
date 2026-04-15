import XCTest
import simd
@testable import MetalSplatter

final class CountingSortBoundsTests: XCTestCase {
    func testRadialDepthBoundsClampNearDistanceToZeroInsideBounds() {
        let bounds = (
            min: SIMD3<Float>(-1, -1, -1),
            max: SIMD3<Float>(1, 1, 1)
        )

        let estimated = SplatRenderer.estimateCountingSortDepthBounds(
            from: bounds,
            cameraPosition: .zero,
            cameraForward: SIMD3<Float>(0, 0, -1),
            sortByDistance: true
        )

        XCTAssertEqual(estimated.min, 0, accuracy: 0.0001)
        XCTAssertEqual(estimated.max, sqrt(3) + sqrt(3) * 0.001, accuracy: 0.0001)
    }

    func testRadialDepthBoundsUseNearestPointAndFarthestCorner() {
        let bounds = (
            min: SIMD3<Float>(0, 0, 0),
            max: SIMD3<Float>(1, 1, 1)
        )
        let cameraPosition = SIMD3<Float>(2, 0.5, 0.5)

        let estimated = SplatRenderer.estimateCountingSortDepthBounds(
            from: bounds,
            cameraPosition: cameraPosition,
            cameraForward: SIMD3<Float>(0, 0, -1),
            sortByDistance: true
        )

        let nearestDistance: Float = 1
        let farthestDistance: Float = sqrt(4.0 + 0.25 + 0.25)
        let padding = max((farthestDistance - nearestDistance) * 0.001, 0.001)

        XCTAssertEqual(estimated.min, nearestDistance - padding, accuracy: 0.0001)
        XCTAssertEqual(estimated.max, farthestDistance + padding, accuracy: 0.0001)
    }

    func testLinearDepthBoundsProjectAABBAlongViewDirection() {
        let bounds = (
            min: SIMD3<Float>(-2, -1, -5),
            max: SIMD3<Float>(2, 1, -1)
        )

        let estimated = SplatRenderer.estimateCountingSortDepthBounds(
            from: bounds,
            cameraPosition: .zero,
            cameraForward: SIMD3<Float>(0, 0, -1),
            sortByDistance: false
        )

        let minimumDepth: Float = 1
        let maximumDepth: Float = 5
        let padding = (maximumDepth - minimumDepth) * 0.001

        XCTAssertEqual(estimated.min, minimumDepth - padding, accuracy: 0.0001)
        XCTAssertEqual(estimated.max, maximumDepth + padding, accuracy: 0.0001)
    }
}
