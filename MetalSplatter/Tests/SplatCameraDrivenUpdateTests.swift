import XCTest
import simd
@testable import MetalSplatter

final class SplatCameraDrivenUpdateTests: XCTestCase {
    func testDirtyStateForcesUpdate() {
        let shouldRun = SplatRenderer.shouldRunCameraDrivenUpdate(
            dirty: true,
            now: 5.0,
            lastUpdateTime: 4.99,
            minimumInterval: 1.0,
            currentPosition: SIMD3<Float>(0, 0, 0),
            currentForward: SIMD3<Float>(0, 0, -1),
            lastPosition: SIMD3<Float>(0, 0, 0),
            lastForward: SIMD3<Float>(0, 0, -1),
            positionEpsilon: 0.1,
            directionEpsilon: 0.1
        )

        XCTAssertTrue(shouldRun)
    }

    func testMinimumIntervalSuppressesCameraDrivenUpdate() {
        let shouldRun = SplatRenderer.shouldRunCameraDrivenUpdate(
            dirty: false,
            now: 10.0,
            lastUpdateTime: 9.99,
            minimumInterval: 0.5,
            currentPosition: SIMD3<Float>(0.5, 0, 0),
            currentForward: simd_normalize(SIMD3<Float>(0.2, 0, -1)),
            lastPosition: SIMD3<Float>(0, 0, 0),
            lastForward: SIMD3<Float>(0, 0, -1),
            positionEpsilon: 0.01,
            directionEpsilon: 0.0001
        )

        XCTAssertFalse(shouldRun)
    }

    func testCameraMotionThresholdsGateUpdate() {
        let belowThreshold = SplatRenderer.shouldRunCameraDrivenUpdate(
            dirty: false,
            now: 10.0,
            lastUpdateTime: 9.0,
            minimumInterval: 0.0,
            currentPosition: SIMD3<Float>(0.001, 0, 0),
            currentForward: simd_normalize(SIMD3<Float>(0.001, 0, -1)),
            lastPosition: SIMD3<Float>(0, 0, 0),
            lastForward: SIMD3<Float>(0, 0, -1),
            positionEpsilon: 0.01,
            directionEpsilon: 0.0001
        )
        let aboveThreshold = SplatRenderer.shouldRunCameraDrivenUpdate(
            dirty: false,
            now: 10.0,
            lastUpdateTime: 9.0,
            minimumInterval: 0.0,
            currentPosition: SIMD3<Float>(0.02, 0, 0),
            currentForward: SIMD3<Float>(0, 0, -1),
            lastPosition: SIMD3<Float>(0, 0, 0),
            lastForward: SIMD3<Float>(0, 0, -1),
            positionEpsilon: 0.01,
            directionEpsilon: 0.0001
        )

        XCTAssertFalse(belowThreshold)
        XCTAssertTrue(aboveThreshold)
    }
}
