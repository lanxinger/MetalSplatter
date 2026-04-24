import XCTest
@testable import MetalSplatter

final class SortPerformanceTelemetryTests: XCTestCase {
    func testLogMessageIncludesPathCountsTimingsAndState() {
        let sample = SplatRenderer.SortPerformanceSample(
            path: .counting,
            splatCount: 246_821,
            renderableCount: 240_000,
            wallTime: 0.006,
            callbackWallTime: 0.006,
            gpuTime: 0.004,
            mainQueueDelay: 0.0065,
            inFlightSortsAtStart: 1,
            inFlightSortsAtCompletion: 0,
            interactionMode: true,
            sortByDistance: false,
            status: "completed"
        )

        let message = sample.logMessage

        XCTAssertTrue(message.contains("path=counting"))
        XCTAssertTrue(message.contains("splats=246821"))
        XCTAssertTrue(message.contains("renderable=240000"))
        XCTAssertTrue(message.contains("wallMs=6.00"))
        XCTAssertTrue(message.contains("callbackWallMs=6.00"))
        XCTAssertTrue(message.contains("gpuMs=4.00"))
        XCTAssertTrue(message.contains("overheadMs=2.00"))
        XCTAssertTrue(message.contains("mainQueueMs=6.50"))
        XCTAssertTrue(message.contains("inFlightStart=1"))
        XCTAssertTrue(message.contains("inFlightEnd=0"))
        XCTAssertTrue(message.contains("interaction=true"))
        XCTAssertTrue(message.contains("sortByDistance=false"))
        XCTAssertTrue(message.contains("status=completed"))
    }
}
