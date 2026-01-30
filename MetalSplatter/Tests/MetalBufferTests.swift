import XCTest
import Metal
@testable import MetalSplatter

final class MetalBufferTests: XCTestCase {

    var device: MTLDevice!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device should be available for testing")
    }

    override func tearDown() {
        device = nil
        super.tearDown()
    }

    // MARK: - Basic Functionality Tests

    func testBasicCreation() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 100)
        XCTAssertEqual(buffer.capacity, 100)
        XCTAssertEqual(buffer.count, 0)
    }

    func testAppend() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 10)

        let index1 = buffer.append(1.0)
        XCTAssertEqual(index1, 0)
        XCTAssertEqual(buffer.count, 1)

        let index2 = buffer.append(2.0)
        XCTAssertEqual(index2, 1)
        XCTAssertEqual(buffer.count, 2)
    }

    func testAppendArray() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 10)

        let index = buffer.append([1.0, 2.0, 3.0])
        XCTAssertEqual(index, 0)
        XCTAssertEqual(buffer.count, 3)
    }

    func testEnsureCapacity() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 10)
        XCTAssertEqual(buffer.capacity, 10)

        try buffer.ensureCapacity(50)
        XCTAssertGreaterThanOrEqual(buffer.capacity, 50)
    }

    func testSetCapacity() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 10)
        _ = buffer.append([1.0, 2.0, 3.0])

        try buffer.setCapacity(100)
        XCTAssertGreaterThanOrEqual(buffer.capacity, 100)
        XCTAssertEqual(buffer.count, 3) // Data should be preserved
    }

    // MARK: - Thread Safety Tests

    func testConcurrentCountAccess() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 1000)

        // Pre-populate with some data
        for i in 0..<100 {
            buffer.append(Float(i))
        }

        let expectation = self.expectation(description: "Concurrent count access")
        expectation.expectedFulfillmentCount = 10

        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)

        // Multiple threads reading count concurrently
        for _ in 0..<10 {
            queue.async {
                for _ in 0..<1000 {
                    let count = buffer.count
                    XCTAssertGreaterThanOrEqual(count, 100)
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 10.0)
    }

    func testConcurrentAppend() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 10000)

        let expectation = self.expectation(description: "Concurrent append")
        expectation.expectedFulfillmentCount = 10

        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)
        let appendsPerThread = 100

        for threadIndex in 0..<10 {
            queue.async {
                for i in 0..<appendsPerThread {
                    _ = buffer.append(Float(threadIndex * 1000 + i))
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 10.0)

        // All appends should have succeeded
        XCTAssertEqual(buffer.count, 10 * appendsPerThread)
    }

    func testConcurrentEnsureCapacity() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 10)

        let expectation = self.expectation(description: "Concurrent ensureCapacity")
        expectation.expectedFulfillmentCount = 10

        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)

        // Multiple threads trying to grow the buffer simultaneously
        for i in 0..<10 {
            queue.async {
                do {
                    try buffer.ensureCapacity(100 * (i + 1))
                } catch {
                    XCTFail("ensureCapacity should not throw: \(error)")
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 10.0)

        // Buffer should have grown to at least the largest requested capacity
        XCTAssertGreaterThanOrEqual(buffer.capacity, 1000)
    }

    func testConcurrentReadWrite() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 1000)

        // Pre-populate
        for i in 0..<100 {
            buffer.append(Float(i))
        }

        let expectation = self.expectation(description: "Concurrent read/write")
        expectation.expectedFulfillmentCount = 20

        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)

        // 10 reader threads
        for _ in 0..<10 {
            queue.async {
                for _ in 0..<100 {
                    let _ = buffer.count
                    let _ = buffer.capacity
                }
                expectation.fulfill()
            }
        }

        // 10 writer threads
        for _ in 0..<10 {
            queue.async {
                for i in 0..<10 {
                    _ = buffer.append(Float(i))
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 10.0)
    }

    // MARK: - withLockedValues Tests

    func testWithLockedValues() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 100)
        _ = buffer.append([1.0, 2.0, 3.0])

        let result = buffer.withLockedValues { values, count in
            var sum: Float = 0
            for i in 0..<count {
                sum += values[i]
            }
            return sum
        }

        XCTAssertEqual(result, 6.0)
    }

    func testWithLockedValuesProtectsDuringResize() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 10)
        _ = buffer.append([1.0, 2.0, 3.0])

        let expectation = self.expectation(description: "Locked values during resize")
        expectation.expectedFulfillmentCount = 2

        let queue = DispatchQueue(label: "test.concurrent", attributes: .concurrent)

        // Thread 1: Hold lock while reading values
        queue.async {
            buffer.withLockedValues { values, count in
                // Simulate work while holding lock
                var sum: Float = 0
                for i in 0..<count {
                    sum += values[i]
                }
                // Brief pause to increase chance of race
                Thread.sleep(forTimeInterval: 0.01)

                // Values should still be valid
                for i in 0..<count {
                    XCTAssertFalse(values[i].isNaN)
                }
            }
            expectation.fulfill()
        }

        // Thread 2: Try to resize while Thread 1 holds lock
        queue.async {
            do {
                try buffer.ensureCapacity(1000)
            } catch {
                XCTFail("ensureCapacity should not throw: \(error)")
            }
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 10.0)
    }

    // MARK: - withLockedBuffer Tests

    func testWithLockedBuffer() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 100)
        _ = buffer.append([1.0, 2.0, 3.0])

        buffer.withLockedBuffer { mtlBuffer, count in
            XCTAssertEqual(count, 3)
            XCTAssertGreaterThan(mtlBuffer.length, 0)
        }
    }

    // MARK: - Error Handling Tests

    func testCapacityExceedsMaximum() {
        let maxCapacity = MetalBuffer<Float>.maxCapacity(for: device)

        XCTAssertThrowsError(try MetalBuffer<Float>(device: device, capacity: maxCapacity + 1)) { error in
            guard case MetalBuffer<Float>.Error.capacityGreatedThanMaxCapacity = error else {
                XCTFail("Expected capacityGreatedThanMaxCapacity error")
                return
            }
        }
    }

    func testEnsureCapacityExceedsMaximum() throws {
        let buffer = try MetalBuffer<Float>(device: device, capacity: 10)
        let maxCapacity = MetalBuffer<Float>.maxCapacity(for: device)

        XCTAssertThrowsError(try buffer.ensureCapacity(maxCapacity + 1)) { error in
            guard case MetalBuffer<Float>.Error.capacityGreatedThanMaxCapacity = error else {
                XCTFail("Expected capacityGreatedThanMaxCapacity error")
                return
            }
        }
    }
}
