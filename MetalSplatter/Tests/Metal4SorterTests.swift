import XCTest
import Metal
import simd
@testable import MetalSplatter

/// GPU correctness tests for Metal4Sorter: both the legacy multi-dispatch radix
/// sort and the OneSweep (MSL 4.1 decoupled lookback) path, which only exists
/// when runtime compilation succeeds on this OS.
final class Metal4SorterTests: XCTestCase {

    @available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
    private struct Harness {
        let device: MTLDevice
        let queue: MTLCommandQueue
        let sorter: Metal4Sorter

        init?() throws {
            guard let device = MTLCreateSystemDefaultDevice(),
                  device.supportsFamily(.apple9),
                  let queue = device.makeCommandQueue() else { return nil }
            let library = try device.makeDefaultLibrary(bundle: Bundle.module)
            self.device = device
            self.queue = queue
            self.sorter = try Metal4Sorter(device: device, library: library)
        }

        /// Sort splats at the given positions (camera at origin, forward -Z)
        /// and return the sorted indices.
        func sort(positions: [SIMD3<Float>], useOneSweep: Bool) throws -> [Int32] {
            let count = positions.count
            let splats = positions.map {
                SplatRenderer.Splat(position: $0,
                                    color: SIMD4<Float>(1, 1, 1, 1),
                                    scale: SIMD3<Float>(1, 1, 1),
                                    rotation: simd_quatf(angle: 0, axis: SIMD3<Float>(0, 1, 0)))
            }
            let splatsBuffer = splats.withUnsafeBytes { bytes in
                device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
            }
            let indicesBuffer = device.makeBuffer(length: count * MemoryLayout<Int32>.stride,
                                                  options: .storageModeShared)
            guard let splatsBuffer, let indicesBuffer,
                  let commandBuffer = queue.makeCommandBuffer() else {
                throw XCTSkip("Failed to create Metal resources")
            }

            sorter.useOneSweep = useOneSweep
            try sorter.sort(splats: splatsBuffer,
                            count: count,
                            cameraPosition: SIMD3<Float>(0, 0, 0),
                            cameraForward: SIMD3<Float>(0, 0, -1),
                            sortByDistance: true,
                            outputIndices: indicesBuffer,
                            commandBuffer: commandBuffer)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            XCTAssertEqual(commandBuffer.status, .completed,
                           "GPU sort failed: \(commandBuffer.error.map(String.init(describing:)) ?? "unknown")")

            return Array(UnsafeBufferPointer(start: indicesBuffer.contents().bindMemory(to: Int32.self, capacity: count),
                                             count: count))
        }
    }

    /// Distinct distances in shuffled order; expects exact back-to-front
    /// (descending distance) output. Sizes cross the OneSweep tile boundary
    /// (1024) and stretch the lookback chain across hundreds of tiles.
    private func assertSortsDescending(useOneSweep: Bool) throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            throw XCTSkip("Metal4Sorter requires OS 26")
        }
        guard let harness = try Harness() else {
            throw XCTSkip("No Metal device with Apple9 GPU family")
        }
        if useOneSweep, !harness.sorter.oneSweepAvailable {
            throw XCTSkip("OneSweep kernels unavailable (MSL 4.1 runtime compilation unsupported)")
        }

        for count in [1, 7, 1000, 1024, 4096, 250_000] {
            // Distances 1...count, shuffled deterministically
            var rng = SplitMix64(seed: 0x5EED + UInt64(count))
            var order = Array(0..<count)
            order.shuffle(using: &rng)
            // order[i] is the rank of splat i: distance = rank + 1
            let positions = order.map { SIMD3<Float>(Float($0 + 1), 0, 0) }

            let indices = try harness.sort(positions: positions, useOneSweep: useOneSweep)

            XCTAssertEqual(indices.count, count)
            for (outputSlot, splatIndex) in indices.enumerated() {
                // Back-to-front: slot 0 holds the farthest splat (rank count-1)
                let expectedRank = count - 1 - outputSlot
                XCTAssertEqual(order[Int(splatIndex)], expectedRank,
                               "count \(count): slot \(outputSlot) has splat \(splatIndex) with rank \(order[Int(splatIndex)]), expected \(expectedRank)")
            }
        }
    }

    /// Equal keys must keep their original relative order (LSD radix sorts
    /// require stability; rendering relies on it for temporal coherence).
    private func assertStableOnEqualKeys(useOneSweep: Bool) throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            throw XCTSkip("Metal4Sorter requires OS 26")
        }
        guard let harness = try Harness() else {
            throw XCTSkip("No Metal device with Apple9 GPU family")
        }
        if useOneSweep, !harness.sorter.oneSweepAvailable {
            throw XCTSkip("OneSweep kernels unavailable (MSL 4.1 runtime compilation unsupported)")
        }

        // 3000 splats in 3 interleaved groups at distances 10, 20, 30
        let count = 3000
        let positions = (0..<count).map { SIMD3<Float>(Float(10 * ($0 % 3 + 1)), 0, 0) }

        let indices = try harness.sort(positions: positions, useOneSweep: useOneSweep)

        // Descending distance: group 30 first, then 20, then 10; ascending
        // original index within each group.
        var expected: [Int32] = []
        for group in [2, 1, 0] {
            expected.append(contentsOf: stride(from: Int32(group), to: Int32(count), by: 3))
        }
        XCTAssertEqual(indices, expected)
    }

    func testLegacySortsDescending() throws {
        try assertSortsDescending(useOneSweep: false)
    }

    func testOneSweepSortsDescending() throws {
        try assertSortsDescending(useOneSweep: true)
    }

    func testLegacyStableOnEqualKeys() throws {
        try assertStableOnEqualKeys(useOneSweep: false)
    }

    func testOneSweepStableOnEqualKeys() throws {
        try assertStableOnEqualKeys(useOneSweep: true)
    }

    /// Both paths are stable sorts over the same key transform, so their
    /// outputs must match exactly on arbitrary input.
    func testOneSweepMatchesLegacy() throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            throw XCTSkip("Metal4Sorter requires OS 26")
        }
        guard let harness = try Harness() else {
            throw XCTSkip("No Metal device with Apple9 GPU family")
        }
        guard harness.sorter.oneSweepAvailable else {
            throw XCTSkip("OneSweep kernels unavailable (MSL 4.1 runtime compilation unsupported)")
        }

        var rng = SplitMix64(seed: 0xDECAF)
        let positions = (0..<100_000).map { _ in
            SIMD3<Float>(Float(rng.next() % 2000) / 7 - 100,
                         Float(rng.next() % 2000) / 7 - 100,
                         Float(rng.next() % 2000) / 7 - 100)
        }

        let legacy = try harness.sort(positions: positions, useOneSweep: false)
        let oneSweep = try harness.sort(positions: positions, useOneSweep: true)
        XCTAssertEqual(legacy, oneSweep)
    }
}

/// Deterministic RNG so failures reproduce.
private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
