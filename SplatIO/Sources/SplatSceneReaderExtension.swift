import Foundation

/// Error thrown when scene reading times out
public enum SplatSceneReaderError: LocalizedError {
    case timeout

    public var errorDescription: String? {
        switch self {
        case .timeout:
            return "Scene reading timed out"
        }
    }
}

/// Default implementation of `readScene()` for existing readers that use the delegate pattern
extension SplatSceneReader {
    public func readScene() throws -> [SplatScenePoint] {
        let collector = PointCollector()
        read(to: collector)

        // Wait for completion (handles both sync and potential async implementations)
        // Current implementations are synchronous, so this returns immediately.
        // Timeout of 5 minutes handles very large files while preventing infinite hangs.
        try collector.waitForCompletion(timeout: 300.0)

        if let error = collector.error {
            throw error
        }

        return collector.points
    }

    /// Reads the scene and reorders points using Morton code ordering for improved GPU cache coherency.
    ///
    /// Morton ordering clusters spatially nearby 3D points together in memory, which can significantly
    /// improve rendering performance for large scenes by reducing GPU cache misses.
    ///
    /// - Parameter useParallel: If true, uses parallel computation for large datasets (>100K points)
    /// - Returns: Array of splat scene points reordered by Morton code
    public func readSceneWithMortonOrdering(useParallel: Bool = true) throws -> [SplatScenePoint] {
        let points = try readScene()

        guard points.count > 1 else { return points }

        // Use parallel version for large datasets
        if useParallel && points.count > 100_000 {
            return MortonOrder.reorderParallel(points)
        } else {
            return MortonOrder.reorder(points)
        }
    }
}

/// Helper class to collect points from a delegate-based reader
private class PointCollector: SplatSceneReaderDelegate {
    var points: [SplatScenePoint] = []
    var error: Error?
    private let semaphore = DispatchSemaphore(value: 0)
    private var completed = false

    func didStartReading(withPointCount pointCount: UInt32?) {
        if let count = pointCount {
            points.reserveCapacity(Int(count))
        }
    }

    func didRead(points: [SplatScenePoint]) {
        self.points.append(contentsOf: points)
    }

    func didFinishReading() {
        completed = true
        semaphore.signal()
    }

    func didFailReading(withError error: Error?) {
        self.error = error
        completed = true
        semaphore.signal()
    }

    /// Wait for the reading to complete with a timeout
    func waitForCompletion(timeout: TimeInterval) throws {
        // If already completed (synchronous read), return immediately
        if completed { return }

        let result = semaphore.wait(timeout: .now() + timeout)
        if result == .timedOut {
            throw SplatSceneReaderError.timeout
        }
    }
}
