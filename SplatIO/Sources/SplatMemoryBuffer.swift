import Foundation

public struct SplatMemoryBuffer {
    /// Internal delegate for accumulating splat points during async reading.
    ///
    /// Thread Safety:
    /// - Point accumulation is protected by an NSLock for defensive thread safety.
    /// - The continuation is resumed exactly once, guarded by `didResume` flag.
    /// - Marked as `@unchecked Sendable` because thread safety is enforced via NSLock.
    private final class BufferReader: SplatSceneReaderDelegate, @unchecked Sendable {
        enum Error: Swift.Error {
            case unknown
        }

        private let continuation: CheckedContinuation<[SplatScenePoint], Swift.Error>
        private var points: [SplatScenePoint] = []
        private var didResume = false
        private let lock = NSLock()

        public init(continuation: CheckedContinuation<[SplatScenePoint], Swift.Error>) {
            self.continuation = continuation
        }

        public func didStartReading(withPointCount pointCount: UInt32?) {}

        public func didRead(points: [SplatIO.SplatScenePoint]) {
            lock.lock()
            defer { lock.unlock() }
            guard !didResume else { return }
            self.points.append(contentsOf: points)
        }

        public func didFinishReading() {
            lock.lock()
            guard !didResume else {
                lock.unlock()
                return
            }
            didResume = true
            let result = points
            lock.unlock()
            continuation.resume(returning: result)
        }

        public func didFailReading(withError error: Swift.Error?) {
            lock.lock()
            guard !didResume else {
                lock.unlock()
                return
            }
            didResume = true
            lock.unlock()
            continuation.resume(throwing: error ?? BufferReader.Error.unknown)
        }
    }

    public var points: [SplatScenePoint] = []

    public init() {}

    /** Replace the content of points with the content read from the given SplatSceneReader. */
    mutating public func read(from reader: SplatSceneReader) async throws {
        // Try to use the direct readScene() method first if available
        do {
            points = try reader.readScene()
        } catch {
            // Fall back to the delegate-based approach if direct reading fails
            points = try await withCheckedThrowingContinuation { continuation in
                reader.read(to: BufferReader(continuation: continuation))
            }
        }
    }
}
