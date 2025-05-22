import Foundation

/// Default implementation of `readScene()` for existing readers that use the delegate pattern
extension SplatSceneReader {
    public func readScene() throws -> [SplatScenePoint] {
        let collector = PointCollector()
        read(to: collector)
        
        if let error = collector.error {
            throw error
        }
        
        return collector.points
    }
}

/// Helper class to collect points from a delegate-based reader
private class PointCollector: SplatSceneReaderDelegate {
    var points: [SplatScenePoint] = []
    var error: Error?
    private let semaphore = DispatchSemaphore(value: 0)
    
    func didStartReading(withPointCount pointCount: UInt32?) {
        if let count = pointCount {
            points.reserveCapacity(Int(count))
        }
    }
    
    func didRead(points: [SplatScenePoint]) {
        self.points.append(contentsOf: points)
    }
    
    func didFinishReading() {
        semaphore.signal()
    }
    
    func didFailReading(withError error: Error?) {
        self.error = error
        semaphore.signal()
    }
    
    deinit {
        // Ensure semaphore is signaled if this object is deallocated
        semaphore.signal()
    }
}
