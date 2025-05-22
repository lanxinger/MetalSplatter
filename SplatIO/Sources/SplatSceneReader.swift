import Foundation

public protocol SplatSceneReaderDelegate: AnyObject {
    func didStartReading(withPointCount pointCount: UInt32?)
    func didRead(points: [SplatScenePoint])
    func didFinishReading()
    func didFailReading(withError error: Error?)
}

public protocol SplatSceneReader {
    /// Read a scene directly into an array of points
    func readScene() throws -> [SplatScenePoint]
    
    /// For backward compatibility - implementations can be added via extension
    func read(to delegate: SplatSceneReaderDelegate)
}
