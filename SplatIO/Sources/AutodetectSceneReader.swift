import Foundation

public class AutodetectSceneReader: SplatSceneReader {
    public enum Error: Swift.Error {
        case cannotDetermineFormat
    }

    private let reader: SplatSceneReader

    public init(_ url: URL) throws {
        print("AutodetectSceneReader: Trying to load file: \(url.path)")
        print("AutodetectSceneReader: File extension: \(url.pathExtension)")
        
        let format = SplatFileFormat(for: url)
        print("AutodetectSceneReader: Detected format: \(String(describing: format))")
        
        switch format {
        case .ply:
            print("AutodetectSceneReader: Loading as PLY")
            reader = try SplatPLYSceneReader(url)
        case .dotSplat:
            print("AutodetectSceneReader: Loading as dotSplat")
            reader = try DotSplatSceneReader(url)
        case .spz: 
            print("AutodetectSceneReader: Loading as SPZ")
            do {
                reader = try SPZSceneReader(contentsOf: url)
                print("AutodetectSceneReader: Successfully created SPZSceneReader")
            } catch {
                print("AutodetectSceneReader: Failed to create SPZSceneReader: \(error)")
                throw Error.cannotDetermineFormat
            }
        case .sogs:
            print("AutodetectSceneReader: Loading as SOGS")
            do {
                reader = try SplatSOGSSceneReader(url)
                print("AutodetectSceneReader: Successfully created SplatSOGSSceneReader")
            } catch {
                print("AutodetectSceneReader: Failed to create SplatSOGSSceneReader: \(error)")
                throw Error.cannotDetermineFormat
            }
        case .none:
            print("AutodetectSceneReader: Unknown format")
            throw Error.cannotDetermineFormat
        }
    }

    public func readScene() throws -> [SplatScenePoint] {
        return try reader.readScene()
    }
    
    public func read(to delegate: SplatSceneReaderDelegate) {
        reader.read(to: delegate)
    }
}
