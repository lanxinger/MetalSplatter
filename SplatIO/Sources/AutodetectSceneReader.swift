import Foundation

public class AutodetectSceneReader: SplatSceneReader {
    public enum Error: Swift.Error {
        case cannotDetermineFormat
    }

    private let reader: SplatSceneReader

    /// Initialize with default settings
    public convenience init(_ url: URL) throws {
        try self.init(url, useOptimizedSOGS: true)
    }
    
    /// Initialize with option to use optimized SOGS reader
    /// - Parameters:
    ///   - url: File URL to read
    ///   - useOptimizedSOGS: Whether to use the optimized SOGS reader (default: true)
    public init(_ url: URL, useOptimizedSOGS: Bool) throws {
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
            print("AutodetectSceneReader: Loading as SOGS (optimized: \(useOptimizedSOGS))")
            do {
                // Check if it's a ZIP file or regular SOGS folder
                if url.pathExtension.lowercased() == "zip" {
                    reader = try SplatSOGSZipReader(url)
                    print("AutodetectSceneReader: Successfully created SplatSOGSZipReader")
                } else {
                    // Use the standard SplatSOGSSceneReader (now with optimizations built-in)
                    reader = try SplatSOGSSceneReader(url)
                    print("AutodetectSceneReader: Successfully created SplatSOGSSceneReader")
                }
            } catch {
                print("AutodetectSceneReader: Failed to create SOGS reader: \(error)")
                throw Error.cannotDetermineFormat
            }
        case .spx:
            print("AutodetectSceneReader: Loading as SPX")
            do {
                reader = try SPXSceneReader(contentsOf: url)
                print("AutodetectSceneReader: Successfully created SPXSceneReader")
            } catch {
                print("AutodetectSceneReader: Failed to create SPXSceneReader: \(error)")
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
