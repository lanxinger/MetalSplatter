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
                // Check the file extension to determine which SOGS reader to use
                let fileExtension = url.pathExtension.lowercased()
                if fileExtension == "sog" {
                    // SOGS v2 bundled format - use the v2 reader directly
                    reader = try SplatSOGSSceneReaderV2(url)
                    print("AutodetectSceneReader: Successfully created SplatSOGSSceneReaderV2 for .sog file")
                } else if fileExtension == "zip" {
                    // Legacy ZIP format - use ZIP reader
                    reader = try SplatSOGSZipReader(url)
                    print("AutodetectSceneReader: Successfully created SplatSOGSZipReader")
                } else {
                    // Standard SOGS format (folder/meta.json) - use main reader with auto-detection
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
        case .gltf, .glb:
            print("AutodetectSceneReader: Loading as glTF Gaussian splats")
            do {
                reader = try GltfGaussianSplatSceneReader(url)
                print("AutodetectSceneReader: Successfully created GltfGaussianSplatSceneReader")
            } catch {
                print("AutodetectSceneReader: Failed to create GltfGaussianSplatSceneReader: \(error)")
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
