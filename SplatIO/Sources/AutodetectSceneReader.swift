import Foundation

public class AutodetectSceneReader: SplatSceneReader {
    public enum Error: Swift.Error {
        case cannotDetermineFormat
    }

    /// Whether the loaded PLY was trained with Brush mip splatting (COV_BLUR = 0.1).
    /// Check this after init to set `SplatRenderer.covarianceBlur` accordingly.
    public private(set) var isMipSplatting: Bool = false

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

        // Detect Brush mip splatting render mode from file metadata.
        // Scans the first 8 KB for the marker string — works across all formats
        // since conversion tools may preserve the original PLY comment/metadata.
        isMipSplatting = Self.detectMipSplatting(url: url)
    }

    /// Scan the first bytes of any splat file for Brush's `SplatRenderMode mip` marker.
    /// In PLY this lives in the ASCII header; converters may preserve it in other formats.
    private static func detectMipSplatting(url: URL) -> Bool {
        let scanSize = 8192
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: scanSize), !data.isEmpty else { return false }
        guard let text = String(data: data, encoding: .ascii) ?? String(data: data, encoding: .utf8) else {
            return false
        }
        let isMip = text.contains("SplatRenderMode mip")
        if isMip {
            print("AutodetectSceneReader: Detected Brush mip splatting render mode")
        }
        return isMip
    }

    public func readScene() throws -> [SplatScenePoint] {
        return try reader.readScene()
    }
    
    public func read(to delegate: SplatSceneReaderDelegate) {
        reader.read(to: delegate)
    }
}
