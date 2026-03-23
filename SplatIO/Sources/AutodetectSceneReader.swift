import Foundation

public class AutodetectSceneReader: SplatSceneReader {
    public enum Error: Swift.Error {
        case cannotDetermineFormat
    }

    public enum RenderMode: Sendable, Equatable {
        case standard
        case mip
    }

    /// Brush metadata-derived render mode.
    public private(set) var renderMode: RenderMode = .standard

    /// Whether the loaded asset should use Brush MIP rendering semantics.
    public var isMipSplatting: Bool { renderMode == .mip }

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

        // Detect Brush render mode from file metadata.
        // Scans the first 8 KB for the marker string since conversion tools may preserve
        // the original PLY comment/metadata across container formats.
        renderMode = Self.detectRenderMode(url: url)
    }

    /// Scan for Brush's `SplatRenderMode: ...` marker.
    /// For binary PLYs, only decode the ASCII header bytes up to `end_header` so
    /// binary payload data cannot break UTF-8/ASCII decoding.
    private static func detectRenderMode(url: URL) -> RenderMode {
        let scanSize = 8192
        guard let handle = try? FileHandle(forReadingFrom: url) else { return .standard }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: scanSize), !data.isEmpty else { return .standard }

        let headerData: Data
        if let range = data.range(of: Data("end_header".utf8)) {
            let upperBound = min(data.count, range.upperBound + 1)
            headerData = data.subdata(in: 0..<upperBound)
        } else {
            headerData = data
        }

        guard let text = String(data: headerData, encoding: .ascii) ?? String(data: headerData, encoding: .utf8) else {
            return .standard
        }

        // Brush currently exports `SplatRenderMode: mip`; accept the legacy marker
        // without the colon as well to remain permissive with converted assets.
        if text.contains("SplatRenderMode: mip") || text.contains("SplatRenderMode mip") {
            print("AutodetectSceneReader: Detected Brush mip render mode")
            return .mip
        }
        return .standard
    }

    public func readScene() throws -> [SplatScenePoint] {
        return try reader.readScene()
    }
    
    public func read(to delegate: SplatSceneReaderDelegate) {
        reader.read(to: delegate)
    }
}
