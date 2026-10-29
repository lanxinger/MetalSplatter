import Foundation

public class AutodetectSceneReader: SplatSceneReader {
    public enum Error: Swift.Error {
        case cannotDetermineFormat
    }

    public enum RenderMode: Sendable, Equatable {
        case standard
        case mip
    }

    /// Render mode from the file's metadata: PLY `SplatRenderMode` (Brush) and `antialiased`
    /// (Postshot) comments, the SPZ antialiased flag, SOG `model`/`antialias`, and the glTF root
    /// `extras.SplatRenderMode`.
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
        var detectedRenderMode: RenderMode?
        print("AutodetectSceneReader: Detected format: \(String(describing: format))")
        
        switch format {
        case .ply:
            print("AutodetectSceneReader: Loading as PLY")
            reader = try SplatPLYSceneReader(url)
            detectedRenderMode = Self.plyRenderMode(url: url)
        case .dotSplat:
            print("AutodetectSceneReader: Loading as dotSplat")
            reader = try DotSplatSceneReader(url)
        case .spz: 
            print("AutodetectSceneReader: Loading as SPZ")
            do {
                let spzReader = try SPZSceneReader(contentsOf: url)
                reader = spzReader
                detectedRenderMode = spzReader.isAntialiased ? .mip : .standard
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
                    let sogReader = try SplatSOGSSceneReaderV2(url)
                    reader = sogReader
                    detectedRenderMode = sogReader.isAntialiased ? .mip : .standard
                    print("AutodetectSceneReader: Successfully created SplatSOGSSceneReaderV2 for .sog file")
                } else if fileExtension == "zip" {
                    // Legacy ZIP format - use ZIP reader
                    let zipReader = try SplatSOGSZipReader(url)
                    reader = zipReader
                    detectedRenderMode = zipReader.isAntialiased ? .mip : .standard
                    print("AutodetectSceneReader: Successfully created SplatSOGSZipReader")
                } else {
                    // Standard SOGS format (folder/meta.json) - use main reader with auto-detection
                    let folderReader = try SplatSOGSSceneReader(url)
                    reader = folderReader
                    detectedRenderMode = folderReader.isAntialiased ? .mip : .standard
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
                let gltfReader = try GltfGaussianSplatSceneReader(url)
                reader = gltfReader
                detectedRenderMode = gltfReader.isAntialiased ? .mip : .standard
                print("AutodetectSceneReader: Successfully created GltfGaussianSplatSceneReader")
            } catch {
                print("AutodetectSceneReader: Failed to create GltfGaussianSplatSceneReader: \(error)")
                throw Error.cannotDetermineFormat
            }
        case .none:
            print("AutodetectSceneReader: Unknown format")
            throw Error.cannotDetermineFormat
        }

        // Formats without their own render mode metadata (SPX, .splat) may still carry a
        // converted PLY comment in their header bytes.
        renderMode = detectedRenderMode ?? Self.scannedRenderMode(url: url)
    }

    /// The render mode set by PLY header comments: Brush's `SplatRenderMode: default | mip | 2dgs`
    /// and Postshot's `antialiased 0 | 1`, case-insensitive. The last recognized comment wins.
    static func renderMode(plyHeader text: String) -> RenderMode {
        var mode = RenderMode.standard
        text.enumerateLines { line, _ in
            var words = line.split(whereSeparator: \.isWhitespace).map { $0.lowercased() }
            if words.first == "comment" { words.removeFirst() }
            guard words.count >= 2 else { return }
            switch (words[0], words[1]) {
            case ("splatrendermode:", "mip"), ("splatrendermode", "mip"), ("antialiased", "1"):
                mode = .mip
            case ("splatrendermode:", "default"), ("splatrendermode", "default"),
                 ("splatrendermode:", "2dgs"), ("splatrendermode", "2dgs"), ("antialiased", "0"):
                mode = .standard
            default:
                break
            }
        }
        return mode
    }

    /// Reads the PLY header (up to `end_header`) for its render mode comments.
    private static func plyRenderMode(url: URL) -> RenderMode {
        guard let text = headerText(url: url, maxBytes: 1024 * 1024) else { return .standard }
        return renderMode(plyHeader: text)
    }

    private static func scannedRenderMode(url: URL) -> RenderMode {
        guard let text = headerText(url: url, maxBytes: 8192) else { return .standard }
        return renderMode(plyHeader: text)
    }

    /// The file's leading bytes up to `end_header`, if present, decoded leniently so binary data
    /// or non-UTF-8 comments can't hide the text around them.
    private static func headerText(url: URL, maxBytes: Int) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var data = Data()
        while data.count < maxBytes {
            guard let chunk = try? handle.read(upToCount: min(65536, maxBytes - data.count)), !chunk.isEmpty else { break }
            data.append(chunk)
            if let range = data.range(of: Data("end_header".utf8)) {
                data = data.subdata(in: data.startIndex..<range.upperBound)
                break
            }
        }
        guard !data.isEmpty else { return nil }
        return String(decoding: data, as: UTF8.self)
    }

    public func readScene() throws -> [SplatScenePoint] {
        return try reader.readScene()
    }
    
    public func read(to delegate: SplatSceneReaderDelegate) {
        reader.read(to: delegate)
    }
}
