import Foundation
import ZIPFoundation

public enum SplatFileFormat {
    case ply
    case dotSplat
    case spz
    case sogs
    case spx
    case gltf
    case glb

    public init?(for url: URL) {
        switch url.pathExtension.lowercased() {
        case "ply": self = .ply
        case "splat": self = .dotSplat
        case "spz":
            self = .spz
        case "spx":
            self = .spx
        case "gltf":
            self = .gltf
        case "glb":
            self = .glb
        case "gz":
            // Check if this is a compressed SPZ file (e.g., file.spz.gz)
            if url.deletingPathExtension().pathExtension.lowercased() == "spz" {
                self = .spz
            } else {
                return nil
            }
        case "json": 
            // Check if this is a SOGS meta.json file by looking for SOGS-specific keys
            if let data = try? Data(contentsOf: url),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               json["means"] != nil && json["scales"] != nil && json["quats"] != nil {
                self = .sogs
            } else {
                return nil
            }
        case "sog":
            // SOGS v2 bundled format - single .sog file (ZIP archive)
            self = .sogs
        case "zip":
            // Check if this is a SOGS ZIP file by examining its contents
            if SplatFileFormat.isSOGSZipFile(url: url) {
                self = .sogs
            } else {
                return nil
            }
        default: return nil
        }
    }
    
    private static func isSOGSZipFile(url: URL) -> Bool {
        // Check if the ZIP file contains SOGS files
        guard let archive = Archive(url: url, accessMode: .read) else { return false }
        
        var hasMetaJson = false
        var hasWebPFiles = false
        
        for entry in archive {
            let filename = entry.path.lowercased()
            if filename.hasSuffix("meta.json") {
                hasMetaJson = true
            } else if filename.hasSuffix(".webp") {
                hasWebPFiles = true
            }
            
            if hasMetaJson && hasWebPFiles {
                return true
            }
        }
        
        return false
    }
}
