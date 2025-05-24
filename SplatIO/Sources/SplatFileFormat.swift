import Foundation

public enum SplatFileFormat {
    case ply
    case dotSplat
    case spz
    case sogs

    public init?(for url: URL) {
        switch url.pathExtension.lowercased() {
        case "ply": self = .ply
        case "splat": self = .dotSplat
        case "spz":
            self = .spz
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
        default: return nil
        }
    }
}
