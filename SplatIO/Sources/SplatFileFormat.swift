import Foundation

public enum SplatFileFormat {
    case ply
    case dotSplat
    case spz

    public init?(for url: URL) {
        switch url.pathExtension.lowercased() {
        case "ply": self = .ply
        case "splat": self = .dotSplat
        case "spz", "spz.gz", "gz" where url.deletingPathExtension().pathExtension.lowercased() == "spz":
            self = .spz
        default: return nil
        }
    }
}
