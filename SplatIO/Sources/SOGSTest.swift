import Foundation

// MARK: - SOGS Test/Example Usage

/// Example showing how to load SOGS compressed splat data
public struct SOGSTest {
    
    public static func loadTestSOGSData() {
        // Example usage with your test data
        let sogsTestDir = URL(fileURLWithPath: "sogs_test") // Adjust path as needed
        let metaURL = sogsTestDir.appendingPathComponent("meta.json")
        
        loadSOGSData(from: metaURL)
    }
    
    public static func loadSOGSData(from metaURL: URL) {
        
        do {
            print("=== SOGS Test: Loading compressed splat data ===")
            
            // Create the SOGS reader
            let reader = try SplatSOGSSceneReader(metaURL)
            
            // Load the scene
            let startTime = Date()
            let points = try reader.readScene()
            let loadTime = Date().timeIntervalSince(startTime)
            
            print("Successfully loaded \(points.count) splats in \(String(format: "%.2f", loadTime)) seconds")
            
            // Print some statistics
            printSOGSStatistics(points)
            
        } catch {
            print("Failed to load SOGS data: \(error)")
        }
    }
    
    private static func printSOGSStatistics(_ points: [SplatScenePoint]) {
        guard !points.isEmpty else { return }
        
        print("\n=== SOGS Statistics ===")
        print("Total splats: \(points.count)")
        
        // Calculate bounding box
        var minPos = points[0].position
        var maxPos = points[0].position
        
        for point in points {
            minPos = SIMD3<Float>(
                min(minPos.x, point.position.x),
                min(minPos.y, point.position.y),
                min(minPos.z, point.position.z)
            )
            maxPos = SIMD3<Float>(
                max(maxPos.x, point.position.x),
                max(maxPos.y, point.position.y),
                max(maxPos.z, point.position.z)
            )
        }
        
        let size = maxPos - minPos
        print("Bounding box: min(\(minPos.x), \(minPos.y), \(minPos.z)) max(\(maxPos.x), \(maxPos.y), \(maxPos.z))")
        print("Scene size: \(size.x) x \(size.y) x \(size.z)")
        
        // Check color format
        let firstPoint = points[0]
        switch firstPoint.color {
        case .sphericalHarmonic(let sh):
            print("Color format: Spherical Harmonics with \(sh.count) coefficients")
        case .linearFloat:
            print("Color format: Linear Float")
        case .linearFloat256:
            print("Color format: Linear Float256")
        case .linearUInt8:
            print("Color format: Linear UInt8")
        }
        
        print("Opacity format: \(type(of: firstPoint.opacity))")
        print("Scale format: \(type(of: firstPoint.scale))")
        print("======================\n")
    }
}

// MARK: - SOGS Loading Helper Extension

public extension AutodetectSceneReader {
    /// Convenience method to load SOGS data from a directory containing meta.json
    static func loadSOGS(from directory: URL) throws -> [SplatScenePoint] {
        let metaURL = directory.appendingPathComponent("meta.json")
        let reader = try AutodetectSceneReader(metaURL)
        return try reader.readScene()
    }
    
    /// Convenience method to check if a directory contains SOGS data
    static func containsSOGSData(_ directory: URL) -> Bool {
        let metaURL = directory.appendingPathComponent("meta.json")
        return SplatFileFormat(for: metaURL) == .sogs
    }
} 