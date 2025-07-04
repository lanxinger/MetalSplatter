import Foundation
import simd
import ZIPFoundation

public class SplatSOGSZipReader: SplatSceneReader {
    public enum SOGSZipError: Error {
        case notAZipFile
        case failedToExtract
        case missingMetaJson
        case invalidSOGSStructure
    }
    
    private let zipURL: URL
    private let extractedURL: URL
    private var actualReader: SplatSOGSSceneReader?
    
    public init(_ zipURL: URL) throws {
        self.zipURL = zipURL
        
        // Create a temporary directory for extraction
        let tempDir = FileManager.default.temporaryDirectory
        self.extractedURL = tempDir.appendingPathComponent("sogs_\(UUID().uuidString)")
        
        print("SplatSOGSZipReader: Initializing with ZIP: \(zipURL.path)")
        print("SplatSOGSZipReader: Will extract to: \(extractedURL.path)")
        
        // Validate it's a ZIP file
        guard zipURL.pathExtension.lowercased() == "zip" else {
            throw SOGSZipError.notAZipFile
        }
        
        // Extract the ZIP file
        try extractZipFile()
        
        // Find and validate meta.json
        let metaURL = try findMetaJson()
        
        // Create the actual SOGS reader with the extracted files
        self.actualReader = try SplatSOGSSceneReader(metaURL)
    }
    
    deinit {
        // Clean up extracted files
        cleanupExtractedFiles()
    }
    
    private func extractZipFile() throws {
        print("SplatSOGSZipReader: Extracting ZIP file...")
        
        // Create extraction directory
        try FileManager.default.createDirectory(at: extractedURL, withIntermediateDirectories: true)
        
        // Use ZIPFoundation to extract the archive
        guard let archive = Archive(url: zipURL, accessMode: .read) else {
            throw SOGSZipError.failedToExtract
        }
        
        for entry in archive {
            let entryURL = extractedURL.appendingPathComponent(entry.path)
            
            // Create intermediate directories if needed
            let entryDirectory = entryURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: entryDirectory, withIntermediateDirectories: true)
            
            // Extract the file
            _ = try archive.extract(entry, to: entryURL)
            print("SplatSOGSZipReader: Extracted: \(entry.path)")
        }
        
        print("SplatSOGSZipReader: Successfully extracted ZIP")
    }
    
    private func findMetaJson() throws -> URL {
        print("SplatSOGSZipReader: Looking for meta.json in extracted files...")
        
        let fileManager = FileManager.default
        let enumerator = fileManager.enumerator(at: extractedURL, 
                                               includingPropertiesForKeys: nil,
                                               options: [.skipsHiddenFiles])
        
        while let fileURL = enumerator?.nextObject() as? URL {
            if fileURL.lastPathComponent.lowercased() == "meta.json" {
                print("SplatSOGSZipReader: Found meta.json at: \(fileURL.path)")
                
                // Validate it's a SOGS meta.json
                if let data = try? Data(contentsOf: fileURL),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   json["means"] != nil && json["scales"] != nil && json["quats"] != nil {
                    return fileURL
                }
            }
        }
        
        throw SOGSZipError.missingMetaJson
    }
    
    private func cleanupExtractedFiles() {
        do {
            if FileManager.default.fileExists(atPath: extractedURL.path) {
                try FileManager.default.removeItem(at: extractedURL)
                print("SplatSOGSZipReader: Cleaned up extracted files")
            }
        } catch {
            print("SplatSOGSZipReader: Failed to cleanup extracted files: \(error)")
        }
    }
    
    public func readScene() throws -> [SplatScenePoint] {
        guard let reader = actualReader else {
            throw SOGSZipError.invalidSOGSStructure
        }
        return try reader.readScene()
    }
    
    public func read(to delegate: SplatSceneReaderDelegate) {
        guard let reader = actualReader else {
            delegate.didFailReading(withError: SOGSZipError.invalidSOGSStructure)
            return
        }
        reader.read(to: delegate)
    }
}

