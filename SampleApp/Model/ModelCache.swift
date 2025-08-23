import Foundation
import MetalSplatter
import SplatIO
import os

#if canImport(UIKit)
import UIKit
#endif

/// Shared cache for loaded models to avoid duplicate loading between renderers
@MainActor
final class ModelCache {
    static let shared = ModelCache()
    
    private var cache: [ModelIdentifier: CachedModel] = [:]
    private let log = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "com.metalsplatter.sampleapp",
        category: "ModelCache"
    )
    
    private init() {
        // Setup memory pressure monitoring (iOS only)
        #if canImport(UIKit)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.clearCache()
            }
        }
        #endif
    }
    
    /// Get cached model or load if not cached
    func getModel(_ identifier: ModelIdentifier) async throws -> CachedModel {
        if var cached = cache[identifier] {
            cached.lastAccessed = Date()
            cache[identifier] = cached // Update the cache with new timestamp
            log.info("Using cached model: \(identifier)")
            return cached
        }
        
        log.info("Loading new model: \(identifier)")
        let model = try await loadModel(identifier)
        cache[identifier] = model
        
        // Clean old entries if cache is getting large
        if cache.count > 3 {
            cleanOldEntries()
        }
        
        return model
    }
    
    /// Load model from file
    private func loadModel(_ identifier: ModelIdentifier) async throws -> CachedModel {
        switch identifier {
        case .gaussianSplat(let url):
            let reader = try AutodetectSceneReader(url)
            let points = try reader.readScene()
            
            return CachedModel(
                identifier: identifier,
                points: points,
                lastAccessed: Date()
            )
            
        case .sampleBox:
            // Create empty points for sample box
            return CachedModel(
                identifier: identifier,
                points: [],
                lastAccessed: Date()
            )
        }
    }
    
    /// Clean old cache entries
    private func cleanOldEntries() {
        let cutoff = Date().addingTimeInterval(-300) // 5 minutes
        let oldKeys = cache.compactMap { key, model in
            model.lastAccessed < cutoff ? key : nil
        }
        
        for key in oldKeys {
            cache.removeValue(forKey: key)
            log.info("Removed old cached model: \(key)")
        }
    }
    
    /// Clear all cached models
    func clearCache() {
        let count = cache.count
        cache.removeAll()
        if count > 0 {
            log.info("Cleared \(count) cached models due to memory pressure")
        }
    }
    
    /// Remove specific model from cache
    func invalidate(_ identifier: ModelIdentifier) {
        if cache.removeValue(forKey: identifier) != nil {
            log.info("Invalidated cached model: \(identifier)")
        }
    }
}

/// Cached model data
struct CachedModel {
    let identifier: ModelIdentifier
    let points: [SplatScenePoint]
    var lastAccessed: Date
}