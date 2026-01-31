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
    /// - Parameters:
    ///   - identifier: The model identifier
    ///   - securityScopedURL: Optional URL that needs security-scoped access. If provided, the cache will manage the access lifecycle.
    ///   - hasSecurityScopedAccess: Whether startAccessingSecurityScopedResource() has already been called on the URL
    func getModel(_ identifier: ModelIdentifier, securityScopedURL: URL? = nil, hasSecurityScopedAccess: Bool = false) async throws -> CachedModel {
        if var cached = cache[identifier] {
            cached.lastAccessed = Date()
            cache[identifier] = cached // Update the cache with new timestamp
            log.info("Using cached model: \(identifier)")
            // If caller passed a security-scoped URL but we're using cache, release the caller's access
            // since the cached model already has its own access (or doesn't need it)
            if hasSecurityScopedAccess, let url = securityScopedURL {
                url.stopAccessingSecurityScopedResource()
                log.debug("Released redundant security-scoped access for cached model")
            }
            return cached
        }

        log.info("Loading new model: \(identifier)")
        let model = try await loadModel(identifier, securityScopedURL: securityScopedURL, hasSecurityScopedAccess: hasSecurityScopedAccess)
        cache[identifier] = model

        // Clean old entries if cache is getting large
        if cache.count > 3 {
            cleanOldEntries()
        }

        return model
    }
    
    /// Load model from file
    private func loadModel(_ identifier: ModelIdentifier, securityScopedURL: URL?, hasSecurityScopedAccess: Bool) async throws -> CachedModel {
        switch identifier {
        case .gaussianSplat(let url):
            let reader = try AutodetectSceneReader(url)
            let points = try reader.readScene()

            return CachedModel(
                identifier: identifier,
                points: points,
                lastAccessed: Date(),
                securityScopedURL: securityScopedURL,
                hasSecurityScopedAccess: hasSecurityScopedAccess
            )

        case .sampleBox:
            // Create empty points for sample box
            // Release any security-scoped access since we don't need it
            if hasSecurityScopedAccess, let url = securityScopedURL {
                url.stopAccessingSecurityScopedResource()
            }
            return CachedModel(
                identifier: identifier,
                points: [],
                lastAccessed: Date(),
                securityScopedURL: nil,
                hasSecurityScopedAccess: false
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
            if let model = cache.removeValue(forKey: key) {
                // Release security-scoped access when evicting from cache
                releaseSecurityAccess(for: model)
                log.info("Removed old cached model: \(key)")
            }
        }
    }

    /// Release security-scoped resource access for a model
    private func releaseSecurityAccess(for model: CachedModel) {
        if model.hasSecurityScopedAccess, let url = model.securityScopedURL {
            url.stopAccessingSecurityScopedResource()
            log.debug("Released security-scoped access for: \(url.lastPathComponent)")
        }
    }
    
    /// Clear all cached models
    func clearCache() {
        let count = cache.count
        // Release security-scoped access for all cached models before clearing
        for model in cache.values {
            releaseSecurityAccess(for: model)
        }
        cache.removeAll()
        if count > 0 {
            log.info("Cleared \(count) cached models due to memory pressure")
        }
    }
    
    /// Remove specific model from cache
    func invalidate(_ identifier: ModelIdentifier) {
        if let model = cache.removeValue(forKey: identifier) {
            releaseSecurityAccess(for: model)
            log.info("Invalidated cached model: \(identifier)")
        }
    }
}

/// Cached model data
struct CachedModel {
    let identifier: ModelIdentifier
    let points: [SplatScenePoint]
    var lastAccessed: Date
    /// Security-scoped URL that should be released when this model is evicted
    let securityScopedURL: URL?
    /// Whether security-scoped access was started for this URL
    let hasSecurityScopedAccess: Bool
}