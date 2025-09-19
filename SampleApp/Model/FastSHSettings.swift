import Foundation
import SwiftUI

/// Observable settings manager for Fast SH configuration
@MainActor
public class FastSHSettings: ObservableObject {
    
    // MARK: - Configuration Properties
    
    /// Enable fast SH evaluation (vs per-splat evaluation)
    @Published public var enabled: Bool = false {
        didSet {
            if !enabled {
                isActive = false
                performanceGain = ""
            }
        }
    }
    
    /// Update SH evaluation every N frames (1 = every frame)
    @Published public var updateFrequency: Int = 1
    
    /// Maximum number of unique SH coefficient sets (palette size)
    @Published public var maxPaletteSize: Int = 65536
    
    // MARK: - Status Properties
    
    /// Whether Fast SH is currently active and processing
    @Published public private(set) var isActive: Bool = false
    
    /// Current palette size being used
    @Published public private(set) var paletteSize: Int = 0
    
    /// SH degree of current model
    @Published public private(set) var shDegree: Int = 0
    
    /// Performance gain description
    @Published public private(set) var performanceGain: String = ""
    
    // MARK: - Internal tracking
    
    private var modelSplatCount: Int = 0
    private var hasShData: Bool = false
    
    public init() {}
    
    // MARK: - Model Analysis & Auto-Configuration
    
    /// Analyze a model and configure settings based on its characteristics
    public func analyzeAndConfigure(splatCount: Int, uniqueShSets: Int, shDegree: Int) {
        self.modelSplatCount = splatCount
        self.paletteSize = uniqueShSets
        self.shDegree = shDegree
        self.hasShData = uniqueShSets > 0 && shDegree >= 0
        
        // Auto-configure based on model characteristics
        if hasShData {
            applyRecommendedSettings()
            updatePerformanceEstimate()
            isActive = enabled
        } else {
            // No SH data - disable Fast SH
            enabled = false
            isActive = false
            performanceGain = ""
        }
    }
    
    /// Apply recommended settings based on model characteristics
    private func applyRecommendedSettings() {
        // Enable for models with SH data
        if !enabled && hasShData {
            enabled = true
        }
        
        // Adjust settings based on model size and complexity
        if modelSplatCount > 1_000_000 {
            // Large models - prioritize performance
            updateFrequency = 2
            maxPaletteSize = min(maxPaletteSize, 32768)
        } else if modelSplatCount > 100_000 {
            // Medium models - balanced approach
            updateFrequency = 1
            maxPaletteSize = min(maxPaletteSize, 65536)
        } else {
            // Small models - prioritize quality
            updateFrequency = 1
            maxPaletteSize = 131072
        }
        
        // Adjust based on palette size (compression efficiency)
        let compressionRatio = Float(paletteSize) / Float(modelSplatCount)
        if compressionRatio < 0.1 {
            // Highly compressible - can afford higher quality settings
            updateFrequency = 1
        }
    }
    
    /// Update performance gain estimate
    private func updatePerformanceEstimate() {
        guard hasShData && enabled else {
            performanceGain = ""
            return
        }
        
        // Calculate estimated performance improvement
        let compressionRatio = Float(paletteSize) / Float(max(1, modelSplatCount))
        let memoryReduction = Int((1.0 - compressionRatio) * 100)
        
        let frameRateGain: Int
        if updateFrequency == 1 {
            frameRateGain = 15 + Int(Float(shDegree) * 5.0)
        } else {
            frameRateGain = 20 + Int(Float(shDegree) * 7.0) + (updateFrequency - 1) * 5
        }
        
        if memoryReduction > 50 {
            performanceGain = "~\(frameRateGain)% faster, \(memoryReduction)% less memory"
        } else {
            performanceGain = "~\(frameRateGain)% performance gain"
        }
    }
    
    // MARK: - SOGS File Detection
    
    /// Apply SOGS-specific optimizations
    public func configureForSOGS() {
        // SOGS files typically have excellent SH compression
        enabled = true
        updateFrequency = 1
        maxPaletteSize = 65536
    }
    
    // MARK: - Settings Validation
    
    /// Validate and clamp settings to safe ranges
    public func validateSettings() {
        updateFrequency = max(1, min(updateFrequency, 10))
        maxPaletteSize = max(1024, min(maxPaletteSize, 131072))
        
        // Update active status
        isActive = enabled && hasShData
        
        // Update performance estimate
        updatePerformanceEstimate()
    }
    
    // MARK: - Reset to Defaults
    
    /// Reset all settings to default values
    public func resetToDefaults() {
        enabled = true
        updateFrequency = 1
        maxPaletteSize = 65536
        validateSettings()
    }
}

// MARK: - Settings Presets

public extension FastSHSettings {
    
    /// Performance-focused preset
    func applyPerformancePreset() {
        enabled = true
        updateFrequency = 3
        maxPaletteSize = 16384
        validateSettings()
    }
    
    /// Quality-focused preset
    func applyQualityPreset() {
        enabled = true
        updateFrequency = 1
        maxPaletteSize = 131072
        validateSettings()
    }
    
    /// Balanced preset
    func applyBalancedPreset() {
        enabled = true
        updateFrequency = 1
        maxPaletteSize = 65536
        validateSettings()
    }
}
