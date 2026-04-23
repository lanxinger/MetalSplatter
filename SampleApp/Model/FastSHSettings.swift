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

    /// Maximum number of unique SH coefficient sets (palette size)
    @Published public var maxPaletteSize: Int = 65536

    /// Minimum camera direction delta required before Fast SH recomputes view-dependent lighting.
    @Published public var shDirectionEpsilon: Float = 0.001

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

    /// Analyze a model and update informational status for the current Fast SH settings.
    public func analyzeAndConfigure(splatCount: Int, uniqueShSets: Int, shDegree: Int) {
        self.modelSplatCount = splatCount
        self.paletteSize = uniqueShSets
        self.shDegree = shDegree
        self.hasShData = uniqueShSets > 0 && shDegree >= 0

        if hasShData {
            updatePerformanceEstimate()
            isActive = enabled
        } else {
            enabled = false
            isActive = false
            performanceGain = ""
        }
    }

    /// Update performance gain estimate
    private func updatePerformanceEstimate() {
        guard hasShData && enabled else {
            performanceGain = ""
            return
        }

        let compressionRatio = Float(paletteSize) / Float(max(1, modelSplatCount))
        let memoryReduction = Int((1.0 - compressionRatio) * 100)
        let thresholdGain: Int
        switch shDirectionEpsilon {
        case ..<0.001:
            thresholdGain = 12
        case ..<0.0025:
            thresholdGain = 18
        default:
            thresholdGain = 24
        }
        let frameRateGain = thresholdGain + Int(Float(shDegree) * 4.0)

        if memoryReduction > 50 {
            performanceGain = "~\(frameRateGain)% faster, \(memoryReduction)% less SH palette memory"
        } else {
            performanceGain = "~\(frameRateGain)% performance gain"
        }
    }

    // MARK: - SOGS File Detection

    /// Apply SOGS-specific optimizations
    public func configureForSOGS() {
        enabled = true
        maxPaletteSize = 65536
        shDirectionEpsilon = 0.001
    }

    // MARK: - Settings Validation

    /// Validate and clamp settings to safe ranges
    public func validateSettings() {
        maxPaletteSize = max(1024, min(maxPaletteSize, 131072))
        shDirectionEpsilon = max(0.0001, min(shDirectionEpsilon, 0.01))
        isActive = enabled && hasShData
        updatePerformanceEstimate()
    }

    // MARK: - Reset to Defaults

    /// Reset all settings to default values
    public func resetToDefaults() {
        enabled = true
        maxPaletteSize = 65536
        shDirectionEpsilon = 0.001
        validateSettings()
    }
}

// MARK: - Settings Presets

public extension FastSHSettings {

    /// Performance-focused preset
    func applyPerformancePreset() {
        enabled = true
        maxPaletteSize = 16384
        shDirectionEpsilon = 0.004
        validateSettings()
    }

    /// Quality-focused preset
    func applyQualityPreset() {
        enabled = true
        maxPaletteSize = 131072
        shDirectionEpsilon = 0.0005
        validateSettings()
    }

    /// Balanced preset
    func applyBalancedPreset() {
        enabled = true
        maxPaletteSize = 65536
        shDirectionEpsilon = 0.0015
        validateSettings()
    }
}
