import Foundation
import Metal
import simd
import os
import SplatIO

/// Actor that manages asynchronous loading and unloading of octree nodes.
/// Implements memory budget enforcement and priority-based loading.
public actor StreamingLODManager {

    private static let log = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "MetalSplatter",
        category: "StreamingLODManager"
    )

    /// The octree being managed
    private let octree: SplatOctree

    /// Metal device for buffer allocation
    private let device: MTLDevice

    /// Memory budget for loaded splats (bytes)
    public var memoryBudget: Int {
        get { octree.memoryBudget }
        set { octree.memoryBudget = newValue }
    }

    /// Maximum concurrent load operations
    public var maxConcurrentLoads: Int = 4

    /// Currently loading node IDs
    private var loadingNodes: Set<String> = []

    /// Queue of nodes waiting to load
    private var loadQueue: [String] = []

    /// Callback when a node finishes loading
    public var onNodeLoaded: ((String, Int) -> Void)?

    /// Callback when a node is unloaded
    public var onNodeUnloaded: ((String) -> Void)?

    /// Whether the manager is actively streaming
    public private(set) var isStreaming: Bool = false

    /// Base URL for loading splat data files
    public var baseURL: URL?

    public init(octree: SplatOctree, device: MTLDevice) {
        self.octree = octree
        self.device = device
    }

    /// Updates the streaming state based on current camera view.
    /// Call this once per frame after updating octree visibility.
    ///
    /// - Parameters:
    ///   - viewProjectionMatrix: Combined view-projection matrix
    ///   - cameraPosition: Camera world position
    ///   - screenHeight: Screen height in pixels
    public func update(
        viewProjectionMatrix: simd_float4x4,
        cameraPosition: SIMD3<Float>,
        screenHeight: Float
    ) async {
        // Update octree visibility
        octree.updateVisibility(
            viewProjectionMatrix: viewProjectionMatrix,
            cameraPosition: cameraPosition,
            screenHeight: screenHeight
        )

        // Check memory pressure and unload if needed
        await enforceMemoryBudget()

        // Queue new loads
        await processLoadQueue()
    }

    /// Enforces memory budget by unloading least important nodes
    private func enforceMemoryBudget() async {
        let stats = octree.getStatistics()

        // If over budget, unload until under
        while octree.loadedMemoryBytes > memoryBudget {
            let candidates = octree.getUnloadCandidates()
            guard let nodeID = candidates.first else {
                Self.log.warning("Over memory budget but no unload candidates")
                break
            }

            await unloadNode(nodeID: nodeID)
        }

        if stats.budgetUtilization > 0.9 {
            Self.log.debug("Memory budget at \(String(format: "%.1f", stats.budgetUtilization * 100))%")
        }
    }

    /// Processes the load queue, starting new loads as capacity allows
    private func processLoadQueue() async {
        // Get nodes that need loading
        let pendingLoads = octree.getLoadQueue()

        // Filter out already loading nodes
        let newLoads = pendingLoads.filter { !loadingNodes.contains($0) }

        // Start new loads up to capacity
        // Guard against negative capacity if loadingNodes.count exceeds maxConcurrentLoads
        // (can happen if maxConcurrentLoads is lowered while loads are in flight)
        let availableCapacity = max(0, maxConcurrentLoads - loadingNodes.count)
        for nodeID in newLoads.prefix(availableCapacity) {
            await startLoading(nodeID: nodeID)
        }
    }

    /// Starts loading a node asynchronously
    private func startLoading(nodeID: String) async {
        guard !loadingNodes.contains(nodeID) else { return }

        loadingNodes.insert(nodeID)
        isStreaming = true

        Task {
            do {
                let memoryUsed = try await loadNodeData(nodeID: nodeID)
                await completeLoading(nodeID: nodeID, success: true, memoryBytes: memoryUsed)
            } catch {
                Self.log.error("Failed to load node \(nodeID): \(error)")
                await completeLoading(nodeID: nodeID, success: false, memoryBytes: 0)
            }
        }
    }

    /// Loads splat data for a node
    private func loadNodeData(nodeID: String) async throws -> Int {
        guard let node = octree.scene.nodes[nodeID] else {
            throw StreamingError.nodeNotFound(nodeID)
        }

        // Find the LOD level to load
        // For now, load the finest available LOD
        guard let lod = node.lodLevels.first else {
            throw StreamingError.noLODData(nodeID)
        }

        if let url = lod.resourceURL {
            // Load from external file
            return try await loadFromURL(url: url, splatCount: lod.splatCount)
        } else if let range = lod.splatRange {
            // Data is already inline - just mark as loaded
            let memoryEstimate = range.count * MemoryLayout<Float>.stride * 20  // ~20 floats per splat estimate
            return memoryEstimate
        } else {
            throw StreamingError.noLODData(nodeID)
        }
    }

    /// Loads splat data from a URL
    private func loadFromURL(url: URL, splatCount: Int) async throws -> Int {
        // Resolve relative URLs against base URL if needed
        let resolvedURL: URL
        if url.isFileURL || url.scheme != nil {
            resolvedURL = url
        } else if let base = baseURL {
            resolvedURL = base.appendingPathComponent(url.path)
        } else {
            resolvedURL = url
        }

        // Load data
        let data = try Data(contentsOf: resolvedURL)

        // Estimate memory usage
        let memoryUsed = data.count

        Self.log.debug("Loaded \(splatCount) splats from \(resolvedURL.lastPathComponent), \(memoryUsed / 1024)KB")

        return memoryUsed
    }

    /// Completes a loading operation
    private func completeLoading(nodeID: String, success: Bool, memoryBytes: Int) async {
        loadingNodes.remove(nodeID)

        if success {
            // Get the LOD level that was loaded
            let lodLevel = octree.scene.nodes[nodeID]?.lodLevels.first?.level ?? 0
            octree.markAsLoaded(nodeID: nodeID, lodLevel: lodLevel, memoryBytes: memoryBytes)
            onNodeLoaded?(nodeID, lodLevel)
            Self.log.debug("Node \(nodeID) loaded, LOD \(lodLevel)")
        }

        isStreaming = !loadingNodes.isEmpty
    }

    /// Unloads a node to free memory
    private func unloadNode(nodeID: String) async {
        octree.markAsUnloaded(nodeID: nodeID)
        onNodeUnloaded?(nodeID)
        Self.log.debug("Node \(nodeID) unloaded")
    }

    /// Forces immediate unloading of all non-visible nodes
    public func unloadInvisibleNodes() async {
        let candidates = octree.getUnloadCandidates()
        for nodeID in candidates {
            await unloadNode(nodeID: nodeID)
        }
    }

    /// Returns current streaming statistics
    public func getStatistics() -> StreamingStatistics {
        let octreeStats = octree.getStatistics()
        return StreamingStatistics(
            isStreaming: isStreaming,
            loadingCount: loadingNodes.count,
            queuedCount: octree.getLoadQueue().count,
            loadedNodes: octreeStats.loadedNodes,
            visibleNodes: octreeStats.visibleNodes,
            memoryUsedMB: octreeStats.loadedMemoryMB,
            memoryBudgetMB: octreeStats.memoryBudgetMB,
            budgetUtilization: octreeStats.budgetUtilization
        )
    }

    /// Statistics about streaming state
    public struct StreamingStatistics: Sendable {
        public let isStreaming: Bool
        public let loadingCount: Int
        public let queuedCount: Int
        public let loadedNodes: Int
        public let visibleNodes: Int
        public let memoryUsedMB: Float
        public let memoryBudgetMB: Float
        public let budgetUtilization: Float
    }

    /// Errors that can occur during streaming
    public enum StreamingError: Error, LocalizedError {
        case nodeNotFound(String)
        case noLODData(String)
        case loadFailed(String, Error)

        public var errorDescription: String? {
            switch self {
            case .nodeNotFound(let id):
                return "Node not found: \(id)"
            case .noLODData(let id):
                return "No LOD data available for node: \(id)"
            case .loadFailed(let id, let error):
                return "Failed to load node \(id): \(error.localizedDescription)"
            }
        }
    }
}
