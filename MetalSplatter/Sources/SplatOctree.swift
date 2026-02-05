import Metal
import simd
import os
import SplatIO

/// Runtime octree for efficient LOD management and spatial queries.
/// Wraps OctreeScene from SplatIO with rendering-specific state.
public class SplatOctree {

    private static let log = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "MetalSplatter",
        category: "SplatOctree"
    )

    /// The underlying octree scene data
    private(set) var scene: OctreeScene

    /// Current frame number for cooldown tracking
    private(set) var currentFrame: UInt64 = 0

    /// Mutable node state for visibility and loading
    private var nodeState: [String: NodeRuntimeState] = [:]

    /// Nodes currently marked as visible
    private(set) var visibleNodes: Set<String> = []

    /// Nodes currently loaded in memory
    private(set) var loadedNodes: Set<String> = []

    /// Total memory used by loaded nodes (bytes)
    private(set) var loadedMemoryBytes: Int = 0

    /// Memory budget for loaded splats (bytes)
    public var memoryBudget: Int

    /// Cooldown frames before unloading invisible nodes
    public var unloadCooldownFrames: Int = 100

    /// Minimum screen-space error for LOD selection
    public var minScreenSpaceError: Float = 0.001

    /// Runtime state for each node
    private struct NodeRuntimeState {
        var isVisible: Bool = false
        var lastVisibleFrame: UInt64 = 0
        var selectedLOD: Int = 0
        var isLoading: Bool = false
        var isLoaded: Bool = false
        var loadedLOD: Int = -1
        var memoryUsage: Int = 0
    }

    public init(scene: OctreeScene, memoryBudget: Int = 512 * 1024 * 1024) {
        self.scene = scene
        self.memoryBudget = memoryBudget

        // Initialize runtime state for all nodes
        for nodeID in scene.nodes.keys {
            nodeState[nodeID] = NodeRuntimeState()
        }
    }

    /// Updates visibility for all nodes based on camera frustum.
    /// Should be called once per frame before rendering.
    ///
    /// - Parameters:
    ///   - viewProjectionMatrix: Combined view-projection matrix
    ///   - cameraPosition: Camera world position
    ///   - screenHeight: Screen height in pixels (for screen-space error calculation)
    public func updateVisibility(
        viewProjectionMatrix: simd_float4x4,
        cameraPosition: SIMD3<Float>,
        screenHeight: Float
    ) {
        currentFrame += 1
        visibleNodes.removeAll()

        // Reset visibility flags for all nodes at the start of each frame
        // This ensures nodes that fall out of frustum are properly marked as not visible
        for nodeID in nodeState.keys {
            nodeState[nodeID]?.isVisible = false
        }

        // Start from root and traverse
        guard let rootID = scene.rootNode?.id else { return }
        traverseForVisibility(
            nodeID: rootID,
            viewProjectionMatrix: viewProjectionMatrix,
            cameraPosition: cameraPosition,
            screenHeight: screenHeight
        )

        // Update cooldown state for nodes no longer visible
        for nodeID in loadedNodes {
            if !visibleNodes.contains(nodeID) {
                // Node is loaded but not visible - start/continue cooldown
                if let state = nodeState[nodeID] {
                    let framesSinceVisible = currentFrame - state.lastVisibleFrame
                    if framesSinceVisible > UInt64(unloadCooldownFrames) {
                        // Mark for potential unloading
                        markForUnload(nodeID: nodeID)
                    }
                }
            }
        }
    }

    /// Recursive visibility traversal
    private func traverseForVisibility(
        nodeID: String,
        viewProjectionMatrix: simd_float4x4,
        cameraPosition: SIMD3<Float>,
        screenHeight: Float
    ) {
        guard let node = scene.nodes[nodeID] else { return }

        // Frustum culling check
        if !isNodeVisible(node: node, viewProjectionMatrix: viewProjectionMatrix) {
            return
        }

        // Calculate screen-space error for LOD selection
        let distance = simd_length(cameraPosition - node.bounds.center)
        let screenError = calculateScreenSpaceError(
            node: node,
            distance: distance,
            screenHeight: screenHeight
        )

        // Update node state
        nodeState[nodeID]?.isVisible = true
        nodeState[nodeID]?.lastVisibleFrame = currentFrame
        visibleNodes.insert(nodeID)

        // Select appropriate LOD based on screen error
        if let selectedLOD = node.selectLOD(forScreenSpaceError: screenError) {
            nodeState[nodeID]?.selectedLOD = selectedLOD
        }

        // Recurse to children if this node has them and we need more detail
        if let childIDs = node.childIDs, !childIDs.isEmpty {
            // Only traverse children if we need finer detail
            // screenError < minScreenSpaceError means we're close enough to need more detail
            // selectedLOD < node's finest LOD means the selected LOD requires finer detail than this node can provide
            let selectedLOD = nodeState[nodeID]?.selectedLOD ?? 0
            let nodeFinestLOD = node.lodLevels.first?.level ?? 0
            let needsChildren = screenError < minScreenSpaceError || selectedLOD < nodeFinestLOD

            if needsChildren {
                for childID in childIDs {
                    traverseForVisibility(
                        nodeID: childID,
                        viewProjectionMatrix: viewProjectionMatrix,
                        cameraPosition: cameraPosition,
                        screenHeight: screenHeight
                    )
                }
            }
        }
    }

    /// Checks if a node is visible in the frustum using its AABB
    private func isNodeVisible(node: OctreeNode, viewProjectionMatrix: simd_float4x4) -> Bool {
        // Transform AABB corners to clip space and check against frustum
        let corners = [
            SIMD4<Float>(node.bounds.min.x, node.bounds.min.y, node.bounds.min.z, 1),
            SIMD4<Float>(node.bounds.max.x, node.bounds.min.y, node.bounds.min.z, 1),
            SIMD4<Float>(node.bounds.min.x, node.bounds.max.y, node.bounds.min.z, 1),
            SIMD4<Float>(node.bounds.max.x, node.bounds.max.y, node.bounds.min.z, 1),
            SIMD4<Float>(node.bounds.min.x, node.bounds.min.y, node.bounds.max.z, 1),
            SIMD4<Float>(node.bounds.max.x, node.bounds.min.y, node.bounds.max.z, 1),
            SIMD4<Float>(node.bounds.min.x, node.bounds.max.y, node.bounds.max.z, 1),
            SIMD4<Float>(node.bounds.max.x, node.bounds.max.y, node.bounds.max.z, 1)
        ]

        // Check each frustum plane
        // If all corners are outside any single plane, the box is culled
        var allOutside = [true, true, true, true, true, true]  // -X, +X, -Y, +Y, -Z, +Z

        for corner in corners {
            let clip = viewProjectionMatrix * corner
            let w = clip.w

            if clip.x >= -w { allOutside[0] = false }
            if clip.x <= w { allOutside[1] = false }
            if clip.y >= -w { allOutside[2] = false }
            if clip.y <= w { allOutside[3] = false }
            if clip.z >= 0 { allOutside[4] = false }  // Near plane (0 for Metal)
            if clip.z <= w { allOutside[5] = false }  // Far plane
        }

        // If any plane has all corners outside, the box is culled
        return !allOutside.contains(true)
    }

    /// Calculates screen-space error for a node at a given distance
    private func calculateScreenSpaceError(
        node: OctreeNode,
        distance: Float,
        screenHeight: Float
    ) -> Float {
        guard distance > 0 else { return .infinity }

        // Screen-space error based on node diagonal size and distance
        // This is a simplified metric - actual implementation might use projected area
        let nodeSize = node.bounds.diagonalLength
        let projectedSize = (nodeSize / distance) * screenHeight

        // Normalize by LOD level's expected detail
        let baseLOD = node.lodLevels.first?.screenSpaceErrorThreshold ?? 1.0
        return projectedSize * baseLOD
    }

    /// Marks a node for potential unloading
    private func markForUnload(nodeID: String) {
        // Don't immediately unload - StreamingLODManager handles actual unloading
        Self.log.debug("Node \(nodeID) marked for potential unload")
    }

    /// Returns nodes that should be loaded based on current visibility
    public func getLoadQueue() -> [String] {
        var queue: [String] = []

        for nodeID in visibleNodes {
            guard let state = nodeState[nodeID],
                  let node = scene.nodes[nodeID] else { continue }

            // Check if we need to load or upgrade this node's LOD
            if !state.isLoaded || state.loadedLOD > state.selectedLOD {
                queue.append(nodeID)
            }
        }

        // Sort by importance (closer nodes first, then by LOD level)
        queue.sort { a, b in
            let stateA = nodeState[a]
            let stateB = nodeState[b]
            return (stateA?.selectedLOD ?? 0) < (stateB?.selectedLOD ?? 0)
        }

        return queue
    }

    /// Returns nodes that can be unloaded to free memory
    public func getUnloadCandidates() -> [String] {
        var candidates: [String] = []

        for nodeID in loadedNodes {
            guard let state = nodeState[nodeID] else { continue }

            // Only unload if not visible and past cooldown
            if !state.isVisible {
                let framesSinceVisible = currentFrame - state.lastVisibleFrame
                if framesSinceVisible > UInt64(unloadCooldownFrames) {
                    candidates.append(nodeID)
                }
            }
        }

        // Sort by longest time since visible (oldest first)
        candidates.sort { a, b in
            let stateA = nodeState[a]
            let stateB = nodeState[b]
            return (stateA?.lastVisibleFrame ?? 0) < (stateB?.lastVisibleFrame ?? 0)
        }

        return candidates
    }

    /// Marks a node as loaded with specific LOD
    public func markAsLoaded(nodeID: String, lodLevel: Int, memoryBytes: Int) {
        if let state = nodeState[nodeID], state.isLoaded {
            loadedMemoryBytes -= state.memoryUsage
        }
        nodeState[nodeID]?.isLoaded = true
        nodeState[nodeID]?.loadedLOD = lodLevel
        nodeState[nodeID]?.memoryUsage = memoryBytes
        loadedNodes.insert(nodeID)
        loadedMemoryBytes += memoryBytes
    }

    /// Marks a node as unloaded
    public func markAsUnloaded(nodeID: String) {
        if let state = nodeState[nodeID] {
            loadedMemoryBytes -= state.memoryUsage
        }
        nodeState[nodeID]?.isLoaded = false
        nodeState[nodeID]?.loadedLOD = -1
        nodeState[nodeID]?.memoryUsage = 0
        loadedNodes.remove(nodeID)
    }

    /// Returns the IntervalManager compatible interval list for active nodes
    public func getActiveIntervals() -> [SplatInterval] {
        var intervals: [SplatInterval] = []

        for nodeID in visibleNodes.sorted() {
            guard let node = scene.nodes[nodeID],
                  let state = nodeState[nodeID],
                  state.isLoaded,
                  let lod = node.lodLevels.first(where: { $0.level == state.loadedLOD }),
                  let range = lod.splatRange else {
                continue
            }

            let interval = SplatInterval(
                sourceStart: range.lowerBound,
                sourceEnd: range.upperBound,
                targetStart: 0,  // Will be computed by IntervalManager
                priority: 1.0 / Float(state.selectedLOD + 1),
                lodLevel: state.loadedLOD,
                isVisible: true
            )
            intervals.append(interval)
        }

        return intervals
    }

    /// Statistics about the octree state
    public struct Statistics {
        public let totalNodes: Int
        public let visibleNodes: Int
        public let loadedNodes: Int
        public let loadedMemoryMB: Float
        public let memoryBudgetMB: Float
        public let budgetUtilization: Float
    }

    public func getStatistics() -> Statistics {
        let memoryMB = Float(loadedMemoryBytes) / (1024 * 1024)
        let budgetMB = Float(memoryBudget) / (1024 * 1024)
        return Statistics(
            totalNodes: scene.nodes.count,
            visibleNodes: visibleNodes.count,
            loadedNodes: loadedNodes.count,
            loadedMemoryMB: memoryMB,
            memoryBudgetMB: budgetMB,
            budgetUtilization: memoryMB / budgetMB
        )
    }
}
