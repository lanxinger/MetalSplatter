import Foundation
import simd

/// Axis-Aligned Bounding Box for octree nodes
public struct AABB: Sendable, Codable, Hashable {
    public var min: SIMD3<Float>
    public var max: SIMD3<Float>

    public init(min: SIMD3<Float>, max: SIMD3<Float>) {
        self.min = min
        self.max = max
    }

    public init() {
        self.min = SIMD3<Float>(repeating: .infinity)
        self.max = SIMD3<Float>(repeating: -.infinity)
    }

    /// Center point of the bounding box
    public var center: SIMD3<Float> {
        (min + max) * 0.5
    }

    /// Size of the bounding box
    public var size: SIMD3<Float> {
        max - min
    }

    /// Half-extents (size / 2)
    public var halfExtents: SIMD3<Float> {
        size * 0.5
    }

    /// Diagonal length of the bounding box
    public var diagonalLength: Float {
        simd_length(size)
    }

    /// Expands the bounding box to include a point
    public mutating func expand(toInclude point: SIMD3<Float>) {
        min = simd_min(min, point)
        max = simd_max(max, point)
    }

    /// Expands the bounding box to include another bounding box
    public mutating func expand(toInclude other: AABB) {
        min = simd_min(min, other.min)
        max = simd_max(max, other.max)
    }

    /// Checks if the bounding box contains a point
    public func contains(_ point: SIMD3<Float>) -> Bool {
        return point.x >= min.x && point.x <= max.x &&
               point.y >= min.y && point.y <= max.y &&
               point.z >= min.z && point.z <= max.z
    }

    /// Checks if this bounding box intersects another
    public func intersects(_ other: AABB) -> Bool {
        return min.x <= other.max.x && max.x >= other.min.x &&
               min.y <= other.max.y && max.y >= other.min.y &&
               min.z <= other.max.z && max.z >= other.min.z
    }

    /// Returns the octant index (0-7) for a point within this bounding box
    public func octantIndex(for point: SIMD3<Float>) -> Int {
        let c = center
        var index = 0
        if point.x >= c.x { index |= 1 }
        if point.y >= c.y { index |= 2 }
        if point.z >= c.z { index |= 4 }
        return index
    }

    /// Returns the bounding box for a specific octant (0-7)
    public func octantBounds(index: Int) -> AABB {
        let c = center
        var newMin = min
        var newMax = max

        if index & 1 != 0 {
            newMin.x = c.x
        } else {
            newMax.x = c.x
        }

        if index & 2 != 0 {
            newMin.y = c.y
        } else {
            newMax.y = c.y
        }

        if index & 4 != 0 {
            newMin.z = c.z
        } else {
            newMax.z = c.z
        }

        return AABB(min: newMin, max: newMax)
    }
}

/// LOD level configuration for octree nodes
public struct OctreeLODLevel: Sendable, Codable {
    /// URL to the splat data file for this LOD level (nil if inline)
    public var resourceURL: URL?

    /// Range of splat indices in the parent buffer (for inline data)
    public var splatRange: Range<Int>?

    /// Approximate number of splats in this LOD level
    public var splatCount: Int

    /// LOD level (0 = highest detail)
    public var level: Int

    /// Screen-space error threshold for this LOD (higher = coarser)
    public var screenSpaceErrorThreshold: Float

    /// Whether this LOD data is currently loaded in memory
    public var isLoaded: Bool = false

    public init(
        resourceURL: URL? = nil,
        splatRange: Range<Int>? = nil,
        splatCount: Int,
        level: Int,
        screenSpaceErrorThreshold: Float
    ) {
        self.resourceURL = resourceURL
        self.splatRange = splatRange
        self.splatCount = splatCount
        self.level = level
        self.screenSpaceErrorThreshold = screenSpaceErrorThreshold
    }
}

/// Represents a node in the octree hierarchy
public struct OctreeNode: Sendable, Codable {
    /// Unique identifier for this node
    public var id: String

    /// Bounding box for this node
    public var bounds: AABB

    /// LOD levels available for this node (sorted by level, 0 = highest detail)
    public var lodLevels: [OctreeLODLevel]

    /// Child node IDs (nil for leaf nodes)
    public var childIDs: [String]?

    /// Parent node ID (nil for root)
    public var parentID: String?

    /// Depth in the octree (0 = root)
    public var depth: Int

    /// Reference count for resource management
    public var referenceCount: Int32 = 0

    /// Frame number when this node was last used (for cooldown-based unloading)
    public var lastUsedFrame: UInt64 = 0

    /// Number of cooldown ticks before unloading (default 100 frames)
    public var cooldownTicks: Int = 100

    /// Whether this node is currently visible in the frustum
    public var isVisible: Bool = false

    public init(
        id: String,
        bounds: AABB,
        lodLevels: [OctreeLODLevel] = [],
        childIDs: [String]? = nil,
        parentID: String? = nil,
        depth: Int = 0
    ) {
        self.id = id
        self.bounds = bounds
        self.lodLevels = lodLevels
        self.childIDs = childIDs
        self.parentID = parentID
        self.depth = depth
    }

    /// Returns the appropriate LOD level for the given screen-space error
    public func selectLOD(forScreenSpaceError error: Float) -> Int? {
        // Find the coarsest LOD that meets the error threshold
        for lod in lodLevels.reversed() {
            if error >= lod.screenSpaceErrorThreshold {
                return lod.level
            }
        }
        // Return finest LOD if error is very small
        return lodLevels.first?.level
    }

    /// Returns true if this is a leaf node (no children)
    public var isLeaf: Bool {
        childIDs == nil || childIDs!.isEmpty
    }

    /// Returns the total splat count across all LOD levels
    public var totalSplatCount: Int {
        lodLevels.reduce(0) { $0 + $1.splatCount }
    }
}

/// Header for octree scene files
public struct OctreeSceneHeader: Sendable, Codable {
    /// File format version
    public var version: Int = 1

    /// Total number of nodes in the octree
    public var nodeCount: Int

    /// Total number of splats across all nodes
    public var totalSplatCount: Int

    /// Number of LOD levels
    public var lodLevelCount: Int

    /// Root node ID
    public var rootNodeID: String

    /// Overall scene bounds
    public var sceneBounds: AABB

    /// Maximum depth of the octree
    public var maxDepth: Int

    /// Memory budget hint (bytes)
    public var memoryBudgetHint: Int?

    public init(
        nodeCount: Int,
        totalSplatCount: Int,
        lodLevelCount: Int,
        rootNodeID: String,
        sceneBounds: AABB,
        maxDepth: Int,
        memoryBudgetHint: Int? = nil
    ) {
        self.nodeCount = nodeCount
        self.totalSplatCount = totalSplatCount
        self.lodLevelCount = lodLevelCount
        self.rootNodeID = rootNodeID
        self.sceneBounds = sceneBounds
        self.maxDepth = maxDepth
        self.memoryBudgetHint = memoryBudgetHint
    }
}

/// Complete octree scene representation
public struct OctreeScene: Sendable, Codable {
    public var header: OctreeSceneHeader
    public var nodes: [String: OctreeNode]

    public init(header: OctreeSceneHeader, nodes: [String: OctreeNode]) {
        self.header = header
        self.nodes = nodes
    }

    /// Returns the root node
    public var rootNode: OctreeNode? {
        nodes[header.rootNodeID]
    }

    /// Returns all leaf nodes
    public var leafNodes: [OctreeNode] {
        nodes.values.filter { $0.isLeaf }
    }

    /// Returns nodes at a specific depth
    public func nodes(atDepth depth: Int) -> [OctreeNode] {
        nodes.values.filter { $0.depth == depth }
    }
}
