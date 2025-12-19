import Foundation
import Metal
import simd
import SplatIO

/// Prototype unified renderer that aggregates multiple splat sources into a single draw.
/// This keeps a single global splat buffer and sort order, similar in spirit to a work-buffer path.
public final class UnifiedSplatRenderer {
    public struct SourceRange {
        public let start: Int
        public let count: Int
    }

    private let renderer: SplatRenderer
    private var sources: [SourceRange] = []
    private var points: [SplatScenePoint] = []

    public init(device: MTLDevice,
                colorFormat: MTLPixelFormat,
                depthFormat: MTLPixelFormat,
                sampleCount: Int,
                maxViewCount: Int,
                maxSimultaneousRenders: Int) throws {
        renderer = try SplatRenderer(device: device,
                                     colorFormat: colorFormat,
                                     depthFormat: depthFormat,
                                     sampleCount: sampleCount,
                                     maxViewCount: maxViewCount,
                                     maxSimultaneousRenders: maxSimultaneousRenders)
    }

    public var usePackedSplats: Bool {
        get { renderer.usePackedSplats }
        set { renderer.usePackedSplats = newValue }
    }

    public var useBinnedSorting: Bool {
        get { renderer.useBinnedSorting }
        set { renderer.useBinnedSorting = newValue }
    }

    public var useChunkHistogramBinning: Bool {
        get { renderer.useChunkHistogramBinning }
        set { renderer.useChunkHistogramBinning = newValue }
    }

    public var splatCount: Int { renderer.splatCount }

    /// Adds a splat source and returns the range reserved for it.
    /// If a transform is provided, it is applied on the CPU before aggregation.
    @discardableResult
    public func add(_ newPoints: [SplatScenePoint], transform: simd_float4x4? = nil) throws -> SourceRange {
        let start = points.count
        let transformed = transform.map { applyTransform($0, to: newPoints) } ?? newPoints
        points.append(contentsOf: transformed)
        let range = SourceRange(start: start, count: transformed.count)
        sources.append(range)
        try rebuild()
        return range
    }

    /// Rebuilds the unified buffers from all sources.
    public func rebuild() throws {
        renderer.reset()
        try renderer.add(points)
    }

    public func render(viewports: [SplatRenderer.ViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorLoadAction: MTLLoadAction,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       depthStoreAction: MTLStoreAction,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        try renderer.render(viewports: viewports,
                            colorTexture: colorTexture,
                            colorLoadAction: colorLoadAction,
                            colorStoreAction: colorStoreAction,
                            depthTexture: depthTexture,
                            depthStoreAction: depthStoreAction,
                            rasterizationRateMap: rasterizationRateMap,
                            renderTargetArrayLength: renderTargetArrayLength,
                            to: commandBuffer)
    }

    private func applyTransform(_ transform: simd_float4x4,
                                to points: [SplatScenePoint]) -> [SplatScenePoint] {
        let rotation = simd_quatf(transform)
        let c0 = SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z)
        let c1 = SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z)
        let c2 = SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        let scale = SIMD3<Float>(simd_length(c0), simd_length(c1), simd_length(c2))

        return points.map { point in
            var updated = point
            let worldPos = transform * SIMD4<Float>(point.position, 1)
            updated.position = SIMD3<Float>(worldPos.x, worldPos.y, worldPos.z)
            updated.rotation = rotation * point.rotation
            updated.scale = .linearFloat(point.scale.asLinearFloat * scale)
            return updated
        }
    }
}
