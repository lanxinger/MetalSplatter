import Foundation
import Metal
import simd

final class SplatSelectionEngine: @unchecked Sendable {
    private struct NearestSelectionCandidate {
        var index: Int32
        var distanceSquared: Float
    }

    private enum QueryMode: UInt32 {
        case point = 0
        case rect = 1
        case mask = 2
        case sphere = 3
        case box = 4
    }

    private struct QueryParameters {
        var projectionMatrix: simd_float4x4
        var viewMatrix: simd_float4x4
        var point: SIMD2<Float>
        var pointRadius: Float
        var mode: UInt32
        var splatCount: UInt32
        var rect: SIMD4<Float>
        var sphereCenter: SIMD3<Float>
        var sphereRadius: Float
        var boxCenter: SIMD3<Float>
        var padding0: Float
        var boxExtents: SIMD3<Float>
        var padding1: Float
        var maskSize: SIMD2<UInt32>
        var padding2: SIMD2<UInt32>
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue?
    private let pipelineState: MTLComputePipelineState?
    private let nearestPointPipelineState: MTLComputePipelineState?
    private let dummyMaskTexture: MTLTexture?
    private var queryBuffer: MTLBuffer?
    private var selectedIndexBuffer: MTLBuffer?
    private var selectedCountBuffer: MTLBuffer?
    private var nearestCandidateBuffer: MTLBuffer?
    private var cachedMaskTextures: [String: MTLTexture] = [:]

    init(device: MTLDevice) throws {
        self.device = device
        self.commandQueue = device.makeCommandQueue()

        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        if let function = library.makeFunction(name: "selectEditableSplats") {
            self.pipelineState = try device.makeComputePipelineState(function: function)
        } else {
            self.pipelineState = nil
        }
        if let function = library.makeFunction(name: "pickNearestEditableSplat") {
            self.nearestPointPipelineState = try device.makeComputePipelineState(function: function)
        } else {
            self.nearestPointPipelineState = nil
        }

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: 1,
            height: 1,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        self.dummyMaskTexture = device.makeTexture(descriptor: descriptor)
        dummyMaskTexture?.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: [UInt8(0)], bytesPerRow: 1)
    }

    func select(query: SplatSelectionQuery,
                viewport: SplatRenderer.ViewportDescriptor,
                renderer: SplatRenderer) async throws -> [Int] {
        guard let commandQueue, let pipelineState else {
            throw SplatEditorError.selectionEngineUnavailable
        }

        let splatCount = renderer.splatCount
        guard splatCount > 0 else { return [] }

        try ensureSelectionResources(splatCount: splatCount)

        guard let outputBuffer = selectedIndexBuffer,
              let selectedCountBuffer,
              let queryBuffer,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder(),
              let editStateBuffer = renderer.editStateBuffer,
              let editTransformIndexBuffer = renderer.editTransformIndexBuffer,
              let editTransformPaletteBuffer = renderer.editTransformPaletteBuffer else {
            throw SplatEditorError.selectionEngineUnavailable
        }

        var maskTexture = dummyMaskTexture
        var parameters = QueryParameters(
            projectionMatrix: viewport.projectionMatrix,
            viewMatrix: viewport.viewMatrix,
            point: SIMD2<Float>(0, 0),
            pointRadius: 0,
            mode: QueryMode.rect.rawValue,
            splatCount: UInt32(splatCount),
            rect: SIMD4<Float>(0, 0, 0, 0),
            sphereCenter: .zero,
            sphereRadius: 0,
            boxCenter: .zero,
            padding0: 0,
            boxExtents: .zero,
            padding1: 0,
            maskSize: .zero,
            padding2: .zero
        )

        switch query {
        case let .point(normalized, radius):
            parameters.mode = QueryMode.point.rawValue
            parameters.point = normalized
            parameters.pointRadius = radius
        case let .rect(normalizedMin, normalizedMax):
            parameters.mode = QueryMode.rect.rawValue
            let minimum = simd_min(normalizedMin, normalizedMax)
            let maximum = simd_max(normalizedMin, normalizedMax)
            parameters.rect = SIMD4<Float>(minimum.x, minimum.y, maximum.x, maximum.y)
        case let .mask(alphaMask, size):
            let pixelCount = size.x * size.y
            guard size.x > 0, size.y > 0, alphaMask.count == pixelCount else {
                throw SplatEditorError.invalidMaskSize
            }
            parameters.mode = QueryMode.mask.rawValue
            parameters.maskSize = SIMD2<UInt32>(UInt32(size.x), UInt32(size.y))
            maskTexture = try makeMaskTexture(data: alphaMask, size: size)
        case let .sphere(center, radius):
            parameters.mode = QueryMode.sphere.rawValue
            parameters.sphereCenter = center
            parameters.sphereRadius = radius
        case let .box(center, extents):
            parameters.mode = QueryMode.box.rawValue
            parameters.boxCenter = center
            parameters.boxExtents = extents
        }

        memcpy(queryBuffer.contents(), &parameters, MemoryLayout<QueryParameters>.stride)
        selectedCountBuffer.contents().storeBytes(of: UInt32(0), as: UInt32.self)

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(renderer.splatBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(editStateBuffer, offset: 0, index: 1)
        encoder.setBuffer(editTransformIndexBuffer, offset: 0, index: 2)
        encoder.setBuffer(editTransformPaletteBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)
        encoder.setBuffer(selectedCountBuffer, offset: 0, index: 5)
        encoder.setBuffer(queryBuffer, offset: 0, index: 6)
        if let maskTexture {
            encoder.setTexture(maskTexture, index: 0)
        }

        let gridSize = MTLSize(width: splatCount, height: 1, depth: 1)
        let threadWidth = min(pipelineState.maxTotalThreadsPerThreadgroup, splatCount)
        let threadgroupSize = MTLSize(width: max(1, threadWidth), height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if let error = buffer.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            }
            commandBuffer.commit()
        }

        let selectedCount = Int(selectedCountBuffer.contents().load(as: UInt32.self))
        guard selectedCount > 0 else { return [] }

        let indices = outputBuffer.contents().bindMemory(to: UInt32.self, capacity: max(selectedCount, 1))
        return (0..<selectedCount).map { Int(indices[$0]) }
    }

    func pickNearest(normalized: SIMD2<Float>,
                     radius: Float,
                     viewport: SplatRenderer.ViewportDescriptor,
                     renderer: SplatRenderer) async throws -> Int? {
        guard let commandQueue, let nearestPointPipelineState else {
            throw SplatEditorError.selectionEngineUnavailable
        }

        let splatCount = renderer.splatCount
        guard splatCount > 0 else { return nil }

        let maxThreads = min(256, nearestPointPipelineState.maxTotalThreadsPerThreadgroup, splatCount)
        var threadsPerGroup = 1
        while threadsPerGroup * 2 <= maxThreads {
            threadsPerGroup *= 2
        }
        let threadgroupCount = max(1, (splatCount + threadsPerGroup - 1) / threadsPerGroup)

        try ensureNearestResources(threadgroupCount: threadgroupCount)
        try ensureQueryBuffer()

        guard let queryBuffer,
              let nearestCandidateBuffer,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder(),
              let editStateBuffer = renderer.editStateBuffer,
              let editTransformIndexBuffer = renderer.editTransformIndexBuffer,
              let editTransformPaletteBuffer = renderer.editTransformPaletteBuffer else {
            throw SplatEditorError.selectionEngineUnavailable
        }

        var parameters = QueryParameters(
            projectionMatrix: viewport.projectionMatrix,
            viewMatrix: viewport.viewMatrix,
            point: normalized,
            pointRadius: radius,
            mode: QueryMode.point.rawValue,
            splatCount: UInt32(splatCount),
            rect: SIMD4<Float>(0, 0, 0, 0),
            sphereCenter: .zero,
            sphereRadius: 0,
            boxCenter: .zero,
            padding0: 0,
            boxExtents: .zero,
            padding1: 0,
            maskSize: .zero,
            padding2: .zero
        )
        memcpy(queryBuffer.contents(), &parameters, MemoryLayout<QueryParameters>.stride)

        let candidates = nearestCandidateBuffer.contents().bindMemory(to: NearestSelectionCandidate.self, capacity: threadgroupCount)
        for index in 0..<threadgroupCount {
            candidates[index] = NearestSelectionCandidate(index: -1, distanceSquared: .greatestFiniteMagnitude)
        }

        encoder.setComputePipelineState(nearestPointPipelineState)
        encoder.setBuffer(renderer.splatBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(editStateBuffer, offset: 0, index: 1)
        encoder.setBuffer(editTransformIndexBuffer, offset: 0, index: 2)
        encoder.setBuffer(editTransformPaletteBuffer, offset: 0, index: 3)
        encoder.setBuffer(nearestCandidateBuffer, offset: 0, index: 4)
        encoder.setBuffer(queryBuffer, offset: 0, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroupCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            commandBuffer.addCompletedHandler { buffer in
                if let error = buffer.error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: ())
                }
            }
            commandBuffer.commit()
        }

        var bestCandidate = NearestSelectionCandidate(index: -1, distanceSquared: .greatestFiniteMagnitude)
        for index in 0..<threadgroupCount {
            let candidate = candidates[index]
            guard candidate.index >= 0 else { continue }
            if candidate.distanceSquared < bestCandidate.distanceSquared {
                bestCandidate = candidate
            }
        }

        return bestCandidate.index >= 0 ? Int(bestCandidate.index) : nil
    }

    private func makeMaskTexture(data: Data, size: SIMD2<Int>) throws -> MTLTexture? {
        let key = "\(size.x)x\(size.y)"
        let texture: MTLTexture
        if let cachedTexture = cachedMaskTextures[key] {
            texture = cachedTexture
        } else {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .r8Unorm,
                width: size.x,
                height: size.y,
                mipmapped: false
            )
            descriptor.usage = [.shaderRead]
            guard let newTexture = device.makeTexture(descriptor: descriptor) else {
                throw SplatEditorError.selectionEngineUnavailable
            }
            cachedMaskTextures[key] = newTexture
            texture = newTexture
        }
        data.withUnsafeBytes { rawBuffer in
            if let baseAddress = rawBuffer.baseAddress {
                texture.replace(
                    region: MTLRegionMake2D(0, 0, size.x, size.y),
                    mipmapLevel: 0,
                    withBytes: baseAddress,
                    bytesPerRow: size.x
                )
            }
        }
        return texture
    }

    private func ensureQueryBuffer() throws {
        if queryBuffer == nil {
            guard let buffer = device.makeBuffer(length: MemoryLayout<QueryParameters>.stride, options: .storageModeShared) else {
                throw SplatEditorError.selectionEngineUnavailable
            }
            queryBuffer = buffer
        }
    }

    private func ensureSelectionResources(splatCount: Int) throws {
        try ensureQueryBuffer()

        let indexLength = max(splatCount, 1) * MemoryLayout<UInt32>.stride
        if selectedIndexBuffer == nil || selectedIndexBuffer?.length != indexLength {
            selectedIndexBuffer = device.makeBuffer(length: indexLength, options: .storageModeShared)
        }
        if selectedCountBuffer == nil {
            selectedCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        }

        guard selectedIndexBuffer != nil, selectedCountBuffer != nil else {
            throw SplatEditorError.selectionEngineUnavailable
        }
    }

    private func ensureNearestResources(threadgroupCount: Int) throws {
        try ensureQueryBuffer()

        let length = max(threadgroupCount, 1) * MemoryLayout<NearestSelectionCandidate>.stride
        if nearestCandidateBuffer == nil || nearestCandidateBuffer?.length != length {
            nearestCandidateBuffer = device.makeBuffer(length: length, options: .storageModeShared)
        }

        guard nearestCandidateBuffer != nil else {
            throw SplatEditorError.selectionEngineUnavailable
        }
    }
}
