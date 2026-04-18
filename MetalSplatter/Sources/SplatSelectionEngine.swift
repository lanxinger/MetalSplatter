import Foundation
import Metal
import simd

final class SplatSelectionEngine: @unchecked Sendable {
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
    private let dummyMaskTexture: MTLTexture?

    init(device: MTLDevice) throws {
        self.device = device
        self.commandQueue = device.makeCommandQueue()

        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        if let function = library.makeFunction(name: "selectEditableSplats") {
            self.pipelineState = try device.makeComputePipelineState(function: function)
        } else {
            self.pipelineState = nil
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

        guard let outputBuffer = device.makeBuffer(length: splatCount, options: .storageModeShared),
              let queryBuffer = device.makeBuffer(length: MemoryLayout<QueryParameters>.stride, options: .storageModeShared),
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

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(renderer.splatBuffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(editStateBuffer, offset: 0, index: 1)
        encoder.setBuffer(editTransformIndexBuffer, offset: 0, index: 2)
        encoder.setBuffer(editTransformPaletteBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)
        encoder.setBuffer(queryBuffer, offset: 0, index: 5)
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

        let bytes = outputBuffer.contents().bindMemory(to: UInt8.self, capacity: splatCount)
        return (0..<splatCount).compactMap { index in
            bytes[index] == 0 ? nil : index
        }
    }

    private func makeMaskTexture(data: Data, size: SIMD2<Int>) throws -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Unorm,
            width: size.x,
            height: size.y,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead]
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw SplatEditorError.selectionEngineUnavailable
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
}
