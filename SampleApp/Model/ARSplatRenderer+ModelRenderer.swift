#if os(iOS)

import Metal
import MetalSplatter

extension ARSplatRenderer: ModelRenderer {
    public func render(viewports: [ModelRendererViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        try renderAsModelRenderer(viewports: viewports,
                                 colorTexture: colorTexture,
                                 colorStoreAction: colorStoreAction,
                                 depthTexture: depthTexture,
                                 rasterizationRateMap: rasterizationRateMap,
                                 renderTargetArrayLength: renderTargetArrayLength,
                                 to: commandBuffer)
    }
}

#endif // os(iOS)