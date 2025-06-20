import Metal
import SampleBoxRenderer

extension SampleBoxRenderer: ModelRenderer {
    public func render(viewports: [ModelRendererViewportDescriptor],
                       colorTexture: MTLTexture,
                       colorStoreAction: MTLStoreAction,
                       depthTexture: MTLTexture?,
                       rasterizationRateMap: MTLRasterizationRateMap?,
                       renderTargetArrayLength: Int,
                       to commandBuffer: MTLCommandBuffer) throws {
        let remappedViewports = viewports.map { viewport -> ViewportDescriptor in
            ViewportDescriptor(viewport: viewport.viewport,
                               projectionMatrix: viewport.projectionMatrix,
                               viewMatrix: viewport.viewMatrix,
                               screenSize: viewport.screenSize)
        }
        try render(viewports: remappedViewports,
                   colorTexture: colorTexture,
                   colorStoreAction: colorStoreAction,
                   depthTexture: depthTexture,
                   rasterizationRateMap: rasterizationRateMap,
                   renderTargetArrayLength: renderTargetArrayLength,
                   to: commandBuffer)
    }
    
    public var bounds: (min: SIMD3<Float>, max: SIMD3<Float>) {
        // Sample box is 2x2x2 centered at origin
        (min: SIMD3<Float>(-1, -1, -1), max: SIMD3<Float>(1, 1, 1))
    }
}
