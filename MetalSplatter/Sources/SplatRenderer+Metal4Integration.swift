import Foundation
import Metal
import MetalKit
import os

// MARK: - Metal 4.0 Integration for SplatRenderer

extension SplatRenderer {
    
    // MARK: - Metal 4.0 Feature Detection
    
    /// Check if Metal 4.0 optimizations are available
    internal var isMetal4OptimizationsAvailable: Bool {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            return device.supportsFamily(.apple9)
        }
        return false
    }
    
    /// Setup Metal 4.0 integration during initialization
    internal func setupMetal4Integration() {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            if device.supportsFamily(.apple9) {
                Self.log.info("Metal 4.0: Integration enabled for Apple GPU Family 9+")
                // Pipeline states will be created lazily when needed
            } else {
                Self.log.info("Metal 4.0: Device doesn't support Apple GPU Family 9+, optimizations disabled")
            }
        } else {
            Self.log.info("Metal 4.0: iOS 26.0+ required for advanced features")
        }
    }
    
    // MARK: - Metal 4.0 Optimized Rendering
    
    /// Metal 4.0 compute preprocessing phase
    @available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
    internal func renderWithMetal4ComputeOptimization(
        computeEncoder: MTLComputeCommandEncoder?,
        splatBuffer: MTLBuffer,
        uniformsBuffer: MTLBuffer,
        splatCount: Int
    ) {
        guard device.supportsFamily(.apple9), let computeEncoder = computeEncoder else {
            Self.log.warning("Metal 4.0: Compute optimization not available")
            return
        }
        
        Self.log.debug("Metal 4.0: Starting compute preprocessing for \(splatCount) splats")
        
        // Choose compute path based on scene complexity
        if splatCount > 10000 {
            renderWithTensorOperations(
                encoder: computeEncoder,
                splatBuffer: splatBuffer,
                uniformsBuffer: uniformsBuffer,
                splatCount: splatCount
            )
        } else {
            renderWithSIMDGroupOperations(
                encoder: computeEncoder,
                splatBuffer: splatBuffer,
                uniformsBuffer: uniformsBuffer,
                splatCount: splatCount
            )
        }
        
        Self.log.debug("Metal 4.0: Compute preprocessing completed")
    }
    
    /// Metal 4.0 rendering phase
    @available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
    internal func renderWithMetal4RenderOptimization(
        renderEncoder: MTLRenderCommandEncoder,
        splatBuffer: MTLBuffer,
        uniformsBuffer: MTLBuffer,
        splatCount: Int
    ) {
        guard device.supportsFamily(.apple9) else {
            Self.log.warning("Metal 4.0: Render optimization not available")
            return
        }
        
        Self.log.debug("Metal 4.0: Starting render phase for \(splatCount) splats")
        
        // For now, use existing render pipeline - mesh shaders would require more complex setup
        // The compute preprocessing has already optimized the data
        Self.log.debug("Metal 4.0: Using standard render path with preprocessed data")
    }
    
    // MARK: - Individual Metal 4.0 Rendering Paths

    @available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private func cachedMetal4ComputePipeline(functionName: String) -> MTLComputePipelineState? {
        metal4PipelineCacheLock.lock()
        if let cached = metal4ComputePipelineCache[functionName] {
            metal4PipelineCacheLock.unlock()
            return cached
        }
        metal4PipelineCacheLock.unlock()

        guard let function = library.makeFunction(name: functionName),
              let pipelineState = try? device.makeComputePipelineState(function: function) else {
            return nil
        }

        metal4PipelineCacheLock.lock()
        metal4ComputePipelineCache[functionName] = pipelineState
        metal4PipelineCacheLock.unlock()
        return pipelineState
    }
    
    /// Use Metal 4.0 SIMD-group operations for splat transformation
    /// Uses batchTransformPositionsSIMD kernel from Metal4TensorOperations.metal
    @available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private func renderWithSIMDGroupOperations(
        encoder: MTLComputeCommandEncoder,
        splatBuffer: MTLBuffer,
        uniformsBuffer: MTLBuffer,
        splatCount: Int
    ) {
        // Cache the pipeline so we don't compile during render.
        guard let pipelineState = cachedMetal4ComputePipeline(functionName: "batchTransformPositionsSIMD") else {
            Self.log.warning("Metal 4.0: SIMD-group compute pipeline not available (batchTransformPositionsSIMD)")
            return
        }

        let viewPositionsSize = splatCount * MemoryLayout<SIMD4<Float>>.stride
        let clipPositionsSize = splatCount * MemoryLayout<SIMD4<Float>>.stride
        let depthsSize = splatCount * MemoryLayout<Float>.stride

        if var outputs = metal4SIMDOutputs,
           outputs.viewPositions.length >= viewPositionsSize,
           outputs.clipPositions.length >= clipPositionsSize,
           outputs.depths.length >= depthsSize {
            outputs.count = splatCount
            metal4SIMDOutputs = outputs
        } else {
            guard let viewPositionsBuffer = device.makeBuffer(length: viewPositionsSize, options: .storageModePrivate),
                  let clipPositionsBuffer = device.makeBuffer(length: clipPositionsSize, options: .storageModePrivate),
                  let depthsBuffer = device.makeBuffer(length: depthsSize, options: .storageModePrivate) else {
                Self.log.error("Metal 4.0: Failed to create output buffers for SIMD transform")
                return
            }
            metal4SIMDOutputs = Metal4SIMDOutputs(viewPositions: viewPositionsBuffer,
                                                  clipPositions: clipPositionsBuffer,
                                                  depths: depthsBuffer,
                                                  count: splatCount)
        }

        guard let outputs = metal4SIMDOutputs else {
            Self.log.error("Metal 4.0: SIMD output buffers unavailable")
            return
        }

        var count = UInt32(splatCount)

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(splatBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputs.viewPositions, offset: 0, index: 1)
        encoder.setBuffer(outputs.clipPositions, offset: 0, index: 2)
        encoder.setBuffer(outputs.depths, offset: 0, index: 3)
        encoder.setBuffer(uniformsBuffer, offset: uniformBufferOffset, index: 4)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 5)

        // Each thread processes 4 splats
        let splatsPerThread = 4
        let threadCount = (splatCount + splatsPerThread - 1) / splatsPerThread
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (threadCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        Self.log.debug("Metal 4.0: SIMD-group batch transform dispatched for \(splatCount) splats")
    }
    
    /// Use Metal 4.0 tensor operations for batch processing
    /// Uses batchPrecomputeSplats kernel from Metal4TensorOperations.metal
    @available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private func renderWithTensorOperations(
        encoder: MTLComputeCommandEncoder,
        splatBuffer: MTLBuffer,
        uniformsBuffer: MTLBuffer,
        splatCount: Int
    ) {
        // Cache the pipeline so we don't compile during render.
        guard let pipelineState = cachedMetal4ComputePipeline(functionName: "batchPrecomputeSplats") else {
            Self.log.warning("Metal 4.0: Batch precompute pipeline not available (batchPrecomputeSplats)")
            return
        }

        // PrecomputedSplat alignment in Metal:
        //   float4 clipPosition: offset 0, size 16 (16-byte aligned)
        //   float3 cov2D:        offset 16, size 12 (16-byte aligned, 4 bytes padding follows)
        //   float2 axis1:        offset 32, size 8  (8-byte aligned)
        //   float2 axis2:        offset 40, size 8
        //   float depth:         offset 48, size 4
        //   uint visible:        offset 52, size 4
        //   Total: 56 bytes, rounded to 64 due to struct's 16-byte alignment requirement
        let requiredSize = splatCount * Self.precomputedSplatStride
        guard let precomputedBuffer = ensurePrecomputedSplatBuffer(requiredSize: requiredSize) else {
            Self.log.error("Metal 4.0: Failed to create precomputed buffer")
            return
        }

        var count = UInt32(splatCount)

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(splatBuffer, offset: 0, index: 0)
        encoder.setBuffer(precomputedBuffer, offset: 0, index: 1)
        encoder.setBuffer(uniformsBuffer, offset: uniformBufferOffset, index: 2)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 3)

        // batchPrecomputeSplats uses max_total_threads_per_threadgroup(256)
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (splatCount + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        Self.log.debug("Metal 4.0: Batch precompute dispatched for \(splatCount) splats")

        precomputedDataDirty = false
    }
    
    /// Use Metal 4.0 mesh shaders for rendering (placeholder - mesh shaders need render pass)
    @available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private func renderWithMeshShaders(
        encoder: MTLRenderCommandEncoder,
        splatBuffer: MTLBuffer,
        uniformsBuffer: MTLBuffer,
        splatCount: Int
    ) {
        // For now, fall back to regular rendering as mesh shaders require more complex setup
        Self.log.debug("Metal 4.0: Mesh shader rendering (fallback to standard path)")
        
        // TODO: Implement actual mesh shader pipeline when mesh shader APIs are available
        // This would require MTLRenderPipelineDescriptor with mesh/object functions
    }
    
    // MARK: - Metal 4.0 Capability Reporting
    
    /// Get Metal 4.0 capabilities status
    internal func getMetal4Capabilities() -> Metal4Capabilities {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            let available = device.supportsFamily(.apple9)

            // Check for actual kernel names from Metal4TensorOperations.metal and Metal4MeshShaders.metal
            return Metal4Capabilities(
                available: available,
                simdGroupOperations: available && library.makeFunction(name: "batchTransformPositionsSIMD") != nil,
                tensorOperations: available && library.makeFunction(name: "batchPrecomputeSplats") != nil,
                advancedAtomics: available && library.makeFunction(name: "countVisibleSplats") != nil,
                meshShaders: available && library.makeFunction(name: "splatMeshShader") != nil
            )
        } else {
            return Metal4Capabilities(
                available: false,
                simdGroupOperations: false,
                tensorOperations: false,
                advancedAtomics: false,
                meshShaders: false
            )
        }
    }
    
    /// Metal 4.0 capabilities structure
    struct Metal4Capabilities {
        let available: Bool
        let simdGroupOperations: Bool
        let tensorOperations: Bool
        let advancedAtomics: Bool
        let meshShaders: Bool
        
        var description: String {
            var features: [String] = []
            if simdGroupOperations { features.append("SIMD-Groups") }
            if tensorOperations { features.append("Tensors") }
            if advancedAtomics { features.append("Advanced Atomics") }
            if meshShaders { features.append("Mesh Shaders") }
            
            return available ? "Metal 4.0 Available: \(features.joined(separator: ", "))" : "Metal 4.0 Not Available"
        }
    }
}

// MARK: - Supporting Notes
//
// The Metal 4 compute kernels use the following structures defined in Metal4TensorOperations.metal:
// - PrecomputedSplat: clipPosition, cov2D, axis1, axis2, depth, visible (64 bytes with alignment)
// - batchPrecomputeSplats: Main batch precompute kernel for large scenes (>10K splats)
// - batchTransformPositionsSIMD: SIMD-optimized position transform for smaller scenes
