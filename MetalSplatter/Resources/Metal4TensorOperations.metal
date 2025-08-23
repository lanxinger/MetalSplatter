#include <metal_stdlib>
#include "ShaderCommon.h"
using namespace metal;

#if __METAL_VERSION__ >= 400
using namespace mpp::tensor_ops;

// Tensor-based batch processing for splat data
namespace tensor_batch_ops {
    
    // Batch process splat transformations using tensors
    [[user_annotation("tensor_batch_transform")]]
    kernel void batch_transform_splats(
        constant SplatArgumentBuffer &argumentBuffer [[buffer(0)]],
        constant Uniforms &uniforms [[buffer(1)]],
        device TransformedSplat *transformedSplats [[buffer(2)]],
        uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
        uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
        uint3 threads_per_threadgroup [[threads_per_threadgroup]]
    ) {
        uint batch_size = threads_per_threadgroup.x * threads_per_threadgroup.y;
        uint batch_start = threadgroup_position_in_grid.x * batch_size;
        uint local_id = thread_position_in_threadgroup.x + 
                       thread_position_in_threadgroup.y * threads_per_threadgroup.x;
        uint global_id = batch_start + local_id;
        
        if (global_id >= argumentBuffer.splatCount) return;
        
        // Create tensor views for batch processing
        tensor<float, extents<int, 4>, threadgroup_descriptor> position_batch;
        tensor<float, extents<int, 4>, threadgroup_descriptor> scale_batch;
        tensor<float, extents<int, 4>, threadgroup_descriptor> rotation_batch;
        
        // Load batch data into tensors
        if (local_id < batch_size && (batch_start + local_id) < argumentBuffer.splatCount) {
            Splat splat = argumentBuffer.splatBuffer[batch_start + local_id];
            
            // Store in tensor format for batch operations
            position_batch[local_id] = float4(splat.position, 1.0);
            scale_batch[local_id] = float4(splat.scale, 1.0);
            rotation_batch[local_id] = splat.rotation;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Batch transform using tensor operations
        tensor<float, extents<int, 4, 4>, threadgroup_descriptor> transform_matrix;
        
        // Load view-projection matrix into tensor
        for (uint i = 0; i < 4; ++i) {
            for (uint j = 0; j < 4; ++j) {
                transform_matrix[i * 4 + j] = uniforms.projectionMatrix[i][j] * uniforms.viewMatrix[i][j];
            }
        }
        
        // Cooperative tensor multiply for batch transformation
        tensor<float, extents<int, 4>, threadgroup_descriptor> transformed_positions;
        cooperative_tensor_multiply(transform_matrix, position_batch, transformed_positions, local_id);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Store transformed results
        if (local_id < batch_size && (batch_start + local_id) < argumentBuffer.splatCount) {
            TransformedSplat& output = transformedSplats[batch_start + local_id];
            output.screenPosition = transformed_positions[local_id];
            output.scale = scale_batch[local_id].xyz;
            output.rotation = rotation_batch[local_id];
            
            // Compute depth for sorting
            output.depth = transformed_positions[local_id].z / transformed_positions[local_id].w;
        }
    }
    
    // Tensor-based batch covariance computation
    [[user_annotation("tensor_batch_covariance")]]
    kernel void batch_compute_covariances(
        device TransformedSplat *transformedSplats [[buffer(0)]],
        device float3x3 *covarianceMatrices [[buffer(1)]],
        uint splatCount [[buffer(2)]],
        uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
        uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
        uint3 threads_per_threadgroup [[threads_per_threadgroup]]
    ) {
        uint batch_size = threads_per_threadgroup.x;
        uint batch_start = threadgroup_position_in_grid.x * batch_size;
        uint local_id = thread_position_in_threadgroup.x;
        uint global_id = batch_start + local_id;
        
        if (global_id >= splatCount) return;
        
        // Batch load rotation quaternions into tensor
        tensor<float, extents<int, 4>, threadgroup_descriptor> quaternion_batch;
        tensor<float, extents<int, 3>, threadgroup_descriptor> scale_batch;
        
        if (local_id < batch_size && global_id < splatCount) {
            quaternion_batch[local_id] = transformedSplats[global_id].rotation;
            scale_batch[local_id] = transformedSplats[global_id].scale;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Cooperative quaternion to rotation matrix conversion
        tensor<float, extents<int, 3, 3>, threadgroup_descriptor> rotation_matrices;
        cooperative_quaternion_to_matrix(quaternion_batch, rotation_matrices, local_id);
        
        // Apply scaling to get covariance
        tensor<float, extents<int, 3, 3>, threadgroup_descriptor> covariance_batch;
        cooperative_scale_matrix(rotation_matrices, scale_batch, covariance_batch, local_id);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Store results
        if (local_id < batch_size && global_id < splatCount) {
            float3x3 covariance;
            for (uint i = 0; i < 3; ++i) {
                for (uint j = 0; j < 3; ++j) {
                    covariance[i][j] = covariance_batch[local_id * 9 + i * 3 + j];
                }
            }
            covarianceMatrices[global_id] = covariance;
        }
    }
    
    // Helper functions for cooperative tensor operations
    template<typename Scope>
    void cooperative_tensor_multiply(
        tensor<float, extents<int, 4, 4>, threadgroup_descriptor>& matrix,
        tensor<float, extents<int, 4>, threadgroup_descriptor>& vectors,
        tensor<float, extents<int, 4>, threadgroup_descriptor>& result,
        uint local_id
    ) {
        // Each thread computes one element of the result vector
        uint vector_idx = local_id / 4;
        uint element_idx = local_id % 4;
        
        float sum = 0.0;
        for (uint k = 0; k < 4; ++k) {
            sum += matrix[element_idx * 4 + k] * vectors[vector_idx * 4 + k];
        }
        
        result[vector_idx * 4 + element_idx] = sum;
    }
    
    template<typename Scope>
    void cooperative_quaternion_to_matrix(
        tensor<float, extents<int, 4>, threadgroup_descriptor>& quaternions,
        tensor<float, extents<int, 3, 3>, threadgroup_descriptor>& matrices,
        uint local_id
    ) {
        uint quat_idx = local_id / 9;
        uint matrix_element = local_id % 9;
        uint row = matrix_element / 3;
        uint col = matrix_element % 3;
        
        if (quat_idx * 4 < quaternions.size() && quaternions[quat_idx * 4 + 3] != 0) {
            float4 q = normalize(float4(
                quaternions[quat_idx * 4],
                quaternions[quat_idx * 4 + 1], 
                quaternions[quat_idx * 4 + 2],
                quaternions[quat_idx * 4 + 3]
            ));
            
            float element;
            if (row == 0 && col == 0) element = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
            else if (row == 0 && col == 1) element = 2.0 * (q.x * q.y + q.w * q.z);
            else if (row == 0 && col == 2) element = 2.0 * (q.x * q.z - q.w * q.y);
            else if (row == 1 && col == 0) element = 2.0 * (q.x * q.y - q.w * q.z);
            else if (row == 1 && col == 1) element = 1.0 - 2.0 * (q.x * q.x + q.z * q.z);
            else if (row == 1 && col == 2) element = 2.0 * (q.y * q.z + q.w * q.x);
            else if (row == 2 && col == 0) element = 2.0 * (q.x * q.z + q.w * q.y);
            else if (row == 2 && col == 1) element = 2.0 * (q.y * q.z - q.w * q.x);
            else element = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
            
            matrices[quat_idx * 9 + matrix_element] = element;
        }
    }
    
    template<typename Scope>
    void cooperative_scale_matrix(
        tensor<float, extents<int, 3, 3>, threadgroup_descriptor>& matrices,
        tensor<float, extents<int, 3>, threadgroup_descriptor>& scales,
        tensor<float, extents<int, 3, 3>, threadgroup_descriptor>& result,
        uint local_id
    ) {
        uint matrix_idx = local_id / 9;
        uint element_idx = local_id % 9;
        uint row = element_idx / 3;
        uint col = element_idx % 3;
        
        if (matrix_idx * 9 < matrices.size()) {
            float scale = scales[matrix_idx * 3 + col];
            result[matrix_idx * 9 + element_idx] = matrices[matrix_idx * 9 + element_idx] * scale;
        }
    }
}

// Additional tensor structure for transformed splat data
struct TransformedSplat {
    float4 screenPosition;
    float3 scale;
    float4 rotation;
    float depth;
};

#endif // __METAL_VERSION__ >= 400