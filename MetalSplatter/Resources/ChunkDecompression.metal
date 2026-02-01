#include <metal_stdlib>
using namespace metal;

#include "ShaderCommon.h"

// Chunk header structure (must match ChunkedSplatFormat.swift GPUChunkHeader)
struct ChunkHeader {
    float3 minPosition;
    float padding1;

    float3 maxPosition;
    float padding2;

    float3 minScale;
    float padding3;

    float3 maxScale;
    uint splatCount;
};

// Packed splat structure (16 bytes, must match ChunkedSplatFormat.swift PackedSplat)
struct PackedSplat {
    uint positionPacked;   // 11-10-11 bits
    uint rotationPacked;   // 2-bit selector + 3×10-bit components
    uint scalePacked;      // 11-10-11 bits with exponential mapping
    uint colorPacked;      // RGBA8
};

// Unpack position from 11-10-11 bits using chunk bounds
inline float3 unpackPosition(uint packed, float3 minPos, float3 maxPos) {
    float x = float((packed >> 21) & 0x7FF) / 2047.0;
    float y = float((packed >> 11) & 0x3FF) / 1023.0;
    float z = float(packed & 0x7FF) / 2047.0;

    float3 range = maxPos - minPos;
    return minPos + float3(x, y, z) * range;
}

// Unpack quaternion from 2-bit selector + 3×10-bit components
// Uses smallest-three encoding
inline float4 unpackRotation(uint packed) {
    uint largestIdx = (packed >> 30) & 0x3;
    float a = float((packed >> 20) & 0x3FF) / 1023.0;
    float b = float((packed >> 10) & 0x3FF) / 1023.0;
    float c = float(packed & 0x3FF) / 1023.0;

    // Map back from [0, 1] to [-0.707, 0.707]
    float3 components = float3(
        (a * 2.0 - 1.0) * 0.707,
        (b * 2.0 - 1.0) * 0.707,
        (c * 2.0 - 1.0) * 0.707
    );

    // Reconstruct largest component from unit quaternion constraint
    float sumSq = dot(components, components);
    float largest = sqrt(max(1.0 - sumSq, 0.0));

    // Build quaternion
    float4 q;
    uint j = 0;
    for (uint i = 0; i < 4; i++) {
        if (i == largestIdx) {
            q[i] = largest;
        } else {
            q[i] = components[j];
            j++;
        }
    }

    return q;
}

// Unpack scale from 11-10-11 bits with exponential mapping
inline float3 unpackScale(uint packed, float3 minScale, float3 maxScale) {
    float x = float((packed >> 21) & 0x7FF) / 2047.0;
    float y = float((packed >> 11) & 0x3FF) / 1023.0;
    float z = float(packed & 0x7FF) / 2047.0;

    // Use log-space interpolation for exponential mapping
    float3 logMin = log(max(minScale, float3(0.0001)));
    float3 logMax = log(max(maxScale, float3(0.0001)));

    float3 logRange = logMax - logMin;
    float3 logScale = logMin + float3(x, y, z) * logRange;

    return exp(logScale);
}

// Unpack RGBA color from 32 bits (8 bits per channel)
inline half4 unpackColor(uint packed) {
    float r = float((packed >> 24) & 0xFF) / 255.0;
    float g = float((packed >> 16) & 0xFF) / 255.0;
    float b = float((packed >> 8) & 0xFF) / 255.0;
    float a = float(packed & 0xFF) / 255.0;

    return half4(r, g, b, a);
}

// Convert quaternion and scale to covariance matrix components
inline void quaternionScaleToCovariance(float4 quat, float3 scale, thread half3& covA, thread half3& covB) {
    // Build rotation matrix from quaternion
    float x = quat.x, y = quat.y, z = quat.z, w = quat.w;

    float3x3 R;
    R[0][0] = 1.0 - 2.0*(y*y + z*z);
    R[0][1] = 2.0*(x*y - w*z);
    R[0][2] = 2.0*(x*z + w*y);
    R[1][0] = 2.0*(x*y + w*z);
    R[1][1] = 1.0 - 2.0*(x*x + z*z);
    R[1][2] = 2.0*(y*z - w*x);
    R[2][0] = 2.0*(x*z - w*y);
    R[2][1] = 2.0*(y*z + w*x);
    R[2][2] = 1.0 - 2.0*(x*x + y*y);

    // Build scale matrix
    float3x3 S = float3x3(float3(scale.x, 0, 0),
                          float3(0, scale.y, 0),
                          float3(0, 0, scale.z));

    // Transform = R * S
    float3x3 M = R * S;

    // Covariance = M * M^T
    float3x3 cov3D = M * transpose(M);

    // Pack into covA (upper triangle) and covB (lower diagonal)
    covA = half3(cov3D[0][0], cov3D[0][1], cov3D[0][2]);
    covB = half3(cov3D[1][1], cov3D[1][2], cov3D[2][2]);
}

// Kernel to decompress a chunk of splats
// Decompresses packed splat data into the standard Splat format
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void decompressChunk(
    constant ChunkHeader& header [[buffer(0)]],
    constant PackedSplat* packedSplats [[buffer(1)]],
    device Splat* outputSplats [[buffer(2)]],
    constant uint& outputOffset [[buffer(3)]],  // Offset in output buffer
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= header.splatCount) return;

    PackedSplat packed = packedSplats[gid];

    // Decompress position
    float3 position = unpackPosition(packed.positionPacked, header.minPosition, header.maxPosition);

    // Decompress rotation and scale
    float4 rotation = unpackRotation(packed.rotationPacked);
    float3 scale = unpackScale(packed.scalePacked, header.minScale, header.maxScale);

    // Decompress color
    half4 color = unpackColor(packed.colorPacked);

    // Convert to covariance
    half3 covA, covB;
    quaternionScaleToCovariance(rotation, scale, covA, covB);

    // Write output splat
    uint outputIdx = outputOffset + gid;
    outputSplats[outputIdx].position = packed_float3(position.x, position.y, position.z);
    outputSplats[outputIdx].color = packed_half4(color.r, color.g, color.b, color.a);
    outputSplats[outputIdx].covA = packed_half3(covA.x, covA.y, covA.z);
    outputSplats[outputIdx].covB = packed_half3(covB.x, covB.y, covB.z);
}

// Kernel to decompress multiple chunks in parallel
// Each threadgroup processes one chunk
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void decompressChunks(
    constant ChunkHeader* headers [[buffer(0)]],
    constant PackedSplat* packedSplats [[buffer(1)]],  // Contiguous array of all packed splats
    device Splat* outputSplats [[buffer(2)]],
    constant uint* chunkOffsets [[buffer(3)]],  // Offset of each chunk in packedSplats array
    constant uint* outputOffsets [[buffer(4)]],  // Offset of each chunk in output buffer
    constant uint& chunkCount [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],  // x = local thread, y = chunk index
    uint2 tid [[threadgroup_position_in_grid]]
) {
    uint chunkIdx = tid.y;
    uint localIdx = gid.x;

    if (chunkIdx >= chunkCount) return;

    ChunkHeader header = headers[chunkIdx];
    if (localIdx >= header.splatCount) return;

    uint packedOffset = chunkOffsets[chunkIdx];
    PackedSplat packed = packedSplats[packedOffset + localIdx];

    // Decompress
    float3 position = unpackPosition(packed.positionPacked, header.minPosition, header.maxPosition);
    float4 rotation = unpackRotation(packed.rotationPacked);
    float3 scale = unpackScale(packed.scalePacked, header.minScale, header.maxScale);
    half4 color = unpackColor(packed.colorPacked);

    half3 covA, covB;
    quaternionScaleToCovariance(rotation, scale, covA, covB);

    // Write output
    uint outputIdx = outputOffsets[chunkIdx] + localIdx;
    outputSplats[outputIdx].position = packed_float3(position.x, position.y, position.z);
    outputSplats[outputIdx].color = packed_half4(color.r, color.g, color.b, color.a);
    outputSplats[outputIdx].covA = packed_half3(covA.x, covA.y, covA.z);
    outputSplats[outputIdx].covB = packed_half3(covB.x, covB.y, covB.z);
}

// Streaming decompression kernel - decompresses on-the-fly during rendering
// Instead of decompressing to a separate buffer, this could be integrated
// into the vertex shader for memory-bound scenarios
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void streamDecompressSplats(
    constant ChunkHeader* headers [[buffer(0)]],
    constant PackedSplat* packedSplats [[buffer(1)]],
    constant uint* chunkLookup [[buffer(2)]],  // Maps global splat index to chunk index
    constant uint* chunkOffsets [[buffer(3)]],
    device Splat* outputSplats [[buffer(4)]],
    constant uint& splatCount [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= splatCount) return;

    // Find which chunk this splat belongs to
    uint chunkIdx = chunkLookup[gid];
    ChunkHeader header = headers[chunkIdx];

    // Calculate local index within chunk
    uint chunkStart = chunkOffsets[chunkIdx];
    uint localIdx = gid - chunkStart;

    if (localIdx >= header.splatCount) return;

    PackedSplat packed = packedSplats[chunkStart + localIdx];

    // Decompress
    float3 position = unpackPosition(packed.positionPacked, header.minPosition, header.maxPosition);
    float4 rotation = unpackRotation(packed.rotationPacked);
    float3 scale = unpackScale(packed.scalePacked, header.minScale, header.maxScale);
    half4 color = unpackColor(packed.colorPacked);

    half3 covA, covB;
    quaternionScaleToCovariance(rotation, scale, covA, covB);

    outputSplats[gid].position = packed_float3(position.x, position.y, position.z);
    outputSplats[gid].color = packed_half4(color.r, color.g, color.b, color.a);
    outputSplats[gid].covA = packed_half3(covA.x, covA.y, covA.z);
    outputSplats[gid].covB = packed_half3(covB.x, covB.y, covB.z);
}
