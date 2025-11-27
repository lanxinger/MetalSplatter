#include <metal_stdlib>
#include <simd/simd.h>
#include "ShaderCommon.h"

using namespace metal;

// Spherical harmonics basis functions
// Based on: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/sh_utils.py
// Coefficient order (shared with SphericalHarmonicsEvaluator.swift/FastSHRenderPath.metal):
// 0: dc,
// 1: y, 2: z, 3: x,
// 4: xy, 5: yz, 6: 2zz-xx-yy, 7: xz, 8: xx-yy,
// 9: y*(3xx-yy), 10: xy*z, 11: y*(4zz-xx-yy), 12: z*(2zz-3xx-3yy),
// 13: x*(4zz-xx-yy), 14: z*(xx-yy), 15: x*(xx-3yy)

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[5] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
};
constant float SH_C3[7] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
};

struct SHEvaluateParams {
    float3 viewDirection;  // Normalized view direction (camera forward)
    uint paletteSize;      // Number of unique SH coefficient sets (e.g., 64K)
    uint degree;           // SH degree (0-3)
};

// Evaluate spherical harmonics for a given direction
float4 evaluateSH(float3 dir, device const float3* sh_coeffs, uint degree) {
    // Normalize direction
    float3 d = normalize(dir);
    
    // Band 0 (DC term) plus ambient offset (see SOG spec §3.4)
    float3 result = float3(0.5f) + SH_C0 * sh_coeffs[0];
    
    if (degree >= 1) {
        // Band 1
        result += SH_C1 * d.y * sh_coeffs[1];
        result += SH_C1 * d.z * sh_coeffs[2];
        result += SH_C1 * d.x * sh_coeffs[3];
    }
    
    if (degree >= 2) {
        // Band 2
        float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
        float xy = d.x * d.y, yz = d.y * d.z, xz = d.x * d.z;
        
        result += SH_C2[0] * xy * sh_coeffs[4];
        result += SH_C2[1] * yz * sh_coeffs[5];
        result += SH_C2[2] * (2.0f * zz - xx - yy) * sh_coeffs[6];
        result += SH_C2[3] * xz * sh_coeffs[7];
        result += SH_C2[4] * (xx - yy) * sh_coeffs[8];
    }
    
    if (degree >= 3) {
        // Band 3
        float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
        float xy = d.x * d.y;
        
        result += SH_C3[0] * d.y * (3.0f * xx - yy) * sh_coeffs[9];
        result += SH_C3[1] * xy * d.z * sh_coeffs[10];
        result += SH_C3[2] * d.y * (4.0f * zz - xx - yy) * sh_coeffs[11];
        result += SH_C3[3] * d.z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_coeffs[12];
        result += SH_C3[4] * d.x * (4.0f * zz - xx - yy) * sh_coeffs[13];
        result += SH_C3[5] * d.z * (xx - yy) * sh_coeffs[14];
        result += SH_C3[6] * d.x * (xx - 3.0f * yy) * sh_coeffs[15];
    }
    
    // Clamp negative values and limit to [0, 1]
    result = clamp(result, 0.0f, 1.0f);
    return float4(result, 1.0f);
}

// Compute kernel to pre-evaluate SH for all palette entries
kernel void evaluateSphericalHarmonicsPalette(
    device const float3* sh_palette [[buffer(0)]],           // Input: SH coefficients palette
    device float4* evaluated_sh [[buffer(1)]],               // Output: Evaluated RGB colors
    constant SHEvaluateParams& params [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= params.paletteSize) return;
    
    // Get pointer to this palette entry's SH coefficients
    uint coeffsPerEntry = (params.degree + 1) * (params.degree + 1);
    device const float3* sh_coeffs = sh_palette + (idx * coeffsPerEntry);
    
    // Evaluate SH for the current view direction
    float4 color = evaluateSH(params.viewDirection, sh_coeffs, params.degree);
    
    // Store the evaluated color
    evaluated_sh[idx] = color;
}

// Alternative: Evaluate SH for a 2D texture of directions (for more accuracy at edges)
kernel void evaluateSphericalHarmonicsDirectional(
    device const float3* sh_palette [[buffer(0)]],           // Input: SH coefficients palette
    texture2d<float, access::write> evaluated_sh [[texture(0)]], // Output: Evaluated RGB colors
    constant SHEvaluateParams& params [[buffer(2)]],
    device const float3* viewDirections [[buffer(3)]],       // Per-pixel view directions
    uint2 tid [[thread_position_in_grid]])
{
    uint width = evaluated_sh.get_width();
    uint height = evaluated_sh.get_height();
    
    if (tid.x >= width || tid.y >= height) return;
    
    // Calculate palette index from texture coordinates
    uint idx = tid.y * width + tid.x;
    if (idx >= params.paletteSize) return;
    
    // Get pointer to this palette entry's SH coefficients
    uint coeffsPerEntry = (params.degree + 1) * (params.degree + 1);
    device const float3* sh_coeffs = sh_palette + (idx * coeffsPerEntry);
    
    // Get view direction for this pixel (could be per-pixel or use params.viewDirection)
    float3 dir = params.viewDirection; // Simple version: use single direction
    
    // Evaluate SH
    float4 color = evaluateSH(dir, sh_coeffs, params.degree);
    
    // Write to texture
    evaluated_sh.write(color, tid);
}
