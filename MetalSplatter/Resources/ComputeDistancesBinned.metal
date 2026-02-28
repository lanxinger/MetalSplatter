#include "ShaderCommon.h"

// Camera-relative binned precision sorting inspired by PlayCanvas approach
// Allocates more sort precision to near-camera splats where visual quality matters most

constant uint NUM_BINS = 32;

struct BinParameters {
    uint binBase[NUM_BINS + 1];      // Starting index for each bin (extra entry for safety)
    uint binDivider[NUM_BINS + 1];   // Precision allocation per bin
};

// Weight tiers for camera-relative precision (distance from camera bin -> weight multiplier)
struct WeightTier {
    uint maxDistance;
    float weight;
};

constant WeightTier WEIGHT_TIERS[5] = {
    {0, 40.0},              // Camera bin
    {2, 20.0},              // Adjacent bins
    {5, 8.0},               // Nearby bins
    {10, 3.0},              // Medium distance
    {0xFFFFFFFF, 1.0}       // Far bins (maxDistance = UINT_MAX)
};

// Pre-calculate weight lookup table by distance from camera
kernel void setupCameraRelativeBins(constant float& minDist [[buffer(0)]],
                                   constant float& maxDist [[buffer(1)]],
                                   constant float3& cameraPosition [[buffer(2)]],
                                   constant bool& sortByDistance [[buffer(3)]],
                                   constant uint& compareBits [[buffer(4)]],
                                   device BinParameters& binParams [[buffer(5)]],
                                   uint index [[thread_position_in_grid]]) {
    if (index > 0) return; // Only one thread needed

    float range = maxDist - minDist;
    uint bucketCount = (1 << compareBits) + 1;

    // Determine camera bin based on sort mode
    uint cameraBin;
    if (sortByDistance) {
        // For radial sort with inverted distances, camera (dist=0) maps to the last bin
        cameraBin = NUM_BINS - 1;
    } else {
        // For linear sort, calculate where camera falls in the projected distance range
        float cameraOffsetFromRangeStart = 0 - minDist;
        float cameraBinFloat = (cameraOffsetFromRangeStart / range) * NUM_BINS;
        cameraBin = clamp((uint)floor(cameraBinFloat), 0u, NUM_BINS - 1);
    }

    // Calculate weight by distance lookup
    float weightByDistance[NUM_BINS];
    for (uint dist = 0; dist < NUM_BINS; ++dist) {
        float weight = 1.0;
        for (uint j = 0; j < 5; ++j) {
            if (dist <= WEIGHT_TIERS[j].maxDistance) {
                weight = WEIGHT_TIERS[j].weight;
                break;
            }
        }
        weightByDistance[dist] = weight;
    }

    // Assign weights to bins based on distance from camera
    float bitsPerBin[NUM_BINS];
    for (uint i = 0; i < NUM_BINS; ++i) {
        uint distFromCamera = abs((int)i - (int)cameraBin);
        bitsPerBin[i] = weightByDistance[distFromCamera];
    }

    // Normalize to fit within budget
    float totalWeight = 0;
    for (uint i = 0; i < NUM_BINS; ++i) {
        totalWeight += bitsPerBin[i];
    }

    uint accumulated = 0;
    for (uint i = 0; i < NUM_BINS; ++i) {
        binParams.binDivider[i] = max(1u, (uint)floor((bitsPerBin[i] / totalWeight) * bucketCount));
        binParams.binBase[i] = accumulated;
        accumulated += binParams.binDivider[i];
    }

    // Adjust last bin to fit exactly
    if (accumulated > bucketCount) {
        uint excess = accumulated - bucketCount;
        binParams.binDivider[NUM_BINS - 1] = max(1u, binParams.binDivider[NUM_BINS - 1] - excess);
    }

    // Add safety entry for edge case where bin >= numBins
    binParams.binBase[NUM_BINS] = binParams.binBase[NUM_BINS - 1] + binParams.binDivider[NUM_BINS - 1];
    binParams.binDivider[NUM_BINS] = 0;
}

// Compute binned distances with camera-relative precision
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeSplatDistancesBinned(uint index [[thread_position_in_grid]],
                                       constant Splat* splatArray [[buffer(0)]],
                                       device uint* binnedDistances [[buffer(1)]],
                                       constant float3& cameraPosition [[buffer(2)]],
                                       constant float3& cameraForward [[buffer(3)]],
                                       constant bool& sortByDistance [[buffer(4)]],
                                       constant uint& splatCount [[buffer(5)]],
                                       constant float& minDist [[buffer(6)]],
                                       constant float& range [[buffer(7)]],
                                       constant BinParameters& binParams [[buffer(8)]]) {

    if (index >= splatCount) return;

    Splat splat = splatArray[index];
    float3 splatPos = float3(splat.position);

    // Calculate raw distance
    float dist;
    if (sortByDistance) {
        float3 delta = splatPos - cameraPosition;
        dist = length(delta);
        // Invert for radial sort (far objects rendered first)
        dist = range - dist;
    } else {
        // Project onto forward vector for depth sorting
        float3 delta = splatPos - cameraPosition;
        dist = dot(delta, cameraForward);
    }

    // Normalize distance to [0, range]
    dist = dist - minDist;

    // Map to bin with precision allocation
    float invBinRange = NUM_BINS / range;
    float d = dist * invBinRange;
    uint bin = min((uint)d, NUM_BINS - 1);
    float binFraction = d - (float)bin;

    // Calculate sort key with camera-relative precision
    uint sortKey = binParams.binBase[bin] + (uint)(binParams.binDivider[bin] * binFraction);

    binnedDistances[index] = sortKey;
}

// Simple version without binning for comparison
[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void computeSplatDistancesSimple(uint index [[thread_position_in_grid]],
                                       constant Splat* splatArray [[buffer(0)]],
                                       device float* distances [[buffer(1)]],
                                       constant float3& cameraPosition [[buffer(2)]],
                                       constant float3& cameraForward [[buffer(3)]],
                                       constant bool& sortByDistance [[buffer(4)]],
                                       constant uint& splatCount [[buffer(5)]]) {

    if (index >= splatCount) return;

    Splat splat = splatArray[index];
    float3 splatPos = float3(splat.position);

    if (sortByDistance) {
        float3 delta = splatPos - cameraPosition;
        float distanceSquared = dot(delta, delta);
        distances[index] = distanceSquared;
    } else {
        // Project onto forward vector
        distances[index] = dot(splatPos - cameraPosition, cameraForward);
    }
}
