#include "ShaderCommon.h"

// Counting Sort Implementation for O(n) Gaussian Splat Sorting
// Inspired by PlayCanvas gsplat-sort-worker.js
//
// This replaces the O(n log n) MPS argSort with an O(n) counting sort
// that uses quantized depth buckets for faster sorting.
//
// Configuration notes:
// - Bin count is passed via CountingSortParams.binCount (typically 4K-64K)
// - 16 bits provides good depth precision while keeping histogram small
// - Threadgroup size is 256 threads for optimal GPU occupancy
//
// Camera-relative binning (optional):
// - Allocates more precision to near-camera bins where visual quality matters most
// - Weight tiers: camera=40x, adjacent=20x, nearby=8x, medium=3x, far=1x
// - Inspired by PlayCanvas gsplat-sort-bin-weights approach

// Parameters structure passed from CPU
struct CountingSortParams {
    float minDepth;
    float maxDepth;
    float invRange;       // binCount / (maxDepth - minDepth)
    uint splatCount;
    uint binCount;        // Number of bins to use (can be less than max for small scenes)
};

// Camera-relative bin weighting parameters
// Allocates more sort precision to splats near the camera
constant uint NUM_DISTANCE_BINS = 32;

struct CameraRelativeBinParams {
    uint binBase[NUM_DISTANCE_BINS + 1];     // Starting index for each distance bin
    uint binDivider[NUM_DISTANCE_BINS + 1];  // Precision allocation per distance bin
    uint cameraBin;                           // Which distance bin contains the camera
    float invRange;                           // NUM_DISTANCE_BINS / (maxDepth - minDepth)
    float minDepth;
    uint totalBuckets;                        // Total number of sort buckets
};

// Pass 1: Compute histogram of depth values AND cache bin indices
// Each thread processes multiple splats and atomically increments histogram bins
// The bin indices are cached to avoid recomputing depth in the scatter pass
[[kernel]]
void countingSortHistogram(
    device const Splat* splats [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant CountingSortParams& params [[buffer(2)]],
    constant float3& cameraPosition [[buffer(3)]],
    constant float3& cameraForward [[buffer(4)]],
    constant bool& sortByDistance [[buffer(5)]],
    device ushort* cachedBins [[buffer(6)]],  // NEW: Cache bin indices (ushort saves memory vs uint)
    uint tid [[thread_position_in_grid]],
    uint threadCount [[threads_per_grid]]
) {
    // Each thread processes multiple splats for better efficiency
    for (uint i = tid; i < params.splatCount; i += threadCount) {
        float3 splatPos = float3(splats[i].position);

        float depth;
        if (sortByDistance) {
            float3 delta = splatPos - cameraPosition;
            depth = length(delta);
        } else {
            float3 delta = splatPos - cameraPosition;
            depth = dot(delta, cameraForward);
        }

        // Quantize depth to bin index
        // Map depth to [0, binCount-1] range
        float normalizedDepth = (depth - params.minDepth) * params.invRange;
        uint bin = clamp(uint(normalizedDepth), 0u, params.binCount - 1);

        // For back-to-front rendering, invert the bin order
        bin = params.binCount - 1 - bin;

        // Cache the bin index for the scatter pass (avoids recomputing depth)
        cachedBins[i] = ushort(bin);

        atomic_fetch_add_explicit(&histogram[bin], 1, memory_order_relaxed);
    }
}

// Camera-relative weighted histogram kernel
// Uses PlayCanvas-style bin weighting for better near-camera precision
[[kernel]]
void countingSortHistogramWeighted(
    device const Splat* splats [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant CountingSortParams& params [[buffer(2)]],
    constant float3& cameraPosition [[buffer(3)]],
    constant float3& cameraForward [[buffer(4)]],
    constant bool& sortByDistance [[buffer(5)]],
    device ushort* cachedBins [[buffer(6)]],
    constant CameraRelativeBinParams& binParams [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint threadCount [[threads_per_grid]]
) {
    for (uint i = tid; i < params.splatCount; i += threadCount) {
        float3 splatPos = float3(splats[i].position);

        float depth;
        if (sortByDistance) {
            float3 delta = splatPos - cameraPosition;
            depth = length(delta);
        } else {
            float3 delta = splatPos - cameraPosition;
            depth = dot(delta, cameraForward);
        }

        // Map depth to distance bin [0, NUM_DISTANCE_BINS-1]
        float normalizedDist = (depth - binParams.minDepth) * binParams.invRange;
        uint distBin = clamp(uint(normalizedDist), 0u, NUM_DISTANCE_BINS - 1);
        float binFraction = normalizedDist - float(distBin);

        // Apply camera-relative precision weighting
        // Bins near camera get more precision (more sub-buckets)
        uint sortKey = binParams.binBase[distBin] + uint(float(binParams.binDivider[distBin]) * binFraction);
        sortKey = min(sortKey, binParams.totalBuckets - 1);

        // For back-to-front rendering, invert the sort key order
        uint bin = binParams.totalBuckets - 1 - sortKey;

        // Cache the bin index for the scatter pass
        cachedBins[i] = ushort(bin);

        atomic_fetch_add_explicit(&histogram[bin], 1, memory_order_relaxed);
    }
}

// Legacy version without caching (for compatibility)
[[kernel]]
void countingSortHistogramNoCaching(
    device const Splat* splats [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant CountingSortParams& params [[buffer(2)]],
    constant float3& cameraPosition [[buffer(3)]],
    constant float3& cameraForward [[buffer(4)]],
    constant bool& sortByDistance [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint threadCount [[threads_per_grid]]
) {
    for (uint i = tid; i < params.splatCount; i += threadCount) {
        float3 splatPos = float3(splats[i].position);

        float depth;
        if (sortByDistance) {
            float3 delta = splatPos - cameraPosition;
            depth = length(delta);
        } else {
            float3 delta = splatPos - cameraPosition;
            depth = dot(delta, cameraForward);
        }

        float normalizedDepth = (depth - params.minDepth) * params.invRange;
        uint bin = clamp(uint(normalizedDepth), 0u, params.binCount - 1);
        bin = params.binCount - 1 - bin;

        atomic_fetch_add_explicit(&histogram[bin], 1, memory_order_relaxed);
    }
}

// Pass 2: Parallel prefix sum on histogram
// This converts counts to starting indices for each bin
// For small bin counts, this can also be done on CPU
[[kernel]]
void countingSortPrefixSum(
    device uint* histogram [[buffer(0)]],
    device uint* prefixSum [[buffer(1)]],
    constant uint& binCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    // Simple single-thread prefix sum for now
    // For larger bin counts, use parallel prefix sum (Hillis-Steele or Blelloch)
    if (tid != 0) return;

    uint sum = 0;
    for (uint i = 0; i < binCount; i++) {
        prefixSum[i] = sum;
        sum += histogram[i];
    }
}

// Optimized parallel prefix sum using Blelloch algorithm (work-efficient)
// Two phases: up-sweep (reduce) and down-sweep
// More efficient for larger bin counts and correctly handles in-place updates
[[kernel]]
void countingSortPrefixSumParallel(
    device uint* histogram [[buffer(0)]],
    device uint* prefixSum [[buffer(1)]],
    constant uint& binCount [[buffer(2)]],
    threadgroup uint* temp [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    // Load into threadgroup memory
    for (uint index = tid; index < binCount; index += tgSize) {
        temp[index] = histogram[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase - build sum tree
    // Each iteration combines pairs of elements with increasing stride
    for (uint stride = 1; stride < binCount; stride *= 2) {
        uint strideDouble = stride * 2;
        for (uint index = tid * strideDouble + strideDouble - 1; index < binCount; index += tgSize * strideDouble) {
            temp[index] += temp[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Clear the last element (root of tree) for exclusive scan
    if (tid == 0) {
        temp[binCount - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep phase - propagate partial sums down the tree
    for (uint stride = binCount / 2; stride >= 1; stride /= 2) {
        uint strideDouble = stride * 2;
        for (uint index = tid * strideDouble + strideDouble - 1; index < binCount; index += tgSize * strideDouble) {
            uint t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back to global memory
    for (uint index = tid; index < binCount; index += tgSize) {
        prefixSum[index] = temp[index];
    }
}

// MARK: - Blocked Parallel Prefix Sum (for large bin counts)
// When binCount exceeds threadgroup memory limits, we use a 3-phase blocked approach:
// Phase 1: Local prefix sum within each block, output block totals
// Phase 2: Prefix sum of block totals (small array, single-thread OK)
// Phase 3: Add block prefix to each element

// Phase 1: Compute local prefix sum per block and output block totals
[[kernel]]
void countingSortBlockPrefixSum(
    device const uint* histogram [[buffer(0)]],
    device uint* prefixSum [[buffer(1)]],
    device uint* blockSums [[buffer(2)]],
    constant uint& binCount [[buffer(3)]],
    constant uint& blockSize [[buffer(4)]],
    threadgroup uint* temp [[threadgroup(0)]],
    uint blockId [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    uint blockStart = blockId * blockSize;
    uint blockEnd = min(blockStart + blockSize, binCount);
    uint localCount = blockEnd - blockStart;

    // Load this block's data into threadgroup memory
    for (uint i = tid; i < localCount; i += tgSize) {
        temp[i] = histogram[blockStart + i];
    }
    // Zero out any remaining threadgroup memory for power-of-2 alignment
    for (uint i = localCount + tid; i < blockSize; i += tgSize) {
        temp[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Up-sweep (reduce) phase
    for (uint stride = 1; stride < blockSize; stride *= 2) {
        uint strideDouble = stride * 2;
        for (uint index = tid * strideDouble + strideDouble - 1; index < blockSize; index += tgSize * strideDouble) {
            temp[index] += temp[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Save block total before zeroing for exclusive scan
    if (tid == 0) {
        blockSums[blockId] = temp[blockSize - 1];
        temp[blockSize - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep phase
    for (uint stride = blockSize / 2; stride >= 1; stride /= 2) {
        uint strideDouble = stride * 2;
        for (uint index = tid * strideDouble + strideDouble - 1; index < blockSize; index += tgSize * strideDouble) {
            uint t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write local prefix sums back to global memory
    for (uint i = tid; i < localCount; i += tgSize) {
        prefixSum[blockStart + i] = temp[i];
    }
}

// Phase 2: Prefix sum of block totals (small array, single-thread is fine)
[[kernel]]
void countingSortBlockSumsPrefixSum(
    device uint* blockSums [[buffer(0)]],
    constant uint& blockCount [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    uint sum = 0;
    for (uint i = 0; i < blockCount; i++) {
        uint val = blockSums[i];
        blockSums[i] = sum;
        sum += val;
    }
}

// Phase 3: Add block prefix to each element
[[kernel]]
void countingSortAddBlockPrefix(
    device uint* prefixSum [[buffer(0)]],
    device const uint* blockSums [[buffer(1)]],
    constant uint& binCount [[buffer(2)]],
    constant uint& blockSize [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= binCount) return;
    uint blockId = gid / blockSize;
    prefixSum[gid] += blockSums[blockId];
}

// Pass 3: Scatter splat indices to sorted positions (optimized with cached bins)
// Uses cached bin indices from histogram pass - no depth recomputation needed
[[kernel]]
void countingSortScatter(
    device const ushort* cachedBins [[buffer(0)]],  // Cached bin indices from histogram pass
    device atomic_uint* binOffsets [[buffer(1)]],   // Current write position per bin (initialized from prefix sum)
    device int32_t* sortedIndices [[buffer(2)]],
    constant CountingSortParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint threadCount [[threads_per_grid]]
) {
    for (uint i = tid; i < params.splatCount; i += threadCount) {
        // Use cached bin index - no depth calculation needed!
        uint bin = uint(cachedBins[i]);

        // Atomically get position and increment
        uint pos = atomic_fetch_add_explicit(&binOffsets[bin], 1, memory_order_relaxed);
        sortedIndices[pos] = int32_t(i);
    }
}

// Legacy scatter without caching (recomputes depth)
[[kernel]]
void countingSortScatterNoCaching(
    device const Splat* splats [[buffer(0)]],
    device atomic_uint* binOffsets [[buffer(1)]],
    device int32_t* sortedIndices [[buffer(2)]],
    constant CountingSortParams& params [[buffer(3)]],
    constant float3& cameraPosition [[buffer(4)]],
    constant float3& cameraForward [[buffer(5)]],
    constant bool& sortByDistance [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint threadCount [[threads_per_grid]]
) {
    for (uint i = tid; i < params.splatCount; i += threadCount) {
        float3 splatPos = float3(splats[i].position);

        float depth;
        if (sortByDistance) {
            float3 delta = splatPos - cameraPosition;
            depth = length(delta);
        } else {
            float3 delta = splatPos - cameraPosition;
            depth = dot(delta, cameraForward);
        }

        float normalizedDepth = (depth - params.minDepth) * params.invRange;
        uint bin = clamp(uint(normalizedDepth), 0u, params.binCount - 1);
        bin = params.binCount - 1 - bin;

        uint pos = atomic_fetch_add_explicit(&binOffsets[bin], 1, memory_order_relaxed);
        sortedIndices[pos] = int32_t(i);
    }
}

// Combined single-pass counting sort for small scenes
// Uses threadgroup memory to build local histogram, then global reduction
// This avoids multiple kernel launches for small-to-medium scenes
[[kernel]]
void countingSortCombined(
    device const Splat* splats [[buffer(0)]],
    device atomic_uint* globalHistogram [[buffer(1)]],
    device uint* prefixSum [[buffer(2)]],
    device atomic_uint* binOffsets [[buffer(3)]],
    device int32_t* sortedIndices [[buffer(4)]],
    constant CountingSortParams& params [[buffer(5)]],
    constant float3& cameraPosition [[buffer(6)]],
    constant float3& cameraForward [[buffer(7)]],
    constant bool& sortByDistance [[buffer(8)]],
    threadgroup uint* localHistogram [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]],
    uint numThreadgroups [[threadgroups_per_grid]]
) {
    // Initialize local histogram
    for (uint i = tid; i < params.binCount; i += tgSize) {
        localHistogram[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build local histogram for this threadgroup's splats
    uint splatsPerGroup = (params.splatCount + numThreadgroups - 1) / numThreadgroups;
    uint startSplat = tgid * splatsPerGroup;
    uint endSplat = min(startSplat + splatsPerGroup, params.splatCount);

    for (uint i = startSplat + tid; i < endSplat; i += tgSize) {
        float3 splatPos = float3(splats[i].position);

        float depth;
        if (sortByDistance) {
            float3 delta = splatPos - cameraPosition;
            depth = length(delta);
        } else {
            float3 delta = splatPos - cameraPosition;
            depth = dot(delta, cameraForward);
        }

        float normalizedDepth = (depth - params.minDepth) * params.invRange;
        uint bin = clamp(uint(normalizedDepth), 0u, params.binCount - 1);
        bin = params.binCount - 1 - bin;

        atomic_fetch_add_explicit((threadgroup atomic_uint*)&localHistogram[bin], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Merge local histogram into global histogram
    for (uint i = tid; i < params.binCount; i += tgSize) {
        if (localHistogram[i] > 0) {
            atomic_fetch_add_explicit(&globalHistogram[i], localHistogram[i], memory_order_relaxed);
        }
    }
}

// Reset histogram buffer to zeros
[[kernel]]
void countingSortResetHistogram(
    device uint* histogram [[buffer(0)]],
    constant uint& binCount [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < binCount) {
        histogram[tid] = 0;
    }
}

// Copy prefix sum to bin offsets (for scatter pass initialization)
[[kernel]]
void countingSortInitBinOffsets(
    device const uint* prefixSum [[buffer(0)]],
    device uint* binOffsets [[buffer(1)]],
    constant uint& binCount [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < binCount) {
        binOffsets[tid] = prefixSum[tid];
    }
}
