#include "ShaderCommon.h"

// Structure for key-value pairs (distance, index)
struct SortKeyValue {
    float key;      // distance/depth
    uint value;     // original splat index
};

// Convert float to sortable unsigned int (handles negative values correctly)
uint floatAsUint(float f) {
    uint ui = as_type<uint>(f);
    return (ui >> 31) ? (~ui) : (ui | 0x80000000u);
}

// Simple counting sort for one digit (256 buckets)
kernel void countingSort(device SortKeyValue* output [[buffer(0)]],
                        constant SortKeyValue* input [[buffer(1)]],
                        device uint* histogram [[buffer(2)]],
                        constant uint& count [[buffer(3)]],
                        constant uint& shift [[buffer(4)]],
                        uint threadId [[thread_position_in_grid]]) {
    
    if (threadId >= count) return;
    
    SortKeyValue element = input[threadId];
    uint sortKey = floatAsUint(element.key);
    uint digit = (sortKey >> shift) & 0xFF; // 8 bits = 256 buckets
    
    // Find position by counting elements with smaller digits
    uint position = 0;
    for (uint i = 0; i < digit; i++) {
        position += histogram[i];
    }
    
    // Count elements with same digit that come before this one
    for (uint i = 0; i < threadId; i++) {
        uint otherSortKey = floatAsUint(input[i].key);
        uint otherDigit = (otherSortKey >> shift) & 0xFF;
        if (otherDigit == digit) {
            position++;
        }
    }
    
    output[position] = element;
}

// Compute histogram for counting sort
kernel void computeHistogramSimple(device uint* histogram [[buffer(0)]],
                                  constant SortKeyValue* input [[buffer(1)]],
                                  constant uint& count [[buffer(2)]],
                                  constant uint& shift [[buffer(3)]],
                                  uint threadId [[thread_position_in_grid]]) {
    
    // Clear histogram
    if (threadId < 256) {
        histogram[threadId] = 0;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Count elements
    if (threadId < count) {
        uint sortKey = floatAsUint(input[threadId].key);
        uint digit = (sortKey >> shift) & 0xFF;
        
        // Use atomic increment to avoid race conditions
        atomic_fetch_add_explicit((device atomic_uint*)&histogram[digit], 1, memory_order_relaxed);
    }
}

// Reorder splats based on sorted indices
kernel void reorderSplats(device Splat* output [[buffer(0)]],
                         constant Splat* input [[buffer(1)]],
                         constant SortKeyValue* sortedPairs [[buffer(2)]],
                         constant uint& count [[buffer(3)]],
                         uint threadId [[thread_position_in_grid]]) {
    
    if (threadId >= count) return;
    
    uint originalIndex = sortedPairs[threadId].value;
    output[threadId] = input[originalIndex];
}

// Alternative version for optimized splats
kernel void reorderOptimizedSplats(device SplatOptimized* outputGeometry [[buffer(0)]],
                                  device PackedColor* outputColor [[buffer(1)]],
                                  constant SplatOptimized* inputGeometry [[buffer(2)]],
                                  constant PackedColor* inputColor [[buffer(3)]],
                                  constant SortKeyValue* sortedPairs [[buffer(4)]],
                                  constant uint& count [[buffer(5)]],
                                  uint threadId [[thread_position_in_grid]]) {
    
    if (threadId >= count) return;
    
    uint originalIndex = sortedPairs[threadId].value;
    outputGeometry[threadId] = inputGeometry[originalIndex];
    outputColor[threadId] = inputColor[originalIndex];
}

// Initialize key-value pairs from distance buffer
kernel void initializeKeyValuePairs(device SortKeyValue* output [[buffer(0)]],
                                   constant float* distances [[buffer(1)]],
                                   constant uint& count [[buffer(2)]],
                                   uint threadId [[thread_position_in_grid]]) {
    
    if (threadId >= count) return;
    
    // For descending sort, negate the distance
    output[threadId].key = -distances[threadId];
    output[threadId].value = threadId;
}

// Simple bitonic sort for smaller arrays (more reliable than radix sort)
kernel void bitonicSortStep(device SortKeyValue* data [[buffer(0)]],
                           constant uint& count [[buffer(1)]],
                           constant uint& k [[buffer(2)]],
                           constant uint& j [[buffer(3)]],
                           uint threadId [[thread_position_in_grid]]) {
    
    uint i = threadId;
    if (i >= count) return;
    
    uint ixj = i ^ j;
    if (ixj > i) {
        if ((i & k) == 0) {
            // Ascending
            if (data[i].key > data[ixj].key) {
                SortKeyValue temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        } else {
            // Descending
            if (data[i].key < data[ixj].key) {
                SortKeyValue temp = data[i];
                data[i] = data[ixj];
                data[ixj] = temp;
            }
        }
    }
}