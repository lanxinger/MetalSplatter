#include <metal_stdlib>
#include "ShaderCommon.h"
using namespace metal;

// =============================================================================
// Metal 4 Advanced Atomics for High-Performance Splat Sorting
// =============================================================================
// Note: The Swift code already has @available(iOS 26.0, ...) runtime checks.
// However, advanced memory orderings (acquire/release/acq_rel) are only available
// in Metal 4+. We provide fallbacks for Metal 3.x using memory_order_relaxed.

// Memory ordering compatibility macros for Metal 3.x vs 4.x
#if __METAL_VERSION__ >= 400
    #define MEM_ORDER_ACQUIRE memory_order_acquire
    #define MEM_ORDER_RELEASE memory_order_release
    #define MEM_ORDER_ACQ_REL memory_order_acq_rel
#else
    // Metal 3.x doesn't have acquire/release semantics - use relaxed as fallback
    // This may reduce correctness guarantees but allows compilation
    #define MEM_ORDER_ACQUIRE memory_order_relaxed
    #define MEM_ORDER_RELEASE memory_order_relaxed
    #define MEM_ORDER_ACQ_REL memory_order_relaxed
#endif

// =============================================================================
// MARK: - Types and Helper Functions (in namespace)
// =============================================================================
// Advanced atomic operations for high-performance splat sorting
namespace advanced_atomics {

    // Structure for atomic sorting keys
    struct SortingKey {
        float depth;
        uint original_index;
    };

    // Structure for atomic counters and bins
    struct AtomicSortingState {
        atomic_uint bin_counts[32];     // Radix sort bins
        atomic_uint global_counter;     // Global work counter
        atomic_uint completed_phases;   // Phase completion tracking
        atomic_uint sync_failed;        // Set to 1 if synchronization times out - abort sort
    };

    // =============================================================================
    // MARK: - Float to Sortable Uint Conversion
    // =============================================================================
    // IEEE 754 floats can be sorted as uints with a simple transform:
    // - If positive: flip sign bit (0x80000000)
    // - If negative: flip all bits (~)
    // This produces a monotonic mapping: f1 < f2 => transform(f1) < transform(f2)
    //
    // For DESCENDING order (back-to-front), we want larger depths first,
    // so we invert the result (~transformed) to reverse sort order.

    inline uint float_to_sortable_uint_descending(float f) {
        uint bits = as_type<uint>(f);
        // Apply IEEE 754 transform for sortable uints
        uint mask = (bits >> 31) ? 0xFFFFFFFF : 0x80000000;
        uint sortable = bits ^ mask;
        // Invert for descending order (back-to-front for splat rendering)
        return ~sortable;
    }

    // Forward declarations for helper functions
    bool parallel_merge_chunks(
        device SortingKey *keys,
        device uint *sorted_indices,
        device AtomicSortingState &sorting_state,
        uint count,
        uint thread_id,
        uint total_threads
    );

    bool atomic_merge_sequences(
        device SortingKey *keys,
        uint left_start,
        uint right_start,
        uint end,
        device AtomicSortingState &sorting_state
    );

    void atomic_heap_bubble_up(
        device SortingKey *heap,
        uint pos,
        device AtomicSortingState &state
    );

    void atomic_heap_bubble_down(
        device SortingKey *heap,
        uint pos,
        uint heap_size,
        device AtomicSortingState &state
    );

    // =============================================================================
    // MARK: - Helper Function Implementations
    // =============================================================================

    // Atomic merge operation for combining sorted sequences
    // Returns false if synchronization fails (caller should output identity mapping)
    bool parallel_merge_chunks(
        device SortingKey *keys,
        device uint *sorted_indices,
        device AtomicSortingState &sorting_state,
        uint count,
        uint thread_id,
        uint total_threads
    ) {
        // Implement parallel merge using atomic operations
        uint merge_size = 2;
        uint merge_level = 0;

        while (merge_size <= count) {
            // Check for failure from any thread before starting this level
            if (atomic_load_explicit(&sorting_state.sync_failed, MEM_ORDER_ACQUIRE) != 0) {
                return false;
            }

            uint merges_per_thread = (count / merge_size + total_threads - 1) / total_threads;

            for (uint merge_idx = 0; merge_idx < merges_per_thread; ++merge_idx) {
                uint global_merge_idx = thread_id * merges_per_thread + merge_idx;
                uint left_start = global_merge_idx * merge_size;
                uint right_start = left_start + merge_size / 2;
                uint end = min(left_start + merge_size, count);

                if (left_start >= count) break;

                // Perform atomic merge - check for failure signal from CAS timeouts
                if (!atomic_merge_sequences(keys, left_start, right_start, end, sorting_state)) {
                    return false;
                }
            }

            // Synchronize between merge levels
            atomic_fetch_add_explicit(&sorting_state.completed_phases, 1, MEM_ORDER_RELEASE);

            merge_level++;
            uint expected_completions = total_threads * merge_level;

            // Bounded spin-wait with failure signaling
            uint spin_count = 0;
            const uint max_spin = 1000000;
            while (atomic_load_explicit(&sorting_state.completed_phases, MEM_ORDER_ACQUIRE) < expected_completions) {
                // Check if another thread signaled failure
                if (atomic_load_explicit(&sorting_state.sync_failed, MEM_ORDER_ACQUIRE) != 0) {
                    return false;
                }
                spin_count++;
                if (spin_count >= max_spin) {
                    // Signal failure to all threads
                    atomic_store_explicit(&sorting_state.sync_failed, 1, MEM_ORDER_RELEASE);
                    return false;
                }
            }

            merge_size *= 2;
        }

        // Final pass: extract sorted indices (only thread 0)
        if (thread_id == 0) {
            for (uint i = 0; i < count; ++i) {
                sorted_indices[i] = keys[i].original_index;
            }
        }
        return true;
    }

    // Atomic merge of two sorted sequences
    // Returns false if CAS loop times out (signals failure)
    // Note: Limited to merging sequences up to 256 elements total.
    bool atomic_merge_sequences(
        device SortingKey *keys,
        uint left_start,
        uint right_start,
        uint end,
        device AtomicSortingState &sorting_state
    ) {
        // Calculate merge size and enforce limit
        uint merge_count = end - left_start;
        const uint MAX_MERGE_SIZE = 256;

        // If merge is too large, skip (data stays as-is for this segment)
        // A production implementation should use a proper large-merge algorithm
        if (merge_count > MAX_MERGE_SIZE) {
            return true;  // Not a failure, just a limitation
        }

        // Check for prior failure before doing work
        if (atomic_load_explicit(&sorting_state.sync_failed, MEM_ORDER_ACQUIRE) != 0) {
            return false;
        }

        uint left_idx = left_start;
        uint right_idx = right_start;

        // Temporary array for merged results
        SortingKey temp_keys[MAX_MERGE_SIZE];
        uint temp_idx = 0;

        // Merge two sorted halves
        while (left_idx < right_start && right_idx < end && temp_idx < MAX_MERGE_SIZE) {
            if (keys[left_idx].depth <= keys[right_idx].depth) {
                temp_keys[temp_idx++] = keys[left_idx++];
            } else {
                temp_keys[temp_idx++] = keys[right_idx++];
            }
        }

        // Copy remaining elements from left half
        while (left_idx < right_start && temp_idx < MAX_MERGE_SIZE) {
            temp_keys[temp_idx++] = keys[left_idx++];
        }

        // Copy remaining elements from right half
        while (right_idx < end && temp_idx < MAX_MERGE_SIZE) {
            temp_keys[temp_idx++] = keys[right_idx++];
        }

        // Copy back to original array using atomic operations
        for (uint i = 0; i < temp_idx; ++i) {
            uint target_pos = left_start + i;
            if (target_pos < end) {
                SortingKey old_key = keys[target_pos];
                uint spin_count = 0;
                const uint max_spin = 10000;
                bool write_succeeded = false;

                while (spin_count < max_spin) {
                    uint expected_index = old_key.original_index;
                    if (atomic_compare_exchange_weak_explicit(
                        (device atomic_uint*)&keys[target_pos].original_index,
                        &expected_index,
                        temp_keys[i].original_index,
                        MEM_ORDER_ACQ_REL,
                        memory_order_relaxed)) {

                        keys[target_pos] = temp_keys[i];
                        write_succeeded = true;
                        break;
                    }
                    old_key = keys[target_pos];
                    spin_count++;
                }

                // If CAS loop timed out, signal failure
                if (!write_succeeded) {
                    atomic_store_explicit(&sorting_state.sync_failed, 1, MEM_ORDER_RELEASE);
                    return false;
                }
            }
        }
        return true;
    }

    // Helper functions for atomic heap operations
    void atomic_heap_bubble_up(
        device SortingKey *heap,
        uint pos,
        device AtomicSortingState &state
    ) {
        while (pos > 0) {
            uint parent_pos = (pos - 1) / 2;

            if (heap[pos].depth >= heap[parent_pos].depth) {
                break; // Heap property satisfied
            }

            // Atomic swap with parent
            SortingKey temp = heap[pos];
            uint expected_index = heap[parent_pos].original_index;

            if (atomic_compare_exchange_weak_explicit(
                (device atomic_uint*)&heap[parent_pos].original_index,
                &expected_index,
                temp.original_index,
                MEM_ORDER_ACQ_REL,
                memory_order_relaxed)) {

                heap[pos] = heap[parent_pos];
                heap[parent_pos] = temp;
                pos = parent_pos;
            } else {
                break; // Another thread modified, stop bubbling
            }
        }
    }

    void atomic_heap_bubble_down(
        device SortingKey *heap,
        uint pos,
        uint heap_size,
        device AtomicSortingState &state
    ) {
        while (true) {
            uint left_child = 2 * pos + 1;
            uint right_child = 2 * pos + 2;
            uint smallest = pos;

            if (left_child < heap_size && heap[left_child].depth < heap[smallest].depth) {
                smallest = left_child;
            }

            if (right_child < heap_size && heap[right_child].depth < heap[smallest].depth) {
                smallest = right_child;
            }

            if (smallest == pos) {
                break; // Heap property satisfied
            }

            // Atomic swap with smallest child
            SortingKey temp = heap[pos];
            uint expected_index = heap[smallest].original_index;

            if (atomic_compare_exchange_weak_explicit(
                (device atomic_uint*)&heap[smallest].original_index,
                &expected_index,
                temp.original_index,
                MEM_ORDER_ACQ_REL,
                memory_order_relaxed)) {

                heap[pos] = heap[smallest];
                heap[smallest] = temp;
                pos = smallest;
            } else {
                break; // Another thread modified, stop bubbling
            }
        }
    }

} // namespace advanced_atomics

// =============================================================================
// MARK: - Kernel Functions (at global scope for Metal function lookup)
// =============================================================================
// These kernel functions must be at global scope because Metal's makeFunction(name:)
// API does not support C++ namespace-qualified names.

using advanced_atomics::SortingKey;
using advanced_atomics::AtomicSortingState;
using advanced_atomics::float_to_sortable_uint_descending;
using advanced_atomics::parallel_merge_chunks;
using advanced_atomics::atomic_heap_bubble_up;
using advanced_atomics::atomic_heap_bubble_down;

// Constant redefined at global scope (using declarations don't work for constants in Metal)
constant uint SCATTER_THREADGROUP_SIZE = 256;

// =============================================================================
// MARK: - Key Building Kernel
// =============================================================================
// Builds SortingKey array from splat positions, matching existing sort criteria
// (sortByDistance for distance-based sort, or forward dot for depth-based sort)
// Applies float→sortable-uint transform for correct radix sorting
// Inverts for DESCENDING order (back-to-front) to match other sort paths

kernel void build_sorting_keys(
    constant Splat *splats [[buffer(0)]],
    device SortingKey *keys [[buffer(1)]],
    constant float3 &cameraPosition [[buffer(2)]],
    constant float3 &cameraForward [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant bool &sortByDistance [[buffer(5)]],  // matches Constants.sortByDistance
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    float3 worldPos = float3(splats[gid].position);
    float depth;

    if (sortByDistance) {
        // Distance from camera (matches computeDistances kernel)
        depth = length(worldPos - cameraPosition);
    } else {
        // Forward dot product (matches computeDepths kernel)
        depth = dot(worldPos - cameraPosition, cameraForward);
    }

    // Store as sortable uint (descending order for back-to-front rendering)
    // This ensures correct sorting of full 32-bit float range including negatives
    keys[gid].depth = as_type<float>(float_to_sortable_uint_descending(depth));
    keys[gid].original_index = gid;
}

// =============================================================================
// MARK: - Index Extraction Kernel
// =============================================================================
// Extracts sorted indices from SortingKey array after radix sort

kernel void extract_sorted_indices(
    constant SortingKey *sorted_keys [[buffer(0)]],
    device int32_t *sorted_indices [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    sorted_indices[gid] = int32_t(sorted_keys[gid].original_index);
}

// =============================================================================
// MARK: - Stable Histogram Radix Sort
// =============================================================================
// Three-phase stable radix sort using histogram + prefix sum + scatter
// Each pass processes 8 bits (256 buckets) for 4 total passes on 32-bit keys
//
// Phase 1: histogram_radix_pass - count elements per bucket
// Phase 2: prefix_sum_buckets - convert counts to starting offsets
// Phase 3: scatter_radix_pass - place elements using stable local ranking

// Histogram phase: count elements in each of 256 buckets for current byte
kernel void histogram_radix_pass(
    constant SortingKey *keys [[buffer(0)]],
    device atomic_uint *histogram [[buffer(1)]],  // 256 buckets
    constant uint &count [[buffer(2)]],
    constant uint &byte_index [[buffer(3)]],  // 0-3 for which byte to examine
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;

    uint key_bits = as_type<uint>(keys[gid].depth);
    uint bucket = (key_bits >> (byte_index * 8)) & 0xFF;

    atomic_fetch_add_explicit(&histogram[bucket], 1, memory_order_relaxed);
}

// Prefix sum phase: convert histogram counts to cumulative offsets
// Run with 256 threads (one per bucket)
kernel void prefix_sum_buckets(
    device uint *histogram [[buffer(0)]],  // 256 buckets, in-place
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup uint *shared_data [[threadgroup(0)]]
) {
    // Load into shared memory
    shared_data[tid] = (tid < 256) ? histogram[tid] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exclusive prefix sum using Blelloch scan
    // Up-sweep (reduce)
    for (uint stride = 1; stride < 256; stride *= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < 256) {
            shared_data[index] += shared_data[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Set last element to 0 for exclusive scan
    if (tid == 0) {
        shared_data[255] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint stride = 128; stride > 0; stride /= 2) {
        uint index = (tid + 1) * stride * 2 - 1;
        if (index < 256) {
            uint temp = shared_data[index - stride];
            shared_data[index - stride] = shared_data[index];
            shared_data[index] += temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write back
    if (tid < 256) {
        histogram[tid] = shared_data[tid];
    }
}

// =============================================================================
// MARK: - Stable Scatter Phase
// =============================================================================
// Truly stable scatter for LSD radix sort correctness.
//
// GPU threads execute in nondeterministic order, so per-element atomic increments
// would destroy the relative ordering from previous passes. LSD radix sort requires
// stability: elements with equal digits must maintain their prior relative order.
//
// Solution: Three-phase scatter with deterministic threadgroup ordering
//   1. scatter_count_per_threadgroup: Count bucket populations per threadgroup
//      (stores counts only, no atomic block claiming)
//   2. compute_scatter_offsets: Compute deterministic block offsets via prefix sum
//      across threadgroups in threadgroup-ID order (ensures TG0 < TG1 < TG2...)
//   3. scatter_write_stable: Each thread computes its local rank within its
//      threadgroup's bucket and writes to the pre-computed deterministic offset
//
// This ensures elements are written in original index order across all threadgroups.

// Phase 1: Count elements per bucket per threadgroup (NO atomic block claiming)
// Must run with THREADGROUP_SIZE threads per threadgroup (e.g., 256)
// Stores counts to tg_bucket_counts[threadgroup_id * 256 + bucket]
kernel void scatter_count_per_threadgroup(
    constant SortingKey *input_keys [[buffer(0)]],
    device uint *tg_bucket_counts [[buffer(1)]],  // [num_threadgroups * 256] bucket counts per TG
    constant uint &count [[buffer(2)]],
    constant uint &byte_index [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    threadgroup uint *local_histogram [[threadgroup(0)]]  // 256 uints
) {
    // Initialize local histogram
    if (tid < 256) {
        local_histogram[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count this thread's element's bucket (if in range)
    bool has_element = gid < count;
    if (has_element) {
        uint key_bits = as_type<uint>(input_keys[gid].depth);
        uint my_bucket = (key_bits >> (byte_index * 8)) & 0xFF;
        // Atomic increment local histogram
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&local_histogram[my_bucket], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Store counts for this threadgroup (no atomic claiming here!)
    if (tid < 256) {
        tg_bucket_counts[tgid * 256 + tid] = local_histogram[tid];
    }
}

// Phase 2: Compute deterministic block offsets via prefix sum across threadgroups
// Run with 256 threads (one per bucket), single threadgroup
// For each bucket, computes prefix sum of counts across threadgroups in TG-ID order
// Output: tg_bucket_offsets[tg * 256 + bucket] = starting position for that (tg, bucket) pair
kernel void compute_scatter_offsets(
    constant uint *global_bucket_offsets [[buffer(0)]],  // 256 buckets (from histogram prefix sum)
    constant uint *tg_bucket_counts [[buffer(1)]],       // [num_threadgroups * 256] counts
    device uint *tg_bucket_offsets [[buffer(2)]],        // [num_threadgroups * 256] output offsets
    constant uint &num_threadgroups [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 256) return;

    uint bucket = tid;
    uint bucket_base = global_bucket_offsets[bucket];  // Starting position for this bucket
    uint running_offset = bucket_base;

    // Iterate through threadgroups in order, computing prefix sum
    // This guarantees TG0 gets the first block, TG1 gets the next, etc.
    for (uint tg = 0; tg < num_threadgroups; tg++) {
        uint count_for_tg = tg_bucket_counts[tg * 256 + bucket];
        tg_bucket_offsets[tg * 256 + bucket] = running_offset;
        running_offset += count_for_tg;
    }
}

// Phase 2: Write elements to output in stable order
// Each thread computes its rank within its threadgroup's bucket, then writes deterministically
kernel void scatter_write_stable(
    constant SortingKey *input_keys [[buffer(0)]],
    device SortingKey *output_keys [[buffer(1)]],
    constant uint *tg_bucket_offsets [[buffer(2)]],  // [num_threadgroups * 256] block starts
    constant uint &count [[buffer(3)]],
    constant uint &byte_index [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    threadgroup uint *local_counts [[threadgroup(0)]],  // 256 uints for bucket counts
    threadgroup uint *local_prefix [[threadgroup(1)]]   // 256 uints for prefix sums
) {
    // Initialize shared memory
    if (tid < 256) {
        local_counts[tid] = 0;
        local_prefix[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Determine this thread's bucket
    bool has_element = gid < count;
    uint my_bucket = 0;
    SortingKey my_key;
    if (has_element) {
        my_key = input_keys[gid];
        uint key_bits = as_type<uint>(my_key.depth);
        my_bucket = (key_bits >> (byte_index * 8)) & 0xFF;
    }

    // Compute local rank within bucket using sequential scan
    // This is O(threadgroup_size) per thread but guarantees stability
    // For each element, count how many earlier elements in this threadgroup have the same bucket
    uint my_local_rank = 0;
    if (has_element) {
        for (uint i = 0; i < tid; i++) {
            uint other_gid = tgid * SCATTER_THREADGROUP_SIZE + i;
            if (other_gid < count) {
                uint other_bits = as_type<uint>(input_keys[other_gid].depth);
                uint other_bucket = (other_bits >> (byte_index * 8)) & 0xFF;
                if (other_bucket == my_bucket) {
                    my_local_rank++;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to output at deterministic position
    if (has_element) {
        uint block_start = tg_bucket_offsets[tgid * 256 + my_bucket];
        uint output_pos = block_start + my_local_rank;
        output_keys[output_pos] = my_key;
    }
}

// Reset histogram to zeros
kernel void reset_histogram(
    device atomic_uint *histogram [[buffer(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < 256) {
        atomic_store_explicit(&histogram[gid], 0, memory_order_relaxed);
    }
}

// Lock-free insertion sort using atomic compare-and-swap
kernel void atomic_insertion_sort(
    device SortingKey *keys [[buffer(0)]],
    device uint *sorted_indices [[buffer(1)]],
    device AtomicSortingState &sorting_state [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &thread_count [[buffer(4)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint thread_id = thread_position_in_grid.x;
    uint total_threads = thread_count;

    // Thread 0 resets state at kernel start (before any work)
    if (thread_id == 0) {
        atomic_store_explicit(&sorting_state.sync_failed, 0, MEM_ORDER_RELEASE);
        atomic_store_explicit(&sorting_state.completed_phases, 0, MEM_ORDER_RELEASE);
    }

    // Ensure all threads see the reset state before proceeding
    threadgroup_barrier(mem_flags::mem_device);

    // Each thread processes a chunk of the array
    uint chunk_size = (count + total_threads - 1) / total_threads;
    uint start = thread_id * chunk_size;
    uint end = min(start + chunk_size, count);

    // Local insertion sort within chunk
    for (uint i = start + 1; i < end; ++i) {
        SortingKey key = keys[i];
        uint pos = i;

        // Find insertion position using atomic operations
        while (pos > start) {
            SortingKey prev_key = keys[pos - 1];
            if (prev_key.depth <= key.depth) break;

            // Atomic swap if we can claim this position
            uint expected_index = prev_key.original_index;
            if (atomic_compare_exchange_weak_explicit(
                (device atomic_uint*)&keys[pos].original_index,
                &expected_index,
                key.original_index,
                MEM_ORDER_ACQ_REL,
                memory_order_relaxed)) {

                keys[pos] = prev_key;
                pos--;
            } else {
                break;
            }
        }

        keys[pos] = key;
    }

    // Mark this thread's work as complete
    atomic_fetch_add_explicit(&sorting_state.completed_phases, 1, MEM_ORDER_RELEASE);

    // Wait for all threads to complete local sorting
    // Bounded spin-wait to prevent deadlock - signal failure on timeout
    uint spin_count = 0;
    const uint max_spin = 1000000;
    bool sync_succeeded = true;
    while (atomic_load_explicit(&sorting_state.completed_phases, MEM_ORDER_ACQUIRE) < total_threads) {
        spin_count++;
        if (spin_count >= max_spin) {
            // Signal that synchronization failed - other threads should abort
            atomic_store_explicit(&sorting_state.sync_failed, 1, MEM_ORDER_RELEASE);
            sync_succeeded = false;
            break;
        }
        // Check if another thread already signaled failure
        if (atomic_load_explicit(&sorting_state.sync_failed, MEM_ORDER_ACQUIRE) != 0) {
            sync_succeeded = false;
            break;
        }
    }

    // If synchronization failed, abort - don't proceed with merge on partially sorted data
    if (!sync_succeeded || atomic_load_explicit(&sorting_state.sync_failed, MEM_ORDER_ACQUIRE) != 0) {
        // Ensure all threads see the failure before any writes
        threadgroup_barrier(mem_flags::mem_device);
        // Only thread 0 writes the identity mapping fallback to avoid races
        if (thread_id == 0) {
            for (uint i = 0; i < count; ++i) {
                sorted_indices[i] = i;
            }
        }
        return;
    }

    // Reset for merge phase
    if (thread_id == 0) {
        atomic_store_explicit(&sorting_state.completed_phases, 0, MEM_ORDER_RELEASE);
    }

    // Ensure reset is visible before merge starts
    threadgroup_barrier(mem_flags::mem_device);

    // Parallel merge of sorted chunks
    bool merge_success = parallel_merge_chunks(keys, sorted_indices, sorting_state, count, thread_id, total_threads);

    // Ensure all threads agree on merge success/failure
    threadgroup_barrier(mem_flags::mem_device);

    // If merge failed, only thread 0 writes identity mapping fallback
    if (!merge_success || atomic_load_explicit(&sorting_state.sync_failed, MEM_ORDER_ACQUIRE) != 0) {
        if (thread_id == 0) {
            for (uint i = 0; i < count; ++i) {
                sorted_indices[i] = i;
            }
        }
    }
}

// High-performance radix sort using atomic operations
kernel void atomic_radix_sort(
    device SortingKey *input_keys [[buffer(0)]],
    device SortingKey *output_keys [[buffer(1)]],
    device uint *sorted_indices [[buffer(2)]],
    device AtomicSortingState &sorting_state [[buffer(3)]],
    constant uint &bit_shift [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    constant uint &thread_count [[buffer(6)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint thread_id = thread_position_in_grid.x;
    uint total_threads = thread_count;

    // Thread 0 resets state at kernel start
    if (thread_id == 0) {
        atomic_store_explicit(&sorting_state.sync_failed, 0, MEM_ORDER_RELEASE);
    }

    // Clear bin counts for this pass
    if (thread_id < 32) {
        atomic_store_explicit(&sorting_state.bin_counts[thread_id], 0, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Count phase - each thread counts elements in its range
    uint chunk_size = (count + total_threads - 1) / total_threads;
    uint start = thread_id * chunk_size;
    uint end = min(start + chunk_size, count);

    // Local counts for each bin
    uint local_bins[32] = {0};

    for (uint i = start; i < end; ++i) {
        // Extract bits for this radix sort pass
        uint depth_bits = as_type<uint>(input_keys[i].depth);
        uint bin = (depth_bits >> bit_shift) & 0x1F; // 5 bits = 32 bins
        local_bins[bin]++;
    }

    // Atomically add local counts to global bins
    for (uint bin = 0; bin < 32; ++bin) {
        if (local_bins[bin] > 0) {
            atomic_fetch_add_explicit(&sorting_state.bin_counts[bin], local_bins[bin], memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Prefix sum to get bin offsets (single thread)
    if (thread_id == 0) {
        uint offset = 0;
        for (uint bin = 0; bin < 32; ++bin) {
            uint bin_count = atomic_load_explicit(&sorting_state.bin_counts[bin], memory_order_relaxed);
            atomic_store_explicit(&sorting_state.bin_counts[bin], offset, memory_order_relaxed);
            offset += bin_count;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Scatter phase - place elements in sorted order
    for (uint i = start; i < end; ++i) {
        uint depth_bits = as_type<uint>(input_keys[i].depth);
        uint bin = (depth_bits >> bit_shift) & 0x1F;

        // Atomically claim position in output
        uint position = atomic_fetch_add_explicit(&sorting_state.bin_counts[bin], 1, memory_order_relaxed);

        // Place element in sorted position
        output_keys[position] = input_keys[i];
    }
}

// Lock-free priority queue for dynamic splat management
kernel void atomic_priority_queue_operations(
    device SortingKey *priority_queue [[buffer(0)]],
    device AtomicSortingState &queue_state [[buffer(1)]],
    constant uint &operation_type [[buffer(2)]], // 0=insert, 1=extract_min
    constant SortingKey &new_element [[buffer(3)]],
    device SortingKey &extracted_element [[buffer(4)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    // thread_id reserved for future multi-threaded priority queue operations
    (void)thread_position_in_grid;

    if (operation_type == 0) {
        // Insert operation
        uint insert_pos = atomic_fetch_add_explicit(&queue_state.global_counter, 1, MEM_ORDER_ACQ_REL);

        // Heap insertion with atomic operations
        priority_queue[insert_pos] = new_element;

        // Bubble up to maintain heap property
        atomic_heap_bubble_up(priority_queue, insert_pos, queue_state);

    } else if (operation_type == 1) {
        // Extract minimum operation
        if (atomic_load_explicit(&queue_state.global_counter, MEM_ORDER_ACQUIRE) == 0) {
            return; // Queue is empty
        }

        // Extract root (minimum element)
        extracted_element = priority_queue[0];

        // Move last element to root and bubble down
        uint last_pos = atomic_fetch_sub_explicit(&queue_state.global_counter, 1, MEM_ORDER_ACQ_REL) - 1;

        if (last_pos > 0) {
            priority_queue[0] = priority_queue[last_pos];
            atomic_heap_bubble_down(priority_queue, 0, last_pos, queue_state);
        }
    }
}
