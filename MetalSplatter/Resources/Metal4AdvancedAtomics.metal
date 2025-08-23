#include <metal_stdlib>
#include "ShaderCommon.h"
using namespace metal;

#if __METAL_VERSION__ >= 400

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
    };
    
    // Lock-free insertion sort using atomic compare-and-swap
    [[user_annotation("atomic_insertion_sort")]]
    kernel void atomic_insertion_sort(
        device SortingKey *keys [[buffer(0)]],
        device uint *sorted_indices [[buffer(1)]],
        device AtomicSortingState &sorting_state [[buffer(2)]],
        uint count [[buffer(3)]],
        uint3 thread_position_in_grid [[thread_position_in_grid]]
    ) {
        uint thread_id = thread_position_in_grid.x;
        uint total_threads = thread_position_in_grid.x; // Should be thread_count
        
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
                    memory_order_acq_rel,
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
        atomic_fetch_add_explicit(&sorting_state.completed_phases, 1, memory_order_release);
        
        // Wait for all threads to complete local sorting
        while (atomic_load_explicit(&sorting_state.completed_phases, memory_order_acquire) < total_threads) {
            // Spin wait
        }
        
        // Merge phase using atomic operations
        if (thread_id == 0) {
            atomic_store_explicit(&sorting_state.completed_phases, 0, memory_order_release);
        }
        
        // Parallel merge of sorted chunks
        parallel_merge_chunks(keys, sorted_indices, sorting_state, count, thread_id, total_threads);
    }
    
    // High-performance radix sort using atomic operations
    [[user_annotation("atomic_radix_sort")]]
    kernel void atomic_radix_sort(
        device SortingKey *input_keys [[buffer(0)]],
        device SortingKey *output_keys [[buffer(1)]],
        device uint *sorted_indices [[buffer(2)]],
        device AtomicSortingState &sorting_state [[buffer(3)]],
        constant uint &bit_shift [[buffer(4)]],
        uint count [[buffer(5)]],
        uint3 thread_position_in_grid [[thread_position_in_grid]]
    ) {
        uint thread_id = thread_position_in_grid.x;
        uint total_threads = thread_position_in_grid.x; // Should be from thread count
        
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
    
    // Atomic merge operation for combining sorted sequences
    void parallel_merge_chunks(
        device SortingKey *keys,
        device uint *sorted_indices,
        device AtomicSortingState &sorting_state,
        uint count,
        uint thread_id,
        uint total_threads
    ) {
        // Implement parallel merge using atomic operations
        uint merge_size = 2;
        
        while (merge_size <= count) {
            uint chunks_per_merge = merge_size;
            uint merges_per_thread = (count / merge_size + total_threads - 1) / total_threads;
            
            for (uint merge_idx = 0; merge_idx < merges_per_thread; ++merge_idx) {
                uint global_merge_idx = thread_id * merges_per_thread + merge_idx;
                uint left_start = global_merge_idx * merge_size;
                uint right_start = left_start + merge_size / 2;
                uint end = min(left_start + merge_size, count);
                
                if (left_start >= count) break;
                
                // Perform atomic merge of two sorted sequences
                atomic_merge_sequences(keys, left_start, right_start, end, sorting_state);
            }
            
            // Synchronize between merge levels
            atomic_fetch_add_explicit(&sorting_state.completed_phases, 1, memory_order_release);
            
            // Wait for all threads to complete this merge level
            uint expected_completions = total_threads * (merge_idx + 1);
            while (atomic_load_explicit(&sorting_state.completed_phases, memory_order_acquire) < expected_completions) {
                // Spin wait
            }
            
            merge_size *= 2;
        }
        
        // Final pass: extract sorted indices
        if (thread_id == 0) {
            for (uint i = 0; i < count; ++i) {
                sorted_indices[i] = keys[i].original_index;
            }
        }
    }
    
    // Atomic merge of two sorted sequences
    void atomic_merge_sequences(
        device SortingKey *keys,
        uint left_start,
        uint right_start,
        uint end,
        device AtomicSortingState &sorting_state
    ) {
        uint left_idx = left_start;
        uint right_idx = right_start;
        uint merge_pos = left_start;
        
        // Temporary array for merged results (should be allocated properly)
        SortingKey temp_keys[256]; // Limited size for demonstration
        
        uint temp_idx = 0;
        
        // Merge two sorted halves
        while (left_idx < right_start && right_idx < end && temp_idx < 256) {
            if (keys[left_idx].depth <= keys[right_idx].depth) {
                temp_keys[temp_idx++] = keys[left_idx++];
            } else {
                temp_keys[temp_idx++] = keys[right_idx++];
            }
        }
        
        // Copy remaining elements
        while (left_idx < right_start && temp_idx < 256) {
            temp_keys[temp_idx++] = keys[left_idx++];
        }
        
        while (right_idx < end && temp_idx < 256) {
            temp_keys[temp_idx++] = keys[right_idx++];
        }
        
        // Copy back to original array using atomic operations
        for (uint i = 0; i < temp_idx; ++i) {
            uint target_pos = left_start + i;
            if (target_pos < end) {
                // Use atomic exchange to ensure consistency
                SortingKey old_key = keys[target_pos];
                while (true) {
                    uint expected_index = old_key.original_index;
                    if (atomic_compare_exchange_weak_explicit(
                        (device atomic_uint*)&keys[target_pos].original_index,
                        &expected_index,
                        temp_keys[i].original_index,
                        memory_order_acq_rel,
                        memory_order_relaxed)) {
                        
                        keys[target_pos] = temp_keys[i];
                        break;
                    }
                    old_key = keys[target_pos];
                }
            }
        }
    }
    
    // Lock-free priority queue for dynamic splat management
    [[user_annotation("atomic_priority_queue")]]
    kernel void atomic_priority_queue_operations(
        device SortingKey *priority_queue [[buffer(0)]],
        device AtomicSortingState &queue_state [[buffer(1)]],
        constant uint &operation_type [[buffer(2)]], // 0=insert, 1=extract_min
        constant SortingKey &new_element [[buffer(3)]],
        device SortingKey &extracted_element [[buffer(4)]],
        uint3 thread_position_in_grid [[thread_position_in_grid]]
    ) {
        uint thread_id = thread_position_in_grid.x;
        
        if (operation_type == 0) {
            // Insert operation
            uint insert_pos = atomic_fetch_add_explicit(&queue_state.global_counter, 1, memory_order_acq_rel);
            
            // Heap insertion with atomic operations
            priority_queue[insert_pos] = new_element;
            
            // Bubble up to maintain heap property
            atomic_heap_bubble_up(priority_queue, insert_pos, queue_state);
            
        } else if (operation_type == 1) {
            // Extract minimum operation
            if (atomic_load_explicit(&queue_state.global_counter, memory_order_acquire) == 0) {
                return; // Queue is empty
            }
            
            // Extract root (minimum element)
            extracted_element = priority_queue[0];
            
            // Move last element to root and bubble down
            uint last_pos = atomic_fetch_sub_explicit(&queue_state.global_counter, 1, memory_order_acq_rel) - 1;
            
            if (last_pos > 0) {
                priority_queue[0] = priority_queue[last_pos];
                atomic_heap_bubble_down(priority_queue, 0, last_pos, queue_state);
            }
        }
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
                memory_order_acq_rel,
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
                memory_order_acq_rel,
                memory_order_relaxed)) {
                
                heap[pos] = heap[smallest];
                heap[smallest] = temp;
                pos = smallest;
            } else {
                break; // Another thread modified, stop bubbling
            }
        }
    }
}

#endif // __METAL_VERSION__ >= 400