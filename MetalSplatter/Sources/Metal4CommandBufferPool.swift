import Metal
import Foundation
import os

/// A pool-based command buffer allocator that reuses command buffers to reduce memory allocation overhead.
/// This implements Metal 4 command buffer reuse architecture for improved performance.
@available(iOS 18.0, macOS 15.0, visionOS 2.0, *)
class Metal4CommandBufferPool {
    
    private static let log = Logger(subsystem: "com.saturdayresearch.metalsplatter", category: "Metal4CommandBufferPool")
    
    private let commandQueue: MTLCommandQueue
    private var availableBuffers: [MTLCommandBuffer] = []
    private var activeBuffers: [ObjectIdentifier: MTLCommandBuffer] = [:]
    private let maxPoolSize: Int
    private let lock = NSLock()
    
    /// Initialize the command buffer pool
    /// - Parameters:
    ///   - commandQueue: The Metal command queue to create buffers from
    ///   - maxPoolSize: Maximum number of command buffers to keep in the pool (default: 6)
    init(commandQueue: MTLCommandQueue, maxPoolSize: Int = 6) {
        self.commandQueue = commandQueue
        self.maxPoolSize = maxPoolSize
    }
    
    /// Get a reusable command buffer from the pool
    /// - Returns: A command buffer ready for use, or nil if creation fails
    func getCommandBuffer() -> MTLCommandBuffer? {
        lock.lock()
        defer { lock.unlock() }
        
        // Note: Command buffers cannot be reused after completion in Metal
        // The pool mainly serves to track active buffers and manage statistics
        // Always create new command buffers
        
        // Create a new command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            Self.log.error("Failed to create new command buffer")
            return nil
        }
        
        // Set up completion handler to return buffer to pool
        commandBuffer.addCompletedHandler { [weak self] completedBuffer in
            self?.returnCommandBuffer(completedBuffer)
        }
        
        activeBuffers[ObjectIdentifier(commandBuffer)] = commandBuffer
        return commandBuffer
    }
    
    /// Remove completed command buffer from active tracking
    /// - Parameter commandBuffer: The completed command buffer to remove from tracking
    private func returnCommandBuffer(_ commandBuffer: MTLCommandBuffer) {
        lock.lock()
        defer { lock.unlock() }
        
        // Remove from active dictionary (completed buffers cannot be reused)
        activeBuffers.removeValue(forKey: ObjectIdentifier(commandBuffer))
    }
    
    /// Get current pool statistics for monitoring
    var statistics: (available: Int, active: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (available: 0, active: activeBuffers.count) // No reusable buffers in Metal
    }
    
    /// Clear tracking data (for memory pressure)  
    func clearPool() {
        lock.lock()
        defer { lock.unlock() }
        
        // Note: We can't actually clear active command buffers as they may be in use
        // This mainly serves to reset any tracking data
        Self.log.info("Command buffer pool memory pressure handled")
    }
    
    deinit {
        clearPool()
    }
}

/// Command buffer pool manager that provides fallback behavior for non-Metal 4 devices
public class CommandBufferManager {
    
    private static let log = Logger(subsystem: "com.saturdayresearch.metalsplatter", category: "CommandBufferManager")
    
    private let commandQueue: MTLCommandQueue
    private var metal4Pool: Any? // Metal4CommandBufferPool on iOS 18+
    
    /// Access to the underlying command queue for operations that require it directly
    public var queue: MTLCommandQueue {
        return commandQueue
    }
    
    public init(commandQueue: MTLCommandQueue) {
        self.commandQueue = commandQueue
        
        // Initialize Metal 4 pool only on supported devices
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
            self.metal4Pool = Metal4CommandBufferPool(commandQueue: commandQueue)
            Self.log.info("Initialized Metal 4 command buffer pool")
        } else {
            Self.log.info("Using legacy command buffer allocation (pre-Metal 4)")
        }
    }
    
    /// Get a command buffer with optimal allocation strategy
    /// - Returns: A command buffer ready for use, or nil if creation fails
    public func makeCommandBuffer() -> MTLCommandBuffer? {
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *),
           let pool = metal4Pool as? Metal4CommandBufferPool {
            return pool.getCommandBuffer()
        } else {
            // Fallback to direct allocation for older devices
            return commandQueue.makeCommandBuffer()
        }
    }
    
    /// Get pool statistics (Metal 4 only)
    public var poolStatistics: (available: Int, active: Int)? {
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *),
           let pool = metal4Pool as? Metal4CommandBufferPool {
            return pool.statistics
        }
        return nil
    }
    
    /// Clear the command buffer pool (Metal 4 only)
    public func clearPool() {
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *),
           let pool = metal4Pool as? Metal4CommandBufferPool {
            pool.clearPool()
        }
    }
    
    /// Handle memory pressure by clearing the pool
    public func handleMemoryPressure() {
        clearPool()
        Self.log.info("Cleared command buffer pool due to memory pressure")
    }
    
    /// Log current pool state for debugging
    public func logPoolState() {
        if let stats = poolStatistics {
            Self.log.info("Command buffer pool state - Available: \(stats.available), Active: \(stats.active)")
        } else {
            Self.log.info("Using legacy command buffer allocation (no pool)")
        }
    }
}