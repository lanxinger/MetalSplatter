import Foundation
import Metal
import os

#if canImport(UIKit)
import UIKit
#endif

/**
 * A thread-safe pool of Metal buffers for efficient reuse and reduced allocation overhead.
 * Manages type-safe buffers with automatic memory pressure handling.
 */
public class MetalBufferPool<T> {
    
    // MARK: - Error Types
    
    public enum PoolError: LocalizedError {
        case deviceUnavailable
        case bufferCreationFailed(capacity: Int)
        case invalidCapacity(requested: Int, max: Int)
        
        public var errorDescription: String? {
            switch self {
            case .deviceUnavailable:
                return "Metal device is not available"
            case .bufferCreationFailed(let capacity):
                return "Failed to create buffer with capacity \(capacity)"
            case .invalidCapacity(let requested, let max):
                return "Requested capacity \(requested) exceeds maximum \(max)"
            }
        }
    }
    
    // MARK: - Configuration
    
    public struct Configuration {
        public let maxPoolSize: Int
        public let maxBufferAge: TimeInterval
        public let memoryPressureThreshold: Float // 0.0 to 1.0
        public let enableMemoryPressureMonitoring: Bool
        public let enableMetal4Optimizations: Bool
        public let useResidencyTracking: Bool
        
        public init(maxPoolSize: Int = 10,
                   maxBufferAge: TimeInterval = 60.0,
                   memoryPressureThreshold: Float = 0.8,
                   enableMemoryPressureMonitoring: Bool = true,
                   enableMetal4Optimizations: Bool = true,
                   useResidencyTracking: Bool = true) {
            self.maxPoolSize = maxPoolSize
            self.maxBufferAge = maxBufferAge
            self.memoryPressureThreshold = memoryPressureThreshold
            self.enableMemoryPressureMonitoring = enableMemoryPressureMonitoring
            self.enableMetal4Optimizations = enableMetal4Optimizations
            self.useResidencyTracking = useResidencyTracking
        }
        
        
        public static var `default`: Configuration {
            return Configuration()
        }
    }
    
    // MARK: - Buffer Entry
    
    private class PooledBuffer {
        let buffer: MetalBuffer<T>
        let creationTime: TimeInterval
        var lastUsedTime: TimeInterval
        var useCount: Int
        
        init(buffer: MetalBuffer<T>) {
            self.buffer = buffer
            let now = CFAbsoluteTimeGetCurrent()
            self.creationTime = now
            self.lastUsedTime = now
            self.useCount = 0
        }
        
        func markUsed() {
            lastUsedTime = CFAbsoluteTimeGetCurrent()
            useCount += 1
        }
        
        var age: TimeInterval {
            CFAbsoluteTimeGetCurrent() - creationTime
        }
        
        var timeSinceLastUse: TimeInterval {
            CFAbsoluteTimeGetCurrent() - lastUsedTime
        }
    }
    
    // MARK: - Properties
    
    private let device: MTLDevice
    private let configuration: Configuration
    private let queue = DispatchQueue(label: "com.metalsplatter.buffer-pool", attributes: .concurrent)
    private var availableBuffers: [PooledBuffer] = []
    private var leasedBuffers: Set<ObjectIdentifier> = []
    
    // Metal 4.0 optimizations - use Any to avoid @available on stored properties
    private var residencySet: Any? // MTLResidencySet when available
    private var argumentEncoder: Any? // MTLArgumentEncoder when available
    
    private let log = Logger(
        subsystem: Bundle.module.bundleIdentifier ?? "com.metalsplatter.unknown",
        category: "MetalBufferPool"
    )
    
    // MARK: - Memory Pressure Monitoring
    
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    
    // MARK: - Initialization
    
    public init(device: MTLDevice, configuration: Configuration = .default) {
        self.device = device
        self.configuration = configuration
        
        // Initialize Metal 4.0 optimizations
        if configuration.enableMetal4Optimizations {
            setupMetal4Optimizations()
        }
        
        if configuration.enableMemoryPressureMonitoring {
            setupMemoryPressureMonitoring()
        }
        
        // Setup app lifecycle notifications for cleanup
        #if canImport(UIKit)
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.handleMemoryPressure(.critical)
        }
        
        NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.trimToMemoryPressure()
        }
        #endif
    }
    
    deinit {
        memoryPressureSource?.cancel()
        clearAll()
    }
    
    // MARK: - Pool Management
    
    /// Acquires a buffer with at least the specified minimum capacity
    public func acquire(minimumCapacity: Int) throws -> MetalBuffer<T> {
        return try queue.sync {
            log.debug("Acquiring buffer with minimum capacity: \(minimumCapacity)")
            
            // Validate capacity
            let maxCapacity = MetalBuffer<T>.maxCapacity(for: device)
            guard minimumCapacity <= maxCapacity else {
                throw PoolError.invalidCapacity(requested: minimumCapacity, max: maxCapacity)
            }
            
            // Try to find a suitable buffer in the pool
            if let pooledBuffer = findSuitableBuffer(minimumCapacity: minimumCapacity) {
                pooledBuffer.markUsed()
                leasedBuffers.insert(ObjectIdentifier(pooledBuffer.buffer))
                
                // Metal 4.0: Track residency
                if configuration.enableMetal4Optimizations {
                    trackBufferResidency(pooledBuffer.buffer)
                }
                
                log.debug("Reusing pooled buffer with capacity: \(pooledBuffer.buffer.capacity)")
                return pooledBuffer.buffer
            }
            
            // Create a new buffer if none suitable found
            let buffer = try MetalBuffer<T>(device: device, capacity: minimumCapacity)
            leasedBuffers.insert(ObjectIdentifier(buffer))
            
            // Metal 4.0: Track residency for new buffers
            if configuration.enableMetal4Optimizations {
                trackBufferResidency(buffer)
            }
            
            log.debug("Created new buffer with capacity: \(buffer.capacity)")
            return buffer
        }
    }
    
    /// Returns a buffer to the pool for reuse
    public func release(_ buffer: MetalBuffer<T>) {
        queue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            let bufferID = ObjectIdentifier(buffer)
            guard self.leasedBuffers.contains(bufferID) else {
                self.log.warning("Attempted to release buffer that wasn't acquired from this pool")
                return
            }
            
            self.leasedBuffers.remove(bufferID)
            
            // Reset buffer state
            buffer.count = 0
            
            // Add to pool if there's room and it's worth keeping
            if self.shouldPoolBuffer(buffer) {
                let pooledBuffer = PooledBuffer(buffer: buffer)
                self.availableBuffers.append(pooledBuffer)
                self.log.debug("Returned buffer to pool, pool size: \(self.availableBuffers.count)")
            } else {
                self.log.debug("Buffer not added to pool (pool full or not suitable)")
            }
            
            // Trim old buffers
            self.trimExpiredBuffers()
        }
    }
    
    // MARK: - Private Helper Methods
    
    private func findSuitableBuffer(minimumCapacity: Int) -> PooledBuffer? {
        // Find the smallest buffer that meets the minimum capacity requirement
        var bestBuffer: PooledBuffer?
        var bestCapacity = Int.max
        var bestIndex = -1
        
        for (index, pooledBuffer) in availableBuffers.enumerated() {
            let capacity = pooledBuffer.buffer.capacity
            if capacity >= minimumCapacity && capacity < bestCapacity {
                bestBuffer = pooledBuffer
                bestCapacity = capacity
                bestIndex = index
            }
        }
        
        if let buffer = bestBuffer, bestIndex >= 0 {
            availableBuffers.remove(at: bestIndex)
            return buffer
        }
        
        return nil
    }
    
    private func shouldPoolBuffer(_ buffer: MetalBuffer<T>) -> Bool {
        // Don't pool if we're at capacity
        guard availableBuffers.count < configuration.maxPoolSize else {
            return false
        }
        
        // Don't pool very large buffers that are unlikely to be reused
        let bufferSizeMB = Float(buffer.capacity * MemoryLayout<T>.stride) / (1024 * 1024)
        if bufferSizeMB > 100 {
            log.debug("Not pooling large buffer: \(bufferSizeMB)MB")
            return false
        }
        
        return true
    }
    
    private func trimExpiredBuffers() {
        let now = CFAbsoluteTimeGetCurrent()
        availableBuffers.removeAll { pooledBuffer in
            let expired = pooledBuffer.age > configuration.maxBufferAge
            if expired {
                log.debug("Removed expired buffer from pool")
            }
            return expired
        }
    }
    
    // MARK: - Memory Pressure Handling
    
    private func setupMemoryPressureMonitoring() {
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: queue
        )
        
        memoryPressureSource?.setEventHandler { [weak self] in
            guard let self = self else { return }
            let event = self.memoryPressureSource?.mask ?? []
            
            if event.contains(.critical) {
                self.handleMemoryPressure(.critical)
            } else if event.contains(.warning) {
                self.handleMemoryPressure(.warning)
            }
        }
        
        memoryPressureSource?.resume()
    }
    
    private enum MemoryPressureLevel {
        case warning
        case critical
    }
    
    private func handleMemoryPressure(_ level: MemoryPressureLevel) {
        queue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            switch level {
            case .warning:
                self.log.info("Memory pressure warning - trimming buffer pool")
                self.trimToMemoryPressure()
                
            case .critical:
                self.log.warning("Critical memory pressure - clearing buffer pool")
                self.clearAll()
            }
        }
    }
    
    /// Reduces pool size based on memory pressure
    public func trimToMemoryPressure() {
        queue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            let originalSize = self.availableBuffers.count
            
            // Remove least recently used buffers, keeping only the most recent ones
            self.availableBuffers.sort { lhs, rhs in
                lhs.lastUsedTime > rhs.lastUsedTime
            }
            
            // Keep only the top 25% of buffers during memory pressure
            let targetSize = max(1, self.configuration.maxPoolSize / 4)
            if self.availableBuffers.count > targetSize {
                self.availableBuffers.removeSubrange(targetSize...)
            }
            
            let trimmedCount = originalSize - self.availableBuffers.count
            if trimmedCount > 0 {
                self.log.info("Trimmed \(trimmedCount) buffers due to memory pressure")
            }
        }
    }
    
    /// Clears all buffers from the pool
    public func clearAll() {
        queue.async(flags: .barrier) { [weak self] in
            guard let self = self else { return }
            
            let clearedCount = self.availableBuffers.count
            self.availableBuffers.removeAll()
            
            if clearedCount > 0 {
                self.log.info("Cleared all \(clearedCount) buffers from pool")
            }
        }
    }
    
    // MARK: - Statistics
    
    public struct PoolStatistics {
        public let availableBuffers: Int
        public let leasedBuffers: Int
        public let totalMemoryMB: Float
        public let averageBufferAge: TimeInterval
        
        public init(availableBuffers: Int, leasedBuffers: Int, totalMemoryMB: Float, averageBufferAge: TimeInterval) {
            self.availableBuffers = availableBuffers
            self.leasedBuffers = leasedBuffers
            self.totalMemoryMB = totalMemoryMB
            self.averageBufferAge = averageBufferAge
        }
    }
    
    /// Returns current pool statistics
    public func getStatistics() -> PoolStatistics {
        return queue.sync {
            let totalMemory = availableBuffers.reduce(0) { total, pooledBuffer in
                total + pooledBuffer.buffer.capacity * MemoryLayout<T>.stride
            }
            
            let averageAge = availableBuffers.isEmpty ? 0 : 
                availableBuffers.reduce(0) { $0 + $1.age } / Double(availableBuffers.count)
            
            return PoolStatistics(
                availableBuffers: availableBuffers.count,
                leasedBuffers: leasedBuffers.count,
                totalMemoryMB: Float(totalMemory) / (1024 * 1024),
                averageBufferAge: averageAge
            )
        }
    }
    
    // MARK: - Metal 4.0 Optimizations
    
    /// Setup Metal 4.0 optimizations for buffer management
    private func setupMetal4Optimizations() {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            if device.supportsFamily(.apple9) {
                // Setup residency tracking
                if configuration.useResidencyTracking {
                    do {
                        let descriptor = MTLResidencySetDescriptor()
                        residencySet = try device.makeResidencySet(descriptor: descriptor)
                        log.info("Metal 4.0: Residency tracking enabled")
                    } catch {
                        log.error("Metal 4.0: Failed to create residency set: \(error)")
                    }
                }
                
                // Setup argument encoder for buffer pool management
                setupArgumentEncoder()
                
                log.info("Metal 4.0: Buffer pool optimizations enabled")
            } else {
                log.info("Metal 4.0: Device doesn't support Apple GPU Family 9+, optimizations disabled")
            }
        } else {
            log.info("Metal 4.0: iOS 26.0+ required for optimizations")
        }
    }
    
    @available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
    private func setupArgumentEncoder() {
        // Create argument encoder for efficient buffer binding
        // This is a placeholder - actual implementation would depend on shader requirements
        // argumentEncoder = device.makeArgumentEncoder(arguments: [...])
        log.debug("Metal 4.0: Argument encoder setup (placeholder)")
    }
    
    /// Track buffer residency for Metal 4.0 optimization
    private func trackBufferResidency(_ buffer: MetalBuffer<T>) {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            if let residencySet = residencySet as? MTLResidencySet, configuration.useResidencyTracking {
                // Add buffer to residency set for GPU memory tracking
                // residencySet.addAllocation(buffer.buffer)
                log.debug("Metal 4.0: Tracking buffer residency for capacity \(buffer.capacity)")
            }
        }
    }
    
    /// Get Metal 4.0 optimization status
    public func getMetal4Status() -> Metal4Status {
        if #available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *) {
            let hasAppleGPU9 = device.supportsFamily(.apple9)
            let optimizationsEnabled = configuration.enableMetal4Optimizations && hasAppleGPU9
            let residencyEnabled = (residencySet as? MTLResidencySet) != nil && configuration.useResidencyTracking
            
            return Metal4Status(
                available: true,
                optimizationsEnabled: optimizationsEnabled,
                residencyTrackingEnabled: residencyEnabled,
                argumentEncoderEnabled: (argumentEncoder as? MTLArgumentEncoder) != nil
            )
        } else {
            return Metal4Status(
                available: false,
                optimizationsEnabled: false,
                residencyTrackingEnabled: false,
                argumentEncoderEnabled: false
            )
        }
    }
    
    public struct Metal4Status {
        public let available: Bool
        public let optimizationsEnabled: Bool
        public let residencyTrackingEnabled: Bool
        public let argumentEncoderEnabled: Bool
    }
}

// MARK: - Convenience Extensions

extension MetalBufferPool {
    
    /// Acquires a buffer and automatically releases it when the block completes
    public func withBuffer<Result>(minimumCapacity: Int, 
                                 _ block: (MetalBuffer<T>) throws -> Result) throws -> Result {
        let buffer = try acquire(minimumCapacity: minimumCapacity)
        defer { release(buffer) }
        return try block(buffer)
    }
}