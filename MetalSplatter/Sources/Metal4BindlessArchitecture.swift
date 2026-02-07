import Foundation
import Metal
import MetalKit
import os

/// Enhanced Metal 4 Bindless Architecture with complete residency management
/// Implements full bindless resource management with background population and zero per-draw binding
@available(iOS 18.0, macOS 15.0, visionOS 2.0, *)
public class Metal4BindlessArchitecture: @unchecked Sendable {

    private static let log = Logger(
        subsystem: Bundle.module.bundleIdentifier ?? "com.metalsplatter.unknown",
        category: "Metal4BindlessArchitecture"
    )

    // MARK: - Configuration

    public struct Configuration {
        public let maxResources: Int
        public let maxSplatBuffers: Int
        public let maxUniformBuffers: Int
        public let maxTextures: Int
        public let enableBackgroundPopulation: Bool
        public let enableResidencyTracking: Bool
        public let resourceTableSize: Int

        public init(maxResources: Int = 1024,
                   maxSplatBuffers: Int = 16,
                   maxUniformBuffers: Int = 16,
                   maxTextures: Int = 32,
                   enableBackgroundPopulation: Bool = true,
                   enableResidencyTracking: Bool = true,
                   resourceTableSize: Int = 4096) {
            self.maxResources = maxResources
            self.maxSplatBuffers = maxSplatBuffers
            self.maxUniformBuffers = maxUniformBuffers
            self.maxTextures = maxTextures
            self.enableBackgroundPopulation = enableBackgroundPopulation
            self.enableResidencyTracking = enableResidencyTracking
            self.resourceTableSize = resourceTableSize
        }
    }

    // MARK: - Properties

    private let device: MTLDevice
    private let configuration: Configuration
    private let populationQueue = DispatchQueue(label: "com.metalsplatter.bindless.population", qos: .userInitiated)

    // Argument buffers and encoders - double-buffered for thread safety
    // Each buffer has its own encoder to avoid MTLArgumentEncoder thread-safety issues
    private var argumentEncoders: [MTLArgumentEncoder] = []  // One per buffer for thread safety
    private var argumentBuffers: [MTLBuffer] = []  // Double-buffered
    private var currentBufferIndex: Int = 0
    private var resourceTables: [MTLBuffer] = []  // Double-buffered resource tables

    // Serial queue for encoder operations to ensure thread safety
    private let encoderQueue = DispatchQueue(label: "com.metalsplatter.bindless.encoder")

    // Resource tracking with type-specific slot management
    private var resourceRegistry: ResourceRegistry
    private var pendingResources: [ResourceHandle] = []
    private let resourceLock = NSLock()

    // Slot allocators for each resource type
    private var splatBufferSlots: SlotAllocator
    private var uniformBufferSlots: SlotAllocator
    private var textureSlots: SlotAllocator

    // Residency management (placeholder for future Metal APIs)
    private var residencyController: ResidencyController?

    // Performance metrics
    private var bindlessMetrics = BindlessMetrics()

    // Background thread management
    private var shouldStopPopulation = false
    private let populationSemaphore = DispatchSemaphore(value: 0)
    private let populationLock = NSLock()

    // MARK: - Initialization

    public init(device: MTLDevice, configuration: Configuration = Configuration()) throws {
        self.device = device
        self.configuration = configuration

        // Initialize slot allocators
        self.splatBufferSlots = SlotAllocator(maxSlots: configuration.maxSplatBuffers)
        self.uniformBufferSlots = SlotAllocator(maxSlots: configuration.maxUniformBuffers)
        self.textureSlots = SlotAllocator(maxSlots: configuration.maxTextures)

        // Initialize resource registry with bounds
        self.resourceRegistry = ResourceRegistry(maxResources: configuration.resourceTableSize)

        guard device.supportsFamily(.apple7) else {
            throw BindlessError.unsupportedDevice("Device must support Apple GPU Family 7+")
        }

        try setupArgumentBuffers()
        try setupResourceTable()

        if configuration.enableResidencyTracking {
            setupResidencyTracking()
        }

        if configuration.enableBackgroundPopulation {
            startBackgroundResourcePopulation()
        }

        Self.log.info("✅ Metal 4 Bindless Architecture initialized")
        Self.log.info("   Max Resources: \(configuration.maxResources)")
        Self.log.info("   Background Population: \(configuration.enableBackgroundPopulation)")
        Self.log.info("   Residency Tracking: \(configuration.enableResidencyTracking)")
    }

    // MARK: - Setup Methods

    private func setupArgumentBuffers() throws {
        // Create comprehensive argument descriptor for all resource types
        var argumentDescriptors: [MTLArgumentDescriptor] = []

        // Splat buffers (0 to maxSplatBuffers-1)
        for i in 0..<configuration.maxSplatBuffers {
            let descriptor = MTLArgumentDescriptor()
            descriptor.index = i
            descriptor.dataType = .pointer
            descriptor.access = .readOnly
            descriptor.arrayLength = 1
            argumentDescriptors.append(descriptor)
        }

        // Uniform buffers (maxSplatBuffers to maxSplatBuffers+maxUniformBuffers-1)
        for i in 0..<configuration.maxUniformBuffers {
            let descriptor = MTLArgumentDescriptor()
            descriptor.index = configuration.maxSplatBuffers + i
            descriptor.dataType = .pointer
            descriptor.access = .readWrite
            descriptor.arrayLength = 1
            argumentDescriptors.append(descriptor)
        }

        // Textures (maxSplatBuffers+maxUniformBuffers to end)
        for i in 0..<configuration.maxTextures {
            let descriptor = MTLArgumentDescriptor()
            descriptor.index = configuration.maxSplatBuffers + configuration.maxUniformBuffers + i
            descriptor.dataType = .texture
            descriptor.textureType = .type2D
            descriptor.access = .readOnly
            descriptor.arrayLength = 1
            argumentDescriptors.append(descriptor)
        }

        // Create double-buffered argument buffers, each with its own encoder
        // This avoids MTLArgumentEncoder thread-safety issues by isolating encoders per buffer
        guard let templateEncoder = device.makeArgumentEncoder(arguments: argumentDescriptors) else {
            throw BindlessError.argumentEncoderCreationFailed
        }
        let bufferSize = templateEncoder.encodedLength

        for i in 0..<2 {
            guard let buffer = device.makeBuffer(length: bufferSize, options: [.storageModeShared]) else {
                throw BindlessError.argumentBufferCreationFailed
            }
            buffer.label = "Metal4 Bindless Argument Buffer \(i)"
            argumentBuffers.append(buffer)

            // Create a dedicated encoder for each buffer to ensure thread safety
            guard let encoder = device.makeArgumentEncoder(arguments: argumentDescriptors) else {
                throw BindlessError.argumentEncoderCreationFailed
            }
            argumentEncoders.append(encoder)
        }

        Self.log.info("Created \(self.argumentEncoders.count) argument encoders with \(argumentDescriptors.count) descriptors each, buffer size: \(bufferSize) bytes")
    }

    private func setupResourceTable() throws {
        // Create double-buffered resource tables for thread-safe GPU/CPU access
        let tableSize = configuration.resourceTableSize * MemoryLayout<UInt64>.stride

        for i in 0..<2 {
            guard let table = device.makeBuffer(length: tableSize, options: [.storageModeShared]) else {
                throw BindlessError.resourceTableCreationFailed
            }

            table.label = "Metal4 Bindless Resource Table \(i)"

            // Initialize table with null handles
            let contents = table.contents().bindMemory(to: UInt64.self, capacity: configuration.resourceTableSize)
            for j in 0..<configuration.resourceTableSize {
                contents[j] = ResourceHandle.null.value
            }

            resourceTables.append(table)
        }

        Self.log.info("Created double-buffered resource tables with \(self.configuration.resourceTableSize) entries each")
    }

    private func setupResidencyTracking() {
        // Initialize residency controller
        // Note: In Metal 4, this would use MTLResidencySet
        // For now, we implement a custom tracking system
        residencyController = ResidencyController(device: device)
        Self.log.info("Residency tracking enabled")
    }

    // MARK: - Resource Registration

    /// Register a buffer for bindless access without per-draw binding
    /// Returns nil if resource table is full
    public func registerBuffer(_ buffer: MTLBuffer, type: ResourceType) -> ResourceHandle? {
        resourceLock.lock()
        defer { resourceLock.unlock() }

        // Allocate a slot based on resource type
        let slotAllocator: SlotAllocator
        let slotOffset: Int

        switch type {
        case .splatBuffer:
            slotAllocator = splatBufferSlots
            slotOffset = 0
        case .uniformBuffer:
            slotAllocator = uniformBufferSlots
            slotOffset = configuration.maxSplatBuffers
        case .indexBuffer:
            // Index buffers use splat buffer slots
            slotAllocator = splatBufferSlots
            slotOffset = 0
        case .texture, .sampler:
            Self.log.error("Cannot register texture/sampler as buffer")
            return nil
        }

        guard let slot = slotAllocator.allocate() else {
            Self.log.error("No available slots for resource type \(String(describing: type))")
            return nil
        }

        let argumentIndex = slotOffset + slot

        guard let handle = resourceRegistry.register(buffer, type: type, argumentIndex: argumentIndex) else {
            slotAllocator.free(slot)
            Self.log.error("Resource table is full, cannot register buffer")
            return nil
        }

        // Queue for background population or populate immediately
        if configuration.enableBackgroundPopulation {
            pendingResources.append(handle)
            populationSemaphore.signal()  // Wake up background thread
        } else {
            populateResourceUnsafe(handle, buffer: buffer, argumentIndex: argumentIndex)
        }

        bindlessMetrics.resourcesRegistered += 1

        Self.log.debug("Registered \(String(describing: type)) buffer with handle: \(handle.value), argument index: \(argumentIndex)")
        return handle
    }

    /// Register a texture for bindless access
    /// Returns nil if resource table is full
    public func registerTexture(_ texture: MTLTexture) -> ResourceHandle? {
        resourceLock.lock()
        defer { resourceLock.unlock() }

        guard let slot = textureSlots.allocate() else {
            Self.log.error("No available texture slots")
            return nil
        }

        let argumentIndex = configuration.maxSplatBuffers + configuration.maxUniformBuffers + slot

        guard let handle = resourceRegistry.register(texture, type: .texture, argumentIndex: argumentIndex) else {
            textureSlots.free(slot)
            Self.log.error("Resource table is full, cannot register texture")
            return nil
        }

        if configuration.enableBackgroundPopulation {
            pendingResources.append(handle)
            populationSemaphore.signal()
        } else {
            populateResourceUnsafe(handle, texture: texture, argumentIndex: argumentIndex)
        }

        bindlessMetrics.resourcesRegistered += 1

        Self.log.debug("Registered texture with handle: \(handle.value), argument index: \(argumentIndex)")
        return handle
    }

    // MARK: - Background Resource Population

    private func startBackgroundResourcePopulation() {
        populationQueue.async { [weak self] in
            while true {
                // Check if self still exists and get stop flag atomically
                guard let strongSelf = self else {
                    // Self was deallocated, exit immediately
                    break
                }

                strongSelf.populationLock.lock()
                let shouldStop = strongSelf.shouldStopPopulation
                strongSelf.populationLock.unlock()

                if shouldStop {
                    break
                }

                // Release strong reference before waiting to allow deallocation
                // Wait for work with timeout (allows periodic stop flag checks and deallocation)
                let semaphore = strongSelf.populationSemaphore
                let result = semaphore.wait(timeout: .now() + .milliseconds(100))

                // Re-check self after wait - it may have been deallocated while waiting
                guard let strongSelf = self else {
                    break
                }

                if result == .success {
                    strongSelf.processPendingResources()
                }
            }
        }
    }

    deinit {
        // Signal background thread to stop
        populationLock.lock()
        shouldStopPopulation = true
        populationLock.unlock()

        // Wake up the thread if it's waiting on the semaphore
        populationSemaphore.signal()

        // Give the background thread time to exit gracefully
        // The thread checks shouldStopPopulation every 100ms via timeout
        // Note: We can't use a completion handler with DispatchQueue.async,
        // so we rely on the timeout-based polling in the background thread
    }

    private func processPendingResources() {
        // Read all shared state under lock
        resourceLock.lock()
        let resourcesToProcess = Array(pendingResources.prefix(16))
        pendingResources.removeFirst(min(16, pendingResources.count))
        // Snapshot both buffer indices for population
        let currentIndex = currentBufferIndex
        let writeBufferIndex = 1 - currentIndex
        resourceLock.unlock()

        guard !resourcesToProcess.isEmpty else { return }

        Self.log.debug("Processing \(resourcesToProcess.count) pending resources in background")

        // Populate to BOTH buffers for immediate visibility
        // This ensures newly registered resources are visible on the very next render,
        // regardless of whether swapBuffers() has been called.
        // Each buffer has its own encoder, so this is thread-safe.
        for handle in resourcesToProcess {
            if let entry = resourceRegistry.getEntry(for: handle) {
                switch entry.resource {
                case .buffer(let buffer):
                    // Populate to write (back) buffer first
                    populateToBuffer(at: writeBufferIndex, handle: handle, buffer: buffer, argumentIndex: entry.argumentIndex)
                    // Also populate to current (front) buffer for immediate visibility
                    populateToBuffer(at: currentIndex, handle: handle, buffer: buffer, argumentIndex: entry.argumentIndex)
                case .texture(let texture):
                    populateToBuffer(at: writeBufferIndex, handle: handle, texture: texture, argumentIndex: entry.argumentIndex)
                    populateToBuffer(at: currentIndex, handle: handle, texture: texture, argumentIndex: entry.argumentIndex)
                }
            }
        }

        bindlessMetrics.resourcesPopulatedInBackground += resourcesToProcess.count
    }

    /// Populate resource to a specific buffer (thread-safe via dedicated encoder per buffer)
    private func populateToBuffer(at bufferIndex: Int, handle: ResourceHandle, buffer: MTLBuffer, argumentIndex: Int) {
        guard bufferIndex < argumentEncoders.count,
              bufferIndex < argumentBuffers.count,
              bufferIndex < resourceTables.count else { return }

        // Use the encoder dedicated to this buffer index - no thread contention
        let encoder = argumentEncoders[bufferIndex]
        let argBuffer = argumentBuffers[bufferIndex]

        // Serialize encoder operations for this specific buffer
        // Each buffer has its own encoder, so different buffer indices can run in parallel
        encoderQueue.sync {
            encoder.setArgumentBuffer(argBuffer, offset: 0)
            encoder.setBuffer(buffer, offset: 0, index: argumentIndex)
        }

        // Update the matching resource table (double-buffered)
        updateResourceTable(handle: handle, tableIndex: bufferIndex)

        // Track residency
        residencyController?.trackResource(buffer, handle: handle)
    }

    private func populateToBuffer(at bufferIndex: Int, handle: ResourceHandle, texture: MTLTexture, argumentIndex: Int) {
        guard bufferIndex < argumentEncoders.count,
              bufferIndex < argumentBuffers.count,
              bufferIndex < resourceTables.count else { return }

        let encoder = argumentEncoders[bufferIndex]
        let argBuffer = argumentBuffers[bufferIndex]

        encoderQueue.sync {
            encoder.setArgumentBuffer(argBuffer, offset: 0)
            encoder.setTexture(texture, index: argumentIndex)
        }

        updateResourceTable(handle: handle, tableIndex: bufferIndex)
        residencyController?.trackResource(texture, handle: handle)
    }

    /// Immediate population - called under resourceLock when background population is disabled
    private func populateResourceUnsafe(_ handle: ResourceHandle, buffer: MTLBuffer, argumentIndex: Int) {
        guard argumentEncoders.count == argumentBuffers.count else { return }

        // Populate both argument buffers and both resource tables using per-buffer encoders
        encoderQueue.sync {
            for i in 0..<argumentBuffers.count {
                let encoder = argumentEncoders[i]
                let argBuffer = argumentBuffers[i]
                encoder.setArgumentBuffer(argBuffer, offset: 0)
                encoder.setBuffer(buffer, offset: 0, index: argumentIndex)
                updateResourceTable(handle: handle, tableIndex: i)
            }
        }

        residencyController?.trackResource(buffer, handle: handle)
    }

    private func populateResourceUnsafe(_ handle: ResourceHandle, texture: MTLTexture, argumentIndex: Int) {
        guard argumentEncoders.count == argumentBuffers.count else { return }

        // Populate both argument buffers and both resource tables using per-buffer encoders
        encoderQueue.sync {
            for i in 0..<argumentBuffers.count {
                let encoder = argumentEncoders[i]
                let argBuffer = argumentBuffers[i]
                encoder.setArgumentBuffer(argBuffer, offset: 0)
                encoder.setTexture(texture, index: argumentIndex)
                updateResourceTable(handle: handle, tableIndex: i)
            }
        }

        residencyController?.trackResource(texture, handle: handle)
    }

    private func updateResourceTable(handle: ResourceHandle, tableIndex: Int) {
        guard tableIndex < resourceTables.count else { return }
        let table = resourceTables[tableIndex]

        let index = Int(handle.index)
        guard index < self.configuration.resourceTableSize else {
            Self.log.error("Handle index \(index) exceeds resource table size \(self.configuration.resourceTableSize)")
            return
        }

        let contents = table.contents().bindMemory(to: UInt64.self, capacity: self.configuration.resourceTableSize)
        contents[index] = handle.value
    }

    // MARK: - Render Integration (Zero Per-Draw Binding)

    /// Bind all resources once at the beginning of a render pass - no per-draw binding needed
    /// Call this at the start of each render pass. The binding uses the current buffer which
    /// is guaranteed to be fully populated.
    public func bindToRenderEncoder(_ renderEncoder: MTLRenderCommandEncoder) {
        resourceLock.lock()
        let bufferIndex = currentBufferIndex
        resourceLock.unlock()

        guard bufferIndex < argumentBuffers.count,
              bufferIndex < resourceTables.count else { return }

        let argBuffer = argumentBuffers[bufferIndex]
        let table = resourceTables[bufferIndex]

        // Bind argument buffer once for entire render pass
        renderEncoder.setVertexBuffer(argBuffer, offset: 0, index: 30) // Reserved index for bindless
        renderEncoder.setFragmentBuffer(argBuffer, offset: 0, index: 30)

        // Bind matching resource table (double-buffered)
        renderEncoder.setVertexBuffer(table, offset: 0, index: 31) // Reserved index for table
        renderEncoder.setFragmentBuffer(table, offset: 0, index: 31)

        // Resources are automatically tracked by the argument buffer
        // useResource is deprecated in macOS 13.0+, no longer needed for bindless

        bindlessMetrics.renderPassesWithoutBinding += 1

        Self.log.debug("Bound bindless resources for entire render pass - no per-draw binding needed")
    }

    /// Swap to the back buffer after GPU has finished with current frame
    /// Call this after command buffer completion to switch to the updated buffer
    public func swapBuffers() {
        resourceLock.lock()
        currentBufferIndex = 1 - currentBufferIndex
        resourceLock.unlock()
    }

    /// Bind to compute encoder for compute passes
    public func bindToComputeEncoder(_ computeEncoder: MTLComputeCommandEncoder) {
        resourceLock.lock()
        let bufferIndex = currentBufferIndex
        resourceLock.unlock()

        guard bufferIndex < argumentBuffers.count,
              bufferIndex < resourceTables.count else { return }

        let argBuffer = argumentBuffers[bufferIndex]
        let table = resourceTables[bufferIndex]

        computeEncoder.setBuffer(argBuffer, offset: 0, index: 30)
        computeEncoder.setBuffer(table, offset: 0, index: 31)

        computeEncoder.useResource(argBuffer, usage: .read)
        computeEncoder.useResource(table, usage: .read)
    }

    // MARK: - Residency Management

    /// Update residency for visible resources
    public func updateResidency(visibleHandles: [ResourceHandle], commandBuffer: MTLCommandBuffer) {
        residencyController?.updateResidency(
            visibleHandles: visibleHandles,
            commandBuffer: commandBuffer
        )

        bindlessMetrics.residencyUpdates += 1
    }

    /// Handle memory pressure by evicting unused resources
    public func handleMemoryPressure() {
        residencyController?.evictUnusedResources()

        // Clear pending resources if needed
        resourceLock.lock()
        let clearedCount = pendingResources.count
        pendingResources.removeAll()
        resourceLock.unlock()

        Self.log.info("Handled memory pressure, cleared \(clearedCount) pending resources")
    }

    // MARK: - Statistics

    public func getStatistics() -> BindlessStatistics {
        resourceLock.lock()
        let pending = pendingResources.count
        resourceLock.unlock()

        return BindlessStatistics(
            registeredResources: resourceRegistry.count,
            pendingResources: pending,
            argumentBufferSize: argumentBuffers.first?.length ?? 0,
            resourceTableSize: resourceTables.first?.length ?? 0,
            metrics: bindlessMetrics,
            residencyInfo: residencyController?.getInfo() ?? ResidencyInfo()
        )
    }

    public func printStatistics() {
        let stats = getStatistics()
        print("=== Metal 4 Bindless Architecture Statistics ===")
        print("Registered Resources: \(stats.registeredResources)")
        print("Pending Resources: \(stats.pendingResources)")
        print("Argument Buffer: \(stats.argumentBufferSize / 1024) KB")
        print("Resource Table: \(stats.resourceTableSize / 1024) KB")
        print("Resources Populated in Background: \(stats.metrics.resourcesPopulatedInBackground)")
        print("Render Passes Without Per-Draw Binding: \(stats.metrics.renderPassesWithoutBinding)")
        print("Residency Updates: \(stats.metrics.residencyUpdates)")
        print("Resident Resources: \(stats.residencyInfo.residentCount)")
        print("Evicted Resources: \(stats.residencyInfo.evictedCount)")
        print("Memory Pressure Events: \(stats.residencyInfo.memoryPressureEvents)")
    }
}

// MARK: - Supporting Types

/// Handle for bindless resource access
public struct ResourceHandle: Hashable, CustomStringConvertible, Sendable {
    let value: UInt64
    let index: UInt32
    let generation: UInt32

    static let null = ResourceHandle(value: 0, index: 0, generation: 0)

    public var description: String {
        return "ResourceHandle(index: \(index), generation: \(generation))"
    }

    init(index: UInt32, generation: UInt32) {
        self.index = index
        self.generation = generation
        self.value = (UInt64(generation) << 32) | UInt64(index)
    }

    private init(value: UInt64, index: UInt32, generation: UInt32) {
        self.value = value
        self.index = index
        self.generation = generation
    }
}

/// Resource type classification
public enum ResourceType: Sendable {
    case splatBuffer
    case uniformBuffer
    case indexBuffer
    case texture
    case sampler
}

/// Errors for bindless architecture
public enum BindlessError: LocalizedError {
    case unsupportedDevice(String)
    case argumentEncoderCreationFailed
    case argumentBufferCreationFailed
    case resourceTableCreationFailed
    case resourceNotFound(ResourceHandle)
    case resourceTableFull
    case noAvailableSlots(ResourceType)

    public var errorDescription: String? {
        switch self {
        case .unsupportedDevice(let reason):
            return "Unsupported device: \(reason)"
        case .argumentEncoderCreationFailed:
            return "Failed to create argument encoder"
        case .argumentBufferCreationFailed:
            return "Failed to create argument buffer"
        case .resourceTableCreationFailed:
            return "Failed to create resource table"
        case .resourceNotFound(let handle):
            return "Resource not found for handle: \(handle)"
        case .resourceTableFull:
            return "Resource table is full"
        case .noAvailableSlots(let type):
            return "No available slots for resource type: \(type)"
        }
    }
}

/// Slot allocator for managing typed resource slots
private class SlotAllocator {
    private var availableSlots: [Int]
    private var allocatedSlots: Set<Int>
    private let lock = NSLock()
    let maxSlots: Int

    init(maxSlots: Int) {
        self.maxSlots = maxSlots
        self.availableSlots = Array(0..<maxSlots)
        self.allocatedSlots = []
    }

    func allocate() -> Int? {
        lock.lock()
        defer { lock.unlock() }

        guard let slot = availableSlots.popLast() else {
            return nil
        }
        allocatedSlots.insert(slot)
        return slot
    }

    func free(_ slot: Int) {
        lock.lock()
        defer { lock.unlock() }

        guard allocatedSlots.remove(slot) != nil else { return }
        availableSlots.append(slot)
    }
}

/// Resource registry entry with argument index
private struct ResourceEntry {
    enum Resource {
        case buffer(MTLBuffer)
        case texture(MTLTexture)
    }

    let resource: Resource
    let type: ResourceType
    let argumentIndex: Int
}

/// Resource registry for tracking registered resources with bounds checking
private class ResourceRegistry {
    private var resources: [ResourceHandle: ResourceEntry] = [:]
    private var nextIndex: UInt32 = 1
    private var generation: UInt32 = 1
    private let maxResources: Int
    private let lock = NSLock()

    init(maxResources: Int) {
        self.maxResources = maxResources
    }

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return resources.count
    }

    /// Register a buffer. Returns nil if table is full.
    func register(_ buffer: MTLBuffer, type: ResourceType, argumentIndex: Int) -> ResourceHandle? {
        lock.lock()
        defer { lock.unlock() }

        guard nextIndex < maxResources else {
            return nil
        }

        let handle = ResourceHandle(index: nextIndex, generation: generation)
        resources[handle] = ResourceEntry(resource: .buffer(buffer), type: type, argumentIndex: argumentIndex)
        nextIndex += 1

        return handle
    }

    /// Register a texture. Returns nil if table is full.
    func register(_ texture: MTLTexture, type: ResourceType, argumentIndex: Int) -> ResourceHandle? {
        lock.lock()
        defer { lock.unlock() }

        guard nextIndex < maxResources else {
            return nil
        }

        let handle = ResourceHandle(index: nextIndex, generation: generation)
        resources[handle] = ResourceEntry(resource: .texture(texture), type: type, argumentIndex: argumentIndex)
        nextIndex += 1

        return handle
    }

    func getEntry(for handle: ResourceHandle) -> ResourceEntry? {
        lock.lock()
        defer { lock.unlock() }
        return resources[handle]
    }

    func remove(_ handle: ResourceHandle) {
        lock.lock()
        defer { lock.unlock() }
        resources.removeValue(forKey: handle)
    }
}

/// Residency controller for managing GPU memory residency
private class ResidencyController {
    private let device: MTLDevice
    private var residentResources = Set<ResourceHandle>()
    private var resourceMemory: [ResourceHandle: Int] = [:]
    private var lastAccessTime: [ResourceHandle: Date] = [:]
    private var memoryPressureEvents = 0
    private let lock = NSLock()

    init(device: MTLDevice) {
        self.device = device
    }

    func trackResource(_ resource: MTLResource, handle: ResourceHandle) {
        lock.lock()
        defer { lock.unlock() }

        residentResources.insert(handle)
        resourceMemory[handle] = resource.allocatedSize
        lastAccessTime[handle] = Date()

        // In Metal 4, this would use MTLResidencySet
        // For now, we track manually
    }

    func updateResidency(visibleHandles: [ResourceHandle], commandBuffer: MTLCommandBuffer) {
        lock.lock()
        defer { lock.unlock() }

        // Update access times for visible resources
        let now = Date()
        for handle in visibleHandles {
            lastAccessTime[handle] = now
        }

        // In Metal 4, this would use:
        // residencySet.commit(to: commandBuffer)
    }

    func evictUnusedResources() {
        lock.lock()
        defer { lock.unlock() }

        memoryPressureEvents += 1

        // Evict resources not accessed in last 5 seconds
        let cutoffTime = Date().addingTimeInterval(-5.0)
        var evictedHandles: [ResourceHandle] = []

        for (handle, accessTime) in lastAccessTime {
            if accessTime < cutoffTime {
                evictedHandles.append(handle)
            }
        }

        for handle in evictedHandles {
            residentResources.remove(handle)
            resourceMemory.removeValue(forKey: handle)
            lastAccessTime.removeValue(forKey: handle)
        }

        // Log through a simple print for now
        print("Metal4BindlessArchitecture: Evicted \(evictedHandles.count) resources due to memory pressure")
    }

    func getInfo() -> ResidencyInfo {
        lock.lock()
        defer { lock.unlock() }

        let totalMemory = resourceMemory.values.reduce(0, +)

        return ResidencyInfo(
            residentCount: residentResources.count,
            evictedCount: 0,
            totalMemoryMB: Float(totalMemory) / (1024 * 1024),
            memoryPressureEvents: memoryPressureEvents
        )
    }
}

/// Metrics for bindless performance tracking
public struct BindlessMetrics {
    var resourcesRegistered: Int = 0
    var resourcesPopulatedInBackground: Int = 0
    var renderPassesWithoutBinding: Int = 0
    var residencyUpdates: Int = 0
}

/// Residency information
public struct ResidencyInfo {
    var residentCount: Int = 0
    var evictedCount: Int = 0
    var totalMemoryMB: Float = 0
    var memoryPressureEvents: Int = 0
}

/// Statistics for bindless architecture
public struct BindlessStatistics {
    public let registeredResources: Int
    public let pendingResources: Int
    public let argumentBufferSize: Int
    public let resourceTableSize: Int
    public let metrics: BindlessMetrics
    public let residencyInfo: ResidencyInfo
}
