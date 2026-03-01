import Foundation
import Metal
import os

/// Real Metal 4 Argument Buffer implementation replacing custom MTL4ArgumentTable abstractions
/// Uses genuine MTLArgumentEncoder and argument buffers for true Metal API compliance
@available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
public class Metal4ArgumentBufferManager {
    
    private static let log = Logger(
        subsystem: Bundle.module.bundleIdentifier ?? "com.metalsplatter.unknown",
        category: "Metal4ArgumentBufferManager"
    )
    
    // MARK: - Properties
    
    private let device: MTLDevice
    private let maxSplatCount: Int
    
    // Real Metal APIs (not custom abstractions)
    private var argumentEncoder: MTLArgumentEncoder?
    private var argumentBuffer: MTLBuffer?
    private var residencySet: MTLResidencySet?
    
    // Resource management
    private var splatBuffers: [MTLBuffer] = []
    private var uniformBuffers: [MTLBuffer] = []
    private var trackedAllocations: Set<ObjectIdentifier> = []
    private var queuesWithAttachedResidencySet: Set<ObjectIdentifier> = []
    private let lock = NSLock()
    
    // MARK: - Initialization
    
    public init(device: MTLDevice, maxSplatCount: Int) throws {
        self.device = device
        self.maxSplatCount = maxSplatCount
        
        try setupArgumentBuffers()
        try setupResidencySet()
    }
    
    // MARK: - Real Metal API Setup
    
    private func setupArgumentBuffers() throws {
        // Create argument descriptor for our resources
        let argumentDescriptors = [
            MTLArgumentDescriptor.init().with {
                $0.index = 0
                $0.dataType = .pointer
                $0.access = .readOnly
                $0.arrayLength = 1
            },
            MTLArgumentDescriptor.init().with {
                $0.index = 1
                $0.dataType = .pointer
                $0.access = .readOnly
                $0.arrayLength = 1
            }
        ]
        
        // Create real MTLArgumentEncoder (not our custom abstraction)
        guard let encoder = device.makeArgumentEncoder(arguments: argumentDescriptors) else {
            throw Metal4Error.argumentEncoderCreationFailed
        }
        
        self.argumentEncoder = encoder
        
        // Create argument buffer using real Metal API
        guard let argBuffer = device.makeBuffer(length: encoder.encodedLength, 
                                               options: [.storageModeShared]) else {
            throw Metal4Error.argumentBufferCreationFailed
        }
        
        self.argumentBuffer = argBuffer
        Self.log.info("✅ Created real MTLArgumentEncoder with length: \(encoder.encodedLength)")
    }
    
    private func setupResidencySet() throws {
        do {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.label = "MetalSplatter Residency Set"
            descriptor.initialCapacity = max(8, maxSplatCount / 4)

            let set = try device.makeResidencySet(descriptor: descriptor)
            set.commit()
            set.requestResidency()
            residencySet = set

            Self.log.info("✅ Metal4 residency management initialized")
        } catch {
            throw Metal4Error.residencySetCreationFailed
        }
    }
    
    // MARK: - Resource Management
    
    public func registerSplatBuffer(_ buffer: MTLBuffer, at index: Int) throws {
        guard let encoder = argumentEncoder,
              let argBuffer = argumentBuffer else {
            throw Metal4Error.notInitialized
        }
        
        lock.lock()
        defer { lock.unlock() }

        // Use real Metal API to encode buffer into argument buffer
        encoder.setArgumentBuffer(argBuffer, offset: 0)
        encoder.setBuffer(buffer, offset: 0, index: index)

        if !splatBuffers.contains(where: { $0 === buffer }) {
            splatBuffers.append(buffer)
        }
        registerAllocationIfNeeded(buffer)

        Self.log.debug("Registered splat buffer at index \(index) using real MTLArgumentEncoder")
    }

    public func registerUniformBuffer(_ buffer: MTLBuffer, at index: Int = 1) throws {
        guard let encoder = argumentEncoder,
              let argBuffer = argumentBuffer else {
            throw Metal4Error.notInitialized
        }

        lock.lock()
        defer { lock.unlock() }

        encoder.setArgumentBuffer(argBuffer, offset: 0)
        encoder.setBuffer(buffer, offset: 0, index: index)

        if !uniformBuffers.contains(where: { $0 === buffer }) {
            uniformBuffers.append(buffer)
        }
        registerAllocationIfNeeded(buffer)

        Self.log.debug("Registered uniform buffer at index \(index)")
    }

    public func registerAdditionalBuffer(_ buffer: MTLBuffer) {
        lock.lock()
        defer { lock.unlock() }
        registerAllocationIfNeeded(buffer)
    }

    private func registerAllocationIfNeeded(_ allocation: any MTLAllocation) {
        guard let residencySet else { return }

        let allocationID = ObjectIdentifier(allocation as AnyObject)
        guard !trackedAllocations.contains(allocationID) else { return }

        residencySet.addAllocation(allocation)
        trackedAllocations.insert(allocationID)
        residencySet.commit()
    }
    
    public func makeResourcesResident(commandBuffer: MTLCommandBuffer) {
        lock.lock()
        defer { lock.unlock() }

        guard let residencySet else { return }

        let queue = commandBuffer.commandQueue
        let queueID = ObjectIdentifier(queue)
        if !queuesWithAttachedResidencySet.contains(queueID) {
            queue.addResidencySet(residencySet)
            queuesWithAttachedResidencySet.insert(queueID)
        }

        residencySet.requestResidency()
    }
    
    // MARK: - Render Pass Integration
    
    public func bindArgumentBuffer(to renderEncoder: MTLRenderCommandEncoder, index: Int = 0) {
        guard let argBuffer = argumentBuffer else {
            Self.log.warning("Argument buffer not available for binding")
            return
        }
        
        // Bind argument buffer using standard Metal API
        renderEncoder.setVertexBuffer(argBuffer, offset: 0, index: index)
        renderEncoder.setFragmentBuffer(argBuffer, offset: 0, index: index)
        
        Self.log.debug("Bound argument buffer at index \(index)")
    }
    
    // MARK: - Statistics
    
    public func getStatistics() -> ArgumentBufferStatistics {
        lock.lock()
        defer { lock.unlock() }

        let bufferCount = splatBuffers.count + uniformBuffers.count
        let totalMemory = splatBuffers.reduce(0) { $0 + $1.length } + 
                         uniformBuffers.reduce(0) { $0 + $1.length }
        
        return ArgumentBufferStatistics(
            argumentBufferSize: argumentBuffer?.length ?? 0,
            resourceCount: bufferCount,
            totalResourceMemoryMB: Float(totalMemory) / (1024 * 1024),
            residencySetSize: residencySet?.allocationCount ?? 0
        )
    }
}

// MARK: - Supporting Types

@available(iOS 26.0, macOS 26.0, tvOS 26.0, visionOS 26.0, *)
public enum Metal4Error: LocalizedError {
    case argumentEncoderCreationFailed
    case argumentBufferCreationFailed
    case residencySetCreationFailed
    case notInitialized
    
    public var errorDescription: String? {
        switch self {
        case .argumentEncoderCreationFailed:
            return "Failed to create MTLArgumentEncoder"
        case .argumentBufferCreationFailed:
            return "Failed to create argument buffer"
        case .residencySetCreationFailed:
            return "Failed to create MTLResidencySet"
        case .notInitialized:
            return "Metal4ArgumentBufferManager not properly initialized"
        }
    }
}

public struct ArgumentBufferStatistics {
    public let argumentBufferSize: Int
    public let resourceCount: Int
    public let totalResourceMemoryMB: Float
    public let residencySetSize: Int
}

// MARK: - Helper Extension

private extension MTLArgumentDescriptor {
    func with(_ configure: (MTLArgumentDescriptor) -> Void) -> MTLArgumentDescriptor {
        configure(self)
        return self
    }
}
