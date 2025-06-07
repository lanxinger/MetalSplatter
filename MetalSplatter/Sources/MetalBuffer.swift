import Foundation
import Metal
import os

fileprivate let log =
    Logger(subsystem: Bundle.module.bundleIdentifier!,
           category: "MetalBuffer")

class MetalBuffer<T> {
    enum Error: LocalizedError {
        case capacityGreatedThanMaxCapacity(requested: Int, max: Int)
        case bufferCreationFailed

        var errorDescription: String? {
            switch self {
            case .capacityGreatedThanMaxCapacity(let requested, let max):
                "Requested metal buffer size (\(requested)) exceeds device maximum (\(max))"
            case .bufferCreationFailed:
                "Failed to create metal buffer"
            }
        }
    }

    let device: MTLDevice
    let storageMode: MTLResourceOptions
    
    var capacity: Int = 0
    var count: Int = 0
    var buffer: MTLBuffer
    var values: UnsafeMutablePointer<T>
    
    // Smart growth constants (computed properties to avoid static stored property limitation)
    private var smallBufferThreshold: Int { 1024 }
    private var mediumBufferThreshold: Int { 32768 }
    private var maxGrowthFactor: Float { 1.5 }

    init(device: MTLDevice, capacity: Int = 1, storageMode: MTLResourceOptions = .storageModeShared) throws {
        let capacity = max(capacity, 1)
        guard capacity <= Self.maxCapacity(for: device) else {
            throw Error.capacityGreatedThanMaxCapacity(requested: capacity, max: Self.maxCapacity(for: device))
        }

        self.device = device
        self.storageMode = storageMode

        self.capacity = capacity
        self.count = 0
        guard let buffer = device.makeBuffer(length: MemoryLayout<T>.stride * self.capacity,
                                             options: .storageModeShared) else {
            throw Error.bufferCreationFailed
        }
        self.buffer = buffer
        self.values = UnsafeMutableRawPointer(self.buffer.contents()).bindMemory(to: T.self, capacity: self.capacity)
    }

    static func maxCapacity(for device: MTLDevice) -> Int {
        device.maxBufferLength / MemoryLayout<T>.stride
    }

    var maxCapacity: Int {
        device.maxBufferLength / MemoryLayout<T>.stride
    }

    func setCapacity(_ newCapacity: Int) throws {
        let newCapacity = max(newCapacity, 1)
        guard newCapacity != capacity else { return }
        guard capacity <= maxCapacity else {
            throw Error.capacityGreatedThanMaxCapacity(requested: capacity, max: maxCapacity)
        }

        // Smart growth strategy based on buffer size
        let actualNewCapacity: Int
        if newCapacity > capacity {
            if capacity < smallBufferThreshold {
                // Double small buffers for quick initial growth
                actualNewCapacity = max(newCapacity, capacity * 2)
            } else if capacity < mediumBufferThreshold {
                // 1.5x growth for medium buffers
                actualNewCapacity = max(newCapacity, Int(Float(capacity) * maxGrowthFactor))
            } else {
                // Conservative growth for large buffers (25% or minimum needed)
                let conservativeGrowth = capacity + max(capacity / 4, newCapacity - capacity)
                actualNewCapacity = min(conservativeGrowth, maxCapacity)
            }
        } else {
            actualNewCapacity = newCapacity
        }

        log.info("Allocating a new buffer of size \(MemoryLayout<T>.stride) * \(actualNewCapacity) = \(Float(MemoryLayout<T>.stride * actualNewCapacity) / (1024.0 * 1024.0))mb")
        
        guard let newBuffer = device.makeBuffer(length: MemoryLayout<T>.stride * actualNewCapacity,
                                                options: self.storageMode) else {
            throw Error.bufferCreationFailed
        }
        let newValues = UnsafeMutableRawPointer(newBuffer.contents()).bindMemory(to: T.self, capacity: actualNewCapacity)
        let newCount = min(count, actualNewCapacity)
        if newCount > 0 {
            memcpy(newValues, values, MemoryLayout<T>.stride * newCount)
        }

        self.capacity = actualNewCapacity
        self.count = newCount
        self.buffer = newBuffer
        self.values = newValues
    }

    func ensureCapacity(_ minimumCapacity: Int) throws {
        guard capacity < minimumCapacity else { return }
        try setCapacity(minimumCapacity)
    }

    /// Assumes capacity is available
    /// Returns the index of the value
    @discardableResult
    func append(_ element: T) -> Int {
        (values + count).pointee = element
        defer { count += 1 }
        return count
    }

    /// Assumes capacity is available.
    /// Returns the index of the first values.
    @discardableResult
    func append(_ elements: [T]) -> Int {
        (values + count).update(from: elements, count: elements.count)
        defer { count += elements.count }
        return count
    }

    /// Assumes capacity is available
    /// Returns the index of the value
    @discardableResult
    func append(_ otherBuffer: MetalBuffer<T>, fromIndex: Int) -> Int {
        (values + count).pointee = (otherBuffer.values + fromIndex).pointee
        defer { count += 1 }
        return count
    }
}
