import Foundation
import Metal
import os

fileprivate let log =
    Logger(subsystem: Bundle.module.bundleIdentifier ?? "MetalSplatter",
           category: "MetalBuffer")

/// Thread-safe Metal buffer wrapper for GPU data storage.
///
/// Thread Safety:
/// - All mutations to `count`, `capacity`, `buffer`, and `values` are protected by an internal lock.
/// - Reading `count` and `capacity` is thread-safe.
/// - **Important**: The `buffer` and `values` properties return references that can become invalid
///   if another thread calls `setCapacity`. For safe access during potential resize operations,
///   use `withLockedValues(_:)` or `withLockedBuffer(_:)` which hold the lock during the closure.
/// - Direct access to `buffer` and `values` for GPU operations should be coordinated externally
///   (e.g., via command buffer completion handlers) to avoid data races with GPU execution.
public class MetalBuffer<T>: @unchecked Sendable {
    public enum Error: LocalizedError {
        case capacityGreatedThanMaxCapacity(requested: Int, max: Int)
        case bufferCreationFailed

        public var errorDescription: String? {
            switch self {
            case .capacityGreatedThanMaxCapacity(let requested, let max):
                "Requested metal buffer size (\(requested)) exceeds device maximum (\(max))"
            case .bufferCreationFailed:
                "Failed to create metal buffer"
            }
        }
    }

    public let device: MTLDevice

    // Lock protecting all mutable state
    private var lock = os_unfair_lock()

    private var _capacity: Int = 0
    private var _count: Int = 0
    private var _buffer: MTLBuffer
    private var _values: UnsafeMutablePointer<T>

    /// Current capacity of the buffer (thread-safe read)
    public var capacity: Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return _capacity
    }

    /// Current number of elements in the buffer (thread-safe read/write)
    public var count: Int {
        get {
            os_unfair_lock_lock(&lock)
            defer { os_unfair_lock_unlock(&lock) }
            return _count
        }
        set {
            os_unfair_lock_lock(&lock)
            _count = newValue
            os_unfair_lock_unlock(&lock)
        }
    }

    /// The underlying Metal buffer.
    /// Note: Direct access should be coordinated with GPU execution via command buffer synchronization.
    public var buffer: MTLBuffer {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return _buffer
    }

    /// Pointer to the buffer contents.
    /// Warning: This pointer can become invalid if `setCapacity` is called concurrently.
    /// For safe access during potential resize operations, use `withLockedValues(_:)`.
    public var values: UnsafeMutablePointer<T> {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return _values
    }

    /// Execute a closure with locked access to the values pointer.
    /// This guarantees the pointer remains valid for the duration of the closure,
    /// even if another thread attempts to resize the buffer.
    ///
    /// - Parameter body: A closure that receives the values pointer and current count.
    /// - Returns: The result of the closure.
    @discardableResult
    public func withLockedValues<R>(_ body: (UnsafeMutablePointer<T>, Int) throws -> R) rethrows -> R {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return try body(_values, _count)
    }

    /// Execute a closure with locked access to the Metal buffer.
    /// This guarantees the buffer reference remains valid for the duration of the closure.
    ///
    /// - Parameter body: A closure that receives the buffer and current count.
    /// - Returns: The result of the closure.
    @discardableResult
    public func withLockedBuffer<R>(_ body: (MTLBuffer, Int) throws -> R) rethrows -> R {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return try body(_buffer, _count)
    }

    public init(device: MTLDevice, capacity: Int = 1) throws {
        let capacity = max(capacity, 1)
        guard capacity <= Self.maxCapacity(for: device) else {
            throw Error.capacityGreatedThanMaxCapacity(requested: capacity, max: Self.maxCapacity(for: device))
        }

        self.device = device

        self._capacity = capacity
        self._count = 0
        guard let buffer = device.makeBuffer(length: MemoryLayout<T>.stride * capacity,
                                             options: .storageModeShared) else {
            throw Error.bufferCreationFailed
        }
        self._buffer = buffer
        self._values = UnsafeMutableRawPointer(buffer.contents()).bindMemory(to: T.self, capacity: capacity)
    }

    public static func maxCapacity(for device: MTLDevice) -> Int {
        device.maxBufferLength / MemoryLayout<T>.stride
    }

    public var maxCapacity: Int {
        device.maxBufferLength / MemoryLayout<T>.stride
    }

    public func setCapacity(_ newCapacity: Int) throws {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        let newCapacity = max(newCapacity, 1)
        guard newCapacity != _capacity else { return }
        let maxCap = self.maxCapacity
        guard newCapacity <= maxCap else {
            throw Error.capacityGreatedThanMaxCapacity(requested: newCapacity, max: maxCap)
        }

        // Use exponential growth strategy to reduce frequent reallocations
        let growthTarget = newCapacity > _capacity ? max(newCapacity, _capacity * 2) : newCapacity
        let actualNewCapacity = min(growthTarget, maxCap)
        if growthTarget > maxCap {
            log.warning("Requested buffer growth to \(growthTarget) exceeds device limit \(maxCap); clamping to \(actualNewCapacity)")
        }

        log.info("Allocating a new buffer of size \(MemoryLayout<T>.stride) * \(actualNewCapacity) = \(Float(MemoryLayout<T>.stride * actualNewCapacity) / (1024.0 * 1024.0))mb")

        // Use shared storage mode for CPU+GPU access
        let storageMode: MTLResourceOptions = .storageModeShared

        guard let newBuffer = device.makeBuffer(length: MemoryLayout<T>.stride * actualNewCapacity,
                                                options: storageMode) else {
            throw Error.bufferCreationFailed
        }
        let newValues = UnsafeMutableRawPointer(newBuffer.contents()).bindMemory(to: T.self, capacity: actualNewCapacity)
        let newCount = min(_count, actualNewCapacity)
        if newCount > 0 {
            memcpy(newValues, _values, MemoryLayout<T>.stride * newCount)
        }

        self._capacity = actualNewCapacity
        self._count = newCount
        self._buffer = newBuffer
        self._values = newValues
    }

    /// Ensure the buffer has at least the specified capacity (grow-only).
    /// This is safe against concurrent capacity increases by other threads.
    public func ensureCapacity(_ minimumCapacity: Int) throws {
        os_unfair_lock_lock(&lock)

        // Check if we already have enough capacity (another thread may have grown it)
        guard _capacity < minimumCapacity else {
            os_unfair_lock_unlock(&lock)
            return
        }

        // Perform growth while holding the lock
        let maxCap = self.maxCapacity
        guard minimumCapacity <= maxCap else {
            os_unfair_lock_unlock(&lock)
            throw Error.capacityGreatedThanMaxCapacity(requested: minimumCapacity, max: maxCap)
        }

        // Use exponential growth strategy
        let growthTarget = max(minimumCapacity, _capacity * 2)
        let actualNewCapacity = min(growthTarget, maxCap)
        if growthTarget > maxCap {
            log.warning("Requested buffer growth to \(growthTarget) exceeds device limit \(maxCap); clamping to \(actualNewCapacity)")
        }

        log.info("Allocating a new buffer of size \(MemoryLayout<T>.stride) * \(actualNewCapacity) = \(Float(MemoryLayout<T>.stride * actualNewCapacity) / (1024.0 * 1024.0))mb")

        guard let newBuffer = device.makeBuffer(length: MemoryLayout<T>.stride * actualNewCapacity,
                                                options: .storageModeShared) else {
            os_unfair_lock_unlock(&lock)
            throw Error.bufferCreationFailed
        }
        let newValues = UnsafeMutableRawPointer(newBuffer.contents()).bindMemory(to: T.self, capacity: actualNewCapacity)
        if _count > 0 {
            memcpy(newValues, _values, MemoryLayout<T>.stride * _count)
        }

        self._capacity = actualNewCapacity
        self._buffer = newBuffer
        self._values = newValues

        os_unfair_lock_unlock(&lock)
    }

    /// Assumes capacity is available
    /// Returns the index of the value
    @discardableResult
    public func append(_ element: T) -> Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        let index = _count
        (_values + _count).pointee = element
        _count += 1
        return index
    }

    /// Assumes capacity is available.
    /// Returns the index of the first values.
    @discardableResult
    public func append(_ elements: [T]) -> Int {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        let index = _count
        (_values + _count).update(from: elements, count: elements.count)
        _count += elements.count
        return index
    }

    /// Assumes capacity is available
    /// Returns the index of the value
    @discardableResult
    public func append(_ otherBuffer: MetalBuffer<T>, fromIndex: Int) -> Int {
        // Guard against self-append deadlock
        precondition(otherBuffer !== self, "Cannot append buffer to itself")

        // Copy element under source lock to ensure pointer validity during read
        let elementToCopy = otherBuffer.withLockedValues { values, count in
            precondition(fromIndex >= 0 && fromIndex < count,
                         "Index \(fromIndex) out of bounds [0..<\(count)]")
            return (values + fromIndex).pointee
        }

        // Now lock self and append
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }

        let index = _count
        (_values + _count).pointee = elementToCopy
        _count += 1
        return index
    }
}
