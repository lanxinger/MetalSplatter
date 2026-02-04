import Foundation
import simd

public final class GltfGaussianSplatSceneReader: SplatSceneReader {
    public enum Error: Swift.Error {
        case unsupportedFileType
        case invalidGLBHeader
        case missingJSONChunk
        case unsupportedGLBVersion(UInt32)
        case missingBuffers
        case missingBufferView(Int)
        case missingAccessor(Int)
        case unsupportedAccessorType(String)
        case unsupportedComponentType(Int)
        case sparseAccessorsNotSupported
        case bufferOutOfBounds
        case nonUniformNodeScale
        case negativeOrZeroScale
        case missingRequiredAttribute(String)
        case mismatchedAccessorCounts
        case missingGaussianExtension
    }

    private struct GltfRoot: Codable {
        var asset: GltfAsset?
        var buffers: [GltfBuffer]?
        var bufferViews: [GltfBufferView]?
        var accessors: [GltfAccessor]?
        var meshes: [GltfMesh]?
        var nodes: [GltfNode]?
        var scenes: [GltfScene]?
        var scene: Int?
    }

    private struct GltfAsset: Codable {
        var version: String?
    }

    private struct GltfBuffer: Codable {
        var uri: String?
        var byteLength: Int
    }

    private struct GltfBufferView: Codable {
        var buffer: Int
        var byteOffset: Int?
        var byteLength: Int
        var byteStride: Int?
    }

    private struct GltfAccessor: Codable {
        var bufferView: Int?
        var byteOffset: Int?
        var componentType: Int
        var normalized: Bool?
        var count: Int
        var type: String
        var sparse: GltfAccessorSparse?
    }

    private struct GltfAccessorSparse: Codable {
        var count: Int
    }

    private struct GltfMesh: Codable {
        var primitives: [GltfPrimitive]
    }

    private struct GltfPrimitive: Codable {
        var attributes: [String: Int]
        var mode: Int?
        var extensions: GltfPrimitiveExtensions?
    }

    private struct GltfPrimitiveExtensions: Codable {
        var khrGaussianSplatting: GltfGaussianSplattingExtension?

        enum CodingKeys: String, CodingKey {
            case khrGaussianSplatting = "KHR_gaussian_splatting"
        }
    }

    private struct GltfGaussianSplattingExtension: Codable {
        var kernel: String?
        var colorSpace: String?
        var sortingMethod: String?
        var projection: String?
    }

    private struct GltfNode: Codable {
        var mesh: Int?
        var children: [Int]?
        var matrix: [Float]?
        var translation: [Float]?
        var rotation: [Float]?
        var scale: [Float]?
    }

    private struct GltfScene: Codable {
        var nodes: [Int]?
    }

    private struct Transform {
        var translation: SIMD3<Float>
        var rotation: simd_quatf
        var scale: Float

        static let identity = Transform(translation: .zero, rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1), scale: 1)

        func combined(with child: Transform) throws -> Transform {
            guard scale > 0 else { throw Error.negativeOrZeroScale }
            guard child.scale > 0 else { throw Error.negativeOrZeroScale }

            let scaledTranslation = child.translation * scale
            let rotatedTranslation = rotation.act(scaledTranslation)
            let combinedTranslation = translation + rotatedTranslation
            let combinedRotation = rotation * child.rotation
            let combinedScale = scale * child.scale
            return Transform(translation: combinedTranslation, rotation: combinedRotation, scale: combinedScale)
        }
    }

    private let url: URL
    private let root: GltfRoot
    private let buffers: [Data]

    public init(_ url: URL) throws {
        self.url = url
        let (root, buffers) = try GltfGaussianSplatSceneReader.loadRootAndBuffers(from: url)
        self.root = root
        self.buffers = buffers
    }

    public func readScene() throws -> [SplatScenePoint] {
        guard let meshes = root.meshes, let accessors = root.accessors, let bufferViews = root.bufferViews else {
            return []
        }

        let accessorReader = AccessorReader(accessors: accessors, bufferViews: bufferViews, buffers: buffers)
        var results: [SplatScenePoint] = []

        if let nodes = root.nodes, let scenes = root.scenes {
            let sceneIndex = root.scene ?? 0
            let sceneNodes = sceneIndex < scenes.count ? (scenes[sceneIndex].nodes ?? []) : []
            let nodeIndices = sceneNodes.isEmpty ? Array(nodes.indices) : sceneNodes
            for nodeIndex in nodeIndices {
                try traverse(nodeIndex: nodeIndex, parentTransform: .identity, nodes: nodes, meshes: meshes, accessorReader: accessorReader, results: &results)
            }
        } else if !meshes.isEmpty {
            for meshIndex in meshes.indices {
                try appendMesh(meshIndex: meshIndex, transform: .identity, meshes: meshes, accessorReader: accessorReader, results: &results)
            }
        }

        return results
    }

    public func read(to delegate: SplatSceneReaderDelegate) {
        do {
            let points = try readScene()
            delegate.didStartReading(withPointCount: UInt32(points.count))
            delegate.didRead(points: points)
            delegate.didFinishReading()
        } catch {
            delegate.didFailReading(withError: error)
        }
    }

    private func traverse(nodeIndex: Int,
                          parentTransform: Transform,
                          nodes: [GltfNode],
                          meshes: [GltfMesh],
                          accessorReader: AccessorReader,
                          results: inout [SplatScenePoint]) throws {
        guard nodeIndex >= 0 && nodeIndex < nodes.count else { return }
        let node = nodes[nodeIndex]
        let nodeTransform = try TransformBuilder.transform(from: node)
        let combinedTransform = try parentTransform.combined(with: nodeTransform)

        if let meshIndex = node.mesh {
            try appendMesh(meshIndex: meshIndex, transform: combinedTransform, meshes: meshes, accessorReader: accessorReader, results: &results)
        }

        for child in node.children ?? [] {
            try traverse(nodeIndex: child, parentTransform: combinedTransform, nodes: nodes, meshes: meshes, accessorReader: accessorReader, results: &results)
        }
    }

    private func appendMesh(meshIndex: Int,
                            transform: Transform,
                            meshes: [GltfMesh],
                            accessorReader: AccessorReader,
                            results: inout [SplatScenePoint]) throws {
        guard meshIndex >= 0 && meshIndex < meshes.count else { return }
        let mesh = meshes[meshIndex]

        for primitive in mesh.primitives {
            let mode = primitive.mode ?? 4
            guard mode == 0 else { continue }

            guard let ext = primitive.extensions?.khrGaussianSplatting else { continue }
            if let kernel = ext.kernel, kernel != "ellipse" { continue }

            guard let positionAccessor = primitive.attributes["POSITION"] else { continue }
            guard let rotationAccessor = primitive.attributes["KHR_gaussian_splatting:ROTATION"] else { continue }
            guard let scaleAccessor = primitive.attributes["KHR_gaussian_splatting:SCALE"] else { continue }
            guard let opacityAccessor = primitive.attributes["KHR_gaussian_splatting:OPACITY"] else { continue }

            let positions = try accessorReader.readVec3(positionAccessor)
            let rotations = try accessorReader.readVec4(rotationAccessor)
            let scales = try accessorReader.readVec3(scaleAccessor)
            let opacities = try accessorReader.readScalar(opacityAccessor)

            let count = positions.count
            if rotations.count != count || scales.count != count || opacities.count != count {
                continue
            }

            let sh0Accessor = primitive.attributes["KHR_gaussian_splatting:SH_DEGREE_0_COEF_0"]
            var sh1Accessors: [Int] = []
            var sh2Accessors: [Int] = []
            var sh3Accessors: [Int] = []

            for i in 0..<3 {
                if let index = primitive.attributes["KHR_gaussian_splatting:SH_DEGREE_1_COEF_\(i)"] {
                    sh1Accessors.append(index)
                }
            }
            for i in 0..<5 {
                if let index = primitive.attributes["KHR_gaussian_splatting:SH_DEGREE_2_COEF_\(i)"] {
                    sh2Accessors.append(index)
                }
            }
            for i in 0..<7 {
                if let index = primitive.attributes["KHR_gaussian_splatting:SH_DEGREE_3_COEF_\(i)"] {
                    sh3Accessors.append(index)
                }
            }

            let color0Accessor = primitive.attributes["COLOR_0"]

            var sh0Values: [SIMD3<Float>]? = nil
            var sh1Values: [[SIMD3<Float>]] = []
            var sh2Values: [[SIMD3<Float>]] = []
            var sh3Values: [[SIMD3<Float>]] = []
            var color0Values: [SIMD4<Float>]? = nil
            var shDataValid = true

            if let sh0Accessor {
                sh0Values = try accessorReader.readVec3(sh0Accessor)
                if sh0Values?.count != count { sh0Values = nil }
            }

            if sh1Accessors.count > 0 {
                guard sh1Accessors.count == 3 else { continue }
                for index in sh1Accessors {
                    let values = try accessorReader.readVec3(index)
                    if values.count != count { shDataValid = false; break }
                    sh1Values.append(values)
                }
            }

            if sh2Accessors.count > 0 {
                guard sh2Accessors.count == 5 else { continue }
                for index in sh2Accessors {
                    let values = try accessorReader.readVec3(index)
                    if values.count != count { shDataValid = false; break }
                    sh2Values.append(values)
                }
            }

            if sh3Accessors.count > 0 {
                guard sh3Accessors.count == 7 else { continue }
                for index in sh3Accessors {
                    let values = try accessorReader.readVec3(index)
                    if values.count != count { shDataValid = false; break }
                    sh3Values.append(values)
                }
            }

            if !shDataValid { continue }
            if sh0Values == nil && (!sh1Values.isEmpty || !sh2Values.isEmpty || !sh3Values.isEmpty) { continue }

            if let color0Accessor {
                color0Values = try accessorReader.readColor(color0Accessor)
                if color0Values?.count != count { color0Values = nil }
            }

            for i in 0..<count {
                var coeffs: [SIMD3<Float>] = []
                if let sh0Values {
                    coeffs.append(sh0Values[i])
                }
                for values in sh1Values { coeffs.append(values[i]) }
                for values in sh2Values { coeffs.append(values[i]) }
                for values in sh3Values { coeffs.append(values[i]) }

                let color: SplatScenePoint.Color
                if !coeffs.isEmpty {
                    color = .sphericalHarmonic(coeffs)
                } else if let color0Values {
                    let rgba = color0Values[i]
                    color = .linearFloat(SIMD3<Float>(rgba.x, rgba.y, rgba.z))
                } else {
                    continue
                }

                var opacity = opacities[i]
                if opacity < 0 || opacity > 1 {
                    opacity = max(0, min(1, opacity))
                }

                let rotationVector = rotations[i]
                var rotation = simd_quatf(ix: rotationVector.x, iy: rotationVector.y, iz: rotationVector.z, r: rotationVector.w)
                rotation = rotation.normalized

                let scaleExp = scales[i]
                let adjustedScale: SIMD3<Float>
                if transform.scale != 1 {
                    let logScale = log(transform.scale)
                    adjustedScale = scaleExp + SIMD3<Float>(repeating: logScale)
                } else {
                    adjustedScale = scaleExp
                }

                let scaledPosition = positions[i] * transform.scale
                let transformedPosition = transform.translation + transform.rotation.act(scaledPosition)
                let transformedRotation = (transform.rotation * rotation).normalized

                let point = SplatScenePoint(position: transformedPosition,
                                            color: color,
                                            opacity: .linearFloat(opacity),
                                            scale: .exponent(adjustedScale),
                                            rotation: transformedRotation)
                results.append(point)
            }
        }
    }

    private static func loadRootAndBuffers(from url: URL) throws -> (GltfRoot, [Data]) {
        let ext = url.pathExtension.lowercased()
        let jsonData: Data
        let glbBinData: Data?

        if ext == "glb" {
            let data = try Data(contentsOf: url)
            let (jsonChunk, binChunk) = try parseGLB(data)
            jsonData = jsonChunk
            glbBinData = binChunk
        } else if ext == "gltf" {
            jsonData = try Data(contentsOf: url)
            glbBinData = nil
        } else {
            throw Error.unsupportedFileType
        }

        let decoder = JSONDecoder()
        let root = try decoder.decode(GltfRoot.self, from: jsonData)
        let buffers = try loadBuffers(root: root, baseURL: url.deletingLastPathComponent(), glbBinData: glbBinData)
        return (root, buffers)
    }

    private static func parseGLB(_ data: Data) throws -> (Data, Data?) {
        guard data.count >= 12 else { throw Error.invalidGLBHeader }
        let magic = readUInt32(data, offset: 0)
        guard magic == 0x46546c67 else { throw Error.invalidGLBHeader }
        let version = readUInt32(data, offset: 4)
        guard version == 2 else { throw Error.unsupportedGLBVersion(version) }
        let totalLength = Int(readUInt32(data, offset: 8))
        if totalLength > data.count { throw Error.invalidGLBHeader }

        var offset = 12
        var jsonChunk: Data?
        var binChunk: Data?

        while offset + 8 <= data.count {
            let chunkLength = Int(readUInt32(data, offset: offset))
            let chunkType = readUInt32(data, offset: offset + 4)
            let chunkStart = offset + 8
            let chunkEnd = chunkStart + chunkLength
            if chunkEnd > data.count { throw Error.invalidGLBHeader }
            let chunkData = data.subdata(in: chunkStart..<chunkEnd)

            if chunkType == 0x4e4f534a { // JSON
                jsonChunk = chunkData
            } else if chunkType == 0x004e4942 { // BIN\0
                binChunk = chunkData
            }

            offset = chunkEnd
        }

        guard let json = jsonChunk else { throw Error.missingJSONChunk }
        return (json, binChunk)
    }

    private static func loadBuffers(root: GltfRoot, baseURL: URL, glbBinData: Data?) throws -> [Data] {
        guard let buffers = root.buffers else { return [] }
        var results: [Data] = []
        results.reserveCapacity(buffers.count)

        for (index, buffer) in buffers.enumerated() {
            if let uri = buffer.uri {
                if uri.hasPrefix("data:") {
                    guard let data = decodeDataURI(uri) else { throw Error.missingBuffers }
                    results.append(data)
                } else {
                    let bufferURL = baseURL.appendingPathComponent(uri)
                    results.append(try Data(contentsOf: bufferURL))
                }
            } else {
                if index == 0, let glbBinData {
                    results.append(glbBinData)
                } else {
                    throw Error.missingBuffers
                }
            }
        }

        return results
    }

    private static func decodeDataURI(_ uri: String) -> Data? {
        guard let range = uri.range(of: "base64,") else { return nil }
        let base64Part = String(uri[range.upperBound...])
        return Data(base64Encoded: base64Part)
    }

    private static func readUInt32(_ data: Data, offset: Int) -> UInt32 {
        let value = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
        return UInt32(littleEndian: value)
    }

    private struct TransformBuilder {
        static func transform(from node: GltfNode) throws -> Transform {
            if let matrix = node.matrix, matrix.count == 16 {
                let m = simd_float4x4(columns: (
                    SIMD4<Float>(matrix[0], matrix[1], matrix[2], matrix[3]),
                    SIMD4<Float>(matrix[4], matrix[5], matrix[6], matrix[7]),
                    SIMD4<Float>(matrix[8], matrix[9], matrix[10], matrix[11]),
                    SIMD4<Float>(matrix[12], matrix[13], matrix[14], matrix[15])
                ))
                return try decomposeUniformTransform(matrix: m)
            }

            let translation = SIMD3<Float>(node.translation?.safe(0) ?? 0,
                                            node.translation?.safe(1) ?? 0,
                                            node.translation?.safe(2) ?? 0)
            let rotationVec = SIMD4<Float>(node.rotation?.safe(0) ?? 0,
                                            node.rotation?.safe(1) ?? 0,
                                            node.rotation?.safe(2) ?? 0,
                                            node.rotation?.safe(3) ?? 1)
            let scaleVec = SIMD3<Float>(node.scale?.safe(0) ?? 1,
                                         node.scale?.safe(1) ?? 1,
                                         node.scale?.safe(2) ?? 1)

            guard isUniform(scaleVec) else { throw Error.nonUniformNodeScale }
            guard scaleVec.x > 0 else { throw Error.negativeOrZeroScale }

            let rotation = simd_quatf(ix: rotationVec.x, iy: rotationVec.y, iz: rotationVec.z, r: rotationVec.w).normalized
            return Transform(translation: translation, rotation: rotation, scale: scaleVec.x)
        }

        private static func decomposeUniformTransform(matrix: simd_float4x4) throws -> Transform {
            let col0 = SIMD3<Float>(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z)
            let col1 = SIMD3<Float>(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z)
            let col2 = SIMD3<Float>(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z)

            let scaleX = simd_length(col0)
            let scaleY = simd_length(col1)
            let scaleZ = simd_length(col2)

            let scaleVec = SIMD3<Float>(scaleX, scaleY, scaleZ)
            guard isUniform(scaleVec) else { throw Error.nonUniformNodeScale }
            guard scaleX > 0 else { throw Error.negativeOrZeroScale }

            let rotationMatrix = simd_float3x3(columns: (
                col0 / scaleX,
                col1 / scaleX,
                col2 / scaleX
            ))
            let rotation = simd_quatf(rotationMatrix).normalized
            let translation = SIMD3<Float>(matrix.columns.3.x, matrix.columns.3.y, matrix.columns.3.z)
            return Transform(translation: translation, rotation: rotation, scale: scaleX)
        }

        private static func isUniform(_ scale: SIMD3<Float>) -> Bool {
            let epsilon: Float = 1e-4
            return abs(scale.x - scale.y) <= epsilon && abs(scale.x - scale.z) <= epsilon
        }
    }

    private struct AccessorReader {
        let accessors: [GltfAccessor]
        let bufferViews: [GltfBufferView]
        let buffers: [Data]

        func readVec3(_ index: Int) throws -> [SIMD3<Float>] {
            let values = try readAccessor(index: index, expectedComponents: 3)
            return values.map { SIMD3<Float>($0[0], $0[1], $0[2]) }
        }

        func readVec4(_ index: Int) throws -> [SIMD4<Float>] {
            let values = try readAccessor(index: index, expectedComponents: 4)
            return values.map { SIMD4<Float>($0[0], $0[1], $0[2], $0[3]) }
        }

        func readScalar(_ index: Int) throws -> [Float] {
            let values = try readAccessor(index: index, expectedComponents: 1)
            return values.map { $0[0] }
        }

        func readColor(_ index: Int) throws -> [SIMD4<Float>] {
            let accessor = try getAccessor(index)
            let componentCount = components(for: accessor.type)
            guard componentCount == 3 || componentCount == 4 else { throw Error.unsupportedAccessorType(accessor.type) }
            let values = try readAccessor(index: index, expectedComponents: componentCount)
            if componentCount == 3 {
                return values.map { SIMD4<Float>($0[0], $0[1], $0[2], 1) }
            }
            return values.map { SIMD4<Float>($0[0], $0[1], $0[2], $0[3]) }
        }

        private func readAccessor(index: Int, expectedComponents: Int) throws -> [[Float]] {
            let accessor = try getAccessor(index)
            if accessor.sparse != nil { throw Error.sparseAccessorsNotSupported }

            let componentCount = components(for: accessor.type)
            guard componentCount > 0 else { throw Error.unsupportedAccessorType(accessor.type) }
            guard componentCount == expectedComponents else { throw Error.unsupportedAccessorType(accessor.type) }

            guard let bufferViewIndex = accessor.bufferView else { throw Error.missingBufferView(index) }
            guard bufferViewIndex >= 0 && bufferViewIndex < bufferViews.count else { throw Error.missingBufferView(bufferViewIndex) }
            let bufferView = bufferViews[bufferViewIndex]

            guard bufferView.buffer >= 0 && bufferView.buffer < buffers.count else { throw Error.missingBuffers }
            let buffer = buffers[bufferView.buffer]

            let componentSize = componentSizeBytes(for: accessor.componentType)
            guard componentSize > 0 else { throw Error.unsupportedComponentType(accessor.componentType) }
            let stride = bufferView.byteStride ?? (componentSize * componentCount)
            if stride < componentSize * componentCount { throw Error.bufferOutOfBounds }

            let baseOffset = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0)
            let totalSize = baseOffset + stride * accessor.count
            if totalSize > buffer.count { throw Error.bufferOutOfBounds }

            var result: [[Float]] = []
            result.reserveCapacity(accessor.count)

            for i in 0..<accessor.count {
                var entry: [Float] = []
                entry.reserveCapacity(componentCount)
                let elementOffset = baseOffset + i * stride
                for c in 0..<componentCount {
                    let offset = elementOffset + c * componentSize
                    let value = try readComponentAsFloat(buffer: buffer, offset: offset, componentType: accessor.componentType, normalized: accessor.normalized ?? false)
                    entry.append(value)
                }
                result.append(entry)
            }

            return result
        }

        private func getAccessor(_ index: Int) throws -> GltfAccessor {
            guard index >= 0 && index < accessors.count else { throw Error.missingAccessor(index) }
            return accessors[index]
        }

        private func components(for type: String) -> Int {
            switch type {
            case "SCALAR": return 1
            case "VEC2": return 2
            case "VEC3": return 3
            case "VEC4": return 4
            default: return 0
            }
        }

        private func componentSizeBytes(for componentType: Int) -> Int {
            switch componentType {
            case 5120, 5121: return 1
            case 5122, 5123: return 2
            case 5125, 5126: return 4
            default: return 0
            }
        }

        private func readComponentAsFloat(buffer: Data, offset: Int, componentType: Int, normalized: Bool) throws -> Float {
            switch componentType {
            case 5120:
                guard offset + 1 <= buffer.count else { throw Error.bufferOutOfBounds }
                let value = Int8(bitPattern: buffer[offset])
                if normalized { return normalizeSigned(Float(value), maxValue: 127) }
                return Float(value)
            case 5121:
                guard offset + 1 <= buffer.count else { throw Error.bufferOutOfBounds }
                let value = buffer[offset]
                if normalized { return Float(value) / 255.0 }
                return Float(value)
            case 5122:
                let value = try readInt16(buffer, offset: offset)
                if normalized { return normalizeSigned(Float(value), maxValue: 32767) }
                return Float(value)
            case 5123:
                let value = try readUInt16(buffer, offset: offset)
                if normalized { return Float(value) / 65535.0 }
                return Float(value)
            case 5125:
                let value = try readUInt32(buffer, offset: offset)
                if normalized { return Float(value) / 4294967295.0 }
                return Float(value)
            case 5126:
                return try readFloat32(buffer, offset: offset)
            default:
                throw Error.unsupportedComponentType(componentType)
            }
        }

        private func normalizeSigned(_ value: Float, maxValue: Float) -> Float {
            let scaled = value / maxValue
            return Swift.max(-1, Swift.min(1, scaled))
        }

        private func readUInt16(_ buffer: Data, offset: Int) throws -> UInt16 {
            guard offset + 2 <= buffer.count else { throw Error.bufferOutOfBounds }
            let value = buffer.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt16.self) }
            return UInt16(littleEndian: value)
        }

        private func readInt16(_ buffer: Data, offset: Int) throws -> Int16 {
            guard offset + 2 <= buffer.count else { throw Error.bufferOutOfBounds }
            let value = buffer.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Int16.self) }
            return Int16(littleEndian: value)
        }

        private func readUInt32(_ buffer: Data, offset: Int) throws -> UInt32 {
            guard offset + 4 <= buffer.count else { throw Error.bufferOutOfBounds }
            let value = buffer.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            return UInt32(littleEndian: value)
        }

        private func readFloat32(_ buffer: Data, offset: Int) throws -> Float {
            let bits = try readUInt32(buffer, offset: offset)
            return Float(bitPattern: bits)
        }
    }
}

private extension Array where Element == Float {
    func safe(_ index: Int) -> Float {
        if index >= 0 && index < count { return self[index] }
        return 0
    }
}
