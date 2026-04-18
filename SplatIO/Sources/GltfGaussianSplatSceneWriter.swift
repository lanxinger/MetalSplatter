import Foundation
import simd

public final class GltfGaussianSplatSceneWriter: SplatSceneWriter {
    public enum Container: Sendable {
        case gltf
        case glb
    }

    public enum Error: Swift.Error {
        case cannotDetermineContainer
        case unsupportedFileExtension(String)
        case failedToEncodeJSON
    }

    private struct Root: Encodable {
        struct Asset: Encodable {
            let version = "2.0"
            let generator = "MetalSplatter"
        }

        struct Buffer: Encodable {
            let byteLength: Int
            let uri: String?
        }

        struct BufferView: Encodable {
            let buffer = 0
            let byteOffset: Int
            let byteLength: Int
        }

        struct Accessor: Encodable {
            let bufferView: Int
            let byteOffset = 0
            let componentType = 5126
            let count: Int
            let type: String
        }

        struct PrimitiveExtensions: Encodable {
            struct GaussianSplatting: Encodable {
                let kernel = "ellipse"
                let colorSpace = "linear"
            }

            let gaussianSplatting = GaussianSplatting()

            enum CodingKeys: String, CodingKey {
                case gaussianSplatting = "KHR_gaussian_splatting"
            }
        }

        struct Primitive: Encodable {
            let attributes: [String: Int]
            let mode = 0
            let extensions = PrimitiveExtensions()
        }

        struct Mesh: Encodable {
            let primitives: [Primitive]
        }

        struct Node: Encodable {
            let mesh = 0
        }

        struct Scene: Encodable {
            let nodes = [0]
        }

        let asset = Asset()
        let extensionsUsed = ["KHR_gaussian_splatting"]
        let buffers: [Buffer]
        let bufferViews: [BufferView]
        let accessors: [Accessor]
        let meshes: [Mesh]
        let nodes = [Node()]
        let scenes = [Scene()]
        let scene = 0
    }

    private struct AttributePayload {
        let name: String
        let componentCount: Int
        let type: String
        let data: Data
    }

    private let preferredContainer: Container?
    private var outputURL: URL?
    private var points: [SplatScenePoint]?

    public init(container: Container? = nil) {
        self.preferredContainer = container
    }

    public func setOutputURL(_ url: URL) {
        outputURL = url
    }

    public func write(_ points: [SplatScenePoint]) throws {
        try SplatDataValidator.validatePoints(points)
        self.points = points
    }

    public func close() throws {
        guard let outputURL, let points else { return }
        try writeScene(points, to: outputURL)
    }

    public func writeScene(_ points: [SplatScenePoint], to url: URL) throws {
        try SplatDataValidator.validatePoints(points)

        let container = try resolveContainer(for: url)
        let payloads = makeAttributePayloads(points: points)
        let packed = try pack(payloads: payloads, binURI: container == .gltf ? url.deletingPathExtension().lastPathComponent + ".bin" : nil)

        switch container {
        case .gltf:
            try packed.jsonData.write(to: url, options: .atomic)
            if let binData = packed.binData {
                try binData.write(to: url.deletingPathExtension().appendingPathExtension("bin"), options: .atomic)
            }
        case .glb:
            let glbData = try makeGLB(jsonData: packed.jsonData, binData: packed.binData ?? Data())
            try glbData.write(to: url, options: .atomic)
        }
    }

    private func resolveContainer(for url: URL) throws -> Container {
        if let preferredContainer {
            return preferredContainer
        }

        switch url.pathExtension.lowercased() {
        case "gltf":
            return .gltf
        case "glb":
            return .glb
        case "":
            throw Error.cannotDetermineContainer
        default:
            throw Error.unsupportedFileExtension(url.pathExtension)
        }
    }

    private func makeAttributePayloads(points: [SplatScenePoint]) -> [AttributePayload] {
        var payloads: [AttributePayload] = []
        payloads.append(AttributePayload(name: "POSITION", componentCount: 3, type: "VEC3", data: encodeVec3(points.map(\.position))))
        payloads.append(AttributePayload(name: "KHR_gaussian_splatting:ROTATION", componentCount: 4, type: "VEC4", data: encodeVec4(points.map { $0.rotation.normalized.vector })))
        payloads.append(AttributePayload(name: "KHR_gaussian_splatting:SCALE", componentCount: 3, type: "VEC3", data: encodeVec3(points.map { $0.scale.asExponent })))
        payloads.append(AttributePayload(name: "KHR_gaussian_splatting:OPACITY", componentCount: 1, type: "SCALAR", data: encodeScalar(points.map { $0.opacity.asLinearFloat })))

        let pointsWithSH = points.filter {
            if case .sphericalHarmonic = $0.color { return true }
            return false
        }

        if pointsWithSH.isEmpty {
            let colors = points.map { color -> SIMD4<Float> in
                let rgb = color.color.asLinearFloat
                return SIMD4<Float>(rgb.x, rgb.y, rgb.z, 1)
            }
            payloads.append(AttributePayload(name: "COLOR_0", componentCount: 4, type: "VEC4", data: encodeVec4(colors)))
            return payloads
        }

        let maxCoefficients = normalizedSHCoefficientCount(for: pointsWithSH.map { $0.color.asSphericalHarmonic.count }.max() ?? 1)
        let shCoefficients = points.map { point -> [SIMD3<Float>] in
            let coeffs = point.color.asSphericalHarmonic
            if coeffs.count >= maxCoefficients {
                return Array(coeffs.prefix(maxCoefficients))
            }
            return coeffs + Array(repeating: .zero, count: maxCoefficients - coeffs.count)
        }

        payloads.append(AttributePayload(name: "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0",
                                         componentCount: 3,
                                         type: "VEC3",
                                         data: encodeVec3(shCoefficients.map { $0[0] })))

        if maxCoefficients >= 4 {
            for coefficientIndex in 0..<3 {
                payloads.append(AttributePayload(
                    name: "KHR_gaussian_splatting:SH_DEGREE_1_COEF_\(coefficientIndex)",
                    componentCount: 3,
                    type: "VEC3",
                    data: encodeVec3(shCoefficients.map { $0[1 + coefficientIndex] })
                ))
            }
        }

        if maxCoefficients >= 9 {
            for coefficientIndex in 0..<5 {
                payloads.append(AttributePayload(
                    name: "KHR_gaussian_splatting:SH_DEGREE_2_COEF_\(coefficientIndex)",
                    componentCount: 3,
                    type: "VEC3",
                    data: encodeVec3(shCoefficients.map { $0[4 + coefficientIndex] })
                ))
            }
        }

        if maxCoefficients >= 16 {
            for coefficientIndex in 0..<7 {
                payloads.append(AttributePayload(
                    name: "KHR_gaussian_splatting:SH_DEGREE_3_COEF_\(coefficientIndex)",
                    componentCount: 3,
                    type: "VEC3",
                    data: encodeVec3(shCoefficients.map { $0[9 + coefficientIndex] })
                ))
            }
        }

        return payloads
    }

    private func normalizedSHCoefficientCount(for count: Int) -> Int {
        if count >= 16 { return 16 }
        if count >= 9 { return 9 }
        if count >= 4 { return 4 }
        return 1
    }

    private func pack(payloads: [AttributePayload], binURI: String?) throws -> (jsonData: Data, binData: Data?) {
        var binData = Data()
        var bufferViews: [Root.BufferView] = []
        var accessors: [Root.Accessor] = []
        var attributes: [String: Int] = [:]

        for payload in payloads {
            let alignedOffset = align4(binData.count)
            if alignedOffset > binData.count {
                binData.append(Data(repeating: 0, count: alignedOffset - binData.count))
            }

            let bufferViewIndex = bufferViews.count
            bufferViews.append(Root.BufferView(byteOffset: alignedOffset, byteLength: payload.data.count))
            binData.append(payload.data)

            let accessorIndex = accessors.count
            accessors.append(Root.Accessor(bufferView: bufferViewIndex,
                                           count: payload.data.count / (payload.componentCount * MemoryLayout<Float>.size),
                                           type: payload.type))
            attributes[payload.name] = accessorIndex
        }

        let root = Root(
            buffers: [Root.Buffer(byteLength: binData.count, uri: binURI)],
            bufferViews: bufferViews,
            accessors: accessors,
            meshes: [Root.Mesh(primitives: [Root.Primitive(attributes: attributes)])]
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let jsonData = try encoder.encode(root)
        return (jsonData: jsonData, binData: binData)
    }

    private func makeGLB(jsonData: Data, binData: Data) throws -> Data {
        guard !jsonData.isEmpty else {
            throw Error.failedToEncodeJSON
        }

        let paddedJSON = padTo4Bytes(jsonData, byte: 0x20)
        let paddedBIN = padTo4Bytes(binData, byte: 0)

        var data = Data()
        data.reserveCapacity(12 + 8 + paddedJSON.count + 8 + paddedBIN.count)

        data.append(contentsOf: [0x67, 0x6C, 0x54, 0x46]) // glTF
        data.append(contentsOf: littleEndianBytes(UInt32(2)))
        data.append(contentsOf: [0, 0, 0, 0]) // Patched below with final length.

        data.append(contentsOf: littleEndianBytes(UInt32(paddedJSON.count)))
        data.append(contentsOf: littleEndianBytes(UInt32(0x4E4F534A)))
        data.append(contentsOf: paddedJSON)

        data.append(contentsOf: littleEndianBytes(UInt32(paddedBIN.count)))
        data.append(contentsOf: littleEndianBytes(UInt32(0x004E4942)))
        data.append(contentsOf: paddedBIN)

        let totalLength = UInt32(data.count)
        data.replaceSubrange(8..<12, with: littleEndianBytes(totalLength))
        return data
    }

    private func align4(_ value: Int) -> Int {
        (value + 3) & ~3
    }

    private func padTo4Bytes(_ data: Data, byte: UInt8) -> Data {
        let padding = align4(data.count) - data.count
        guard padding > 0 else { return data }
        return data + Data(repeating: byte, count: padding)
    }

    private func littleEndianBytes<T: FixedWidthInteger>(_ value: T) -> Data {
        var littleEndian = value.littleEndian
        return withUnsafeBytes(of: &littleEndian) { Data($0) }
    }

    private func encodeScalar(_ values: [Float]) -> Data {
        var data = Data(capacity: values.count * MemoryLayout<Float>.size)
        for value in values {
            data.append(floatBytes(value))
        }
        return data
    }

    private func encodeVec3(_ values: [SIMD3<Float>]) -> Data {
        var data = Data(capacity: values.count * 3 * MemoryLayout<Float>.size)
        for value in values {
            data.append(floatBytes(value.x))
            data.append(floatBytes(value.y))
            data.append(floatBytes(value.z))
        }
        return data
    }

    private func encodeVec4(_ values: [SIMD4<Float>]) -> Data {
        var data = Data(capacity: values.count * 4 * MemoryLayout<Float>.size)
        for value in values {
            data.append(floatBytes(value.x))
            data.append(floatBytes(value.y))
            data.append(floatBytes(value.z))
            data.append(floatBytes(value.w))
        }
        return data
    }

    private func floatBytes(_ value: Float) -> Data {
        var littleEndian = value.bitPattern.littleEndian
        return withUnsafeBytes(of: &littleEndian) { Data($0) }
    }
}
