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
        /// No longer thrown: non-uniform node scale is applied to each splat's covariance.
        case nonUniformNodeScale
        /// No longer thrown: a singular node transform hides its meshes, and a mirroring one is applied.
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
        var extensions: GltfGaussianSplattingNestedExtensions?
    }

    private struct GltfGaussianSplattingNestedExtensions: Codable {
        var spzCompression: GltfSPZCompression?

        enum CodingKeys: String, CodingKey {
            case spzCompression = "KHR_gaussian_splatting_compression_spz_2"
        }
    }

    private struct GltfSPZCompression: Codable {
        var bufferView: Int?
    }

    private struct GltfNode: Codable {
        var mesh: Int?
        var children: [Int]?
        var matrix: [Double]?
        var translation: [Double]?
        var rotation: [Double]?
        var scale: [Double]?
    }

    private struct GltfScene: Codable {
        var nodes: [Int]?
    }

    /// Values the typed model above leaves out: free-form `extras` and the generator.
    private struct FileHints {
        /// Root `extras.SplatRenderMode`, which Plinth writes ("mip" for antialiased scenes).
        var renderMode: String?
        var generator: String?
        /// `asset.extras.splatkitEncoding` ("linear" or "gsbox"), written by SplatKit and Plinth.
        var encodingMarker: String?

        init(json: Data) {
            guard let root = (try? JSONSerialization.jsonObject(with: json)) as? [String: Any] else { return }
            renderMode = (root["extras"] as? [String: Any])?["SplatRenderMode"] as? String
            let asset = root["asset"] as? [String: Any]
            generator = asset?["generator"] as? String
            encodingMarker = (asset?["extras"] as? [String: Any])?["splatkitEncoding"] as? String
        }
    }

    private let url: URL
    private let root: GltfRoot
    private let buffers: [Data]
    private let hints: FileHints

    /// Whether the scene should be rendered with mip (antialiased) splatting: the root
    /// `extras.SplatRenderMode`, or else the antialiased flag of an SPZ-compressed primitive.
    public let isAntialiased: Bool

    public init(_ url: URL) throws {
        self.url = url
        let (root, buffers, hints) = try GltfGaussianSplatSceneReader.loadRootAndBuffers(from: url)
        self.root = root
        self.buffers = buffers
        self.hints = hints
        switch hints.renderMode?.lowercased() {
        case "mip", "antialiased":
            isAntialiased = true
        case "default", "standard", "2dgs":
            isAntialiased = false
        default:
            isAntialiased = Self.embeddedSPZIsAntialiased(root: root, buffers: buffers)
        }
    }

    public func readScene() throws -> [SplatScenePoint] {
        guard let meshes = root.meshes else {
            return []
        }

        let accessorReader = AccessorReader(accessors: root.accessors ?? [], bufferViews: root.bufferViews ?? [], buffers: buffers)
        var results: [SplatScenePoint] = []

        if let nodes = root.nodes, !nodes.isEmpty {
            try traverse(roots: sceneRoots(nodes: nodes), nodes: nodes, meshes: meshes, accessorReader: accessorReader, results: &results)
        } else {
            for meshIndex in meshes.indices {
                try appendMesh(meshIndex: meshIndex, transform: matrix_identity_double4x4, meshes: meshes, accessorReader: accessorReader, results: &results)
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

    /// The default scene's root nodes, or every node that isn't another node's child.
    private func sceneRoots(nodes: [GltfNode]) -> [Int] {
        if let scenes = root.scenes {
            let sceneIndex = root.scene ?? 0
            if scenes.indices.contains(sceneIndex), let sceneNodes = scenes[sceneIndex].nodes, !sceneNodes.isEmpty {
                return sceneNodes
            }
        }
        let children = Set(nodes.flatMap { $0.children ?? [] })
        return nodes.indices.filter { !children.contains($0) }
    }

    /// Depth-first over the node hierarchy, without recursion. Each node is visited once, so a
    /// node listed as its own descendant (invalid glTF) can't loop or overflow the stack.
    private func traverse(roots: [Int],
                          nodes: [GltfNode],
                          meshes: [GltfMesh],
                          accessorReader: AccessorReader,
                          results: inout [SplatScenePoint]) throws {
        var pending: [(node: Int, parent: simd_double4x4)] = roots.reversed().map { ($0, matrix_identity_double4x4) }
        var visited = Set<Int>()
        while let (nodeIndex, parentTransform) = pending.popLast() {
            guard nodes.indices.contains(nodeIndex), visited.insert(nodeIndex).inserted else { continue }
            let node = nodes[nodeIndex]
            let transform = parentTransform * Self.localTransform(of: node)

            if let meshIndex = node.mesh {
                try appendMesh(meshIndex: meshIndex, transform: transform, meshes: meshes, accessorReader: accessorReader, results: &results)
            }

            for child in (node.children ?? []).reversed() {
                pending.append((child, transform))
            }
        }
    }

    private static func localTransform(of node: GltfNode) -> simd_double4x4 {
        if let m = node.matrix, m.count == 16 {
            // glTF matrices are column-major.
            return simd_double4x4(columns: (SIMD4(m[0], m[1], m[2], m[3]),
                                            SIMD4(m[4], m[5], m[6], m[7]),
                                            SIMD4(m[8], m[9], m[10], m[11]),
                                            SIMD4(m[12], m[13], m[14], m[15])))
        }

        let t = node.translation ?? []
        let r = node.rotation ?? []
        let s = node.scale ?? []
        let translation = SIMD3(t.safe(0, default: 0), t.safe(1, default: 0), t.safe(2, default: 0))
        let rotation = simd_quatd(ix: r.safe(0, default: 0), iy: r.safe(1, default: 0), iz: r.safe(2, default: 0), r: r.safe(3, default: 1))
        let scale = SIMD3(s.safe(0, default: 1), s.safe(1, default: 1), s.safe(2, default: 1))

        let linear = simd_double3x3(rotation.normalized) * simd_double3x3(diagonal: scale)
        return simd_double4x4(columns: (SIMD4(linear.columns.0, 0),
                                        SIMD4(linear.columns.1, 0),
                                        SIMD4(linear.columns.2, 0),
                                        SIMD4(translation, 1)))
    }

    private func appendMesh(meshIndex: Int,
                            transform: simd_double4x4,
                            meshes: [GltfMesh],
                            accessorReader: AccessorReader,
                            results: inout [SplatScenePoint]) throws {
        guard meshIndex >= 0 && meshIndex < meshes.count else { return }
        // A singular transform (zero scale) collapses the mesh to nothing, which is how glTF hides nodes.
        guard let pointTransform = PointTransform(transform) else { return }
        let mesh = meshes[meshIndex]

        for primitive in mesh.primitives {
            let points = try readPrimitive(primitive, accessorReader: accessorReader)
            if pointTransform.isIdentity {
                results.append(contentsOf: points)
            } else {
                results.append(contentsOf: points.map(pointTransform.apply))
            }
        }
    }

    private func readPrimitive(_ primitive: GltfPrimitive, accessorReader: AccessorReader) throws -> [SplatScenePoint] {
        let mode = primitive.mode ?? 4
        guard mode == 0 else { return [] }

        guard let ext = primitive.extensions?.khrGaussianSplatting else { return [] }
        if let kernel = ext.kernel, kernel != "ellipse" { return [] }

        if let spzBufferView = ext.extensions?.spzCompression?.bufferView {
            return try readEmbeddedSPZ(bufferView: spzBufferView, accessorReader: accessorReader)
        }

        guard let positionAccessor = primitive.attributes["POSITION"] else { return [] }
        guard let rotationAccessor = primitive.attributes["KHR_gaussian_splatting:ROTATION"] else { return [] }
        guard let scaleAccessor = primitive.attributes["KHR_gaussian_splatting:SCALE"] else { return [] }
        guard let opacityAccessor = primitive.attributes["KHR_gaussian_splatting:OPACITY"] else { return [] }

        let positions = try accessorReader.readVec3(positionAccessor)
        let rotations = try accessorReader.readVec4(rotationAccessor)
        let scales = try accessorReader.readVec3(scaleAccessor)
        let opacities = try accessorReader.readScalar(opacityAccessor)

        let count = positions.count
        if rotations.count != count || scales.count != count || opacities.count != count {
            return []
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
            guard sh1Accessors.count == 3 else { return [] }
            for index in sh1Accessors {
                let values = try accessorReader.readVec3(index)
                if values.count != count { shDataValid = false; break }
                sh1Values.append(values)
            }
        }

        if sh2Accessors.count > 0 {
            guard sh2Accessors.count == 5 else { return [] }
            for index in sh2Accessors {
                let values = try accessorReader.readVec3(index)
                if values.count != count { shDataValid = false; break }
                sh2Values.append(values)
            }
        }

        if sh3Accessors.count > 0 {
            guard sh3Accessors.count == 7 else { return [] }
            for index in sh3Accessors {
                let values = try accessorReader.readVec3(index)
                if values.count != count { shDataValid = false; break }
                sh3Values.append(values)
            }
        }

        if !shDataValid { return [] }
        if sh0Values == nil && (!sh1Values.isEmpty || !sh2Values.isEmpty || !sh3Values.isEmpty) { return [] }

        if let color0Accessor {
            color0Values = try accessorReader.readColor(color0Accessor)
            if color0Values?.count != count { color0Values = nil }
        }

        let encoding = try valueEncoding(scaleAccessor: scaleAccessor, scales: scales,
                                         opacityAccessor: opacityAccessor, opacities: opacities,
                                         accessorReader: accessorReader)

        var results: [SplatScenePoint] = []
        results.reserveCapacity(count)
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

            let opacity: SplatScenePoint.Opacity = encoding.logitOpacity
                ? .logitFloat(opacities[i])
                : .linearFloat(max(0, min(1, opacities[i])))
            // Signed normalized accessors can hold negative values; a linear scale can't be below zero.
            let scale: SplatScenePoint.Scale = encoding.logScale
                ? .exponent(scales[i])
                : .linearFloat(simd_max(scales[i], .zero))

            let rotationVector = rotations[i]
            let rotation = simd_quatf(ix: rotationVector.x, iy: rotationVector.y, iz: rotationVector.z, r: rotationVector.w).normalized

            results.append(SplatScenePoint(position: positions[i],
                                           color: color,
                                           opacity: opacity,
                                           scale: scale,
                                           rotation: rotation))
        }
        return results
    }

    /// Whether SCALE holds log-space values and OPACITY logits (gsbox, and writers from before
    /// `KHR_gaussian_splatting` was ratified) instead of the spec's linear values. Quantized
    /// accessors are linear by definition. Otherwise an explicit marker decides, then gsbox's
    /// generator, then the values: a conforming file can't hold a negative scale or an opacity
    /// outside 0…1. The two are decided separately because some writers (earlier versions of
    /// this package among them) paired log-space scale with linear opacity.
    private func valueEncoding(scaleAccessor: Int,
                               scales: [SIMD3<Float>],
                               opacityAccessor: Int,
                               opacities: [Float],
                               accessorReader: AccessorReader) throws -> (logScale: Bool, logitOpacity: Bool) {
        let floatScale = try accessorReader.componentType(scaleAccessor) == AccessorReader.floatComponentType
        let floatOpacity = try accessorReader.componentType(opacityAccessor) == AccessorReader.floatComponentType

        let legacy: (scale: Bool, opacity: Bool)
        switch hints.encodingMarker?.lowercased() {
        case "gsbox":
            legacy = (true, true)
        case "linear":
            legacy = (false, false)
        default:
            if hints.generator?.lowercased().hasPrefix("gsbox") == true {
                legacy = (true, true)
            } else {
                legacy = (scales.contains { $0.min() < 0 }, opacities.contains { $0 < 0 || $0 > 1 })
            }
        }
        return (legacy.scale && floatScale, legacy.opacity && floatOpacity)
    }

    /// `KHR_gaussian_splatting_compression_spz_2`: the primitive's splats are an SPZ stream in a
    /// buffer view, read as a standalone SPZ file would be.
    private func readEmbeddedSPZ(bufferView: Int, accessorReader: AccessorReader) throws -> [SplatScenePoint] {
        let data = try accessorReader.bufferViewData(bufferView)
        return try SPZSceneReader(data: data).readScene()
    }

    private static func embeddedSPZIsAntialiased(root: GltfRoot, buffers: [Data]) -> Bool {
        let reader = AccessorReader(accessors: root.accessors ?? [], bufferViews: root.bufferViews ?? [], buffers: buffers)
        for mesh in root.meshes ?? [] {
            for primitive in mesh.primitives {
                guard let bufferView = primitive.extensions?.khrGaussianSplatting?.extensions?.spzCompression?.bufferView,
                      let data = try? reader.bufferViewData(bufferView) else { continue }
                return SPZSceneReader.headerIsAntialiased(data)
            }
        }
        return false
    }

    private static func loadRootAndBuffers(from url: URL) throws -> (GltfRoot, [Data], FileHints) {
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
        return (root, buffers, FileHints(json: jsonData))
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
        precondition(offset >= 0 && offset + MemoryLayout<UInt32>.size <= data.count)

        let start = data.startIndex + offset
        let end = start + MemoryLayout<UInt32>.size
        return data[start..<end].enumerated().reduce(into: UInt32(0)) { value, element in
            value |= UInt32(element.element) << UInt32(element.offset * 8)
        }
    }

    /// A node's world transform as applied to splats. A similarity (rotation and uniform positive
    /// scale) moves each splat directly; any other linear part — non-uniform scale, shear, a
    /// mirror — transforms each splat's covariance, which is re-decomposed into scale and rotation.
    /// Spherical harmonics follow the rotation part of the transform's polar decomposition.
    private struct PointTransform {
        let linear: simd_double3x3
        let translation: SIMD3<Double>
        /// Uniform scale and rotation when `linear` is a similarity.
        let similarity: (scale: Double, rotation: simd_quatd)?
        let shRotation: SphericalHarmonicRotation
        /// Whether the transform mirrors, which negates the odd SH bands.
        let mirrors: Bool
        let isIdentity: Bool

        /// Nil for a singular transform.
        init?(_ matrix: simd_double4x4) {
            let c = matrix.columns
            linear = simd_double3x3(columns: (SIMD3(c.0.x, c.0.y, c.0.z),
                                              SIMD3(c.1.x, c.1.y, c.1.z),
                                              SIMD3(c.2.x, c.2.y, c.2.z)))
            translation = SIMD3(c.3.x, c.3.y, c.3.z)
            let determinant = linear.determinant
            let gram = linear.transpose * linear
            let (values, vectors) = symmetricEigendecomposition(gram)
            // Singular relative to its own scale, so a scene in tiny units still counts as regular.
            guard determinant.isFinite, determinant != 0, values.min() > values.max() * 1e-24 else { return nil }
            isIdentity = linear == matrix_identity_double3x3 && translation == .zero

            let squaredScale = (gram[0][0] + gram[1][1] + gram[2][2]) / 3
            let tolerance = 1e-6 * squaredScale
            let isSimilarity = determinant > 0 && (0..<3).allSatisfy { column in
                (0..<3).allSatisfy { row in
                    abs(gram[column][row] - (row == column ? squaredScale : 0)) <= tolerance
                }
            }

            // Polar decomposition linear = U·P, with U = linear·(linearᵀ·linear)^(-1/2).
            let inverseRoot = simd_double3x3(diagonal: SIMD3(1 / values.x.squareRoot(), 1 / values.y.squareRoot(), 1 / values.z.squareRoot()))
            var orthogonal = linear * vectors * inverseRoot * vectors.transpose
            mirrors = determinant < 0
            if mirrors {
                // U = −R with R a proper rotation; the −I part is the inversion handled by `mirrors`.
                orthogonal = -1 * orthogonal
            }
            let rotation = simd_quatd(orthogonal).normalized
            shRotation = SphericalHarmonicRotation(rotation: simd_quatf(rotation))

            if isSimilarity {
                let scale = squaredScale.squareRoot()
                similarity = (scale, rotation)
            } else {
                similarity = nil
            }
        }

        func apply(_ point: SplatScenePoint) -> SplatScenePoint {
            var result = point
            result.position = SIMD3<Float>(linear * SIMD3<Double>(point.position) + translation)

            if let similarity {
                result.rotation = simd_quatf((similarity.rotation * simd_quatd(point.rotation)).normalized)
                switch point.scale {
                case .exponent(let logScale):
                    result.scale = .exponent(logScale + Float(log(similarity.scale)))
                case .linearFloat(let scale):
                    result.scale = .linearFloat(scale * Float(similarity.scale))
                }
            } else {
                // Σ' = A·R·S²·Rᵀ·Aᵀ = M·Mᵀ with M = A·R·S.
                let axes = linear * simd_double3x3(simd_quatd(point.rotation).normalized)
                    * simd_double3x3(diagonal: SIMD3<Double>(point.scale.asLinearFloat))
                let (values, vectors) = symmetricEigendecomposition(axes * axes.transpose)
                result.scale = .linearFloat(SIMD3<Float>(simd_max(values, .zero).squareRoot()))
                result.rotation = simd_quatf(simd_quatd(vectors).normalized)
            }

            result.color = transformedColor(point.color)
            return result
        }

        private func transformedColor(_ color: SplatScenePoint.Color) -> SplatScenePoint.Color {
            let rotated = shRotation.applying(to: color)
            guard mirrors, case var .sphericalHarmonic(coefficients) = rotated else { return rotated }
            // Inversion d → −d flips the sign of odd-degree terms (coefficients 1–3 and 9–15).
            for index in coefficients.indices {
                let degree = Int(Double(index).squareRoot())
                if degree % 2 == 1 { coefficients[index] = -coefficients[index] }
            }
            return .sphericalHarmonic(coefficients)
        }
    }

    private struct AccessorReader {
        static let floatComponentType = 5126

        let accessors: [GltfAccessor]
        let bufferViews: [GltfBufferView]
        let buffers: [Data]

        func componentType(_ index: Int) throws -> Int {
            try getAccessor(index).componentType
        }

        func bufferViewData(_ index: Int) throws -> Data {
            guard index >= 0 && index < bufferViews.count else { throw Error.missingBufferView(index) }
            let bufferView = bufferViews[index]
            guard bufferView.buffer >= 0 && bufferView.buffer < buffers.count else { throw Error.missingBuffers }
            let buffer = buffers[bufferView.buffer]
            let offset = bufferView.byteOffset ?? 0
            guard offset >= 0, bufferView.byteLength >= 0,
                  bufferView.byteLength <= buffer.count, offset <= buffer.count - bufferView.byteLength else {
                throw Error.bufferOutOfBounds
            }
            let start = buffer.startIndex + offset
            return buffer.subdata(in: start..<(start + bufferView.byteLength))
        }

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
            guard accessor.count >= 0 else { throw Error.bufferOutOfBounds }
            let elementSize = componentSize * componentCount
            let stride = bufferView.byteStride ?? (componentSize * componentCount)
            if stride < elementSize { throw Error.bufferOutOfBounds }

            let viewOffset = bufferView.byteOffset ?? 0
            let accessorOffset = accessor.byteOffset ?? 0
            guard viewOffset >= 0, accessorOffset >= 0, bufferView.byteLength >= 0 else {
                throw Error.bufferOutOfBounds
            }

            let (baseOffset, baseOffsetOverflow) = viewOffset.addingReportingOverflow(accessorOffset)
            let (viewEnd, viewEndOverflow) = viewOffset.addingReportingOverflow(bufferView.byteLength)
            guard !baseOffsetOverflow, !viewEndOverflow, viewEnd <= buffer.count else {
                throw Error.bufferOutOfBounds
            }

            let requiredSize: Int
            if accessor.count == 0 {
                requiredSize = baseOffset
            } else {
                let (strideSpan, strideOverflow) = stride.multipliedReportingOverflow(by: accessor.count - 1)
                let (lastElementOffset, offsetOverflow) = baseOffset.addingReportingOverflow(strideSpan)
                let (lastElementEnd, endOverflow) = lastElementOffset.addingReportingOverflow(elementSize)
                guard !strideOverflow, !offsetOverflow, !endOverflow else { throw Error.bufferOutOfBounds }
                requiredSize = lastElementEnd
            }
            if requiredSize > viewEnd { throw Error.bufferOutOfBounds }

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
            let start = buffer.startIndex + offset
            return UInt16(buffer[start])
                | (UInt16(buffer[start + 1]) << 8)
        }

        private func readInt16(_ buffer: Data, offset: Int) throws -> Int16 {
            let value = try readUInt16(buffer, offset: offset)
            return Int16(bitPattern: value)
        }

        private func readUInt32(_ buffer: Data, offset: Int) throws -> UInt32 {
            guard offset + 4 <= buffer.count else { throw Error.bufferOutOfBounds }
            let start = buffer.startIndex + offset
            return UInt32(buffer[start])
                | (UInt32(buffer[start + 1]) << 8)
                | (UInt32(buffer[start + 2]) << 16)
                | (UInt32(buffer[start + 3]) << 24)
        }

        private func readFloat32(_ buffer: Data, offset: Int) throws -> Float {
            let bits = try readUInt32(buffer, offset: offset)
            return Float(bitPattern: bits)
        }
    }
}

private extension Array where Element == Double {
    func safe(_ index: Int, default defaultValue: Double) -> Double {
        if index >= 0 && index < count { return self[index] }
        return defaultValue
    }
}

private extension simd_quatd {
    init(_ q: simd_quatf) {
        self.init(ix: Double(q.imag.x), iy: Double(q.imag.y), iz: Double(q.imag.z), r: Double(q.real))
    }
}

private extension simd_quatf {
    init(_ q: simd_quatd) {
        self.init(ix: Float(q.imag.x), iy: Float(q.imag.y), iz: Float(q.imag.z), r: Float(q.real))
    }
}

/// Eigenvalues and orthonormal eigenvectors (as columns, forming a proper rotation) of a
/// symmetric 3×3 matrix, by cyclic Jacobi rotations.
func symmetricEigendecomposition(_ matrix: simd_double3x3) -> (values: SIMD3<Double>, vectors: simd_double3x3) {
    var a = matrix
    var vectors = matrix_identity_double3x3
    let scale = max(abs(a[0][0]), abs(a[1][1]), abs(a[2][2]), abs(a[1][0]), abs(a[2][0]), abs(a[2][1]))
    if scale > 0 {
        for _ in 0..<32 {
            let offDiagonal = abs(a[1][0]) + abs(a[2][0]) + abs(a[2][1])
            if offDiagonal <= 1e-15 * scale { break }
            for (p, q) in [(0, 1), (0, 2), (1, 2)] {
                let apq = a[q][p]
                if apq == 0 { continue }
                // Rotation J in the (p, q) plane chosen so that (Jᵀ·A·J)[p][q] = 0.
                let theta = (a[q][q] - a[p][p]) / (2 * apq)
                let t = (theta >= 0 ? 1.0 : -1.0) / (abs(theta) + (theta * theta + 1).squareRoot())
                let c = 1 / (t * t + 1).squareRoot()
                let s = t * c
                var rotation = matrix_identity_double3x3
                rotation[p][p] = c
                rotation[q][q] = c
                rotation[q][p] = s
                rotation[p][q] = -s
                a = rotation.transpose * a * rotation
                vectors = vectors * rotation
            }
        }
    }
    if vectors.determinant < 0 {
        vectors.columns.2 = -vectors.columns.2
    }
    return (SIMD3(a[0][0], a[1][1], a[2][2]), vectors)
}
