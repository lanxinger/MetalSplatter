import Foundation
import ZIPFoundation
import simd

public final class SOGSV2SceneWriter: SplatSceneWriter {
    public enum Error: Swift.Error {
        case cannotDetermineOutputFormat
        case unsupportedOutputURL(URL)
        case tooManySphericalHarmonicPalettes(Int)
    }

    private struct Asset: Encodable {
        let generator = "MetalSplatter"
    }

    private struct Document: Encodable {
        let version = 2
        let asset = Asset()
        let count: Int
        let antialias: Bool?
        let means: SOGSMeansInfoV2
        let scales: SOGSScalesInfoV2
        let quats: SOGSQuatsInfoV2
        let sh0: SOGSH0InfoV2
        let shN: SOGSSHNInfoV2?
    }

    private struct TextureDimensions {
        let width: Int
        let height: Int

        var byteCount: Int {
            width * height * 4
        }
    }

    private struct QuantizedPoint {
        let shCoefficients: [SIMD3<Float>]
        let positionEncoded: SIMD3<Float>
    }

    private struct QuantizedPaletteEntry: Hashable {
        let bytes: Data
    }

    private struct FilePayload {
        let name: String
        let data: Data
    }

    private let antialias: Bool
    private var outputURL: URL?
    private var points: [SplatScenePoint]?

    public init(antialias: Bool = false) {
        self.antialias = antialias
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

        let payloads = try buildPayloads(for: points)
        switch try resolvedOutputKind(for: url) {
        case .archive:
            try writeArchive(payloads, to: url)
        case .directory:
            try writeDirectory(payloads, metaURL: url)
        }
    }

    private enum OutputKind {
        case archive
        case directory
    }

    private func resolvedOutputKind(for url: URL) throws -> OutputKind {
        let ext = url.pathExtension.lowercased()
        if ext == "sog" {
            return .archive
        }
        if ext == "json", url.lastPathComponent.lowercased() == "meta.json" {
            return .directory
        }
        if ext.isEmpty {
            throw Error.cannotDetermineOutputFormat
        }
        throw Error.unsupportedOutputURL(url)
    }

    private func buildPayloads(for points: [SplatScenePoint]) throws -> [FilePayload] {
        let baseDimensions = Self.makeBaseTextureDimensions(pointCount: max(points.count, 1))
        let normalizedCoefficientCount = Self.normalizedSphericalHarmonicCount(
            points.map { $0.color.asSphericalHarmonic.count }.max() ?? 1
        )
        let quantizedPoints = points.map { point in
            QuantizedPoint(
                shCoefficients: Self.paddedSphericalHarmonics(point.color.asSphericalHarmonic, count: normalizedCoefficientCount),
                positionEncoded: SIMD3<Float>(
                    Self.sogEncodeLog(point.position.x),
                    Self.sogEncodeLog(point.position.y),
                    Self.sogEncodeLog(point.position.z)
                )
            )
        }

        let meansTextures = buildMeansTextures(from: quantizedPoints, dimensions: baseDimensions)
        let scalesTexture = buildScalesTexture(from: points, dimensions: baseDimensions)
        let quatsTexture = buildQuatsTexture(from: points, dimensions: baseDimensions)
        let sh0Texture = buildSH0Texture(from: points, quantizedPoints: quantizedPoints, dimensions: baseDimensions)
        let shNTextures = try buildSHNTextures(from: quantizedPoints, dimensions: baseDimensions)

        let document = Document(
            count: points.count,
            antialias: antialias ? true : nil,
            means: SOGSMeansInfoV2(mins: meansTextures.mins, maxs: meansTextures.maxs, files: ["means_l.webp", "means_u.webp"]),
            scales: SOGSScalesInfoV2(codebook: Self.scaleCodebook, mins: nil, maxs: nil, files: ["scales.webp"]),
            quats: SOGSQuatsInfoV2(files: ["quats.webp"]),
            sh0: SOGSH0InfoV2(codebook: Self.sh0Codebook, mins: nil, maxs: nil, files: ["sh0.webp"]),
            shN: shNTextures?.metadata
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let metaData = try encoder.encode(document)

        var payloads = [
            FilePayload(name: "meta.json", data: metaData),
            FilePayload(name: "means_l.webp", data: try WebPEncoder.encodeLosslessRGBA(meansTextures.lowerBytes, width: baseDimensions.width, height: baseDimensions.height)),
            FilePayload(name: "means_u.webp", data: try WebPEncoder.encodeLosslessRGBA(meansTextures.upperBytes, width: baseDimensions.width, height: baseDimensions.height)),
            FilePayload(name: "scales.webp", data: try WebPEncoder.encodeLosslessRGBA(scalesTexture, width: baseDimensions.width, height: baseDimensions.height)),
            FilePayload(name: "quats.webp", data: try WebPEncoder.encodeLosslessRGBA(quatsTexture, width: baseDimensions.width, height: baseDimensions.height)),
            FilePayload(name: "sh0.webp", data: try WebPEncoder.encodeLosslessRGBA(sh0Texture, width: baseDimensions.width, height: baseDimensions.height))
        ]

        if let shNTextures {
            payloads.append(FilePayload(
                name: "shN_centroids.webp",
                data: try WebPEncoder.encodeLosslessRGBA(
                    shNTextures.centroids,
                    width: shNTextures.centroidDimensions.width,
                    height: shNTextures.centroidDimensions.height
                )
            ))
            payloads.append(FilePayload(
                name: "shN_labels.webp",
                data: try WebPEncoder.encodeLosslessRGBA(
                    shNTextures.labels,
                    width: baseDimensions.width,
                    height: baseDimensions.height
                )
            ))
        }

        return payloads
    }

    private func buildMeansTextures(from points: [QuantizedPoint], dimensions: TextureDimensions) -> (lowerBytes: Data, upperBytes: Data, mins: [Float], maxs: [Float]) {
        let mins: [Float] = [
            points.map { $0.positionEncoded.x }.min() ?? 0,
            points.map { $0.positionEncoded.y }.min() ?? 0,
            points.map { $0.positionEncoded.z }.min() ?? 0
        ]
        let maxs: [Float] = [
            points.map { $0.positionEncoded.x }.max() ?? 0,
            points.map { $0.positionEncoded.y }.max() ?? 0,
            points.map { $0.positionEncoded.z }.max() ?? 0
        ]

        var lowerBytes = Data(repeating: 0, count: dimensions.byteCount)
        var upperBytes = Data(repeating: 0, count: dimensions.byteCount)

        for (index, point) in points.enumerated() {
            let encoded = point.positionEncoded
            let quantized = [
                Self.quantizePositionComponent(encoded.x, minValue: mins[0], maxValue: maxs[0]),
                Self.quantizePositionComponent(encoded.y, minValue: mins[1], maxValue: maxs[1]),
                Self.quantizePositionComponent(encoded.z, minValue: mins[2], maxValue: maxs[2])
            ]

            let baseOffset = index * 4
            for componentIndex in 0..<3 {
                lowerBytes[baseOffset + componentIndex] = UInt8(quantized[componentIndex] & 0xFF)
                upperBytes[baseOffset + componentIndex] = UInt8(quantized[componentIndex] >> 8)
            }
            lowerBytes[baseOffset + 3] = 255
            upperBytes[baseOffset + 3] = 255
        }

        return (lowerBytes, upperBytes, mins, maxs)
    }

    private func buildScalesTexture(from points: [SplatScenePoint], dimensions: TextureDimensions) -> Data {
        var data = Data(repeating: 0, count: dimensions.byteCount)
        for (index, point) in points.enumerated() {
            let exponent = point.scale.asExponent
            let baseOffset = index * 4
            data[baseOffset + 0] = Self.encodeScale(exponent.x)
            data[baseOffset + 1] = Self.encodeScale(exponent.y)
            data[baseOffset + 2] = Self.encodeScale(exponent.z)
            data[baseOffset + 3] = 255
        }
        return data
    }

    private func buildQuatsTexture(from points: [SplatScenePoint], dimensions: TextureDimensions) -> Data {
        var data = Data(repeating: 0, count: dimensions.byteCount)
        for (index, point) in points.enumerated() {
            let encoded = Self.encodeQuaternion(point.rotation)
            let baseOffset = index * 4
            data[baseOffset + 0] = encoded.0
            data[baseOffset + 1] = encoded.1
            data[baseOffset + 2] = encoded.2
            data[baseOffset + 3] = encoded.3
        }
        return data
    }

    private func buildSH0Texture(from points: [SplatScenePoint], quantizedPoints: [QuantizedPoint], dimensions: TextureDimensions) -> Data {
        var data = Data(repeating: 0, count: dimensions.byteCount)
        for index in points.indices {
            let sh0 = quantizedPoints[index].shCoefficients[0]
            let baseOffset = index * 4
            data[baseOffset + 0] = Self.encodeSH0(sh0.x)
            data[baseOffset + 1] = Self.encodeSH0(sh0.y)
            data[baseOffset + 2] = Self.encodeSH0(sh0.z)
            data[baseOffset + 3] = Self.encodeOpacity(points[index].opacity.asLinearFloat)
        }
        return data
    }

    private func buildSHNTextures(from points: [QuantizedPoint], dimensions: TextureDimensions) throws -> (metadata: SOGSSHNInfoV2, centroids: Data, centroidDimensions: TextureDimensions, labels: Data)? {
        guard let totalCoefficientCount = points.map({ $0.shCoefficients.count }).max(), totalCoefficientCount > 1 else {
            return nil
        }

        let coefficientsPerEntry = totalCoefficientCount - 1
        let bands = Self.bands(forTotalCoefficientCount: totalCoefficientCount)

        var paletteIndices: [UInt16] = []
        paletteIndices.reserveCapacity(points.count)
        var paletteLookup: [QuantizedPaletteEntry: UInt16] = [:]
        var paletteEntries: [QuantizedPaletteEntry] = []

        for point in points {
            let entry = QuantizedPaletteEntry(bytes: Self.encodePaletteEntry(point.shCoefficients.dropFirst()))
            if let existing = paletteLookup[entry] {
                paletteIndices.append(existing)
                continue
            }

            guard paletteEntries.count < 65_536 else {
                throw Error.tooManySphericalHarmonicPalettes(paletteEntries.count + 1)
            }

            let newIndex = UInt16(paletteEntries.count)
            paletteEntries.append(entry)
            paletteLookup[entry] = newIndex
            paletteIndices.append(newIndex)
        }

        let centroidDimensions = TextureDimensions(
            width: 64 * coefficientsPerEntry,
            height: max(1, (paletteEntries.count + 63) / 64)
        )
        var centroidBytes = Data(repeating: 0, count: centroidDimensions.byteCount)
        for (paletteIndex, entry) in paletteEntries.enumerated() {
            let row = paletteIndex / 64
            let column = paletteIndex % 64
            let startX = column * coefficientsPerEntry
            for coefficientIndex in 0..<coefficientsPerEntry {
                let baseOffset = ((row * centroidDimensions.width) + startX + coefficientIndex) * 4
                let sourceOffset = coefficientIndex * 3
                centroidBytes[baseOffset + 0] = entry.bytes[sourceOffset + 0]
                centroidBytes[baseOffset + 1] = entry.bytes[sourceOffset + 1]
                centroidBytes[baseOffset + 2] = entry.bytes[sourceOffset + 2]
                centroidBytes[baseOffset + 3] = 255
            }
        }

        var labelBytes = Data(repeating: 0, count: dimensions.byteCount)
        for (index, paletteIndex) in paletteIndices.enumerated() {
            let baseOffset = index * 4
            labelBytes[baseOffset + 0] = UInt8(paletteIndex & 0xFF)
            labelBytes[baseOffset + 1] = UInt8(paletteIndex >> 8)
            labelBytes[baseOffset + 2] = 0
            labelBytes[baseOffset + 3] = 255
        }

        let metadata = SOGSSHNInfoV2(
            count: paletteEntries.count,
            bands: bands,
            codebook: Self.shNCodebook,
            mins: nil,
            maxs: nil,
            files: ["shN_centroids.webp", "shN_labels.webp"]
        )
        return (metadata, centroidBytes, centroidDimensions, labelBytes)
    }

    private func writeArchive(_ payloads: [FilePayload], to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        let archive = try Archive(url: url, accessMode: .create)

        for payload in payloads {
            try archive.addEntry(
                with: payload.name,
                type: .file,
                uncompressedSize: Int64(payload.data.count),
                compressionMethod: .deflate,
                provider: { position, size in
                    let start = Int(position)
                    let end = min(start + size, payload.data.count)
                    return payload.data.subdata(in: start..<end)
                }
            )
        }
    }

    private func writeDirectory(_ payloads: [FilePayload], metaURL: URL) throws {
        let directoryURL = metaURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        for payload in payloads {
            try payload.data.write(to: directoryURL.appendingPathComponent(payload.name), options: .atomic)
        }
    }

    private static let shC0: Float = 0.28209479177387814
    private static let sqrtHalfTimesTwo = Float(sqrt(2.0))

    private static let scaleCodebook: [Float] = (0..<256).map { Float($0) / 16.0 - 10.0 }
    private static let sh0Codebook: [Float] = (0..<256).map { (Float($0) / 255.0 - 0.5) / shC0 }
    private static let shNCodebook: [Float] = (0..<256).map { (Float($0) - 128.0) / 128.0 }

    private static func makeBaseTextureDimensions(pointCount: Int) -> TextureDimensions {
        let effectiveCount = max(pointCount, 1)
        let width = max(4, roundUpToMultipleOfFour(Int(ceil(sqrt(Double(effectiveCount))))))
        let height = max(4, roundUpToMultipleOfFour(Int(ceil(Double(effectiveCount) / Double(width)))))
        return TextureDimensions(width: width, height: height)
    }

    private static func normalizedSphericalHarmonicCount(_ count: Int) -> Int {
        if count >= 16 { return 16 }
        if count >= 9 { return 9 }
        if count >= 4 { return 4 }
        return 1
    }

    private static func bands(forTotalCoefficientCount count: Int) -> Int {
        switch count {
        case 4: return 1
        case 9: return 2
        case 16: return 3
        default: return 0
        }
    }

    private static func paddedSphericalHarmonics(_ coefficients: [SIMD3<Float>], count: Int) -> [SIMD3<Float>] {
        if coefficients.count >= count {
            return Array(coefficients.prefix(count))
        }
        return coefficients + Array(repeating: .zero, count: count - coefficients.count)
    }

    private static func sogEncodeLog(_ value: Float) -> Float {
        let transformed = log(abs(value) + 1)
        return value < 0 ? -transformed : transformed
    }

    private static func quantizePositionComponent(_ value: Float, minValue: Float, maxValue: Float) -> UInt16 {
        let range = maxValue - minValue
        guard abs(range) > .ulpOfOne else { return 0 }
        let normalized = (value - minValue) / range
        let clamped = min(max(normalized, 0), 1)
        return UInt16((clamped * 65_535).rounded())
    }

    private static func encodeScale(_ value: Float) -> UInt8 {
        clipToUInt8(((value + 10.0) * 16.0).rounded())
    }

    private static func encodeSH0(_ value: Float) -> UInt8 {
        clipToUInt8((0.5 + shC0 * value) * 255.0)
    }

    private static func encodeSHN(_ value: Float) -> UInt8 {
        clipToUInt8((value * 128.0).rounded() + 128.0)
    }

    private static func encodeOpacity(_ value: Float) -> UInt8 {
        clipToUInt8((min(max(value, 0), 1) * 255.0).rounded())
    }

    private static func encodeQuaternion(_ quaternion: simd_quatf) -> (UInt8, UInt8, UInt8, UInt8) {
        let normalized = quaternion.normalized
        var components = [normalized.real, normalized.imag.x, normalized.imag.y, normalized.imag.z]

        var maxIndex = 0
        for index in 1..<components.count where abs(components[index]) > abs(components[maxIndex]) {
            maxIndex = index
        }

        if components[maxIndex] < 0 {
            for index in components.indices {
                components[index].negate()
            }
        }

        let alpha = UInt8(maxIndex + 252)
        switch maxIndex {
        case 0:
            return (encodeQuaternionComponent(components[1]), encodeQuaternionComponent(components[2]), encodeQuaternionComponent(components[3]), alpha)
        case 1:
            return (encodeQuaternionComponent(components[0]), encodeQuaternionComponent(components[2]), encodeQuaternionComponent(components[3]), alpha)
        case 2:
            return (encodeQuaternionComponent(components[0]), encodeQuaternionComponent(components[1]), encodeQuaternionComponent(components[3]), alpha)
        default:
            return (encodeQuaternionComponent(components[0]), encodeQuaternionComponent(components[1]), encodeQuaternionComponent(components[2]), alpha)
        }
    }

    private static func encodeQuaternionComponent(_ value: Float) -> UInt8 {
        clipToUInt8((((value / sqrtHalfTimesTwo) + 0.5) * 255.0).rounded())
    }

    private static func encodePaletteEntry<T: Collection>(_ coefficients: T) -> Data where T.Element == SIMD3<Float> {
        var data = Data(capacity: coefficients.count * 3)
        for coefficient in coefficients {
            data.append(encodeSHN(coefficient.x))
            data.append(encodeSHN(coefficient.y))
            data.append(encodeSHN(coefficient.z))
        }
        return data
    }

    private static func roundUpToMultipleOfFour(_ value: Int) -> Int {
        let remainder = value % 4
        return remainder == 0 ? value : value + (4 - remainder)
    }

    private static func clipToUInt8(_ value: Float) -> UInt8 {
        UInt8(min(max(Int(value), 0), 255))
    }
}
