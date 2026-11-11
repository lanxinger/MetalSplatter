import Foundation
import simd

/// SOG v3 geometry. V2 keeps its original decoder and all writers still default to v2.
/// Unknown extras/blind fields do not affect viewing; compatibility exports deliberately omit them.
enum SOGV3Decoder {
    private struct Meta: Decodable {
        struct Means: Decodable { let mins: [Float]; let maxs: [Float]; let files: [String] }
        struct Book: Decodable { let codebook: [Float?]; let files: [String] }
        struct SH: Decodable { let count: Int; let bands: Int; let codebook: [Float?]; let files: [String] }
        let version: Int; let count: Int; let means: Means; let scales: Book; let sh0: Book; let shN: SH?
    }
    static func rotation(_ a: UInt8, _ b: UInt8, _ c: UInt8) -> simd_quatf {
        let p = SIMD3(Float(Int(a) - 128), Float(Int(b) - 128), Float(Int(c) - 128)) / 127
        let r = simd_length(p), t = Float.pi / 2 * r
        return simd_quatf(vector: SIMD4(p * (r > 1e-12 ? sin(t) / r : .pi / 2), cos(t)))
    }
    static func opacity(_ byte: UInt8) -> Float { let v = Float(byte) / 204; return v * v }

    static func decode(_ json: Data, load: (String) throws -> WebPDecoder.DecodedImage) throws -> [SplatScenePoint] {
        typealias Failure = SplatSOGSSceneReaderV2.SOGSV2Error
        guard json.count <= 1_048_576 else { throw Failure.invalidMetadata }
        let m = try JSONDecoder().decode(Meta.self, from: json)
        guard m.version == 3, m.count > 0, m.count <= Int(Int32.max),
              m.means.mins.count == 3, m.means.maxs.count == 3, m.means.files.count == 2,
              m.scales.files.count == 1, m.sh0.files.count == 1,
              m.scales.codebook.count == 256, m.sh0.codebook.count == 256 else { throw Failure.invalidMetadata }
        for k in 0..<3 {
            guard m.means.mins[k].isFinite, m.means.maxs[k].isFinite,
                  m.means.mins[k] <= m.means.maxs[k], abs(m.means.mins[k]) < 88,
                  abs(m.means.maxs[k]) < 88 else { throw Failure.invalidMetadata }
        }
        let degree = m.shN?.bands ?? 0
        guard (0...4).contains(degree) else { throw Failure.invalidMetadata }
        // Bound CPU scene expansion before loading textures or allocating points.
        let estimatedBytes = m.count.multipliedReportingOverflow(by: 256 + (degree + 1) * (degree + 1) * 16)
        guard !estimatedBytes.overflow, estimatedBytes.partialValue <= 1_500_000_000 else { throw Failure.invalidMetadata }
        func texture(_ name: String, minimum: Int) throws -> WebPDecoder.DecodedImage {
            guard !name.isEmpty, !name.hasPrefix("/"), !name.contains(":"), !name.contains("\\"),
                  !name.contains("\0"), name.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else { throw Failure.invalidMetadata }
            let image = try load(name)
            guard image.bytesPerPixel == 4, image.width > 0, image.height > 0,
                  image.width * image.height >= minimum, image.pixels.count == image.width * image.height * 4 else { throw Failure.invalidTextureData }
            return image
        }
        let lo = try texture(m.means.files[0], minimum: m.count), hi = try texture(m.means.files[1], minimum: m.count)
        let scales = try texture(m.scales.files[0], minimum: m.count), dc = try texture(m.sh0.files[0], minimum: m.count)
        for image in [hi, scales, dc] where image.width != lo.width || image.height != lo.height { throw Failure.invalidTextureData }
        let terms = (degree + 1) * (degree + 1) - 1
        var palette: WebPDecoder.DecodedImage?, labels: WebPDecoder.DecodedImage?
        if let sh = m.shN, degree > 0 {
            guard sh.count > 0, sh.count <= 262_144, sh.files.count == 2, sh.codebook.count == 256 else { throw Failure.invalidMetadata }
            palette = try texture(sh.files[0], minimum: sh.count * terms)
            labels = try texture(sh.files[1], minimum: m.count)
            guard labels?.width == lo.width, labels?.height == lo.height else { throw Failure.invalidTextureData }
        }
        var points: [SplatScenePoint] = []; points.reserveCapacity(m.count)
        for i in 0..<m.count {
            let o = i * 4
            var position = SIMD3<Float>(), scale = SIMD3<Float>(), dcValue = SIMD3<Float>()
            for k in 0..<3 {
                let q = Float(Int(lo.pixels[o+k]) | Int(hi.pixels[o+k]) << 8) / 65_535
                let v = m.means.mins[k] + (m.means.maxs[k] - m.means.mins[k]) * q
                position[k] = copysign(expm1(abs(v)), v)
                scale[k] = m.scales.codebook[Int(scales.pixels[o+k])] ?? -.greatestFiniteMagnitude
                dcValue[k] = m.sh0.codebook[Int(dc.pixels[o+k])] ?? 0
                guard position[k].isFinite, scale[k].isFinite, dcValue[k].isFinite else { throw Failure.invalidTextureData }
            }
            var coefficients = [dcValue]
            if let sh = m.shN, let palette, let labels, terms > 0 {
                let label = Int(labels.pixels[o]) | Int(labels.pixels[o+1]) << 8 | Int(labels.pixels[o+2]) << 16
                for j in 0..<terms {
                    var coefficient = SIMD3<Float>()
                    if label < sh.count {
                        for k in 0..<3 { coefficient[k] = sh.codebook[Int(palette.pixels[(label * terms + j) * 4 + k])] ?? 0 }
                    }
                    guard coefficient.x.isFinite, coefficient.y.isFinite, coefficient.z.isFinite else { throw Failure.invalidTextureData }
                    coefficients.append(coefficient)
                }
            }
            points.append(SplatScenePoint(position: position, color: .sphericalHarmonic(coefficients),
                opacity: .linearFloat(opacity(dc.pixels[o+3])), scale: .exponent(scale),
                rotation: rotation(lo.pixels[o+3], hi.pixels[o+3], scales.pixels[o+3])))
        }
        return points
    }
}
