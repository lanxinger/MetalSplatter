#if os(iOS)

import ARKit
import Metal
import simd

public class ARPerspectiveCamera {
    public let session: ARSession
    public let near: Float
    public let far: Float
    
    public private(set) var viewMatrix: simd_float4x4 = matrix_identity_float4x4
    public private(set) var projectionMatrix: simd_float4x4 = matrix_identity_float4x4
    public private(set) var intrinsics: simd_float3x3 = matrix_identity_float3x3
    public private(set) var transform: simd_float4x4 = matrix_identity_float4x4
    
    public init(session: ARSession, near: Float = 0.001, far: Float = 100.0) {
        self.session = session
        self.near = near
        self.far = far
    }
    
    public func update(viewportSize: CGSize) {
        guard let frame = session.currentFrame,
              let orientation = getOrientation() else { return }
        
        // Get ARKit matrices - use them directly like the reference implementation
        viewMatrix = frame.camera.viewMatrix(for: orientation)
        projectionMatrix = frame.camera.projectionMatrix(
            for: orientation,
            viewportSize: viewportSize,
            zNear: CGFloat(near),
            zFar: CGFloat(far)
        )
        
        // Get world transform
        transform = viewMatrix.inverse * orientationCorrectionMatrix(for: orientation)
        
        // Store intrinsics
        intrinsics = frame.camera.intrinsics
    }
    
    
    private func orientationCorrectionMatrix(for orientation: UIInterfaceOrientation) -> simd_float4x4 {
        // Y and Z flip matrix
        let yzFlipMatrix = simd_float4x4(
            simd_make_float4(1, 0, 0, 0),
            simd_make_float4(0, -1, 0, 0),
            simd_make_float4(0, 0, -1, 0),
            simd_make_float4(0, 0, 0, 1)
        )
        
        // Rotation based on interface orientation
        let rotationAngle = cameraToDisplayRotation(for: orientation)
        let rotationMatrix = matrix4x4_rotation(radians: rotationAngle, axis: simd_make_float3(0, 0, 1))
        
        return yzFlipMatrix * rotationMatrix
    }
    
    private func cameraToDisplayRotation(for orientation: UIInterfaceOrientation) -> Float {
        switch orientation {
        case .landscapeLeft:
            return Float.pi
        case .portrait:
            return Float.pi * 0.5
        case .portraitUpsideDown:
            return -Float.pi * 0.5
        case .landscapeRight:
            return 0
        default:
            return 0
        }
    }
    
    private func getOrientation() -> UIInterfaceOrientation? {
        return UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first?.interfaceOrientation
    }
}

// MARK: - Matrix Utilities

func matrix4x4_rotation(radians: Float, axis: simd_float3) -> simd_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    
    return simd_float4x4(
        simd_make_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
        simd_make_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
        simd_make_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
        simd_make_float4(                  0,                   0,                   0, 1)
    )
}

func matrix4x4_rotation(_ quaternion: simd_quatf) -> simd_float4x4 {
    let q = quaternion.normalized
    let x = q.imag.x, y = q.imag.y, z = q.imag.z, w = q.real
    
    return simd_float4x4(
        simd_make_float4(1 - 2*(y*y + z*z), 2*(x*y + w*z), 2*(x*z - w*y), 0),
        simd_make_float4(2*(x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x), 0),
        simd_make_float4(2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y), 0),
        simd_make_float4(0, 0, 0, 1)
    )
}

func matrix4x4_translation(_ translation: simd_float3) -> simd_float4x4 {
    return matrix4x4_translation(translation.x, translation.y, translation.z)
}

func matrix4x4_translation(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
    return simd_float4x4(
        simd_make_float4(1, 0, 0, 0),
        simd_make_float4(0, 1, 0, 0),
        simd_make_float4(0, 0, 1, 0),
        simd_make_float4(x, y, z, 1)
    )
}

func matrix4x4_scale(_ scale: Float) -> simd_float4x4 {
    return matrix4x4_scale(scale, scale, scale)
}

func matrix4x4_scale(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
    return simd_float4x4(
        simd_make_float4(x, 0, 0, 0),
        simd_make_float4(0, y, 0, 0),
        simd_make_float4(0, 0, z, 0),
        simd_make_float4(0, 0, 0, 1)
    )
}

#endif // os(iOS)