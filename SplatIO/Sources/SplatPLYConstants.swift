struct SplatPLYConstants {
    enum ElementName: String {
        case point = "vertex"
    }

    enum PropertyName {
        // Canonical names (for writing)
        static let positionXName = "x"
        static let positionYName = "y"
        static let positionZName = "z"
        static let normalXName = "nx"
        static let normalYName = "ny"
        static let normalZName = "nz"
        static let sh0_rName = "f_dc_0"
        static let sh0_gName = "f_dc_1"
        static let sh0_bName = "f_dc_2"
        static let sphericalHarmonicsPrefix = "f_rest_"
        static let colorRName = "red"
        static let colorGName = "green"
        static let colorBName = "blue"
        static let scaleXName = "scale_0"
        static let scaleYName = "scale_1"
        static let scaleZName = "scale_2"
        static let opacityName = "opacity"
        static let rotation0Name = "rot_0"
        static let rotation1Name = "rot_1"
        static let rotation2Name = "rot_2"
        static let rotation3Name = "rot_3"

        // Alternative names accepted when reading (arrays for flexibility)
        static let positionX = [positionXName]
        static let positionY = [positionYName]
        static let positionZ = [positionZName]
        static let normalX = [normalXName]
        static let normalY = [normalYName]
        static let normalZ = [normalZName]
        static let sh0_r = [sh0_rName]
        static let sh0_g = [sh0_gName]
        static let sh0_b = [sh0_bName]
        static let colorR = [colorRName]
        static let colorG = [colorGName]
        static let colorB = [colorBName]
        static let scaleX = [scaleXName]
        static let scaleY = [scaleYName]
        static let scaleZ = [scaleZName]
        static let opacity = [opacityName]
        static let rotation0 = [rotation0Name]
        static let rotation1 = [rotation1Name]
        static let rotation2 = [rotation2Name]
        static let rotation3 = [rotation3Name]
    }
}
