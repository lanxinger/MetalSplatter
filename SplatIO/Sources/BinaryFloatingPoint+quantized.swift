extension BinaryFloatingPoint {
    /// Rounds with `rule` and clamps into `range`. Unlike `I(value)`, this never traps:
    /// NaN maps to zero (or the nearest bound if zero is out of range) and ±infinity to the bounds.
    func quantized<I: FixedWidthInteger>(clampedTo range: ClosedRange<I>,
                                         rounding rule: FloatingPointRoundingRule = .toNearestOrAwayFromZero) -> I {
        guard !isNaN else {
            return Swift.min(Swift.max(0, range.lowerBound), range.upperBound)
        }
        let value = rounded(rule)
        if value <= Self(range.lowerBound) { return range.lowerBound }
        if value >= Self(range.upperBound) { return range.upperBound }
        return I(value)
    }
}
