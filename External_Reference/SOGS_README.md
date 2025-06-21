# SOGS (Self-Organizing Gaussians) Support for MetalSplatter

This implementation adds support for loading SOGS-compressed 3D Gaussian Splat data in MetalSplatter. SOGS is a compression technique that dramatically reduces file sizes by reorganizing Gaussian data and compressing it as WebP images.

## Overview

SOGS achieves compression by:
1. **Grid Reorganization**: Reshaping Gaussian attributes into 2D grids/images
2. **Self-Organizing Sort**: Using PLAS algorithm to arrange similar Gaussians as neighbors
3. **Image Compression**: Applying WebP compression to the smooth, organized attribute images

## File Format

A SOGS compressed scene consists of:
- `meta.json` - Metadata describing the compression parameters
- Multiple WebP files containing compressed attribute data:
  - `means_l.webp`, `means_u.webp` - Position data (lower/upper precision)
  - `quats.webp` - Rotation quaternions
  - `scales.webp` - Scale values
  - `sh0.webp` - Base color/spherical harmonics
  - `shN_centroids.webp`, `shN_labels.webp` - Higher-order spherical harmonics (optional)

## Usage

### Basic Loading

```swift
import SplatIO

// Load SOGS data using AutodetectSceneReader
let metaURL = URL(fileURLWithPath: "path/to/meta.json")
let reader = try AutodetectSceneReader(metaURL)
let points = try reader.readScene()

// Or use the convenience method
let sogsDirectory = URL(fileURLWithPath: "path/to/sogs/directory")
let points = try AutodetectSceneReader.loadSOGS(from: sogsDirectory)
```

### Direct SOGS Reader

```swift
// Use the SOGS reader directly
let reader = try SplatSOGSSceneReader(metaURL)
let points = try reader.readScene()
```

### Async Loading with Delegate

```swift
class MyDelegate: SplatSceneReaderDelegate {
    func didStartReading(withPointCount pointCount: UInt32?) {
        print("Starting to read \(pointCount ?? 0) points")
    }
    
    func didRead(points: [SplatScenePoint]) {
        print("Read \(points.count) points")
    }
    
    func didFinishReading() {
        print("Finished reading")
    }
    
    func didFailReading(withError error: Error?) {
        print("Error: \(error?.localizedDescription ?? "Unknown error")")
    }
}

let delegate = MyDelegate()
reader.read(to: delegate)
```

## Implementation Details

### Architecture

The SOGS implementation consists of several key components:

1. **`SOGSMetadata`** - Parses the meta.json file
2. **`SOGSCompressedData`** - Holds decoded WebP texture data
3. **`SOGSIterator`** - Decompresses individual Gaussian splats
4. **`SplatSOGSSceneReader`** - Main reader implementing `SplatSceneReader`
5. **`WebPDecoder`** - Handles WebP image decoding using Core Image

### WebP Decoding

The implementation uses Core Image for WebP decoding (requires iOS 14+/macOS 11+), with ImageIO as a fallback. For older platforms, you may need to integrate a third-party WebP library.

### Compression Algorithm

The decompression process follows the original SOGS specification:

1. **Position Decoding**: Reconstructs 16-bit precision from 8-bit low/high textures, applies min/max scaling, then exponential mapping
2. **Rotation Decoding**: Unpacks quaternions from compressed format with mode-based reconstruction
3. **Scale Decoding**: Applies min/max scaling to compressed scale values
4. **Color/Opacity**: Converts spherical harmonics to linear color and logit opacity to linear
5. **Higher-order SH**: Uses palette-based compression for additional spherical harmonics coefficients

## Testing

Test your SOGS implementation with the provided test data:

```swift
// Test with your sogs_test directory
SOGSTest.loadTestSOGSData()
```

Make sure to update the path in `SOGSTest.swift` to point to your actual `sogs_test` directory.

## Performance Considerations

- **File Size**: SOGS typically achieves 9x+ compression compared to uncompressed splat files
- **Loading Speed**: WebP decoding adds some overhead, but the reduced I/O often compensates
- **Memory Usage**: Decompressed textures are held in memory during processing
- **CPU vs GPU**: Current implementation uses CPU decompression; GPU implementation would be faster

## Requirements

- iOS 14.0+ / macOS 11.0+ (for Core Image WebP support)
- Swift 5.9+
- Core Image framework
- ImageIO framework (fallback)

## Error Handling

The implementation includes comprehensive error handling:

```swift
public enum SOGSError: Error {
    case invalidMetadata
    case missingFile(String)
    case webpDecodingFailed(String)
    case invalidTextureData
}
```

## Integration with MetalSplatter

Once loaded, SOGS data is converted to standard `SplatScenePoint` format, making it compatible with the existing MetalSplatter rendering pipeline. The loaded points can be used with any MetalSplatter renderer.

## Future Enhancements

Potential improvements:
1. **GPU Decompression**: Move decompression to Metal shaders for better performance
2. **Streaming**: Support progressive loading for large datasets
3. **Alternative Codecs**: Support other compressed formats beyond WebP
4. **Memory Optimization**: Reduce peak memory usage during decompression

## Troubleshooting

### WebP Decoding Issues
If WebP decoding fails, ensure:
- Your target platform supports WebP (iOS 14+/macOS 11+)
- The WebP files are valid and not corrupted
- Consider integrating a third-party WebP library for older platforms

### Performance Issues
- Large textures may consume significant memory during decompression
- Consider processing in chunks for very large datasets
- GPU-based decompression would significantly improve performance

### Compatibility
- Ensure your SOGS files follow the expected format
- Check that all required WebP files are present
- Verify the meta.json structure matches the expected schema 