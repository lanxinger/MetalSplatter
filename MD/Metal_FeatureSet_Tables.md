# Metal Feature Set Tables

##  Developer


**Metal GPU’s (Apple silicon)**

```
GPU Metal version Apple family^1
```
```
A8-series Metal Apple
```
```
A9-series Metal Apple
```
```
A10-series Metal Apple
```
```
A11 Bionic Metal Apple
```
```
A12-series Metal Apple
```
```
A13 Bionic Metal Apple
```
```
A14 Bionic Metal 3 & 4 Apple
```
```
A15 Bionic Metal 3 & 4 Apple
```
```
A16 Bionic Metal 3 & 4 Apple
```
```
A17 Pro Metal 3 & 4 Apple
```
```
A18-series Metal 3 & 4 Apple
```
```
M1-series Metal 3 & 4 Apple
```
```
M2-series Metal 3 & 4 Apple
```
```
M3-series Metal 3 & 4 Apple
```
```
M4-series Metal 3 & 4 Apple
```
**Metal GPU’s (Intel Mac)**

```
GPU Metal version Mac family^1
```
```
AMD 500-series Metal Mac
```
```
AMD Vega Metal 3 Mac
```
```
AMD 5000-series Metal 3 Mac
```
```
AMD 6000-series Metal 3 Mac
```
```
Intel UHD Graphics 630 Metal 3 Mac
```
```
Intel Iris Plus Graphics Metal 3 Mac
```
1. See MTLGPUFamily for each GPU family’s enumeration
    constant.

For Mac devices with Apple silicon, the MTLDevice instance
for the Apple GPU reports that it also supports Mac2 GPU
family because the devices support the union of both feature
families.


## Metal feature availability by GPU family

**GPU family**^1 **Metal Apple Mac**

## Metal performance shaders Metal3 Apple2 Mac

**Programmable blending** Metal4 Apple2 —

**PVRTC pixel formats** — Apple2 —

**EAC/ETC pixel formats** Metal4 Apple2 —

## Compressed volume texture formats Metal3 Apple3 Mac

## Depth-16 pixel format Metal3 Apple2 Mac

## Linear textures Metal3 Apple2 Mac

## MSAA depth resolve Metal3 Apple3 Mac

## Array of textures (read) Metal3 Apple3 Mac

## Array of textures (write) Metal3 Apple6 Mac

## Cube map texture arrays Metal3 Apple4 Mac

## Stencil texture views Metal3 Apple2 Mac

## Array of samplers Metal3 Apple3 Mac

## Sampler maximum anisotropy Metal3 Apple2 Mac

## Sampler LOD clamp Metal3 Apple2 Mac

## MTLSamplerState support for comparison functions Metal3 Apple3 Mac

## 16-bit unsigned integer coordinates Metal3 Apple2 Mac

## Border color Metal3 Apple7 Mac

## Counting occlusion query Metal3 Apple3 Mac

## Base vertex/instance drawing Metal3 Apple3 Mac

## Layered rendering Metal3 Apple5 Mac

## Layered rendering to multisample textures Metal3 Apple7 Mac

## Combined MSAA store and resolve action Metal3 Apple3 Mac

## MSAA blits Metal3 Apple2 Mac

## Programmable sample positions Metal3 Apple2 Mac

## Deferred store action Metal3 Apple2 Mac

## Texture barriers — — Mac

## Memory barriers^3 Metal3 Apple3 Mac

## Memory barriers in indirect command buffers (compute) Metal3 Apple3 Mac

## Indirect tessellation arguments Metal3 Apple5 Mac

## Tessellation in indirect command buffers Metal3 Apple5 Mac

## Resource heaps Metal3 Apple2 Mac

## Function specialization Metal3 Apple2 Mac

## Read/Write buffers in functions Metal3 Apple3 Mac

## Read/Write textures in functions Metal3 Apple4 Mac

## Extract, insert, and reverse bits Metal3 Apple2 Mac

## SIMD barrier Metal3 Apple2 Mac

## Indirect draw and dispatch arguments Metal3 Apple3 Mac

## Argument buffers tier 1 Metal3 Apple2 Mac

## Argument buffers tier 2 Metal3 Apple6 Mac

## Indirect command buffers (rendering) Metal3 Apple3 Mac

## Indirect command buffers (compute) Metal3 Apple3 Mac

## Uniform type Metal3 Apple2 Mac

- MetalKit Metal3 Apple2 Mac Feature Available in family
- Metal performance shaders Metal3 Apple2 Mac
- BC pixel formats^2 — Apple9 Mac ASTC pixel formats Metal4 Apple2 —
- Compressed volume texture formats Metal3 Apple3 Mac
- Wide color pixel format Metal3 Apple2 Mac Extended range pixel formats Metal4 Apple3 —
- Depth-16 pixel format Metal3 Apple2 Mac
- Linear textures Metal3 Apple2 Mac
- MSAA depth resolve Metal3 Apple3 Mac
- Array of textures (read) Metal3 Apple3 Mac
- Array of textures (write) Metal3 Apple6 Mac
- Cube map texture arrays Metal3 Apple4 Mac
- Stencil texture views Metal3 Apple2 Mac
- Array of samplers Metal3 Apple3 Mac
- Sampler maximum anisotropy Metal3 Apple2 Mac
- Sampler LOD clamp Metal3 Apple2 Mac
- MTLSamplerState support for comparison functions Metal3 Apple3 Mac
- 16-bit unsigned integer coordinates Metal3 Apple2 Mac
- Border color Metal3 Apple7 Mac
- Counting occlusion query Metal3 Apple3 Mac
- Base vertex/instance drawing Metal3 Apple3 Mac
- Layered rendering Metal3 Apple5 Mac
- Layered rendering to multisample textures Metal3 Apple7 Mac
- Dual-source blending Metal3 Apple2 Mac Memoryless render targets Metal4 Apple2 —
- Combined MSAA store and resolve action Metal3 Apple3 Mac
- MSAA blits Metal3 Apple2 Mac
- Programmable sample positions Metal3 Apple2 Mac
- Deferred store action Metal3 Apple2 Mac
- Texture barriers — — Mac
- Memory barriers^3 Metal3 Apple3 Mac
- Memory barriers in indirect command buffers (compute) Metal3 Apple3 Mac
- Tessellation Metal3 Apple3 Mac Memory barriers in indirect command buffers (rendering) Metal4 Apple9 —
- Indirect tessellation arguments Metal3 Apple5 Mac
- Tessellation in indirect command buffers Metal3 Apple5 Mac
- Resource heaps Metal3 Apple2 Mac
- Function specialization Metal3 Apple2 Mac
- Read/Write buffers in functions Metal3 Apple3 Mac
- Read/Write textures in functions Metal3 Apple4 Mac
- Extract, insert, and reverse bits Metal3 Apple2 Mac
- SIMD barrier Metal3 Apple2 Mac
- Indirect draw and dispatch arguments Metal3 Apple3 Mac
- Argument buffers tier 1 Metal3 Apple2 Mac
- Argument buffers tier 2 Metal3 Apple6 Mac
- Indirect command buffers (rendering) Metal3 Apple3 Mac
- Indirect command buffers (compute) Metal3 Apple3 Mac
- Uniform type Metal3 Apple2 Mac
   - GPU family Imageblocks Metal4 Apple4 —


**Tile shaders** Metal4 Apple4 —

**Imageblock sample coverage control** Metal4 Apple4 —

**Postdepth coverage** Metal4 Apple4 —

**Quad-scoped permute operations** Metal3 Apple4 Mac

**Quad-scoped reduction operations** Metal3 Apple7 Mac

**SIMD-scoped permute operations** Metal3 Apple6 Mac

**SIMD-scoped reduction operations** Metal3 Apple7 Mac

**SIMD-scoped matrix multiply operations** Metal4 Apple7 —

**Raster order groups**^4 Metal3 Apple4 Varies

**Nonuniform threadgroup size** Metal3 Apple4 Mac

**Multiple viewports** Metal3 Apple5 Mac

**Device notifications** — — Mac

**Stencil feedback** Metal3 Apple5 Mac

**Stencil resolve** Metal3 Apple5 Mac

**Nonsquare tile dispatch** Metal4 Apple5 —

**Texture swizzle** Metal3 Apple2 Mac

**Placement heap** Metal3 Apple2 Mac

**Primitive ID** Metal3 Apple7 Mac

**Barycentric coordinates**^5 Metal4 Apple7 Varies

**Read/Write cube map textures in functions** Metal3 Apple4 Mac

**Sparse textures** Metal4 Apple6 —

**Sparse depth and stencil textures**^6 Metal4 Apple7 —

**Variable rasterization rate**^7 Metal4 Apple6 Varies

**Vertex amplification**^8 Metal4 Apple6 Varies

**64-bit integer math** Metal3 Apple3 —

**Lossy texture compression** — Apple8 —

**SIMD shift and fill** — Apple8 —

**Render dynamic libraries** Metal4 Apple6 —

**Compute dynamic libraries** Metal3 Apple6 Mac

**Mesh shading** Metal3 Apple7 Mac

**Indirect mesh draw arguments** — Apple9 —

**Indirect command buffers containing mesh draws** — Apple9 —

**MetalFX spatial upscaling** Metal3 Apple3 Mac

**MetalFX temporal upscaling** Varies Apple7 —

**MetalFX frame interpolation** Metal4 Apple5 —

**MetalFX denoised upscaling** — Apple9 —

**Fast resource loading** Metal3 Apple2 Mac

**Ray tracing in compute pipelines**^9 Metal3 Apple6 Varies

**Ray tracing in render pipelines**^10 Metal4 Apple6 —

**Floating-point atomics** Metal3 Apple7 Mac

**Texture atomics** Metal3 Apple6 Mac

**64-bit atomics**^11 — Apple9 —

**Query texture LOD**^12 — Apple8 —

**Binary archives** Metal3 Apple3 Mac

**Function pointers in compute pipelines**^13 Metal3 Apple6 Varies

**Function pointers in render pipelines**^10 Metal4 Apple6 —

**Depth sample compare bias and gradient** Metal4 Apple2 —

**Nonprivate depth stencil textures** Metal4 Apple2 —

**Dynamic stride for attribute buffers** Metal3 Apple4 Mac

**MTLAttributeFormat.floatRGB9E5 and .floatRG11B10** Metal3 Apple5 Mac

**MTLDataType.bfloat (brain float) scalar and vector cases** Metal3 Apple6 Mac

**Relaxed math** Metal4 Apple4 —

**Global built-ins and bindings** Metal4 Apple6 —

**Memory coherence for textures and buffers in shaders** Metal4 Apple6 —

**Per-pipeline shader validation** Metal4 Apple6 —

**GPU family**^1 **Metal Apple Mac**


```
Shader logging Metal4 Apple6 —
Residency sets Metal4 Apple6 —
Acceleration structures containing row-major matrices — Apple9 —
Ray tracing with per-component motion interpolation — Apple9 —
Direct access to on-chip ray-intersection result storage — Apple9 —
Fragment visibility count accumulation^14 Metal4 Apple7 —
Argument tables Metal4 Apple7 —
Command allocators Metal4 Apple7 —
Decoupled command queues and command buffers Metal4 Apple7 —
Texture view pools Metal4 Apple7 —
Command barriers Metal4 Apple7 —
Placement sparse buffers Metal4 Apple7 —
Placement sparse textures Metal4 Apple7 —
Dedicated compilation contexts Metal4 Apple7 —
Pipeline dataset serialization Metal4 Apple7 —
Flexible render pipeline state Metal4 Apple7 —
Color attachment mapping^14 Metal4 Apple7 —
Machine learning encoding Metal4 Apple7 —
Tensors Metal4 Apple7 —
Performance counter heaps Metal4 Apple7 —
Address-driven acceleration structure builds — Apple9 —
Acceleration structure build options — Apple9 —
Intersection function buffers — Apple9 —
```
**GPU family**^1 **Metal Apple Mac**

1. See MTLGPUFamily for each GPU family’s enumeration constant.
2. Some GPU devices in the Apple7 and Apple8 families support BC texture compression in iPadOS. You can check
    an individual GPU’s support for this feature by inspecting its MTLDevice.supportsBCTextureCompression
    property at runtime. As of Apple9 all GPU’s have support.
3. GPU devices in Apple3 through Apple9 families don’t support memory barriers that include the
    MTLRenderStages.fragment or .tile stages in the after argument, or
    MTLBarrierScope.renderTargets in the scope argument of
    MTLRenderCommandEncoder.memoryBarrier(scope:after:before:) and
    MTLRenderCommandEncoder.memoryBarrier(resources:after:before:).
4. Some GPU devices in the Mac2 family support raster order groups. You can check an individual GPU’s support for
    this feature by inspecting its MTLDevice.rasterOrderGroupsSupported property at runtime.
5. Some GPU devices in the Mac2 and Metal3 families support barycentric coordinates. You can check an individual
    GPU’s support for this feature by inspecting its MTLDevice.supportsShaderBarycentricCoordinates
    property at runtime.
6. GPU devices in the Apple7 family support sparse depth and stencil textures only for placement sparse textures.
    GPU devices in Apple8 through Apple9 support both placement and automatic heap backing for sparse depth
    and stencil textures.
7. Some GPU devices in the Mac2 family support variable rasterization rates. You can check an individual GPU’s
    support for this feature by calling its MTLDevice.supportsRasterizationRateMap(layerCount:) method
    at runtime.
8. Some GPU devices in the Mac2 family support vertex amplification. You can check an individual GPU’s support for
    this feature by calling its MTLDevice.supportsVertexAmplificationCount(_:) method at runtime.
9. Some GPU devices in the Mac2 family support ray tracing in compute pipelines. You can check an individual GPU’s
    support for this feature by inspecting its MTLDevice.supportsRaytracing property at runtime.

10.Support for function pointers and ray tracing in render pipelines isn’t compatible with mesh shading. You can only
use Metal IR linking through MTLLinkedFunctions.privateFunctions in render pipelines using mesh
shading.

11.Some GPU devices in the Apple8 family support 64-bit atomic minimum and maximum using ulong, on both
buffers and textures. You can check an individual GPU’s support for this feature by verifying it supports both the
Mac2 and Apple8 families by separately passing each to the MTLDevice.supportsFamily(_:) method. As of
Apple9 all GPU’s have support.

12.Some GPU devices in the Apple7 family support query texture LOD. You can check an individual GPU’s support for
this feature by inspecting its MTLDevice.supportsQueryTextureLOD property at runtime. As of Apple8 all
GPU’s have support.

13.Some GPU devices in the Mac2 family support function pointers in compute pipelines. You can check an individual
GPU’s support for this feature by inspecting its MTLDevice.supportsFunctionPointers property at runtime.

14.GPU devices supporting fragment visibility count accumulation and color attachment mapping features support
those features in both Metal3 and Metal4 command encoding models.


**GPU implementation limits by family**

**GPU family**^1 **Metal3 Metal4 Apple2 Apple3 Apple4 Apple5 Apple6 Apple7 Apple8 Apple9 Mac**

**Function arguments Function arguments**

**Maximum number of vertex attributes, per vertex
descriptor**^2

### 31 31 31 31 31 31 31 31 31 31 31

**Maximum number of entries in the buffer argument
table, per graphics or kernel function**^2

### 31 31 31 31 31 31 31 31 31 31 31

**Maximum number of entries in the texture argument
table, per graphics or kernel function**^2

### 128 128 31 31 96 96 128 128 128 128 128

**Maximum number of entries in the sampler state
argument table, per graphics or kernel functions** 2 3

### 16 16 16 16 16 16 16 16 16 16 16

**Maximum number of entries in the threadgroup
memory argument table, per kernel function**^2

### 31 31 31 31 31 31 31 31 31 31 31

**Maximum number of constant buffer arguments in
vertex, fragment, tile, or kernel functions**^2

### 14 31 31 31 31 31 31 31 31 31 14

**Maximum length of constant buffer arguments in
vertex, fragment, tile, or kernel functions**^2

### 4 KB 4 KB 4 KB 4 KB 4 KB 4 KB 4 KB 4 KB 4 KB 4 KB 4 KB

**Maximum threads per threadgroup**^4102410245125121024102410241024102410241024

**Maximum total threadgroup memory allocation** 32 KB 32 KB 16,352 B 16 KB 32 KB 32 KB 32 KB 32 KB 32 KB 32 KB 32 KB

**Maximum explicit image block allocation**^5 Not available 32 KB Not available Not available 32 KB 32 KB 32 KB 32 KB 32 KB 32 KB Not available

**Maximum implicit image block allocation**^5 Not available 128 KB Not available Not available 128 KB 128 KB 128 KB 128 KB 128 KB 128 KB Not available

**Threadgroup memory length alignment** 16 B 16 B 16 B 16 B 16 B 16 B 16 B 16 B 16 B 16 B 16 B

**Maximum function memory allocation for a buffer in
the constant address space**
No limit No limit No limit No limit No limit No limit No limit No limit No limit No limit No limit

**Maximum scalar or vector inputs to a fragment
function. (Declare with the** [[stage_in]] **qualifier.)**^6

### 32 124 60 60 124 124 124 124 124 124 32

**Maximum number of input components to a fragment
function. (Declare with the** [[stage_in]] **qualifier.)**^6

### 124 124 60 60 124 124 124 124 124 124 124

**Maximum number of function constants** 65,536 65,536 65,536 65,536 65,536 65,536 65,536 65,536 65,536 65,536 65,

**Maximum tessellation factor** 64 64 Not available 16 16 64 64 64 64 64 64

**Maximum number of viewports and scissor
rectangles, per vertex function**

### 16 16 1 1 1 16 16 16 16 16 16

**Maximum number of raster order groups, per
fragment function**
8 8 Not available Not available 8 8 8 8 8 8 8

**Minimum alignment of buffer layout descriptor stride** 4 B 1 B 4 B 4 B 4 B 1 B 1 B 1 B 1 B 1 B 4 B

**Maximum size of buffer layout descriptor stride** 4 KB No limit No limit No limit No limit No limit No limit No limit No limit No limit 4 KB

**Argument buffers**^7 **Argument buffers**

**Maximum number of buffers you can access, per
stage, from an argument buffer**
No limit No limit 31 31 96 96 No limit No limit No limit No limit No limit

**GPU family**^1


**Maximum number of textures you can access, per
stage, from an argument buffer**

### 1 M 1 M 31 31 96 96 1 M 1 M 1 M 1 M 1 M

**Maximum number of samplers you can access, per
stage, from an argument buffer**

### 1024 1024 16 16 16 16 128 1024 1024 500 K 1024

**Resources Resources**

**Minimum constant buffer offset alignment** 32 B 4 B 4 B 4 B 4 B 4 B 4 B 4 B 4 B 4 B 32 B

**Maximum 1D texture width** 16,384 px 16,384 px 8192 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px

**Maximum 2D texture width and height** 16,384 px 16,384 px 8192 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px

**Maximum cube map texture width and height** 16,384 px 16,384 px 8192 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px 16,384 px

**Maximum 3D texture width, height, and depth** 2048 px 2048 px 2048 px 2048 px 2048 px 2048 px 2048 px 2048 px 2048 px 2048 px 2048 px

**Maximum texture buffer width**^8 256 M px 256 M px 64 M px 256 M px 256 M px 256 M px 256 M px 256 M px 256 M px 256 M px 256 M px

**Maximum number of layers per 1D texture array, 2D
texture array, or 3D texture array**

### 2048 2048 2048 2048 2048 2048 2048 2048 2048 2048 2048

**Buffer alignment for copying an existing texture to a
buffer**

### 256 B 16 B 64 B 16 B 16 B 16 B 16 B 16 B 16 B 16 B 256 B

**Maximum counter sample buffer length** 32 KB 32 KB 32 KB 32 KB 32 KB 32 KB 32 KB 32 KB 32 KB 32 KB No limit

**Maximum number of sample buffers** 32 32 32 32 32 32 32 32 32 32 No limit

**Maximum number of residency sets per queue** 32 32 Not available Not available Not available Not available 32 32 32 32 32

**Maximum number of residency sets per buffer** 32 32 Not available Not available Not available Not available 32 32 32 32 32

**Render targets Render targets**

**Maximum number of color render targets per render
pass descriptor**

### 8 8 8 8 8 8 8 8 8 8 8

**Maximum size of a point primitive** 511 511 511 511 511 511 511 511 511 511 511

**Maximum explicit image block size, per pixel, per
sample, when using multiple color render targets**
Not available 64 B Not available Not available 64 B 64 B 64 B 64 B 64 B 64 B Not available

**Maximum implicit image block size, per pixel, per
sample, when using multiple color render targets**
Not available 128 B 32 B 32 B 64 B 64 B 64 B 128 B 128 B 128 B Not available

**Maximum visibility query offset** 256 KB 256 KB 65,528 B 65,528 B 65,528 B 65,528 B 65,528 B 256 KB 256 KB 256 KB 256 KB

**Maximum tile size in render passes without MSAA** Not available 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 Not available

**Maximum tile size in render passes with 2x MSAA** Not available 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 32 x 32 Not available

**Maximum tile size in render passes with 4x MSAA** Not available 32 x 16 32 x 16 32 x 16 32 x 16 32 x 16 32 x 16 32 x 16 32 x 16 32 x 16 Not available

**Feature limits Feature limits**

**GPU family**^1 **Metal3 Metal4 Apple2 Apple3 Apple4 Apple5 Apple6 Apple7 Apple8 Apple9 Mac**


```
Maximum number of fences 32,768 32,768 32,768 32,768 32,768 32,768 32,768 32,768 32,768 32,768 32,
```
```
Maximum number of I/O commands per buffer 8192 8192 8192 8192 8192 8192 8192 8192 8192 8192 8192
```
```
Maximum vertex count for vertex amplification^9 Varies 8 Not available Not available Not available Not available 2 8 8 8 Varies
```
```
Maximum threadgroups per object shader grid 1024 No limit Not available Not available Not available Not available Not available No limit No limit No limit 1024
```
```
Maximum threadgroups per mesh shader grid^1010241024 Not available Not available Not available Not available Not available 1024 1024 1,048,575 1024
```
```
Maximum payload in mesh shader pipeline^11 16,384 B 16,384 B Not available Not available Not available Not available Not available 16,384 B 16,384 B 16,384 B 16,384 B
```
```
Largest number of levels a ray-tracing intersector can
traverse in an acceleration structure^12
32 32 Not available Not available Not available Not available 32 32 32 32 32
```
```
Largest number of levels a ray-tracing intersection
query can traverse in an acceleration structure^12
16 16 Not available Not available Not available Not available 16 16 16 16 16
```
```
Maximum texture view pool entries Not available 128 million Not available Not available Not available Not available Not available 128 million 128 million 256 million Not available
```
```
Maximum supported tensor rank Not available 16 Not available Not available Not available Not available Not available 16 16 16 Not available
```
```
Maximum supported tensor stride at dimension index
0 for machine learning encoder usage
Not available 1 element Not available Not available Not available Not available Not available 1 element 1 element 1 element Not available
```
```
Minimum alignment of tensor stride at dimension
index 1 for machine learning encoder usage
Not available 64 B Not available Not available Not available Not available Not available 64 B 64 B 64 B Not available
```
```
Maximum performance counter heaps (per process) Not available 32 Not available Not available Not available Not available Not available 32 32 32 Not available
```
```
Minimum alignment of intersection function buffer Not available 64 B Not available Not available Not available Not available Not available 64 B 64 B 64 B Not available
```
```
Minimum alignment of intersection function buffer
stride
Not available 8 B Not available Not available Not available Not available Not available 8 B 8 B 8 B Not available
```
```
Maximum size of intersection function buffer stride Not available 4096 B Not available Not available Not available Not available Not available 4096 B 4096 B 4096 B Not available
```
**GPU family**^1 **Metal3 Metal4 Apple2 Apple3 Apple4 Apple5 Apple6 Apple7 Apple8 Apple9 Mac**

1. See MTLGPUFamily for each GPU family’s enumeration constant.
2. These values are identical to the maximum number of bindings in an MTL4ArgumentTable of the same type.
3. Inline constexpr samplers that you declare in Metal Shading Language (MSL) code count toward the limit. For example, for a feature set limit of 16, you can have 12 API samplers and 4 language samplers (16 total), but you can’t
    have 12 API samplers and 6 language samplers (18 total).
4. The values in this row are the theoretical maximum number of threads per threadgroup. Check the actual maximum by inspecting the MTLComputePipelineState.maxTotalThreadsPerThreadgroup property at runtime.
5. You can allocate memory between imageblock and threadgroup memory, but the sum of these allocations can’t exceed the maximum total image block memory limit. Some feature sets can’t access image block memory directly,
    but they can access threadgroup memory. Which image block memory limit applies depends on the shaders usage of either implicit or explicit image block layout, see the Metal Shading Language specification for details.
6. A vector counts as _n_ scalars, where _n_ is the number of components in the vector. The iOS and tvOS feature sets only reach the maximum number of inputs if you don’t exceed the maximum number of input components. For example,
    you can have 60 float inputs (components), but you can’t have 60 float4 inputs, which total 240 components.
7. The limits apply to the items you place in the argument buffers you bind directly and in the argument buffers you can access indirectly through your bound argument buffers.
8. The maximum texture buffer width, in pixels, is also limited by MTLDevice.maxBufferLength divided by the size of a pixel, in bytes; as well as available memory.
9. Some GPU devices in the Mac2 family support vertex amplification. You can check an individual GPU’s support for this feature by calling its MTLDevice.supportsVertexAmplificationCount(_:) method at runtime.

10.Mesh shaders can use up to 4 GB of payload and mesh geometry per draw for devices in the Apple7 and Apple8 GPU families.

11.Mesh shaders that have a [[threadgroups_per_grid]] or [[threads_per_grid]] parameter reduce the available payload size by 16 bytes. Viewing a mesh shader’s geometry in the Metal debugger (within Xcode) reduces
the available payload by 16 bytes. The total payload size reduction can be 32 bytes.

12.The value includes one level for the primitive acceleration structure, which leaves the remaining levels for instance acceleration structures.


**Texture capabilities by pixel format**

**GPU family**^1 **Metal3 Metal4 Apple2 Apple3 Apple4 Apple5 Apple6 Apple7 Apple8 Apple9 Mac**

```
Ordinary 8-bit pixel formats Texture capabilities for ordinary 8-bit pixel formats by GPU family
```
```
A8Unorm 2,^9 All All Filter All All All All All All All All
```
```
R8Unorm  2 All All All All All All All All All All All
```
```
R8Unorm_sRGB Not available All All All All All All All All All Not available
```
```
R8Snorm All All All All All All All All All All All
```
```
R8Uint  
R8Sint  
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
```
Ordinary 16-bit pixel formats Texture capabilities for ordinary 16-bit pixel formats by GPU family
```
```
R16Unorm
R16Snorm
```
```
All All
```
```
Filter
Write
Color
MSAA
Blend
```
```
Filter
Write
Color
MSAA
Blend
```
```
All All All All All All All
```
```
R16Uint  
R16Sint  
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
```
R16Float  2 All All All All All All All All All All All
```
```
RG8Unorm All All All All All All All All All All All
```
This table lists the GPU’s texture capabilities for each pixel format:

- **Atomic** : The GPU can use atomic operations on textures with the pixel format.
- **All** : The GPU has the following texture capabilities for the pixel format:
    - **Filter** : The GPU can filter a texture with the pixel format during sampling.
    - **Write** : The GPU can write to a texture on a per-pixel basis with the pixel format.^2
    - **Color** : The GPU can use a texture with the pixel format as a color render target.
    - **Blend** : The GPU can blend a texture with the pixel format.
    - **MSAA** : The GPU can use a texture with the pixel format as a destination for multisample antialias (MSAA) data.
    - **Sparse** : The GPU supports sparse-texture allocations for textures with the pixel format.
       **Sparse** is not included in **All** for the Mac2, Metal3 and Apple2 through Apple6 family columns, because those GPUs do not support the sparse texture feature.
    - **Resolve** : The GPU can use a texture with the pixel format as a source for multisample antialias (MSAA) resolve operations.

**Note**
All graphics and compute kernels can read or sample a texture with any pixel format.


RG8Unorm_sRGB Not available All All All All All All All All All Not available

RG8Snorm All All All All All All All All All All All

RG8Uint
RG8Sint

```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
**Packed 16-bit pixel formats**^7 Texture capabilities for **packed 16-bit pixel formats** by GPU family

B5G6R5Unorm
A1BGR5Unorm
ABGR4Unorm
BGR5A1Unorm

```
Not available
```
```
Filter
Color
MSAA
Resolve
Blend
Sparse
```
```
Filter
Color
MSAA
Resolve
Blend
```
```
Filter
Color
MSAA
Resolve
Blend
```
```
Filter
Color
MSAA
Resolve
Blend
```
```
Filter
Color
MSAA
Resolve
Blend
```
```
Filter
Color
MSAA
Resolve
Blend
Sparse
```
```
Filter
Color
MSAA
Resolve
Blend
Sparse
```
```
Filter
Color
MSAA
Resolve
Blend
Sparse
```
```
Filter
Color
MSAA
Resolve
Blend
Sparse
```
```
Not available
```
**Ordinary 32-bit pixel formats** Texture capabilities for **ordinary 32-bit pixel formats** by GPU family

R32Uint  
R32Sint  

```
Atomic
```
```
Write
Color
```
```
Atomic
```
```
Write
Color
Sparse
```
```
Write
Color
```
```
Write
Color
```
```
Write
Color
```
```
Write
Color
```
```
Atomic
```
```
Write
Color
Sparse
```
```
Atomic
```
```
Write
Color
Sparse
```
```
Atomic
```
```
Write
Color
Sparse
```
```
Atomic
```
```
Write
Color
Sparse
```
```
Atomic
```
```
Write
Color
MSAA
```
R32Float 2,

```
Write
Color
MSAA
Blend
```
```
Write
Color
MSAA
Blend
Sparse
```
```
Write
Color
MSAA
Blend
```
```
Write
Color
MSAA
Blend
```
```
Write
Color
MSAA
Blend
```
```
Write
Color
MSAA
Blend
```
```
Write
Color
MSAA
Blend
Sparse
```
```
Write
Color
MSAA
Blend
Sparse
```
```
Write
Color
MSAA
Blend
Sparse
```
```
All All
```
RG16Unorm
RG16Snorm

```
All All
```
```
Filter
Write
Color
MSAA
Blend
```
```
Filter
Write
Color
MSAA
Blend
```
```
All All All All All All All
```
RG16Uint
RG16Sint

```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
RG16Float All All All All All All All All All All All

RGBA8Unorm  2 All All All All All All All All All All All

**GPU family**^1 **Metal3 Metal4 Apple2 Apple3 Apple4 Apple5 Apple6 Apple7 Apple8 Apple9 Mac**


RGBA8Unorm_sRGB

```
Filter
Color
MSAA
Resolve
Blend
```
```
All All All All All All All All All
```
```
Filter
Color
MSAA
Resolve
Blend
```
RGBA8Snorm All All All All All All All All All All All

RGBA8Uint  
RGBA8Sint  

```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
BGRA8Unorm All All All All All All All All All All All

BGRA8Unorm_sRGB

```
Filter
Color
MSAA
Resolve
Blend
```
```
All All All All All All All All All
```
```
Filter
Color
MSAA
Resolve
Blend
```
**Packed 32-bit pixel formats** Texture capabilities for **packed 32-bit pixel formats** by GPU family

RGB10A2Unorm All All

```
Filter
Color
MSAA
Resolve
Blend
```
```
All All All All All All All All
```
BGR10A2Unorm All All All All All All All All All All All

RGB10A2Uint

```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
RG11B10Float  7 All All

```
Filter
Color
MSAA
Resolve
Blend
```
```
All All All All All All All All
```
RGB9E5Float  7 Filter All

```
Filter
Color
MSAA
Resolve
Blend
```
```
All All All All All All All Filter
```
**Ordinary 64-bit pixel formats** Texture capabilities for **ordinary 64-bit pixel formats** by GPU family

**GPU family**^1 **Metal3 Metal4 Apple2 Apple3 Apple4 Apple5 Apple6 Apple7 Apple8 Apple9 Mac**


RG32Uint  
RG32Sint

```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
```
```
Write
Color
```
```
Write
Color
```
```
Write
Color
```
```
Write
Color
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Atomic
```
```
Write
Color
MSAA
Sparse
```
```
Atomic
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
RG32Float  

```
Write
Color
MSAA
Blend
```
```
Write
Color
MSAA
Blend
Sparse
```
```
Write
Color
Blend
```
```
Write
Color
Blend
```
```
Write
Color
Blend
```
```
Write
Color
Blend
```
```
Write
Color
Blend
Sparse
```
```
Write
Color
MSAA
Blend
Sparse
```
```
Write
Color
MSAA
Blend
Sparse
```
```
All All
```
RGBA16Unorm
RGBA16Snorm

```
All All
```
```
Filter
Write
Color
MSAA
Blend
```
```
Filter
Write
Color
MSAA
Blend
```
```
All All All All All All All
```
RGBA16Uint  
RGBA16Sint  

```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
Sparse
```
```
Write
Color
MSAA
```
RGBA16Float  2 All All All All All All All All All All All

**Ordinary 128-bit pixel formats** Texture capabilities for **ordinary 128-bit pixel formats** by GPU family

RGBA32Uint  
RGBA32Sint  

```
Write
Color
```
```
Write
Color
Sparse
```
```
Write
Color
```
```
Write
Color
```
```
Write
Color
```
```
Write
Color
```
```
Write
Color
Sparse
```
```
Write
Color
Sparse
```
```
Write
Color
Sparse
```
```
Write
Color
Sparse
```
```
Write
Color
MSAA
```
RGBA32Float 2,

```
Write
Color
MSAA
Blend
```
```
Write
Color
MSAA
Blend
Sparse
```
```
Write
Color
Blend
```
```
Write
Color
Blend
```
```
Write
Color
Blend
```
```
Write
Color
Blend
```
```
Write
Color
Blend
Sparse
```
```
Write
Color
MSAA
Blend
Sparse
```
```
Write
Color
MSAA
Blend
Sparse
```
```
All All
```
**Compressed pixel formats**^7 Texture capabilities for **compressed pixel formats** by GPU family

PVRTC pixel formats  3 Not available
Filter
Sparse
Filter Filter Filter Filter
Filter
Sparse

```
Filter
Sparse
```
```
Filter
Sparse
```
```
Filter
Sparse
Not available
```
EAC/ETC pixel formats Not available

```
Filter
Sparse
Filter Filter Filter Filter
Filter
Sparse
```
```
Filter
Sparse
```
```
Filter
Sparse
```
```
Filter
Sparse
Not available
```
ASTC pixel formats Not available

```
Filter
Sparse
Filter Filter Filter Filter
Filter
Sparse
```
```
Filter
Sparse
```
```
Filter
Sparse
```
```
Filter
Sparse
Not available
```
HDR ASTC pixel formats Not available

```
Filter
Sparse
Not available Not available Not available Not available
Filter
Sparse
```
```
Filter
Sparse
```
```
Filter
Sparse
```
```
Filter
Sparse
Not available
```
BC pixel formats Varies^8 Varies^8 Not available Not available Not available Not available Not available Varies^8 Varies^8

```
Filter
Sparse
Filter
```
**YUV pixel formats** 4, 7 Texture capabilities for **YUV pixel formats** by GPU family

**GPU family**^1 **Metal3 Metal4 Apple2 Apple3 Apple4 Apple5 Apple6 Apple7 Apple8 Apple9 Mac**


### GBGR

### BGRG

```
Filter Filter Filter Filter Filter Filter Filter Filter Filter Filter Filter
```
```
Depth and stencil pixel formats  7 Texture capabilities for depth and stencil pixel formats by GPU family
```
```
Depth16Unorm
```
```
Filter
MSAA
Resolve
```
```
Filter
MSAA
Resolve
Sparse  
```
```
Filter
MSAA
```
```
Filter
MSAA
Resolve
```
```
Filter
MSAA
Resolve
```
```
Filter
MSAA
Resolve
```
```
Filter
MSAA
Resolve
```
```
Filter
MSAA
Resolve
Sparse  
```
```
Filter
MSAA
Resolve
Sparse
```
```
Filter
MSAA
Resolve
Sparse
```
```
Filter
MSAA
Resolve
```
```
Depth32Float  
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
Sparse  
```
### MSAA

### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
Sparse  
```
### MSAA

```
Resolve
Sparse
```
```
Filter
MSAA
Resolve
Sparse
```
```
Filter
MSAA
Resolve
```
```
Stencil8 Not available
```
### MSAA

```
Resolve
Sparse  
```
### MSAA

### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
Sparse  
```
### MSAA

```
Resolve
Sparse
```
### MSAA

```
Resolve
Sparse
```
```
Not available
```
```
Depth24Unorm_Stencil8  5 Not available Not available Not available Not available Not available Not available Not available Not available Not available Not available
```
```
Filter
MSAA
Resolve
```
```
Depth32Float_Stencil
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
### MSAA

```
Resolve
```
```
Filter
MSAA
Resolve
```
```
Filter
MSAA
Resolve
```
```
X24_Stencil8 Not available Not available Not available Not available Not available Not available Not available Not available Not available Not available MSAA
```
```
X32_Stencil8 MSAA MSAA MSAA MSAA MSAA MSAA MSAA MSAA MSAA MSAA MSAA
```
```
Extended range and wide color pixel formats Texture capabilities for extended range and wide color formats by GPU family
```
### BGRA10_XR

```
BGRA10_XR_sRGB
BGR10_XR
BGR10_XR_sRGB
```
```
Not available All Not available All All All All All All All Not available
```
**GPU family**^1 **Metal3 Metal4 Apple2 Apple3 Apple4 Apple5 Apple6 Apple7 Apple8 Apple9 Mac**

1. See MTLGPUFamily for each GPU family’s enumeration constant.
2. Some GPUs support read-write textures where a kernel can both read from and write to a texture. You can check an individual GPU’s support for this feature by inspecting its
    MTLDevice.readWriteTextureSupport property at runtime.
3. Only the GPUs in Apple3 and Apple4 families support MTLSamplerAddressMode.clampToZero for the PVRTC pixel formats.
4. The GPUs in Apple6 through Apple9 families don’t support sparse textures with YUV pixel formats.
5. Some GPUs support MTLPixelFormat.depth24Unorm_stencil8. You can check an individual GPU’s support for this feature by inspecting its
    MTLDevice.isDepth24Stencil8PixelFormatSupported property at runtime.
6. Some GPUs in the Apple7, and Apple8 families additionally support the **Filter** and **Resolve** texture capabilities for 32-bit floating-point pixel formats in iPadOS. You can check an individual GPU’s support for this
    feature by inspecting the MTLDevice.supports32BitFloatFiltering property at runtime.
7. Formats in this group aren’t compatible with lossy texture compression through MTLTextureDescriptor.compressionType.
8. Some GPU devices in the Apple7 and Apple8 families support filtering and sparse BC compressed textures in iPadOS. You can check an individual GPU’s support for this feature by inspecting its
    MTLDevice.supportsBCTextureCompression property at runtime.
9. The A8Unorm pixel format is incompatible with imageblocks with explicit layout. Use either an R8Unorm texture view, or imageblocks with implicit layout.

10.You can only apply the RG32Uint format to a ulong texture on a GPU that supports the 64-bit atomics feature.

11.GPU devices in the Apple7 family support **Sparse** depth and stencil textures only for placement sparse textures. GPU devices in Apple8 through Apple9 support both placement and automatic heap backing for
sparse depth and stencil textures.


```
Ordinary 32-bit pixel formats
```
```
Format Access
```
```
R32Uint
R32Sint
```
```
All  
```
```
R32Float All
```
```
RG16Unorm
RG16Snorm
```
```
Read
Write
```
```
RG16Uint
RG16Sint
```
```
Read
Write
```
```
RG16Float
```
```
Read
Write
```
```
RGBA8Unorm All
```
```
RGBA8Snorm
```
```
Read
Write
```
```
RGBA8Uint
RGBA8Sint
```
```
All
```
```
BGRA8Unorm Read
```
These tables list the pixel formats that texture buffers support, and the GPU’s read/write access to textures with those formats:

- **All** : The GPU can use the following accesses for a texture buffer with the pixel format:
    - **Read** : The GPU can use read access for a texture buffer with the pixel format.
    - **Write** : The GPU can use write access for a texture buffer with the pixel format.
    - **Read/Write** : The GPU can use read_write access for a texture buffer with the pixel format.  

**Note**
The GPU capabilities are generally the same across all hardware families, but some GPUs have additional options.  

```
Packed 32-bit pixel formats
```
```
Format Access
```
```
RGB10A2Unorm
```
```
Read
Write
```
```
RGB10A2Uint
```
```
Read
Write
```
```
RG11B10Float
```
```
Read
Write
```
**Texture buffer pixel formats**

1. GPUs with the Tier 2 feature set support read_write access to textures. You can check an individual GPU’s support for this feature by inspecting its
    MTLDevice.readWriteTextureSupport property at runtime.
2. Some devices support this pixel format. Check a device by inspecting its MTLDevice.depth24Stencil8PixelFormatSupported property at runtime.
3. GPUs that support texture atomics (see feature availability by GPU family) also support atomics in read/write texture buffers with this pixel format.

```
Ordinary 16-bit pixel formats
```
```
Format Access
```
```
R16Unorm
R16Snorm
```
```
Read
Write
```
```
R16Uint
R16Sint
```
```
All
```
```
R16Float All
```
```
RG8Unorm
```
```
Read
Write
```
```
RG8Snorm
```
```
Read
Write
```
```
RG8Uint
RG8Sint
```
```
Read
Write
```
```
Ordinary 8-bit pixel formats
```
```
Format Access
```
```
A8Unorm All
```
```
R8Unorm All
```
```
R8Snorm
```
```
Read
Write
```
```
R8Uint
R8Sint
```
```
All Ordinary 64-bit pixel formats
```
```
Format Access
```
```
RG32Uint
RG32Sint
```
```
Read
Write
```
```
RG32Float
```
```
Read
Write
```
```
RGBA16Unorm
RGBA16Snorm
```
```
Read
Write
```
```
RGBA16Uint
RGBA16Sint
```
```
All
```
```
RGBA16Float All
```
```
Ordinary 128-bit pixel formats
```
```
Format Access
```
```
RGBA32Uint
RGBA32Sint
```
```
All
```
```
RGBA32Float All
```


Apple Inc.
Copyright © 2014-2025 Apple Inc.
All rights reserved.

No part of this publication may be reproduced, stored in a retrieval system, or
transmitted, in any form or by any means, mechanical, electronic,
photocopying, recording, or otherwise, without prior written permission of
Apple Inc., with the following exceptions: Any person is hereby authorized to
store documentation on a single computer or device for personal use only and
to print copies of documentation for personal use provided that the
documentation contains Apple’s copyright notice.

No licenses, express or implied, are granted with respect to any of the
technology described in this document. Apple retains all intellectual property
rights associated with the technology described in this document. This
document is intended to assist application developers to develop applications
only for Apple-branded products.

Apple Inc.
One Apple Park Way
Cupertino, CA 95014

Apple is a trademark of Apple Inc., registered in the U.S. and other countries.

**APPLE MAKES NO WARRANTY OR REPRESENTATION, EITHER EXPRESS
OR IMPLIED, WITH RESPECT TO THIS DOCUMENT, ITS QUALITY,
ACCURACY, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR
PURPOSE. AS A RESULT, THIS DOCUMENT IS PROVIDED “AS IS,” AND
YOU, THE READER, ARE ASSUMING THE ENTIRE RISK AS TO ITS
QUALITY AND ACCURACY.**

**IN NO EVENT WILL APPLE BE LIABLE FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES RESULTING FROM ANY
DEFECT, ERROR OR INACCURACY IN THIS DOCUMENT, even if advised of
the possibility of such damages.**

**Some jurisdictions do not allow the exclusion of implied warranties or
liability, so the above exclusion may not apply to you.**

```
Page 15 of 15
```

