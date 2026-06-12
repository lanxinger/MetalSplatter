// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MetalSplatter",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
        .visionOS(.v1),
    ],
    products: [
        .library(
            name: "PLYIO",
            targets: [ "PLYIO" ]
        ),
        .library(
            name: "SplatIO",
            targets: [ "SplatIO" ]
        ),
        .library(
            name: "MetalSplatter",
            targets: [ "MetalSplatter" ]
        ),
        .library(
            name: "SampleBoxRenderer",
            targets: [ "SampleBoxRenderer" ]
        ),
        .executable(
            name: "SplatConverter",
            targets: [ "SplatConverter" ]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.7.0"),
        .package(url: "https://github.com/weichsel/ZIPFoundation", from: "0.9.20"),
        .package(url: "https://github.com/the-swift-collective/libwebp.git", from: "1.4.1"),
        .package(url: "https://github.com/facebook/zstd.git", exact: "1.5.7")
    ],
    targets: [
        .target(
            name: "PLYIO",
            path: "PLYIO",
            exclude: [ "Tests", "TestData" ],
            sources: [ "Sources" ]
        ),
        .testTarget(
            name: "PLYIOTests",
            dependencies: [ "PLYIO" ],
            path: "PLYIO",
            exclude: [ "Sources" ],
            sources: [ "Tests" ],
            resources: [ .copy("TestData") ]
        ),
        .target(
            name: "SplatIO",
            dependencies: [
                "PLYIO",
                .product(name: "ZIPFoundation", package: "ZIPFoundation"),
                .product(name: "libwebp", package: "libwebp"),
                .product(name: "libzstd", package: "zstd")
            ],
            path: "SplatIO",
            exclude: [ "Tests", "TestData" ],
            sources: [ "Sources" ]
        ),
        .testTarget(
            name: "SplatIOTests",
            dependencies: [ "SplatIO" ],
            path: "SplatIO",
            exclude: [ "Sources" ],
            sources: [ "Tests" ],
            resources: [ .copy("TestData") ]
        ),
        .target(
            name: "MetalSplatter",
            dependencies: [ "PLYIO", "SplatIO" ],
            path: "MetalSplatter",
            exclude: [ "Tests" ],
            sources: [ "Sources" ],
            resources: [
                .process("Resources"),
                // Shipped as source, not precompiled: requires MSL 4.1, which is
                // newer than the deployment-target-pinned language version the
                // build-time Metal compiler uses. Metal4Sorter compiles it at
                // runtime on OS versions whose compiler supports MSL 4.1.
                .copy("RuntimeShaders/OneSweepSort.metal"),
            ]
        ),
        .testTarget(
            name: "MetalSplatterTests",
            dependencies: [ "MetalSplatter" ],
            path: "MetalSplatter",
            exclude: [ "Sources", "Resources", "RuntimeShaders" ],
            sources: [ "Tests" ]
        ),
        .target(
            name: "SampleBoxRenderer",
            path: "SampleBoxRenderer",
            sources: [ "Sources" ],
            resources: [ .process("Resources") ]
        ),
        .executableTarget(
            name: "SplatConverter",
            dependencies: [
                "SplatIO",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "SplatConverter",
            sources: [ "Sources" ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
