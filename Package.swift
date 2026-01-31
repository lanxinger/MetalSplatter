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
        .package(url: "https://github.com/weichsel/ZIPFoundation", from: "0.9.20")
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
                .product(name: "ZIPFoundation", package: "ZIPFoundation")
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
            resources: [ .process("Resources") ]
        ),
        .testTarget(
            name: "MetalSplatterTests",
            dependencies: [ "MetalSplatter" ],
            path: "MetalSplatter",
            exclude: [ "Sources", "Resources" ],
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
