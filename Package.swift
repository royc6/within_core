// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "WithinCore",
    platforms: [.iOS("27.0"), .macOS("27.0")],
    products: [
        // The engine: the chunked decoder on Core AI and the generation loop.
        .library(name: "WithinCore", targets: ["WithinCore"]),
        // Platform-independent parts: the package format, host-side tables and schedules.
        .library(name: "WithinCoreBase", targets: ["WithinCoreBase"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/coreai-models", revision: "df8119879f125ad1e2e4c6249c2cddded75c190a"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.0"),
    ],
    targets: [
        .target(name: "WithinCoreBase"),
        .target(
            name: "WithinCore",
            dependencies: [
                "WithinCoreBase",
                .product(name: "CoreAILM", package: "coreai-models"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            swiftSettings: [.enableExperimentalFeature("Lifetimes")]),
        .testTarget(name: "WithinCoreBaseTests", dependencies: ["WithinCoreBase"], resources: [.copy("Fixtures")]),
    ]
)
