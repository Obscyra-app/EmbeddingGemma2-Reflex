// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "ReflexRuntime",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "reflex-embed", targets: ["ReflexCLI"])],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.0")
    ],
    targets: [.executableTarget(name: "ReflexCLI", dependencies: [
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "Tokenizers", package: "swift-transformers")
    ])]
)
