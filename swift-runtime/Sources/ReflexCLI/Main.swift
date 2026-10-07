import Foundation
import MLX
import Tokenizers

@main
struct Main {
    static func emit(_ value: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([10]))
    }

    static func main() async throws {
        Memory.cacheLimit = 64 * 1024 * 1024
        let args = CommandLine.arguments
        guard args.count == 3, ["compiled", "compiled-exact", "compiled-fast", "eager"].contains(args[2]) else {
            throw EmbeddingError.invalid("Usage: reflex-embed MODEL_DIRECTORY compiled|compiled-exact|eager")
        }
        let directory = URL(fileURLWithPath: args[1])
        let compiled = args[2] != "eager"
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let model = try TextModel(directory: directory, fastKernels: args[2] == "compiled-fast")
        try emit(["ready": true, "compiled": compiled])
        while let line = readLine() {
            let start = Date()
            do {
                guard let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                      let texts = request["texts"] as? [String], !texts.isEmpty, texts.count <= 64 else {
                    throw EmbeddingError.invalid("Expected 1...64 raw text strings")
                }
                let rows = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
                guard rows.allSatisfy({ !$0.isEmpty && $0.count <= 8192 }) else {
                    throw EmbeddingError.invalid("Invalid text length; silent truncation forbidden")
                }
                let actual = rows.map(\.count).max()!
                let length = args[2] == "compiled" ? ((actual + 15) / 16) * 16 : actual
                let ids = rows.map { $0 + Array(repeating: 0, count: length - $0.count) }
                let mask = rows.map { Array(repeating: 1, count: $0.count) + Array(repeating: 0, count: length - $0.count) }
                let values = try model.encode(ids: ids, mask: mask, dimensions: 768, compiled: compiled).embeddings.asArray(Float.self)
                guard values.allSatisfy(\.isFinite) else { throw EmbeddingError.invalid("Nonfinite embedding") }
                let vectors = stride(from: 0, to: values.count, by: 768).map { Array(values[$0..<$0 + 768]) }
                try emit(["embeddings": vectors, "seconds": Date().timeIntervalSince(start),
                          "compiled": compiled, "padded_tokens": length, "mlx_peak_bytes": Memory.peakMemory])
            } catch {
                try emit(["error": String(describing: error)])
            }
        }
    }
}
