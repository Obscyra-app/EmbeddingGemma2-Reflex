// Derived from EmbeddingGemma2Swift (MIT). Inference weight loader only.
import Foundation
import MLX

struct QuantizationMetadata: Codable {
    let version: Int
    let mode: String
    let bits, groupSize: Int
    let shapes: [String:[Int]]
    static func load(_ directory:URL) throws -> Self? {
        let url=directory.appendingPathComponent("quantization.json")
        guard FileManager.default.fileExists(atPath:url.path) else { return nil }
        let q=try JSONDecoder().decode(Self.self,from:Data(contentsOf:url))
        guard q.version==1,q.mode=="affine",[4,8].contains(q.bits),q.groupSize==64,!q.shapes.isEmpty else { throw EmbeddingError.invalid("Unsupported quantization metadata") }
        return q
    }
}

/// Keeps packed matrices resident; only requested embedding rows are dequantized.
struct WeightStore {
    let values: [String:MLXArray]
    let quantization: QuantizationMetadata?
    let dtype: DType
    init(loaded:[String:MLXArray], shapes:[String:[Int]], directory:URL, dtype:DType) throws {
        self.dtype=dtype
        let q=try QuantizationMetadata.load(directory);quantization=q
        let packed=q?.shapes ?? [:]
        guard Set(packed.keys).isSubset(of:Set(shapes.keys)) else { throw EmbeddingError.invalid("Unknown quantized tensors") }
        var keys=Set(shapes.keys),selected=[String:MLXArray]()
        for key in packed.keys { keys.insert(key+".scales");keys.insert(key+".biases") }
        guard Set(loaded.keys)==keys else { throw EmbeddingError.invalid("Weight keys mismatch") }
        for (key,shape) in shapes {
            let value=loaded[key]!
            if let original=packed[key],let q {
                guard original==shape,shape.count==2,shape[1]%q.groupSize==0,
                      value.shape==[shape[0],shape[1]*q.bits/32],value.dtype == .uint32 else { throw EmbeddingError.invalid("Invalid packed tensor: \(key)") }
                selected[key]=value
                for suffix in [".scales",".biases"] {
                    let a=loaded[key+suffix]!
                    guard a.shape==[shape[0],shape[1]/q.groupSize],a.dtype == .bfloat16 || a.dtype == .float32 else { throw EmbeddingError.invalid("Invalid quantization scales/biases: \(key)") }
                    selected[key+suffix]=a.asType(dtype)
                }
            } else {
                guard value.shape==shape,value.dtype == .bfloat16 || value.dtype == .float32 else { throw EmbeddingError.invalid("Invalid tensor shape/precision: \(key)") }
                selected[key]=value.asType(dtype)
            }
        }
        values=selected;eval(Array(selected.values))
    }
    subscript(_ key:String) -> MLXArray? { values[key] }
    func linear(_ x:MLXArray,_ key:String) -> MLXArray {
        if let q=quantization,q.shapes[key] != nil {
            return quantizedMM(x,values[key]!,scales:values[key+".scales"]!,biases:values[key+".biases"]!,transpose:true,groupSize:q.groupSize,bits:q.bits)
        }
        return matmul(x,values[key]!.T)
    }
    func embedding(_ ids:MLXArray,_ key:String) -> MLXArray {
        if let q=quantization,let shape=q.shapes[key] {
            let rows=ids.reshaped(-1)
            return dequantized(values[key]![rows],scales:values[key+".scales"]![rows],biases:values[key+".biases"]![rows],groupSize:q.groupSize,bits:q.bits,dtype:dtype).reshaped(ids.shape+[shape[1]])
        }
        return values[key]![ids]
    }
    var format:String { quantization.map { "affine\($0.bits)-g\($0.groupSize)" } ?? "unquantized" }
}

