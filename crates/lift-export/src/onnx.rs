// ============================================================================
// LIFT → ONNX Export
// ============================================================================
//
// Generates ONNX-compatible protobuf text format (.onnx.pbtxt) and binary-
// equivalent JSON representation (.onnx.json) from the LIFT IR.
//
// ONNX (Open Neural Network Exchange) is the industry-standard interchange
// format for neural network models, supported by PyTorch, TensorFlow,
// TensorRT, ONNX Runtime, and many others.
//
// The exporter maps LIFT tensor operations to ONNX operator set opset 21.
// ============================================================================

use lift_core::context::Context;
use lift_core::types::{CoreType, TypeData, DataType, Dimension};
use thiserror::Error;
use std::fmt::Write;

#[derive(Debug, Error)]
pub enum OnnxExportError {
    #[error("Unsupported operation for ONNX export: {0}")]
    UnsupportedOp(String),
    #[error("Export error: {0}")]
    General(String),
}

/// ONNX data type enum values (from onnx.proto TensorProto.DataType)
#[derive(Debug, Clone, Copy)]
#[repr(i32)]
enum OnnxDataType {
    Float = 1,
    // Uint8 = 2,
    Int8 = 3,
    // Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    // String = 8,
    Bool = 9,
    Float16 = 10,
    Double = 11,
    // Uint32 = 12,
    // Uint64 = 13,
    // Complex64 = 14,
    // Complex128 = 15,
    BFloat16 = 16,
    // Float8E4M3FN = 17,
    // Float8E5M2 = 19,
}

fn lift_dtype_to_onnx(dt: &DataType) -> OnnxDataType {
    match dt {
        DataType::FP64 => OnnxDataType::Double,
        DataType::FP32 => OnnxDataType::Float,
        DataType::FP16 => OnnxDataType::Float16,
        DataType::BF16 => OnnxDataType::BFloat16,
        DataType::FP8E4M3 | DataType::FP8E5M2 => OnnxDataType::Float16,
        DataType::INT64 => OnnxDataType::Int64,
        DataType::INT32 => OnnxDataType::Int32,
        DataType::INT16 => OnnxDataType::Int16,
        DataType::INT8 | DataType::INT4 | DataType::INT2 => OnnxDataType::Int8,
        DataType::UINT8 => OnnxDataType::Int8,
        DataType::Bool => OnnxDataType::Bool,
        DataType::Index => OnnxDataType::Int64,
    }
}

/// Map a LIFT tensor op name to ONNX op type and domain.
fn lift_op_to_onnx(op_name: &str) -> Option<(&str, &str)> {
    // Returns (onnx_op_type, domain)
    // domain "" = standard ONNX domain
    // domain "com.microsoft" = Microsoft extensions
    let mapping = match op_name {
        // Arithmetic
        "tensor.add" => ("Add", ""),
        "tensor.sub" => ("Sub", ""),
        "tensor.mul" => ("Mul", ""),
        "tensor.div" => ("Div", ""),
        "tensor.neg" => ("Neg", ""),
        "tensor.abs" => ("Abs", ""),
        "tensor.sqrt" => ("Sqrt", ""),
        "tensor.exp" => ("Exp", ""),
        "tensor.log" => ("Log", ""),
        "tensor.pow" => ("Pow", ""),
        "tensor.floor" => ("Floor", ""),
        "tensor.ceil" => ("Ceil", ""),
        "tensor.clip" => ("Clip", ""),
        "tensor.sin" => ("Sin", ""),
        "tensor.cos" => ("Cos", ""),

        // MatMul / Gemm
        "tensor.matmul" => ("MatMul", ""),
        "tensor.linear" => ("Gemm", ""),

        // Activations
        "tensor.relu" => ("Relu", ""),
        "tensor.gelu" => ("Gelu", ""),
        "tensor.silu" => ("Sigmoid", ""), // SiLU = x * sigmoid(x), approximated
        "tensor.sigmoid" => ("Sigmoid", ""),
        "tensor.tanh" => ("Tanh", ""),
        "tensor.softmax" => ("Softmax", ""),
        "tensor.leaky_relu" => ("LeakyRelu", ""),
        "tensor.elu" => ("Elu", ""),

        // Normalisation
        "tensor.layernorm" => ("LayerNormalization", ""),
        "tensor.rmsnorm" => ("SimplifiedLayerNormalization", "com.microsoft"),
        "tensor.batchnorm" => ("BatchNormalization", ""),
        "tensor.groupnorm" => ("GroupNormalization", ""),

        // Convolution
        "tensor.conv1d" => ("Conv", ""),
        "tensor.conv2d" => ("Conv", ""),
        "tensor.conv3d" => ("Conv", ""),

        // Pooling
        "tensor.maxpool2d" => ("MaxPool", ""),
        "tensor.avgpool2d" => ("AveragePool", ""),
        "tensor.global_avgpool" => ("GlobalAveragePool", ""),
        "tensor.global_maxpool" => ("GlobalMaxPool", ""),

        // Shape manipulation
        "tensor.reshape" => ("Reshape", ""),
        "tensor.transpose" => ("Transpose", ""),
        "tensor.concat" => ("Concat", ""),
        "tensor.flatten" => ("Flatten", ""),
        "tensor.squeeze" => ("Squeeze", ""),
        "tensor.unsqueeze" => ("Unsqueeze", ""),
        "tensor.gather" => ("Gather", ""),
        "tensor.scatter" => ("ScatterElements", ""),
        "tensor.split" => ("Split", ""),
        "tensor.slice" => ("Slice", ""),
        "tensor.pad" => ("Pad", ""),

        // Embedding / Reduce
        "tensor.embedding" => ("Gather", ""),
        "tensor.reduce_sum" => ("ReduceSum", ""),
        "tensor.reduce_mean" => ("ReduceMean", ""),
        "tensor.reduce_max" => ("ReduceMax", ""),

        // Attention (use Microsoft extensions)
        "tensor.attention" => ("Attention", "com.microsoft"),
        "tensor.multi_head_attention" => ("MultiHeadAttention", "com.microsoft"),
        "tensor.grouped_query_attention" => ("GroupQueryAttention", "com.microsoft"),
        "tensor.flash_attention" => ("MultiHeadAttention", "com.microsoft"),
        "tensor.cross_attention" => ("MultiHeadAttention", "com.microsoft"),
        "tensor.sliding_window_attention" => ("MultiHeadAttention", "com.microsoft"),
        "tensor.paged_attention" => ("Attention", "com.microsoft"),

        // Quantization
        "tensor.quantize" => ("QuantizeLinear", ""),
        "tensor.dequantize" => ("DequantizeLinear", ""),

        // MoE
        "tensor.moe_dispatch" => ("MoE", "com.microsoft"),
        "tensor.moe_combine" => ("MoE", "com.microsoft"),

        // Fused (map to subgraph or ms domain)
        "tensor.fused_matmul_bias_relu" => ("FusedMatMul", "com.microsoft"),
        "tensor.fused_matmul_bias" => ("FusedMatMul", "com.microsoft"),
        "tensor.fused_linear_gelu" => ("FusedMatMul", "com.microsoft"),
        "tensor.fused_linear_silu" => ("FusedMatMul", "com.microsoft"),

        // Dropout
        "tensor.dropout" => ("Dropout", ""),

        _ => return None,
    };
    Some(mapping)
}

#[derive(Debug)]
pub struct OnnxExporter {
    opset_version: i64,
    ir_version: i64,
    producer: String,
}

impl OnnxExporter {
    pub fn new() -> Self {
        Self {
            opset_version: 21,
            ir_version: 9,
            producer: "LIFT Compiler Framework".to_string(),
        }
    }

    pub fn with_opset(mut self, version: i64) -> Self {
        self.opset_version = version;
        self
    }

    /// Export LIFT IR to ONNX protobuf text format.
    pub fn export(&self, ctx: &Context) -> Result<String, OnnxExportError> {
        let mut out = String::new();

        // ONNX ModelProto header
        let _ = writeln!(out, "# ONNX Model — Generated by LIFT Compiler Framework");
        let _ = writeln!(out, "# Opset version: {}", self.opset_version);
        let _ = writeln!(out, "# IR version: {}", self.ir_version);
        let _ = writeln!(out);

        for module in &ctx.modules {
            let mod_name = ctx.strings.resolve(module.name).to_string();

            let _ = writeln!(out, "ir_version: {}", self.ir_version);
            let _ = writeln!(out, "producer_name: \"{}\"", self.producer);
            let _ = writeln!(out, "producer_version: \"0.3.0\"");
            let _ = writeln!(out, "domain: \"\"");
            let _ = writeln!(out, "model_version: 1");
            let _ = writeln!(out, "doc_string: \"LIFT model: {}\"", mod_name);
            let _ = writeln!(out);

            // Opset imports
            let _ = writeln!(out, "opset_import {{");
            let _ = writeln!(out, "  domain: \"\"");
            let _ = writeln!(out, "  version: {}", self.opset_version);
            let _ = writeln!(out, "}}");
            let _ = writeln!(out, "opset_import {{");
            let _ = writeln!(out, "  domain: \"com.microsoft\"");
            let _ = writeln!(out, "  version: 1");
            let _ = writeln!(out, "}}");
            let _ = writeln!(out);

            // Graph
            let _ = writeln!(out, "graph {{");
            let _ = writeln!(out, "  name: \"{}\"", mod_name);
            let _ = writeln!(out);

            let mut node_counter = 0usize;
            let mut graph_inputs: Vec<String> = Vec::new();
            let mut graph_outputs: Vec<String> = Vec::new();
            let mut value_info_entries: Vec<String> = Vec::new();

            for func in &module.functions {
                let fname = ctx.strings.resolve(func.name).to_string();

                // Collect function params as graph inputs
                for (pi, param_ty) in func.params.iter().enumerate() {
                    let input_name = format!("{}_input{}", fname, pi);
                    graph_inputs.push(self.format_value_info(&input_name, ctx, *param_ty));
                }

                // Collect return types as graph outputs
                for (ri, ret_ty) in func.returns.iter().enumerate() {
                    let output_name = format!("{}_output{}", fname, ri);
                    graph_outputs.push(self.format_value_info(&output_name, ctx, *ret_ty));
                }

                // Walk the ops
                if let Some(region_key) = func.body {
                    if let Some(region) = ctx.get_region(region_key) {
                        for &block_key in &region.blocks {
                            if let Some(block) = ctx.get_block(block_key) {
                                // Map block args to named values
                                let mut val_names = std::collections::HashMap::new();
                                for (ai, &arg_val) in block.args.iter().enumerate() {
                                    val_names.insert(arg_val, format!("{}_input{}", fname, ai));
                                }

                                for &op_key in &block.ops {
                                    if let Some(op) = ctx.get_op(op_key) {
                                        let op_name = ctx.strings.resolve(op.name).to_string();

                                        // Skip core.return — it just identifies output
                                        if op_name == "core.return" {
                                            // Map the inputs of return as graph outputs
                                            for (ri, &inp) in op.inputs.iter().enumerate() {
                                                let out_name = format!("{}_output{}", fname, ri);
                                                if let Some(src_name) = val_names.get(&inp) {
                                                    // Alias
                                                    let _ = writeln!(out, "  # output {} -> {}", out_name, src_name);
                                                }
                                            }
                                            continue;
                                        }

                                        // Map LIFT op to ONNX
                                        let (onnx_op, domain) = match lift_op_to_onnx(&op_name) {
                                            Some(pair) => pair,
                                            None => {
                                                // Emit as custom op
                                                let _ = writeln!(out, "  # WARNING: unmapped op {} — emitted as custom", op_name);
                                                (&op_name as &str, "ai.lift")
                                            }
                                        };

                                        let node_name = format!("{}_{}", fname, node_counter);

                                        // Input names
                                        let input_names: Vec<String> = op.inputs.iter()
                                            .map(|&v| {
                                                val_names.get(&v)
                                                    .cloned()
                                                    .unwrap_or_else(|| format!("v_{:?}", v))
                                            })
                                            .collect();

                                        // Result names
                                        let result_names: Vec<String> = op.results.iter()
                                            .enumerate()
                                            .map(|(i, &v)| {
                                                let name = format!("{}_r{}", node_name, i);
                                                val_names.insert(v, name.clone());

                                                // Collect intermediate value type info
                                                if let Some(val) = ctx.get_value(v) {
                                                    value_info_entries.push(
                                                        self.format_value_info(&name, ctx, val.ty)
                                                    );
                                                }

                                                name
                                            })
                                            .collect();

                                        // Emit node
                                        let _ = writeln!(out, "  node {{");
                                        for iname in &input_names {
                                            let _ = writeln!(out, "    input: \"{}\"", iname);
                                        }
                                        for rname in &result_names {
                                            let _ = writeln!(out, "    output: \"{}\"", rname);
                                        }
                                        let _ = writeln!(out, "    name: \"{}\"", node_name);
                                        let _ = writeln!(out, "    op_type: \"{}\"", onnx_op);
                                        if !domain.is_empty() {
                                            let _ = writeln!(out, "    domain: \"{}\"", domain);
                                        }
                                        let _ = writeln!(out, "  }}");
                                        let _ = writeln!(out);

                                        node_counter += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Graph inputs
            let _ = writeln!(out, "  # === Graph Inputs ===");
            for gi in &graph_inputs {
                let _ = writeln!(out, "{}", gi);
            }
            let _ = writeln!(out);

            // Graph outputs
            let _ = writeln!(out, "  # === Graph Outputs ===");
            for go in &graph_outputs {
                let _ = writeln!(out, "{}", go);
            }
            let _ = writeln!(out);

            // Intermediate value info
            if !value_info_entries.is_empty() {
                let _ = writeln!(out, "  # === Value Info ===");
                for vi in &value_info_entries {
                    let _ = writeln!(out, "{}", vi);
                }
                let _ = writeln!(out);
            }

            let _ = writeln!(out, "}}");
        }

        Ok(out)
    }

    /// Export to ONNX-compatible JSON (for tools that consume JSON ONNX).
    pub fn export_json(&self, ctx: &Context) -> Result<String, OnnxExportError> {
        let mut out = String::new();

        let _ = writeln!(out, "{{");
        let _ = writeln!(out, "  \"irVersion\": {},", self.ir_version);
        let _ = writeln!(out, "  \"producerName\": \"{}\",", self.producer);
        let _ = writeln!(out, "  \"producerVersion\": \"0.3.0\",");
        let _ = writeln!(out, "  \"opsetImport\": [");
        let _ = writeln!(out, "    {{ \"domain\": \"\", \"version\": {} }},", self.opset_version);
        let _ = writeln!(out, "    {{ \"domain\": \"com.microsoft\", \"version\": 1 }}");
        let _ = writeln!(out, "  ],");

        for module in &ctx.modules {
            let mod_name = ctx.strings.resolve(module.name).to_string();

            let _ = writeln!(out, "  \"graph\": {{");
            let _ = writeln!(out, "    \"name\": \"{}\",", mod_name);

            // Nodes
            let _ = writeln!(out, "    \"node\": [");
            let mut nodes: Vec<String> = Vec::new();

            for func in &module.functions {
                let fname = ctx.strings.resolve(func.name).to_string();
                let mut val_names = std::collections::HashMap::new();
                let mut node_counter = 0usize;

                if let Some(region_key) = func.body {
                    if let Some(region) = ctx.get_region(region_key) {
                        for &block_key in &region.blocks {
                            if let Some(block) = ctx.get_block(block_key) {
                                for (ai, &arg_val) in block.args.iter().enumerate() {
                                    val_names.insert(arg_val, format!("{}_input{}", fname, ai));
                                }

                                for &op_key in &block.ops {
                                    if let Some(op) = ctx.get_op(op_key) {
                                        let op_name = ctx.strings.resolve(op.name).to_string();
                                        if op_name == "core.return" { continue; }

                                        let (onnx_op, domain) = lift_op_to_onnx(&op_name)
                                            .unwrap_or((&op_name, "ai.lift"));

                                        let node_name = format!("{}_{}", fname, node_counter);

                                        let input_names: Vec<String> = op.inputs.iter()
                                            .map(|&v| val_names.get(&v).cloned().unwrap_or_else(|| format!("v_{:?}", v)))
                                            .collect();

                                        let result_names: Vec<String> = op.results.iter()
                                            .enumerate()
                                            .map(|(i, &v)| {
                                                let name = format!("{}_r{}", node_name, i);
                                                val_names.insert(v, name.clone());
                                                name
                                            })
                                            .collect();

                                        let mut node_json = String::new();
                                        let _ = write!(node_json, "      {{");
                                        let _ = write!(node_json, " \"name\": \"{}\",", node_name);
                                        let _ = write!(node_json, " \"opType\": \"{}\",", onnx_op);
                                        if !domain.is_empty() {
                                            let _ = write!(node_json, " \"domain\": \"{}\",", domain);
                                        }
                                        let inputs_json: Vec<String> = input_names.iter().map(|n| format!("\"{}\"", n)).collect();
                                        let outputs_json: Vec<String> = result_names.iter().map(|n| format!("\"{}\"", n)).collect();
                                        let _ = write!(node_json, " \"input\": [{}],", inputs_json.join(", "));
                                        let _ = write!(node_json, " \"output\": [{}]", outputs_json.join(", "));
                                        let _ = write!(node_json, " }}");
                                        nodes.push(node_json);

                                        node_counter += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let _ = writeln!(out, "{}", nodes.join(",\n"));
            let _ = writeln!(out, "    ],");

            // Graph inputs
            let _ = writeln!(out, "    \"input\": [");
            let mut inputs: Vec<String> = Vec::new();
            for func in &module.functions {
                let fname = ctx.strings.resolve(func.name).to_string();
                for (pi, param_ty) in func.params.iter().enumerate() {
                    let name = format!("{}_input{}", fname, pi);
                    inputs.push(self.format_json_value_info(&name, ctx, *param_ty));
                }
            }
            let _ = writeln!(out, "{}", inputs.join(",\n"));
            let _ = writeln!(out, "    ],");

            // Graph outputs
            let _ = writeln!(out, "    \"output\": [");
            let mut outputs: Vec<String> = Vec::new();
            for func in &module.functions {
                let fname = ctx.strings.resolve(func.name).to_string();
                for (ri, ret_ty) in func.returns.iter().enumerate() {
                    let name = format!("{}_output{}", fname, ri);
                    outputs.push(self.format_json_value_info(&name, ctx, *ret_ty));
                }
            }
            let _ = writeln!(out, "{}", outputs.join(",\n"));
            let _ = writeln!(out, "    ]");

            let _ = writeln!(out, "  }}");
        }

        let _ = writeln!(out, "}}");
        Ok(out)
    }

    fn format_value_info(&self, name: &str, ctx: &Context, type_key: lift_core::types::TypeKey) -> String {
        let mut info = String::new();
        let _ = writeln!(info, "  input {{");
        let _ = writeln!(info, "    name: \"{}\"", name);
        let _ = writeln!(info, "    type {{");
        let _ = writeln!(info, "      tensor_type {{");

        match ctx.resolve_type(type_key) {
            CoreType::Opaque { data: TypeData::Tensor(tensor_info), .. } => {
                let onnx_dt = lift_dtype_to_onnx(&tensor_info.dtype);
                let _ = writeln!(info, "        elem_type: {}", onnx_dt as i32);
                let _ = writeln!(info, "        shape {{");
                for dim in &tensor_info.shape {
                    match dim {
                        Dimension::Constant(s) => {
                            let _ = writeln!(info, "          dim {{ dim_value: {} }}", s);
                        }
                        _ => {
                            let _ = writeln!(info, "          dim {{ dim_param: \"?\" }}");
                        }
                    }
                }
                let _ = writeln!(info, "        }}");
            }
            _ => {
                let _ = writeln!(info, "        elem_type: 1"); // default float
            }
        }

        let _ = writeln!(info, "      }}");
        let _ = writeln!(info, "    }}");
        let _ = write!(info, "  }}");
        info
    }

    fn format_json_value_info(&self, name: &str, ctx: &Context, type_key: lift_core::types::TypeKey) -> String {
        let mut info = String::new();
        let _ = write!(info, "      {{ \"name\": \"{}\"", name);

        match ctx.resolve_type(type_key) {
            CoreType::Opaque { data: TypeData::Tensor(tensor_info), .. } => {
                let onnx_dt = lift_dtype_to_onnx(&tensor_info.dtype);
                let dims: Vec<String> = tensor_info.shape.iter().map(|d| {
                    match d {
                        Dimension::Constant(s) => s.to_string(),
                        _ => "\"?\"".to_string(),
                    }
                }).collect();
                let _ = write!(info, ", \"type\": {{ \"tensorType\": {{ \"elemType\": {}, \"shape\": [{}] }} }}", onnx_dt as i32, dims.join(", "));
            }
            _ => {
                let _ = write!(info, ", \"type\": {{ \"tensorType\": {{ \"elemType\": 1 }} }}");
            }
        }

        let _ = write!(info, " }}");
        info
    }
}

impl Default for OnnxExporter {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_export_empty() {
        let ctx = Context::new();
        let exporter = OnnxExporter::new();
        let result = exporter.export(&ctx);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("ir_version"));
    }

    #[test]
    fn test_onnx_json_export_empty() {
        let ctx = Context::new();
        let exporter = OnnxExporter::new();
        let result = exporter.export_json(&ctx);
        assert!(result.is_ok());
        let text = result.unwrap();
        assert!(text.contains("irVersion"));
    }

    #[test]
    fn test_lift_op_mapping() {
        assert_eq!(lift_op_to_onnx("tensor.matmul"), Some(("MatMul", "")));
        assert_eq!(lift_op_to_onnx("tensor.relu"), Some(("Relu", "")));
        assert_eq!(lift_op_to_onnx("tensor.layernorm"), Some(("LayerNormalization", "")));
        assert_eq!(lift_op_to_onnx("tensor.grouped_query_attention"), Some(("GroupQueryAttention", "com.microsoft")));
        assert!(lift_op_to_onnx("unknown.op").is_none());
    }

    #[test]
    fn test_onnx_with_context() {
        let model = lift_core::model_builder::ModelBuilder::new("test_onnx")
            .function("forward")
                .param("x", lift_core::model_builder::tensor(&[1, 784], DataType::FP32))
                .param("w", lift_core::model_builder::tensor_2d(784, 10, DataType::FP32))
                .op("tensor.matmul", &["x", "w"], "out", lift_core::model_builder::tensor(&[1, 10], DataType::FP32))
                .returns("out")
                .done();

        let ctx = model.build_context();
        let exporter = OnnxExporter::new();

        let pbtxt = exporter.export(&ctx).unwrap();
        assert!(pbtxt.contains("MatMul"));
        assert!(pbtxt.contains("test_onnx"));
        assert!(pbtxt.contains("opset_import"));

        let json = exporter.export_json(&ctx).unwrap();
        assert!(json.contains("MatMul"));
        assert!(json.contains("test_onnx"));
    }
}
