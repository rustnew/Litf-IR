use crate::ast::*;
use lift_core::context::Context;
use lift_core::types::{DataType, Dimension, MemoryLayout};
use lift_core::attributes::{Attribute, Attributes};
use lift_core::location::Location;
use lift_core::functions::FunctionData;
use lift_core::values::ValueKey;
use std::collections::HashMap;

pub struct IrBuilder {
    value_map: HashMap<String, ValueKey>,
}

impl IrBuilder {
    pub fn new() -> Self {
        Self {
            value_map: HashMap::new(),
        }
    }

    pub fn build_program(&mut self, ctx: &mut Context, program: &Program) -> Result<(), String> {
        for directive in &program.dialect_directives {
            let _ = ctx.intern_string(&directive.name);
        }

        for module_decl in &program.modules {
            self.build_module(ctx, module_decl)?;
        }

        Ok(())
    }

    fn build_module(&mut self, ctx: &mut Context, module: &ModuleDecl) -> Result<(), String> {
        let module_idx = ctx.create_module(&module.name);

        for func_decl in &module.functions {
            let func = self.build_function(ctx, func_decl)?;
            ctx.add_function_to_module(module_idx, func);
        }

        Ok(())
    }

    fn build_function(&mut self, ctx: &mut Context, func: &FuncDecl) -> Result<FunctionData, String> {
        let name_id = ctx.intern_string(&func.name);

        let param_types: Vec<_> = func.params.iter()
            .map(|p| self.resolve_type_expr(ctx, &p.ty))
            .collect::<Result<Vec<_>, _>>()?;

        let return_types: Vec<_> = func.returns.iter()
            .map(|r| self.resolve_type_expr(ctx, r))
            .collect::<Result<Vec<_>, _>>()?;

        let mut func_data = FunctionData::new(name_id, param_types.clone(), return_types);

        // Create the function body region
        let region = ctx.create_region();
        let block = ctx.create_block();
        ctx.add_block_to_region(region, block);

        // Create block arguments for function parameters
        for (i, param) in func.params.iter().enumerate() {
            let val = ctx.create_block_arg(block, param_types[i]);
            self.value_map.insert(param.name.clone(), val);
        }

        // Build statements
        for stmt in &func.body {
            self.build_statement(ctx, block, stmt)?;
        }

        func_data.body = Some(region);
        Ok(func_data)
    }

    fn build_statement(&mut self, ctx: &mut Context, block: lift_core::blocks::BlockKey, stmt: &Statement) -> Result<(), String> {
        match stmt {
            Statement::OpAssign(op_assign) => {
                self.build_op_assign(ctx, block, op_assign)
            }
            Statement::Return(ret) => {
                self.build_return(ctx, block, ret)
            }
        }
    }

    fn build_op_assign(&mut self, ctx: &mut Context, block: lift_core::blocks::BlockKey, op: &OpAssign) -> Result<(), String> {
        // Resolve input operands
        let mut inputs = Vec::new();
        for operand in &op.operands {
            match operand {
                Operand::Value(name) => {
                    let key = self.value_map.get(name)
                        .ok_or_else(|| format!("Undefined value: %{}", name))?;
                    inputs.push(*key);
                }
                Operand::FuncRef(_) => {
                    // Function references are stored as attributes, not inputs
                }
                Operand::Literal(lit) => {
                    let (ty, attr_val) = match lit {
                        LiteralValue::Integer(v) => {
                            (ctx.make_integer_type(64, true), Attribute::Integer(*v))
                        }
                        LiteralValue::Float(v) => {
                            (ctx.make_float_type(64), Attribute::Float(*v))
                        }
                        LiteralValue::Bool(v) => {
                            (ctx.make_bool_type(), Attribute::Bool(*v))
                        }
                        LiteralValue::String(s) => {
                            let sid = ctx.intern_string(s);
                            (ctx.make_index_type(), Attribute::String(sid))
                        }
                    };
                    // Create a constant op for the literal
                    let mut const_attrs = Attributes::new();
                    const_attrs.set("value", attr_val);
                    let (const_op, const_results) = ctx.create_op(
                        "core.constant", "core",
                        vec![], vec![ty],
                        const_attrs, Location::unknown(),
                    );
                    ctx.add_op_to_block(block, const_op);
                    inputs.push(const_results[0]);
                }
            }
        }

        // Resolve result types from type signature
        let result_types = if let Some(sig) = &op.type_sig {
            sig.outputs.iter()
                .map(|t| self.resolve_type_expr(ctx, t))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            Vec::new()
        };

        // Parse dialect from op name
        let (dialect, _op_short) = op.op_name.split_once('.')
            .unwrap_or(("core", &op.op_name));

        // Build attributes
        let mut attrs = Attributes::new();
        for (key, val) in &op.attrs {
            attrs.set(key.clone(), self.convert_attr_value(ctx, val));
        }

        let (op_key, result_vals) = ctx.create_op(
            &op.op_name, dialect,
            inputs, result_types,
            attrs, Location::unknown(),
        );
        ctx.add_op_to_block(block, op_key);

        // Map result names
        for (i, name) in op.results.iter().enumerate() {
            if i < result_vals.len() {
                self.value_map.insert(name.clone(), result_vals[i]);
            }
        }

        Ok(())
    }

    fn build_return(&mut self, ctx: &mut Context, block: lift_core::blocks::BlockKey, ret: &ReturnStmt) -> Result<(), String> {
        let mut inputs = Vec::new();
        for operand in &ret.values {
            match operand {
                Operand::Value(name) => {
                    let key = self.value_map.get(name)
                        .ok_or_else(|| format!("Undefined value in return: %{}", name))?;
                    inputs.push(*key);
                }
                _ => return Err("Return values must be SSA values".into()),
            }
        }

        let (op_key, _) = ctx.create_op(
            "core.return", "core",
            inputs, vec![],
            Attributes::new(), Location::unknown(),
        );
        ctx.add_op_to_block(block, op_key);
        Ok(())
    }

    fn resolve_type_expr(&self, ctx: &mut Context, ty: &TypeExpr) -> Result<lift_core::types::TypeId, String> {
        match ty {
            TypeExpr::Tensor(t) => {
                let shape: Vec<Dimension> = t.shape.iter().map(|d| match d {
                    DimExpr::Constant(n) => Dimension::Constant(*n),
                    DimExpr::Symbolic(s) => Dimension::Symbolic(s.clone()),
                    DimExpr::Dynamic => Dimension::Symbolic("?".into()),
                }).collect();
                let dtype = parse_dtype(&t.dtype)?;
                Ok(ctx.make_tensor_type(shape, dtype, MemoryLayout::Contiguous))
            }
            TypeExpr::Qubit => Ok(ctx.make_qubit_type()),
            TypeExpr::Bit => Ok(ctx.make_bit_type()),
            TypeExpr::Void => Ok(ctx.make_void_type()),
            TypeExpr::Index => Ok(ctx.make_index_type()),
            TypeExpr::Hamiltonian(n) => Ok(ctx.make_hamiltonian_type(*n)),
            TypeExpr::Scalar(s) => {
                match s.name.as_str() {
                    "f64" => Ok(ctx.make_float_type(64)),
                    "f32" => Ok(ctx.make_float_type(32)),
                    "f16" => Ok(ctx.make_float_type(16)),
                    "i64" => Ok(ctx.make_integer_type(64, true)),
                    "i32" => Ok(ctx.make_integer_type(32, true)),
                    "i16" => Ok(ctx.make_integer_type(16, true)),
                    "i8" => Ok(ctx.make_integer_type(8, true)),
                    "u8" => Ok(ctx.make_integer_type(8, false)),
                    "bool" => Ok(ctx.make_bool_type()),
                    _ => Err(format!("Unknown scalar type: {}", s.name)),
                }
            }
            TypeExpr::Function(sig) => {
                let params = sig.inputs.iter()
                    .map(|t| self.resolve_type_expr(ctx, t))
                    .collect::<Result<Vec<_>, _>>()?;
                let returns = sig.outputs.iter()
                    .map(|t| self.resolve_type_expr(ctx, t))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(ctx.make_function_type(params, returns))
            }
        }
    }

    fn convert_attr_value(&self, ctx: &mut Context, val: &AttrValue) -> Attribute {
        match val {
            AttrValue::Integer(v) => Attribute::Integer(*v),
            AttrValue::Float(v) => Attribute::Float(*v),
            AttrValue::Bool(v) => Attribute::Bool(*v),
            AttrValue::String(s) => {
                let sid = ctx.intern_string(s);
                Attribute::String(sid)
            }
            AttrValue::Array(arr) => {
                Attribute::Array(arr.iter().map(|v| self.convert_attr_value(ctx, v)).collect())
            }
        }
    }
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_dtype(s: &str) -> Result<DataType, String> {
    match s {
        "f64" => Ok(DataType::FP64),
        "f32" => Ok(DataType::FP32),
        "f16" => Ok(DataType::FP16),
        "bf16" => Ok(DataType::BF16),
        "fp8e4m3" => Ok(DataType::FP8E4M3),
        "fp8e5m2" => Ok(DataType::FP8E5M2),
        "i64" => Ok(DataType::INT64),
        "i32" => Ok(DataType::INT32),
        "i16" => Ok(DataType::INT16),
        "i8" => Ok(DataType::INT8),
        "i4" => Ok(DataType::INT4),
        "i2" => Ok(DataType::INT2),
        "u8" => Ok(DataType::UINT8),
        "i1" => Ok(DataType::Bool),
        "index" => Ok(DataType::Index),
        _ => Err(format!("Unknown data type: {}", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    #[test]
    fn test_build_simple_program() {
        let src = r#"
#dialect tensor

module @test {
    func @relu(%x: tensor<4xf32>) -> tensor<4xf32> {
        %out = "tensor.relu"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %out
    }
}
"#;
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize().to_vec();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().expect("Parse failed");

        let mut ctx = Context::new();
        let mut builder = IrBuilder::new();
        builder.build_program(&mut ctx, &program).expect("Build failed");

        assert_eq!(ctx.modules.len(), 1);
        assert_eq!(ctx.modules[0].functions.len(), 1);
    }
}
