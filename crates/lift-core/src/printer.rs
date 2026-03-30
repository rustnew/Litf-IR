use crate::context::Context;
use crate::operations::OpKey;
use crate::blocks::BlockKey;
use crate::regions::RegionKey;
use crate::values::ValueKey;
use crate::attributes::Attribute;
use std::fmt::Write;

pub struct Printer<'a> {
    ctx: &'a Context,
    indent: usize,
    output: String,
    value_names: std::collections::HashMap<ValueKey, String>,
    next_value_id: u32,
    next_block_id: u32,
    block_names: std::collections::HashMap<BlockKey, String>,
}

impl<'a> Printer<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self {
            ctx,
            indent: 0,
            output: String::new(),
            value_names: std::collections::HashMap::new(),
            next_value_id: 0,
            next_block_id: 0,
            block_names: std::collections::HashMap::new(),
        }
    }

    pub fn print_all(&mut self) -> &str {
        for module in &self.ctx.modules {
            let name = self.ctx.strings.resolve(module.name);
            self.write_line(&format!("module @{} {{", name));
            self.indent += 1;

            for func in &module.functions {
                self.print_function(func);
            }

            self.indent -= 1;
            self.write_line("}");
            self.write_line("");
        }

        // Print any standalone blocks/ops not in modules
        for (block_key, _block) in &self.ctx.blocks {
            if self.ctx.blocks[block_key].parent_region.is_none() {
                self.print_block(block_key);
            }
        }

        &self.output
    }

    fn print_function(&mut self, func: &crate::functions::FunctionData) {
        let name = self.ctx.strings.resolve(func.name);
        let mut sig = format!("func @{}(", name);

        for (i, &param_ty) in func.params.iter().enumerate() {
            if i > 0 { sig.push_str(", "); }
            let pname = self.fresh_value_name();
            let _ = write!(sig, "%{}: {}", pname, self.format_type(param_ty));
        }

        sig.push_str(") -> ");
        if func.returns.len() == 1 {
            let _ = write!(sig, "{}", self.format_type(func.returns[0]));
        } else {
            sig.push('(');
            for (i, &ret_ty) in func.returns.iter().enumerate() {
                if i > 0 { sig.push_str(", "); }
                let _ = write!(sig, "{}", self.format_type(ret_ty));
            }
            sig.push(')');
        }

        if let Some(body) = func.body {
            sig.push_str(" {");
            self.write_line(&sig);
            self.indent += 1;
            self.print_region(body);
            self.indent -= 1;
            self.write_line("}");
        } else {
            self.write_line(&sig);
        }
        self.write_line("");
    }

    fn print_region(&mut self, region_key: RegionKey) {
        if let Some(region) = self.ctx.get_region(region_key) {
            for &block_key in &region.blocks {
                self.print_block(block_key);
            }
        }
    }

    fn print_block(&mut self, block_key: BlockKey) {
        let block_name = self.get_block_name(block_key);
        if let Some(block) = self.ctx.get_block(block_key) {
            if !block.args.is_empty() {
                let mut args = String::new();
                for (i, &arg) in block.args.iter().enumerate() {
                    if i > 0 { args.push_str(", "); }
                    let vname = self.get_value_name(arg);
                    if let Some(val) = self.ctx.get_value(arg) {
                        let _ = write!(args, "%{}: {}", vname, self.format_type(val.ty));
                    }
                }
                self.write_line(&format!("^{}({}):", block_name, args));
            } else {
                self.write_line(&format!("^{}:", block_name));
            }

            self.indent += 1;
            for &op_key in &block.ops {
                self.print_op(op_key);
            }
            self.indent -= 1;
        }
    }

    fn print_op(&mut self, op_key: OpKey) {
        if let Some(op) = self.ctx.get_op(op_key) {
            let op_name = self.ctx.strings.resolve(op.name).to_string();
            let mut line = String::new();

            // Result values
            if !op.results.is_empty() {
                for (i, &result) in op.results.iter().enumerate() {
                    if i > 0 { line.push_str(", "); }
                    let vname = self.get_value_name(result);
                    let _ = write!(line, "%{}", vname);
                }
                line.push_str(" = ");
            }

            // Operation name
            let _ = write!(line, "\"{}\"", op_name);

            // Inputs
            line.push('(');
            for (i, &input) in op.inputs.iter().enumerate() {
                if i > 0 { line.push_str(", "); }
                let vname = self.get_value_name(input);
                let _ = write!(line, "%{}", vname);
            }
            line.push(')');

            // Attributes
            if !op.attrs.is_empty() {
                line.push_str(" {");
                for (i, (key, val)) in op.attrs.iter().enumerate() {
                    if i > 0 { line.push_str(", "); }
                    let _ = write!(line, "{} = {}", key, self.format_attr(val));
                }
                line.push('}');
            }

            // Type signature
            line.push_str(" : (");
            for (i, &input) in op.inputs.iter().enumerate() {
                if i > 0 { line.push_str(", "); }
                if let Some(val) = self.ctx.get_value(input) {
                    let _ = write!(line, "{}", self.format_type(val.ty));
                }
            }
            line.push_str(") -> ");

            if op.results.len() == 1 {
                if let Some(val) = self.ctx.get_value(op.results[0]) {
                    let _ = write!(line, "{}", self.format_type(val.ty));
                }
            } else if op.results.is_empty() {
                line.push_str("()");
            } else {
                line.push('(');
                for (i, &result) in op.results.iter().enumerate() {
                    if i > 0 { line.push_str(", "); }
                    if let Some(val) = self.ctx.get_value(result) {
                        let _ = write!(line, "{}", self.format_type(val.ty));
                    }
                }
                line.push(')');
            }

            self.write_line(&line);

            // Print nested regions
            if !op.regions.is_empty() {
                for &region in &op.regions {
                    self.indent += 1;
                    self.print_region(region);
                    self.indent -= 1;
                }
            }
        }
    }

    fn format_type(&self, ty_id: crate::types::TypeId) -> String {
        let ty = self.ctx.resolve_type(ty_id);
        format!("{}", ty)
    }

    fn format_attr(&self, attr: &Attribute) -> String {
        match attr {
            Attribute::Integer(v) => format!("{}", v),
            Attribute::Float(v) => format!("{:.6}", v),
            Attribute::String(s) => {
                let resolved = self.ctx.strings.resolve(*s);
                format!("\"{}\"", resolved)
            }
            Attribute::Bool(b) => format!("{}", b),
            Attribute::Type(_) => "type".to_string(),
            Attribute::Array(arr) => {
                let inner: Vec<String> = arr.iter().map(|a| self.format_attr(a)).collect();
                format!("[{}]", inner.join(", "))
            }
            Attribute::Dict(map) => {
                let inner: Vec<String> = map.iter()
                    .map(|(k, v)| format!("{}: {}", k, self.format_attr(v)))
                    .collect();
                format!("{{{}}}", inner.join(", "))
            }
        }
    }

    fn get_value_name(&mut self, key: ValueKey) -> String {
        if let Some(name) = self.value_names.get(&key) {
            return name.clone();
        }

        // Try to use the debug name if available
        let name = if let Some(val) = self.ctx.get_value(key) {
            if let Some(name_id) = val.name {
                self.ctx.strings.resolve(name_id).to_string()
            } else {
                self.fresh_value_name()
            }
        } else {
            self.fresh_value_name()
        };

        self.value_names.insert(key, name.clone());
        name
    }

    fn fresh_value_name(&mut self) -> String {
        let name = format!("v{}", self.next_value_id);
        self.next_value_id += 1;
        name
    }

    fn get_block_name(&mut self, key: BlockKey) -> String {
        if let Some(name) = self.block_names.get(&key) {
            return name.clone();
        }
        let name = format!("bb{}", self.next_block_id);
        self.next_block_id += 1;
        self.block_names.insert(key, name.clone());
        name
    }

    fn write_line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(text);
        self.output.push('\n');
    }

    pub fn into_string(self) -> String {
        self.output
    }
}

pub fn print_ir(ctx: &Context) -> String {
    let mut printer = Printer::new(ctx);
    printer.print_all();
    printer.into_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_print_empty_context() {
        let ctx = Context::new();
        let output = print_ir(&ctx);
        assert!(output.is_empty() || output.trim().is_empty());
    }
}
