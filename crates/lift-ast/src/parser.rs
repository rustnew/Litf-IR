use crate::token::{Token, TokenKind, Span};
use crate::ast::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Unexpected token {found} at line {line}:{col}, expected {expected}")]
    UnexpectedToken { found: String, expected: String, line: u32, col: u32 },

    #[error("Unexpected end of file")]
    UnexpectedEof,

    #[error("Parse error: {0}")]
    General(String),
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    errors: Vec<ParseError>,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0, errors: Vec::new() }
    }

    pub fn parse(&mut self) -> Result<Program, Vec<ParseError>> {
        let mut program = Program {
            dialect_directives: Vec::new(),
            modules: Vec::new(),
        };

        while !self.at_end() {
            match self.peek_kind() {
                TokenKind::HashDialect(_) => {
                    match self.parse_dialect_directive() {
                        Ok(d) => program.dialect_directives.push(d),
                        Err(e) => { self.errors.push(e); self.recover(); }
                    }
                }
                TokenKind::Module => {
                    match self.parse_module() {
                        Ok(m) => program.modules.push(m),
                        Err(e) => { self.errors.push(e); self.recover(); }
                    }
                }
                TokenKind::Eof => break,
                _ => {
                    let tok = self.advance();
                    self.errors.push(ParseError::UnexpectedToken {
                        found: format!("{}", tok.kind),
                        expected: "module or #dialect".to_string(),
                        line: tok.span.line,
                        col: tok.span.column,
                    });
                }
            }
        }

        if self.errors.is_empty() {
            Ok(program)
        } else {
            Err(std::mem::take(&mut self.errors))
        }
    }

    fn parse_dialect_directive(&mut self) -> Result<DialectDirective, ParseError> {
        let tok = self.advance();
        if let TokenKind::HashDialect(name) = &tok.kind {
            Ok(DialectDirective { name: name.clone(), span: tok.span })
        } else {
            Err(ParseError::General("Expected #dialect directive".into()))
        }
    }

    fn parse_module(&mut self) -> Result<ModuleDecl, ParseError> {
        let start = self.expect_kind(&TokenKind::Module)?.span;

        let name = match &self.advance().kind {
            TokenKind::AtIdent(n) => n.clone(),
            other => return Err(ParseError::UnexpectedToken {
                found: format!("{}", other), expected: "@module_name".into(),
                line: start.line, col: start.column,
            }),
        };

        self.expect_kind(&TokenKind::LBrace)?;

        let mut functions = Vec::new();
        while !self.check_kind(&TokenKind::RBrace) && !self.at_end() {
            match self.parse_function() {
                Ok(f) => functions.push(f),
                Err(e) => { self.errors.push(e); self.recover_to_func_or_rbrace(); }
            }
        }

        self.expect_kind(&TokenKind::RBrace)?;

        Ok(ModuleDecl { name, functions, span: start })
    }

    fn parse_function(&mut self) -> Result<FuncDecl, ParseError> {
        let start = self.expect_kind(&TokenKind::Func)?.span;

        let name = match &self.advance().kind {
            TokenKind::AtIdent(n) => n.clone(),
            other => return Err(ParseError::UnexpectedToken {
                found: format!("{}", other), expected: "@func_name".into(),
                line: start.line, col: start.column,
            }),
        };

        self.expect_kind(&TokenKind::LParen)?;

        let mut params = Vec::new();
        while !self.check_kind(&TokenKind::RParen) && !self.at_end() {
            if !params.is_empty() {
                self.expect_kind(&TokenKind::Comma)?;
            }
            params.push(self.parse_param()?);
        }

        self.expect_kind(&TokenKind::RParen)?;

        let mut returns = Vec::new();
        if self.check_kind(&TokenKind::Arrow) {
            self.advance(); // ->
            returns = self.parse_return_types()?;
        }

        self.expect_kind(&TokenKind::LBrace)?;

        let mut body = Vec::new();
        while !self.check_kind(&TokenKind::RBrace) && !self.at_end() {
            match self.parse_statement() {
                Ok(s) => body.push(s),
                Err(e) => { self.errors.push(e); self.recover_to_statement(); }
            }
        }

        self.expect_kind(&TokenKind::RBrace)?;

        Ok(FuncDecl { name, params, returns, body, span: start })
    }

    fn parse_param(&mut self) -> Result<ParamDecl, ParseError> {
        let tok = self.advance();
        let name = match &tok.kind {
            TokenKind::PercentIdent(n) => n.clone(),
            other => return Err(ParseError::UnexpectedToken {
                found: format!("{}", other), expected: "%param_name".into(),
                line: tok.span.line, col: tok.span.column,
            }),
        };

        self.expect_kind(&TokenKind::Colon)?;
        let ty = self.parse_type()?;

        Ok(ParamDecl { name, ty, span: tok.span })
    }

    fn parse_return_types(&mut self) -> Result<Vec<TypeExpr>, ParseError> {
        if self.check_kind(&TokenKind::LParen) {
            self.advance();
            let mut types = Vec::new();
            while !self.check_kind(&TokenKind::RParen) && !self.at_end() {
                if !types.is_empty() {
                    self.expect_kind(&TokenKind::Comma)?;
                }
                types.push(self.parse_type()?);
            }
            self.expect_kind(&TokenKind::RParen)?;
            Ok(types)
        } else {
            let ty = self.parse_type()?;
            Ok(vec![ty])
        }
    }

    fn parse_type(&mut self) -> Result<TypeExpr, ParseError> {
        match self.peek_kind() {
            TokenKind::Tensor => {
                self.advance();
                self.expect_kind(&TokenKind::LAngle)?;
                let mut shape = Vec::new();
                loop {
                    match self.peek_kind() {
                        TokenKind::Integer(n) => {
                            let n = n;
                            self.advance();
                            shape.push(DimExpr::Constant(n as usize));
                        }
                        TokenKind::Ident(s) => {
                            // Check if it's a dtype like f32, i8, etc.
                            if is_dtype(&s) {
                                let dtype = s;
                                self.advance();
                                self.expect_kind(&TokenKind::RAngle)?;
                                return Ok(TypeExpr::Tensor(TensorTypeExpr { shape, dtype }));
                            } else {
                                let s = s;
                                self.advance();
                                shape.push(DimExpr::Symbolic(s));
                            }
                        }
                        TokenKind::Star => {
                            self.advance();
                            shape.push(DimExpr::Dynamic);
                        }
                        _ => {
                            return Err(ParseError::General(
                                format!("Expected dimension or dtype in tensor type at line {}",
                                        self.current_span().line)
                            ));
                        }
                    }
                    // After a dimension, expect 'x' separator or end
                    if self.check_ident("x") {
                        self.advance();
                    } else if self.check_kind(&TokenKind::RAngle) {
                        self.advance();
                        return Ok(TypeExpr::Tensor(TensorTypeExpr {
                            shape,
                            dtype: "f32".into(),
                        }));
                    }
                }
            }
            TokenKind::Qubit => {
                self.advance();
                Ok(TypeExpr::Qubit)
            }
            TokenKind::Bit => {
                self.advance();
                Ok(TypeExpr::Bit)
            }
            TokenKind::Void => {
                self.advance();
                Ok(TypeExpr::Void)
            }
            TokenKind::Index => {
                self.advance();
                Ok(TypeExpr::Index)
            }
            TokenKind::Hamiltonian => {
                self.advance();
                self.expect_kind(&TokenKind::LAngle)?;
                let n = self.expect_integer()?;
                self.expect_kind(&TokenKind::RAngle)?;
                Ok(TypeExpr::Hamiltonian(n as usize))
            }
            TokenKind::Ident(s) if is_scalar_type(&s) => {
                let s = s;
                self.advance();
                Ok(TypeExpr::Scalar(ScalarTypeExpr { name: s }))
            }
            _ => {
                let tok = self.advance();
                Err(ParseError::UnexpectedToken {
                    found: format!("{}", tok.kind),
                    expected: "type".into(),
                    line: tok.span.line,
                    col: tok.span.column,
                })
            }
        }
    }

    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        if self.check_kind(&TokenKind::Return) {
            return self.parse_return().map(Statement::Return);
        }

        // Try to parse op assignment: %v0 = "op"(...) or %v0, %v1 = "op"(...)
        if self.check_percent_ident() {
            return self.parse_op_assign().map(Statement::OpAssign);
        }

        // Bare operation call: "op"(...)
        if self.check_string_literal() {
            return self.parse_bare_op().map(Statement::OpAssign);
        }

        let tok = self.advance();
        Err(ParseError::UnexpectedToken {
            found: format!("{}", tok.kind),
            expected: "statement".into(),
            line: tok.span.line,
            col: tok.span.column,
        })
    }

    fn parse_return(&mut self) -> Result<ReturnStmt, ParseError> {
        let span = self.advance().span; // 'return'

        let mut values = Vec::new();
        while !self.check_kind(&TokenKind::RBrace) && !self.at_end() &&
              !self.check_kind(&TokenKind::Eof) {
            if !values.is_empty() {
                self.expect_kind(&TokenKind::Comma)?;
            }
            values.push(self.parse_operand()?);
        }

        Ok(ReturnStmt { values, span })
    }

    fn parse_op_assign(&mut self) -> Result<OpAssign, ParseError> {
        let span = self.current_span();

        // Parse result values: %v0 or %v0, %v1
        let mut results = Vec::new();
        loop {
            let tok = self.advance();
            match &tok.kind {
                TokenKind::PercentIdent(n) => results.push(n.clone()),
                other => return Err(ParseError::UnexpectedToken {
                    found: format!("{}", other), expected: "%result_name".into(),
                    line: tok.span.line, col: tok.span.column,
                }),
            }
            if self.check_kind(&TokenKind::Comma) {
                self.advance();
                // If next is '=' then the comma was part of a different grammar
                if self.check_kind(&TokenKind::Equal) { break; }
            } else {
                break;
            }
        }

        self.expect_kind(&TokenKind::Equal)?;

        // Parse operation name (string literal)
        let op_name = match &self.advance().kind {
            TokenKind::StringLiteral(s) => s.clone(),
            other => return Err(ParseError::General(
                format!("Expected operation name string, got {}", other)
            )),
        };

        // Parse operands
        self.expect_kind(&TokenKind::LParen)?;
        let mut operands = Vec::new();
        while !self.check_kind(&TokenKind::RParen) && !self.at_end() {
            if !operands.is_empty() {
                self.expect_kind(&TokenKind::Comma)?;
            }
            operands.push(self.parse_operand()?);
        }
        self.expect_kind(&TokenKind::RParen)?;

        // Parse optional attributes
        let mut attrs = Vec::new();
        if self.check_kind(&TokenKind::LBrace) {
            attrs = self.parse_attr_dict()?;
        }

        // Parse optional type signature
        let type_sig = if self.check_kind(&TokenKind::Colon) {
            self.advance();
            Some(self.parse_type_signature()?)
        } else {
            None
        };

        Ok(OpAssign { results, op_name, operands, attrs, type_sig, span })
    }

    fn parse_bare_op(&mut self) -> Result<OpAssign, ParseError> {
        let span = self.current_span();

        let op_name = match &self.advance().kind {
            TokenKind::StringLiteral(s) => s.clone(),
            other => return Err(ParseError::General(
                format!("Expected operation name string, got {}", other)
            )),
        };

        self.expect_kind(&TokenKind::LParen)?;
        let mut operands = Vec::new();
        while !self.check_kind(&TokenKind::RParen) && !self.at_end() {
            if !operands.is_empty() {
                self.expect_kind(&TokenKind::Comma)?;
            }
            operands.push(self.parse_operand()?);
        }
        self.expect_kind(&TokenKind::RParen)?;

        let mut attrs = Vec::new();
        if self.check_kind(&TokenKind::LBrace) {
            attrs = self.parse_attr_dict()?;
        }

        let type_sig = if self.check_kind(&TokenKind::Colon) {
            self.advance();
            Some(self.parse_type_signature()?)
        } else {
            None
        };

        Ok(OpAssign { results: Vec::new(), op_name, operands, attrs, type_sig, span })
    }

    fn parse_operand(&mut self) -> Result<Operand, ParseError> {
        match self.peek_kind() {
            TokenKind::PercentIdent(n) => {
                let n = n;
                self.advance();
                Ok(Operand::Value(n))
            }
            TokenKind::AtIdent(n) => {
                let n = n;
                self.advance();
                Ok(Operand::FuncRef(n))
            }
            TokenKind::Integer(v) => {
                let v = v;
                self.advance();
                Ok(Operand::Literal(LiteralValue::Integer(v)))
            }
            TokenKind::Float(v) => {
                let v = v;
                self.advance();
                Ok(Operand::Literal(LiteralValue::Float(v)))
            }
            TokenKind::True => {
                self.advance();
                Ok(Operand::Literal(LiteralValue::Bool(true)))
            }
            TokenKind::False => {
                self.advance();
                Ok(Operand::Literal(LiteralValue::Bool(false)))
            }
            _ => {
                let tok = self.advance();
                Err(ParseError::UnexpectedToken {
                    found: format!("{}", tok.kind), expected: "operand".into(),
                    line: tok.span.line, col: tok.span.column,
                })
            }
        }
    }

    fn parse_attr_dict(&mut self) -> Result<Vec<(String, AttrValue)>, ParseError> {
        self.expect_kind(&TokenKind::LBrace)?;
        let mut attrs = Vec::new();

        while !self.check_kind(&TokenKind::RBrace) && !self.at_end() {
            if !attrs.is_empty() {
                self.expect_kind(&TokenKind::Comma)?;
            }
            let key = match &self.advance().kind {
                TokenKind::Ident(s) => s.clone(),
                other => return Err(ParseError::General(
                    format!("Expected attribute key, got {}", other)
                )),
            };
            self.expect_kind(&TokenKind::Equal)?;
            let value = self.parse_attr_value()?;
            attrs.push((key, value));
        }

        self.expect_kind(&TokenKind::RBrace)?;
        Ok(attrs)
    }

    fn parse_attr_value(&mut self) -> Result<AttrValue, ParseError> {
        match self.peek_kind() {
            TokenKind::Integer(v) => { let v = v; self.advance(); Ok(AttrValue::Integer(v)) }
            TokenKind::Float(v) => { let v = v; self.advance(); Ok(AttrValue::Float(v)) }
            TokenKind::True => { self.advance(); Ok(AttrValue::Bool(true)) }
            TokenKind::False => { self.advance(); Ok(AttrValue::Bool(false)) }
            TokenKind::StringLiteral(s) => { let s = s; self.advance(); Ok(AttrValue::String(s)) }
            TokenKind::LBracket => {
                self.advance();
                let mut elems = Vec::new();
                while !self.check_kind(&TokenKind::RBracket) && !self.at_end() {
                    if !elems.is_empty() { self.expect_kind(&TokenKind::Comma)?; }
                    elems.push(self.parse_attr_value()?);
                }
                self.expect_kind(&TokenKind::RBracket)?;
                Ok(AttrValue::Array(elems))
            }
            _ => {
                let tok = self.advance();
                Err(ParseError::UnexpectedToken {
                    found: format!("{}", tok.kind), expected: "attribute value".into(),
                    line: tok.span.line, col: tok.span.column,
                })
            }
        }
    }

    fn parse_type_signature(&mut self) -> Result<TypeSignature, ParseError> {
        self.expect_kind(&TokenKind::LParen)?;
        let mut inputs = Vec::new();
        while !self.check_kind(&TokenKind::RParen) && !self.at_end() {
            if !inputs.is_empty() { self.expect_kind(&TokenKind::Comma)?; }
            inputs.push(self.parse_type()?);
        }
        self.expect_kind(&TokenKind::RParen)?;

        self.expect_kind(&TokenKind::Arrow)?;

        let outputs = self.parse_return_types()?;

        Ok(TypeSignature { inputs, outputs })
    }

    // ── Helpers ──

    fn peek_kind(&self) -> TokenKind {
        if self.pos < self.tokens.len() {
            self.tokens[self.pos].kind.clone()
        } else {
            TokenKind::Eof
        }
    }

    fn check_kind(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek_kind()) == std::mem::discriminant(kind)
    }

    fn check_percent_ident(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::PercentIdent(_))
    }

    fn check_string_literal(&self) -> bool {
        matches!(self.peek_kind(), TokenKind::StringLiteral(_))
    }

    fn check_ident(&self, name: &str) -> bool {
        matches!(&self.peek_kind(), TokenKind::Ident(s) if s == name)
    }

    fn advance(&mut self) -> Token {
        if self.pos < self.tokens.len() {
            let tok = self.tokens[self.pos].clone();
            self.pos += 1;
            tok
        } else {
            Token {
                kind: TokenKind::Eof,
                span: Span { start: 0, end: 0, line: 0, column: 0 },
                text: String::new(),
            }
        }
    }

    fn expect_kind(&mut self, expected: &TokenKind) -> Result<Token, ParseError> {
        let tok = self.advance();
        if std::mem::discriminant(&tok.kind) == std::mem::discriminant(expected) {
            Ok(tok)
        } else {
            Err(ParseError::UnexpectedToken {
                found: format!("{}", tok.kind),
                expected: format!("{}", expected),
                line: tok.span.line,
                col: tok.span.column,
            })
        }
    }

    fn expect_integer(&mut self) -> Result<i64, ParseError> {
        let tok = self.advance();
        match tok.kind {
            TokenKind::Integer(v) => Ok(v),
            _ => Err(ParseError::UnexpectedToken {
                found: format!("{}", tok.kind), expected: "integer".into(),
                line: tok.span.line, col: tok.span.column,
            }),
        }
    }

    fn current_span(&self) -> Span {
        if self.pos < self.tokens.len() {
            self.tokens[self.pos].span
        } else {
            Span { start: 0, end: 0, line: 0, column: 0 }
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len() || matches!(self.peek_kind(), TokenKind::Eof)
    }

    fn recover(&mut self) {
        while !self.at_end() {
            match self.peek_kind() {
                TokenKind::Module | TokenKind::Func | TokenKind::HashDialect(_) => return,
                TokenKind::RBrace => { self.advance(); return; }
                _ => { self.advance(); }
            }
        }
    }

    fn recover_to_func_or_rbrace(&mut self) {
        while !self.at_end() {
            match self.peek_kind() {
                TokenKind::Func | TokenKind::RBrace => return,
                _ => { self.advance(); }
            }
        }
    }

    fn recover_to_statement(&mut self) {
        while !self.at_end() {
            match self.peek_kind() {
                TokenKind::PercentIdent(_) | TokenKind::Return |
                TokenKind::StringLiteral(_) | TokenKind::RBrace => return,
                _ => { self.advance(); }
            }
        }
    }

    pub fn errors(&self) -> &[ParseError] {
        &self.errors
    }
}

fn is_dtype(s: &str) -> bool {
    matches!(s, "f64" | "f32" | "f16" | "bf16" | "fp8e4m3" | "fp8e5m2" |
             "i64" | "i32" | "i16" | "i8" | "i4" | "i2" | "u8" | "i1" | "index")
}

fn is_scalar_type(s: &str) -> bool {
    matches!(s, "f64" | "f32" | "f16" | "bf16" | "i64" | "i32" | "i16" | "i8" | "u8" | "bool")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse_source(src: &str) -> Result<Program, Vec<ParseError>> {
        let mut lexer = Lexer::new(src);
        let tokens = lexer.tokenize().to_vec();
        let mut parser = Parser::new(tokens);
        parser.parse()
    }

    #[test]
    fn test_parse_simple_module() {
        let src = r#"
#dialect tensor

module @test {
    func @relu(%x: tensor<4xf32>) -> tensor<4xf32> {
        %out = "tensor.relu"(%x) : (tensor<4xf32>) -> tensor<4xf32>
        return %out
    }
}
"#;
        let result = parse_source(src);
        assert!(result.is_ok(), "Parse errors: {:?}", result.err());
        let program = result.unwrap();
        assert_eq!(program.dialect_directives.len(), 1);
        assert_eq!(program.modules.len(), 1);
        assert_eq!(program.modules[0].functions.len(), 1);
        assert_eq!(program.modules[0].functions[0].name, "relu");
    }

    #[test]
    fn test_parse_quantum_module() {
        let src = r#"
#dialect quantum

module @qc {
    func @bell(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
        %q2 = "quantum.h"(%q0) : (qubit) -> qubit
        %q3, %q4 = "quantum.cx"(%q2, %q1) : (qubit, qubit) -> (qubit, qubit)
        return %q3, %q4
    }
}
"#;
        let result = parse_source(src);
        assert!(result.is_ok(), "Parse errors: {:?}", result.err());
    }
}
