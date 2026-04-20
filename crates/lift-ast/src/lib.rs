//! LIFT AST: Lexer, parser, and AST for the `.lif` source language.
//!
//! This crate provides tokenisation, parsing, and IR construction from
//! LIFT source files. Use [`Lexer`] to tokenise, [`Parser`] to parse,
//! and [`IrBuilder`] to lower the AST into the core IR.

pub mod lexer;
pub mod token;
pub mod parser;
pub mod ast;
pub mod builder;

pub use lexer::Lexer;
pub use token::{Token, TokenKind};
pub use parser::Parser;
pub use ast::*;
pub use builder::IrBuilder;

/// Convenience: parse a `.lif` source string into a Program AST.
pub fn parse_source(source: &str) -> Result<Program, Vec<parser::ParseError>> {
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().to_vec();
    Parser::new(tokens).parse()
}

/// Convenience: lower a Program AST into a populated Context.
pub fn build_context(program: &Program) -> Result<lift_core::Context, String> {
    let mut ctx = lift_core::Context::new();
    let mut builder = IrBuilder::new();
    builder.build_program(&mut ctx, program)?;
    Ok(ctx)
}
