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
