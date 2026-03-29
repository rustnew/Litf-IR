use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    Integer(i64),
    Float(f64),
    StringLiteral(String),

    // Identifiers and keywords
    Ident(String),
    AtIdent(String),      // @name
    PercentIdent(String), // %name
    CaretIdent(String),   // ^block_name
    HashDialect(String),  // #dialect

    // Keywords
    Module,
    Func,
    Return,
    If,
    Else,
    True,
    False,

    // Type keywords
    Tensor,
    Qubit,
    Bit,
    Hamiltonian,
    Void,
    Index,

    // Punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LAngle,
    RAngle,
    Comma,
    Colon,
    Semicolon,
    Arrow,       // ->
    Equal,
    Dot,
    Star,

    // Special
    Comment(String),
    Newline,
    Eof,
    Error(String),
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Integer(v) => write!(f, "{}", v),
            TokenKind::Float(v) => write!(f, "{}", v),
            TokenKind::StringLiteral(s) => write!(f, "\"{}\"", s),
            TokenKind::Ident(s) => write!(f, "{}", s),
            TokenKind::AtIdent(s) => write!(f, "@{}", s),
            TokenKind::PercentIdent(s) => write!(f, "%{}", s),
            TokenKind::CaretIdent(s) => write!(f, "^{}", s),
            TokenKind::HashDialect(s) => write!(f, "#dialect {}", s),
            TokenKind::Module => write!(f, "module"),
            TokenKind::Func => write!(f, "func"),
            TokenKind::Return => write!(f, "return"),
            TokenKind::If => write!(f, "if"),
            TokenKind::Else => write!(f, "else"),
            TokenKind::True => write!(f, "true"),
            TokenKind::False => write!(f, "false"),
            TokenKind::Tensor => write!(f, "tensor"),
            TokenKind::Qubit => write!(f, "qubit"),
            TokenKind::Bit => write!(f, "bit"),
            TokenKind::Hamiltonian => write!(f, "hamiltonian"),
            TokenKind::Void => write!(f, "void"),
            TokenKind::Index => write!(f, "index"),
            TokenKind::LParen => write!(f, "("),
            TokenKind::RParen => write!(f, ")"),
            TokenKind::LBrace => write!(f, "{{"),
            TokenKind::RBrace => write!(f, "}}"),
            TokenKind::LBracket => write!(f, "["),
            TokenKind::RBracket => write!(f, "]"),
            TokenKind::LAngle => write!(f, "<"),
            TokenKind::RAngle => write!(f, ">"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::Equal => write!(f, "="),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Comment(s) => write!(f, "// {}", s),
            TokenKind::Newline => write!(f, "\\n"),
            TokenKind::Eof => write!(f, "EOF"),
            TokenKind::Error(s) => write!(f, "ERROR: {}", s),
        }
    }
}
