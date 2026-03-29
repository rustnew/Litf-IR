use crate::token::{Token, TokenKind, Span};

pub struct Lexer {
    source: Vec<char>,
    pos: usize,
    line: u32,
    column: u32,
    tokens: Vec<Token>,
    errors: Vec<String>,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
            tokens: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn tokenize(&mut self) -> &[Token] {
        while !self.at_end() {
            self.skip_whitespace();
            if self.at_end() { break; }

            let start = self.pos;
            let start_line = self.line;
            let start_col = self.column;

            let kind = self.next_token();

            match &kind {
                TokenKind::Comment(_) | TokenKind::Newline => {}
                _ => {
                    let text = self.source[start..self.pos].iter().collect();
                    self.tokens.push(Token {
                        kind,
                        span: Span { start, end: self.pos, line: start_line, column: start_col },
                        text,
                    });
                }
            }
        }

        self.tokens.push(Token {
            kind: TokenKind::Eof,
            span: Span { start: self.pos, end: self.pos, line: self.line, column: self.column },
            text: String::new(),
        });

        &self.tokens
    }

    fn next_token(&mut self) -> TokenKind {
        let ch = self.peek();

        match ch {
            '/' if self.peek_next() == '/' => self.lex_comment(),
            '\n' => { self.advance(); self.line += 1; self.column = 1; TokenKind::Newline }
            '"' => self.lex_string(),
            '@' => self.lex_at_ident(),
            '%' => self.lex_percent_ident(),
            '^' => self.lex_caret_ident(),
            '#' => self.lex_hash_directive(),
            '(' => { self.advance(); TokenKind::LParen }
            ')' => { self.advance(); TokenKind::RParen }
            '{' => { self.advance(); TokenKind::LBrace }
            '}' => { self.advance(); TokenKind::RBrace }
            '[' => { self.advance(); TokenKind::LBracket }
            ']' => { self.advance(); TokenKind::RBracket }
            '<' => { self.advance(); TokenKind::LAngle }
            '>' => { self.advance(); TokenKind::RAngle }
            ',' => { self.advance(); TokenKind::Comma }
            ':' => { self.advance(); TokenKind::Colon }
            ';' => { self.advance(); TokenKind::Semicolon }
            '=' => { self.advance(); TokenKind::Equal }
            '.' => { self.advance(); TokenKind::Dot }
            '*' => { self.advance(); TokenKind::Star }
            '-' if self.peek_next() == '>' => { self.advance(); self.advance(); TokenKind::Arrow }
            '-' | '0'..='9' => self.lex_number(),
            c if c.is_alphabetic() || c == '_' => self.lex_ident_or_keyword(),
            c => {
                self.advance();
                let msg = format!("Unexpected character '{}' at line {}:{}", c, self.line, self.column - 1);
                self.errors.push(msg.clone());
                TokenKind::Error(msg)
            }
        }
    }

    fn lex_comment(&mut self) -> TokenKind {
        self.advance(); // '/'
        self.advance(); // '/'
        let start = self.pos;
        while !self.at_end() && self.peek() != '\n' {
            self.advance();
        }
        let text: String = self.source[start..self.pos].iter().collect();
        TokenKind::Comment(text.trim().to_string())
    }

    fn lex_string(&mut self) -> TokenKind {
        self.advance(); // opening "
        let start = self.pos;
        while !self.at_end() && self.peek() != '"' {
            if self.peek() == '\n' { self.line += 1; self.column = 1; }
            self.advance();
        }
        let text: String = self.source[start..self.pos].iter().collect();
        if !self.at_end() { self.advance(); } // closing "
        TokenKind::StringLiteral(text)
    }

    fn lex_at_ident(&mut self) -> TokenKind {
        self.advance(); // '@'
        let start = self.pos;
        while !self.at_end() && (self.peek().is_alphanumeric() || self.peek() == '_' || self.peek() == ':') {
            self.advance();
        }
        let name: String = self.source[start..self.pos].iter().collect();
        TokenKind::AtIdent(name)
    }

    fn lex_percent_ident(&mut self) -> TokenKind {
        self.advance(); // '%'
        let start = self.pos;
        while !self.at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            self.advance();
        }
        // Handle indexing like %feat[0]
        if !self.at_end() && self.peek() == '[' {
            self.advance(); // '['
            while !self.at_end() && self.peek() != ']' {
                self.advance();
            }
            if !self.at_end() { self.advance(); } // ']'
        }
        let name: String = self.source[start..self.pos].iter().collect();
        TokenKind::PercentIdent(name)
    }

    fn lex_caret_ident(&mut self) -> TokenKind {
        self.advance(); // '^'
        let start = self.pos;
        while !self.at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            self.advance();
        }
        let name: String = self.source[start..self.pos].iter().collect();
        TokenKind::CaretIdent(name)
    }

    fn lex_hash_directive(&mut self) -> TokenKind {
        self.advance(); // '#'
        let start = self.pos;
        while !self.at_end() && self.peek().is_alphanumeric() {
            self.advance();
        }
        let directive: String = self.source[start..self.pos].iter().collect();
        if directive == "dialect" {
            self.skip_whitespace();
            let name_start = self.pos;
            while !self.at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
                self.advance();
            }
            let name: String = self.source[name_start..self.pos].iter().collect();
            TokenKind::HashDialect(name)
        } else {
            TokenKind::Ident(format!("#{}", directive))
        }
    }

    fn lex_number(&mut self) -> TokenKind {
        let start = self.pos;
        let negative = self.peek() == '-';
        if negative { self.advance(); }

        while !self.at_end() && self.peek().is_ascii_digit() {
            self.advance();
        }

        if !self.at_end() && self.peek() == '.' && self.peek_next().is_ascii_digit() {
            self.advance(); // '.'
            while !self.at_end() && self.peek().is_ascii_digit() {
                self.advance();
            }
            // Handle exponent
            if !self.at_end() && (self.peek() == 'e' || self.peek() == 'E') {
                self.advance();
                if !self.at_end() && (self.peek() == '+' || self.peek() == '-') {
                    self.advance();
                }
                while !self.at_end() && self.peek().is_ascii_digit() {
                    self.advance();
                }
            }
            let text: String = self.source[start..self.pos].iter().collect();
            match text.parse::<f64>() {
                Ok(v) => TokenKind::Float(v),
                Err(e) => TokenKind::Error(format!("Invalid float: {}", e)),
            }
        } else {
            let text: String = self.source[start..self.pos].iter().collect();
            match text.parse::<i64>() {
                Ok(v) => TokenKind::Integer(v),
                Err(e) => TokenKind::Error(format!("Invalid integer: {}", e)),
            }
        }
    }

    fn lex_ident_or_keyword(&mut self) -> TokenKind {
        let start = self.pos;
        // Special case: 'x' as dimension separator in tensor types (e.g. 1x784xf32)
        // Split when followed by digit or dtype prefix (f, i, b for bf16)
        if self.peek() == 'x' && self.pos + 1 < self.source.len() {
            let next = self.source[self.pos + 1];
            if next.is_ascii_digit() || next == 'f' || next == 'i' || next == 'b' {
                self.advance();
                return TokenKind::Ident("x".into());
            }
        }
        while !self.at_end() && (self.peek().is_alphanumeric() || self.peek() == '_') {
            self.advance();
        }
        let text: String = self.source[start..self.pos].iter().collect();

        match text.as_str() {
            "module" => TokenKind::Module,
            "func" => TokenKind::Func,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "tensor" => TokenKind::Tensor,
            "qubit" => TokenKind::Qubit,
            "bit" => TokenKind::Bit,
            "hamiltonian" => TokenKind::Hamiltonian,
            "void" => TokenKind::Void,
            "index" => TokenKind::Index,
            _ => TokenKind::Ident(text),
        }
    }

    fn peek(&self) -> char {
        if self.at_end() { '\0' } else { self.source[self.pos] }
    }

    fn peek_next(&self) -> char {
        if self.pos + 1 >= self.source.len() { '\0' } else { self.source[self.pos + 1] }
    }

    fn advance(&mut self) -> char {
        let ch = self.source[self.pos];
        self.pos += 1;
        self.column += 1;
        ch
    }

    fn at_end(&self) -> bool {
        self.pos >= self.source.len()
    }

    fn skip_whitespace(&mut self) {
        while !self.at_end() {
            match self.peek() {
                ' ' | '\t' | '\r' => { self.advance(); }
                '\n' => { self.advance(); self.line += 1; self.column = 1; }
                _ => break,
            }
        }
    }

    pub fn errors(&self) -> &[String] {
        &self.errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_simple_module() {
        let src = r#"module @test {
    func @relu(%x: tensor<4xf32>) -> tensor<4xf32> {
        return %x
    }
}"#;
        let mut lexer = Lexer::new(src);
        let tokens: Vec<_> = lexer.tokenize().to_vec();
        assert!(lexer.errors().is_empty());
        assert!(tokens.len() > 5);
        assert_eq!(tokens[0].kind, TokenKind::Module);
    }

    #[test]
    fn test_lex_dialect_directive() {
        let mut lexer = Lexer::new("#dialect tensor");
        let tokens = lexer.tokenize();
        assert!(matches!(&tokens[0].kind, TokenKind::HashDialect(s) if s == "tensor"));
    }

    #[test]
    fn test_lex_string_op() {
        let mut lexer = Lexer::new(r#""tensor.matmul"(%a, %b)"#);
        let tokens = lexer.tokenize();
        assert!(matches!(&tokens[0].kind, TokenKind::StringLiteral(s) if s == "tensor.matmul"));
    }

    #[test]
    fn test_lex_arrow() {
        let mut lexer = Lexer::new("-> tensor<4xf32>");
        let tokens = lexer.tokenize();
        assert_eq!(tokens[0].kind, TokenKind::Arrow);
    }

    #[test]
    fn test_lex_numbers() {
        let mut lexer = Lexer::new("42 3.14 -1");
        let tokens = lexer.tokenize();
        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
        assert_eq!(tokens[1].kind, TokenKind::Float(3.14));
        assert_eq!(tokens[2].kind, TokenKind::Integer(-1));
    }
}
