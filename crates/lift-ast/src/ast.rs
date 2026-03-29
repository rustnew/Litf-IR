use crate::token::Span;

#[derive(Debug, Clone)]
pub struct Program {
    pub dialect_directives: Vec<DialectDirective>,
    pub modules: Vec<ModuleDecl>,
}

#[derive(Debug, Clone)]
pub struct DialectDirective {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ModuleDecl {
    pub name: String,
    pub functions: Vec<FuncDecl>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: String,
    pub params: Vec<ParamDecl>,
    pub returns: Vec<TypeExpr>,
    pub body: Vec<Statement>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ParamDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Statement {
    OpAssign(OpAssign),
    Return(ReturnStmt),
}

#[derive(Debug, Clone)]
pub struct OpAssign {
    pub results: Vec<String>,
    pub op_name: String,
    pub operands: Vec<Operand>,
    pub attrs: Vec<(String, AttrValue)>,
    pub type_sig: Option<TypeSignature>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ReturnStmt {
    pub values: Vec<Operand>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Operand {
    Value(String),
    FuncRef(String),
    Literal(LiteralValue),
}

#[derive(Debug, Clone)]
pub enum LiteralValue {
    Integer(i64),
    Float(f64),
    Bool(bool),
    String(String),
}

#[derive(Debug, Clone)]
pub enum AttrValue {
    Integer(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<AttrValue>),
}

#[derive(Debug, Clone)]
pub struct TypeSignature {
    pub inputs: Vec<TypeExpr>,
    pub outputs: Vec<TypeExpr>,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Tensor(TensorTypeExpr),
    Qubit,
    Bit,
    Hamiltonian(usize),
    Scalar(ScalarTypeExpr),
    Void,
    Index,
    Function(Box<TypeSignature>),
}

#[derive(Debug, Clone)]
pub struct TensorTypeExpr {
    pub shape: Vec<DimExpr>,
    pub dtype: String,
}

#[derive(Debug, Clone)]
pub enum DimExpr {
    Constant(usize),
    Symbolic(String),
    Dynamic,
}

#[derive(Debug, Clone)]
pub struct ScalarTypeExpr {
    pub name: String,
}
