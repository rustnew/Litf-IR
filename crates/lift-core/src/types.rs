use serde::{Serialize, Deserialize};
use crate::interning::{StringId, TypeInternId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TypeId(pub TypeInternId);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CoreType {
    Integer { bits: u32, signed: bool },
    Float { bits: u32 },
    Boolean,
    Tuple(Vec<TypeId>),
    Function { params: Vec<TypeId>, returns: Vec<TypeId> },
    Opaque { dialect: StringId, name: StringId, data: TypeData },
    Void,
    Index,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeData {
    None,
    Tensor(TensorTypeInfo),
    Qubit(QubitTypeInfo),
    ClassicalBit,
    Hamiltonian { num_qubits: usize },
    QuantumState { dimension: usize, repr: StateRepr },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorTypeInfo {
    pub shape: Vec<Dimension>,
    pub dtype: DataType,
    pub layout: MemoryLayout,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QubitTypeInfo {
    Logical,
    Physical {
        id: usize,
        t1_us: OrderedFloat,
        t2_us: OrderedFloat,
        freq_ghz: OrderedFloat,
        fidelity: OrderedFloat,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderedFloat(pub u64);

impl OrderedFloat {
    pub fn from_f64(v: f64) -> Self {
        Self(v.to_bits())
    }

    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dimension {
    Constant(usize),
    Symbolic(String),
    Product(Vec<Dimension>),
}

impl Dimension {
    pub fn is_static(&self) -> bool {
        matches!(self, Dimension::Constant(_))
    }

    pub fn static_value(&self) -> Option<usize> {
        match self {
            Dimension::Constant(v) => Some(*v),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    FP64,
    FP32,
    FP16,
    BF16,
    FP8E4M3,
    FP8E5M2,
    INT64,
    INT32,
    INT16,
    INT8,
    INT4,
    INT2,
    UINT8,
    Bool,
    Index,
}

impl DataType {
    pub fn bit_width(&self) -> u32 {
        match self {
            DataType::FP64 | DataType::INT64 => 64,
            DataType::FP32 | DataType::INT32 => 32,
            DataType::FP16 | DataType::BF16 | DataType::INT16 => 16,
            DataType::FP8E4M3 | DataType::FP8E5M2 | DataType::INT8 | DataType::UINT8 => 8,
            DataType::INT4 => 4,
            DataType::INT2 => 2,
            DataType::Bool => 1,
            DataType::Index => 64,
        }
    }

    pub fn byte_size(&self) -> usize {
        (self.bit_width() as usize + 7) / 8
    }

    pub fn is_float(&self) -> bool {
        matches!(self, DataType::FP64 | DataType::FP32 | DataType::FP16 |
                 DataType::BF16 | DataType::FP8E4M3 | DataType::FP8E5M2)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, DataType::INT64 | DataType::INT32 | DataType::INT16 |
                 DataType::INT8 | DataType::INT4 | DataType::INT2 | DataType::UINT8)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLayout {
    Contiguous,
    NCHW,
    NHWC,
    Strided,
    Tiled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StateRepr {
    StateVector,
    DensityMatrix,
    MPS,
    Stabiliser,
}

impl std::fmt::Display for CoreType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreType::Integer { bits, signed } => {
                write!(f, "{}{}", if *signed { "i" } else { "u" }, bits)
            }
            CoreType::Float { bits } => write!(f, "f{}", bits),
            CoreType::Boolean => write!(f, "bool"),
            CoreType::Void => write!(f, "void"),
            CoreType::Index => write!(f, "index"),
            CoreType::Tuple(elems) => {
                write!(f, "tuple<")?;
                for (i, _) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "type_{}", i)?;
                }
                write!(f, ">")
            }
            CoreType::Function { params, returns } => {
                write!(f, "(")?;
                for (i, _) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "type")?;
                }
                write!(f, ") -> (")?;
                for (i, _) in returns.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "type")?;
                }
                write!(f, ")")
            }
            CoreType::Opaque { data, .. } => {
                match data {
                    TypeData::Tensor(info) => {
                        write!(f, "tensor<")?;
                        for (i, dim) in info.shape.iter().enumerate() {
                            if i > 0 { write!(f, "x")?; }
                            match dim {
                                Dimension::Constant(n) => write!(f, "{}", n)?,
                                Dimension::Symbolic(s) => write!(f, "{}", s)?,
                                Dimension::Product(_) => write!(f, "?")?,
                            }
                        }
                        write!(f, "x{}>", format_dtype(info.dtype))
                    }
                    TypeData::Qubit(_) => write!(f, "qubit"),
                    TypeData::ClassicalBit => write!(f, "bit"),
                    TypeData::Hamiltonian { num_qubits } => {
                        write!(f, "hamiltonian<{}>", num_qubits)
                    }
                    TypeData::QuantumState { dimension, repr } => {
                        write!(f, "qstate<{}, {:?}>", dimension, repr)
                    }
                    TypeData::None => write!(f, "opaque"),
                }
            }
        }
    }
}

fn format_dtype(dt: DataType) -> &'static str {
    match dt {
        DataType::FP64 => "f64",
        DataType::FP32 => "f32",
        DataType::FP16 => "f16",
        DataType::BF16 => "bf16",
        DataType::FP8E4M3 => "fp8e4m3",
        DataType::FP8E5M2 => "fp8e5m2",
        DataType::INT64 => "i64",
        DataType::INT32 => "i32",
        DataType::INT16 => "i16",
        DataType::INT8 => "i8",
        DataType::INT4 => "i4",
        DataType::INT2 => "i2",
        DataType::UINT8 => "u8",
        DataType::Bool => "i1",
        DataType::Index => "index",
    }
}
