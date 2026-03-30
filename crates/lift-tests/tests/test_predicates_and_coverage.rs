use lift_tensor::ops::TensorOp;
use lift_quantum::gates::QuantumGate;
use lift_hybrid::ops::HybridOp;

// ═══════════════════════════════════════════════════════════
// TensorOp predicate completeness
// ═══════════════════════════════════════════════════════════

#[test]
fn test_is_convolution_full() {
    let convs = [
        TensorOp::Conv2D, TensorOp::Conv1D, TensorOp::Conv3D,
        TensorOp::ConvTranspose2D, TensorOp::DepthwiseConv2D,
        TensorOp::DilatedConv2D,
    ];
    for op in &convs { assert!(op.is_convolution()); }
    assert!(!TensorOp::MatMul.is_convolution());
    assert!(!TensorOp::ReLU.is_convolution());
    assert!(!TensorOp::Attention.is_convolution());
}

#[test]
fn test_is_attention_full() {
    let attn = [
        TensorOp::Attention, TensorOp::MultiHeadAttention,
        TensorOp::MultiQueryAttention, TensorOp::GroupedQueryAttention,
        TensorOp::FlashAttention, TensorOp::SlidingWindowAttention,
        TensorOp::CrossAttention, TensorOp::PagedAttention,
    ];
    for op in &attn { assert!(op.is_attention()); }
    assert!(!TensorOp::Conv2D.is_attention());
    assert!(!TensorOp::Softmax.is_attention());
}

#[test]
fn test_is_normalisation_full() {
    let norms = [
        TensorOp::LayerNorm, TensorOp::RMSNorm,
        TensorOp::BatchNorm, TensorOp::GroupNorm,
        TensorOp::InstanceNorm,
    ];
    for op in &norms { assert!(op.is_normalisation()); }
    assert!(!TensorOp::Add.is_normalisation());
}

#[test]
fn test_is_fused_full() {
    let fused = [
        TensorOp::FusedMatMulBiasReLU, TensorOp::FusedMatMulBias,
        TensorOp::FusedLinearGeLU, TensorOp::FusedAttentionLayerNorm,
        TensorOp::FusedLinearSiLU, TensorOp::FusedConvBatchNormReLU,
    ];
    for op in &fused { assert!(op.is_fused()); }
    assert!(!TensorOp::Linear.is_fused());
}

#[test]
fn test_is_gradient_full() {
    let grads = [
        TensorOp::GradMatMul, TensorOp::GradReLU,
        TensorOp::GradSoftmax, TensorOp::GradLayerNorm,
        TensorOp::GradAttention, TensorOp::GradConv2D,
        TensorOp::GradLinear, TensorOp::GradGeLU,
    ];
    for op in &grads { assert!(op.is_gradient()); }
    assert!(!TensorOp::MatMul.is_gradient());
}

#[test]
fn test_is_zero_flop_full() {
    let zeros = [
        TensorOp::Reshape, TensorOp::Transpose,
        TensorOp::Squeeze, TensorOp::Unsqueeze,
        TensorOp::Permute, TensorOp::Expand,
        TensorOp::Slice, TensorOp::Pad, TensorOp::Tile,
        TensorOp::Concat, TensorOp::Split, TensorOp::Gather,
        TensorOp::Scatter, TensorOp::Constant, TensorOp::Zeros,
        TensorOp::Ones, TensorOp::Arange, TensorOp::Full,
        TensorOp::Checkpoint, TensorOp::Offload,
        TensorOp::PipelineSend, TensorOp::PipelineReceive,
        TensorOp::ParallelSplit, TensorOp::ParallelAllReduce,
    ];
    for op in &zeros { assert!(op.is_zero_flop(), "{:?} should be zero-flop", op); }
    assert!(!TensorOp::Add.is_zero_flop());
    assert!(!TensorOp::MatMul.is_zero_flop());
}

#[test]
fn test_activation_not_conv() {
    let activations = [
        TensorOp::ReLU, TensorOp::GeLU, TensorOp::SiLU,
        TensorOp::Sigmoid, TensorOp::Tanh,
    ];
    for op in &activations {
        assert!(op.is_activation());
        assert!(!op.is_convolution());
        assert!(!op.is_attention());
        assert!(!op.is_fused());
        assert!(!op.is_gradient());
    }
}

// ═══════════════════════════════════════════════════════════
// TensorOp — every op has a name
// ═══════════════════════════════════════════════════════════

#[test]
fn test_all_basic_ops_have_name() {
    let ops = [
        TensorOp::Add, TensorOp::Sub, TensorOp::Mul, TensorOp::Div,
        TensorOp::Neg, TensorOp::MatMul, TensorOp::Linear,
        TensorOp::Embedding, TensorOp::Softmax,
    ];
    for op in &ops {
        assert!(!op.name().is_empty(), "{:?} name is empty", op);
        assert!(op.name().starts_with("tensor."), "{:?} name should start with tensor.", op);
    }
}

#[test]
fn test_diffusion_ops_have_names() {
    let ops = [
        TensorOp::UNetDownBlock, TensorOp::UNetUpBlock,
        TensorOp::TimestepEmbedding,
    ];
    for op in &ops {
        assert!(op.name().starts_with("tensor."));
        assert!(TensorOp::from_name(op.name()).is_some());
    }
}

#[test]
fn test_gnn_ops_have_names() {
    let ops = [TensorOp::GNNMessagePassing, TensorOp::GNNGlobalPooling];
    for op in &ops {
        assert!(op.name().starts_with("tensor."));
        assert!(TensorOp::from_name(op.name()).is_some());
    }
}

#[test]
fn test_moe_ops_have_names() {
    let ops = [TensorOp::MoEDispatch, TensorOp::MoECombine];
    for op in &ops {
        assert!(op.name().starts_with("tensor."));
        assert!(TensorOp::from_name(op.name()).is_some());
    }
}

// ═══════════════════════════════════════════════════════════
// QuantumGate — qubit count correctness
// ═══════════════════════════════════════════════════════════

#[test]
fn test_1q_gates_qubit_count() {
    let gates_1q = [
        QuantumGate::H, QuantumGate::X, QuantumGate::Y, QuantumGate::Z,
        QuantumGate::S, QuantumGate::Sdg, QuantumGate::T, QuantumGate::Tdg,
        QuantumGate::SX, QuantumGate::RX, QuantumGate::RY, QuantumGate::RZ,
        QuantumGate::P, QuantumGate::U1, QuantumGate::U3,
        QuantumGate::Rx90, QuantumGate::Rx180,
        QuantumGate::GPI, QuantumGate::GPI2, QuantumGate::VirtualRZ,
    ];
    for g in &gates_1q {
        assert_eq!(g.num_qubits(), 1, "{:?} should be 1-qubit", g);
    }
}

#[test]
fn test_2q_gates_qubit_count() {
    let gates_2q = [
        QuantumGate::CX, QuantumGate::CZ, QuantumGate::CY,
        QuantumGate::SWAP, QuantumGate::ISWAP, QuantumGate::ECR,
        QuantumGate::XX, QuantumGate::YY, QuantumGate::ZZ,
        QuantumGate::CPhase, QuantumGate::XY, QuantumGate::CP,
        QuantumGate::MS, QuantumGate::RZX,
    ];
    for g in &gates_2q {
        assert_eq!(g.num_qubits(), 2, "{:?} should be 2-qubit", g);
    }
}

#[test]
fn test_3q_gates_qubit_count() {
    assert_eq!(QuantumGate::CCX.num_qubits(), 3);
    assert_eq!(QuantumGate::CSWAP.num_qubits(), 3);
}

// ═══════════════════════════════════════════════════════════
// QuantumGate — parametric classification
// ═══════════════════════════════════════════════════════════

#[test]
fn test_non_parametric_gates() {
    let non_param = [
        QuantumGate::H, QuantumGate::X, QuantumGate::Y, QuantumGate::Z,
        QuantumGate::S, QuantumGate::Sdg, QuantumGate::T, QuantumGate::Tdg,
        QuantumGate::CX, QuantumGate::CZ, QuantumGate::SWAP,
    ];
    for g in &non_param {
        assert!(!g.is_parametric(), "{:?} should not be parametric", g);
    }
}

#[test]
fn test_parametric_gates() {
    let param = [
        QuantumGate::RX, QuantumGate::RY, QuantumGate::RZ,
        QuantumGate::P, QuantumGate::U1, QuantumGate::U2, QuantumGate::U3,
        QuantumGate::CPhase, QuantumGate::XY,
        QuantumGate::GPI, QuantumGate::GPI2, QuantumGate::MS,
        QuantumGate::GlobalPhase, QuantumGate::VirtualRZ,
    ];
    for g in &param {
        assert!(g.is_parametric(), "{:?} should be parametric", g);
    }
}

// ═══════════════════════════════════════════════════════════
// HybridOp — all ops have names
// ═══════════════════════════════════════════════════════════

#[test]
fn test_all_hybrid_ops_have_names() {
    let ops = [
        HybridOp::Encode, HybridOp::Decode,
        HybridOp::ParameterShift, HybridOp::FiniteDifference,
        HybridOp::SPSA, HybridOp::JointGradient,
        HybridOp::ClassicalPreprocess, HybridOp::QuantumPostprocess,
        HybridOp::HybridForward, HybridOp::HybridBackward,
        HybridOp::CoExecute,
        HybridOp::AdjointDifferentiation,
        HybridOp::StochasticParameterShift,
        HybridOp::VqcLayer, HybridOp::VqeAnsatz,
        HybridOp::QaoaLayer, HybridOp::QuantumKernel,
        HybridOp::GpuToQpu, HybridOp::QpuToGpu,
        HybridOp::MeasureExpectation, HybridOp::MeasureSamples,
    ];
    for op in &ops {
        let name = op.op_name();
        assert!(!name.is_empty());
        assert!(name.starts_with("hybrid."), "{:?} should start with hybrid.", op);
        let recovered = HybridOp::from_name(name);
        assert_eq!(recovered.as_ref(), Some(op), "roundtrip for {:?}", op);
    }
}

#[test]
fn test_hybrid_gradient_and_variational_disjoint() {
    let ops = [
        HybridOp::Encode, HybridOp::Decode,
        HybridOp::ParameterShift, HybridOp::FiniteDifference,
        HybridOp::SPSA, HybridOp::JointGradient,
        HybridOp::AdjointDifferentiation, HybridOp::StochasticParameterShift,
        HybridOp::VqcLayer, HybridOp::VqeAnsatz,
        HybridOp::QaoaLayer, HybridOp::QuantumKernel,
    ];
    for op in &ops {
        // An op should not be both gradient and variational
        assert!(!(op.is_gradient() && op.is_variational()),
            "{:?} is both gradient and variational", op);
    }
}

#[test]
fn test_hybrid_op_debug_format() {
    let op = HybridOp::VqcLayer;
    let dbg = format!("{:?}", op);
    assert!(dbg.contains("VqcLayer"));
}

// ═══════════════════════════════════════════════════════════
// TensorOp — num_inputs range correctness
// ═══════════════════════════════════════════════════════════

#[test]
fn test_num_inputs_min_leq_max() {
    let ops = [
        TensorOp::Add, TensorOp::MatMul, TensorOp::ReLU,
        TensorOp::Attention, TensorOp::Conv2D, TensorOp::Linear,
        TensorOp::LayerNorm, TensorOp::LSTMCell,
        TensorOp::FlashAttention, TensorOp::FusedMatMulBiasReLU,
        TensorOp::GradMatMul, TensorOp::PagedAttention,
    ];
    for op in &ops {
        let (min, max) = op.num_inputs();
        assert!(min <= max, "{:?}: min {} > max {}", op, min, max);
        assert!(min >= 1, "{:?}: min is 0", op);
    }
}

#[test]
fn test_creation_ops_zero_input() {
    let ops = [
        TensorOp::Constant, TensorOp::Zeros, TensorOp::Ones,
        TensorOp::Arange, TensorOp::Full,
    ];
    for op in &ops {
        let (min, _max) = op.num_inputs();
        assert_eq!(min, 0, "{:?} should accept 0 inputs", op);
    }
}

// ═══════════════════════════════════════════════════════════
// Cross-domain: gate name doesn't collide with tensor name
// ═══════════════════════════════════════════════════════════

#[test]
fn test_no_name_collision_tensor_quantum() {
    let tensor_ops = [
        TensorOp::MatMul, TensorOp::ReLU, TensorOp::Conv2D,
        TensorOp::Attention, TensorOp::Linear,
    ];
    let quantum_gates = [
        QuantumGate::H, QuantumGate::CX, QuantumGate::RZ,
        QuantumGate::Measure,
    ];
    for top in &tensor_ops {
        for qg in &quantum_gates {
            assert_ne!(top.name(), qg.op_name());
        }
    }
}

#[test]
fn test_no_name_collision_tensor_hybrid() {
    let tensor_ops = [TensorOp::MatMul, TensorOp::ReLU];
    let hybrid_ops = [HybridOp::Encode, HybridOp::VqcLayer];
    for top in &tensor_ops {
        for hop in &hybrid_ops {
            assert_ne!(top.name(), hop.op_name());
        }
    }
}

#[test]
fn test_no_name_collision_quantum_hybrid() {
    let quantum_gates = [QuantumGate::H, QuantumGate::CX];
    let hybrid_ops = [HybridOp::Encode, HybridOp::VqcLayer];
    for qg in &quantum_gates {
        for hop in &hybrid_ops {
            assert_ne!(qg.op_name(), hop.op_name());
        }
    }
}

// ═══════════════════════════════════════════════════════════
// TensorOp — flops_formula returns non-empty string for all
// ═══════════════════════════════════════════════════════════

#[test]
fn test_flops_formula_non_empty_basic() {
    let ops = [
        TensorOp::MatMul, TensorOp::Add, TensorOp::Conv2D,
        TensorOp::ReLU, TensorOp::Attention, TensorOp::LayerNorm,
        TensorOp::Reshape, TensorOp::Softmax,
    ];
    for op in &ops {
        assert!(!op.flops_formula().is_empty(), "{:?} has empty formula", op);
    }
}

#[test]
fn test_flops_formula_fused_ops() {
    let ops = [
        TensorOp::FusedMatMulBiasReLU, TensorOp::FusedLinearGeLU,
        TensorOp::FusedAttentionLayerNorm,
    ];
    for op in &ops {
        let f = op.flops_formula();
        assert!(!f.is_empty());
    }
}
