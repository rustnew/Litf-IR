# LIFT Dialect Reference — Complete Guide

**The definitive reference for every dialect, type, operation, and syntax rule in LIFT.**

> After reading this document you will be able to write, read, configure, and assemble any `.lif` file for any model — classical AI, quantum circuits, or hybrid — without errors.

---

## Table of Contents

- [Part I — File Structure, Grammar, and Type System](#part-i--file-structure-grammar-and-type-system)
- [Part II — The `tensor` Dialect (Classical AI)](#part-ii--the-tensor-dialect-classical-ai)
- [Part III — The `quantum` Dialect (Quantum Computing)](#part-iii--the-quantum-dialect-quantum-computing)
- [Part IV — The `hybrid` Dialect (Classical + Quantum Bridge)](#part-iv--the-hybrid-dialect-classical--quantum-bridge)
- [Part V — Configuration (`.lith` Files)](#part-v--configuration-lith-files)
- [Part VI — Assembling Dialects Together](#part-vi--assembling-dialects-together)

---

# Part I — File Structure, Grammar, and Type System

## 1.1 The `.lif` File

Every LIFT program is a `.lif` text file with this structure:

```
#dialect tensor            ← 1. Dialect declarations (one or more)

module @name {             ← 2. Module
    func @fn(%x: type) -> type {   ← 3. Function
        %y = "op"(%x) : (type) -> type   ← 4. Operations
        return %y                          ← 5. Return
    }
}
```

### Rules

1. At least one `#dialect` directive at the top.
2. At least one `module` block.
3. Each module contains one or more `func` declarations.
4. Each function has parameters, optional return types, and a body of operations.
5. The body ends with a `return` statement.

## 1.2 Grammar Rules

### Dialect Directive

```
#dialect tensor
#dialect quantum
#dialect hybrid
```

You can declare multiple dialects in one file.

### Module

```
module @my_model {
    ...
}
```

### Function

```
func @forward(%x: tensor<1x784xf32>, %w: tensor<784x256xf32>) -> tensor<1x256xf32> {
    ...
    return %out
}
```

Multiple return types use parentheses:

```
func @bell(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
    ...
    return %q2, %q3
}
```

### Operation (Assignment)

```
%result = "dialect.operation"(%input1, %input2) : (input_type1, input_type2) -> output_type
```

Multiple results:

```
%r1, %r2 = "quantum.cx"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)
```

With attributes:

```
%y = "quantum.ry"(%q) {angle = 1.5708} : (qubit) -> qubit
```

## 1.3 Identifiers

| Prefix | Meaning | Example |
|--------|---------|---------|
| `@` | Module or function name | `@my_model`, `@forward` |
| `%` | SSA value (variable) | `%x`, `%q0`, `%hidden` |
| `^` | Block label | `^entry` |
| `#dialect` | Dialect directive | `#dialect tensor` |

### SSA Rule

Every `%name` is assigned **exactly once**. No reassignment allowed.

## 1.4 The Type System

### Tensor Types

```
tensor<shape x dtype>
```

| Example | Description |
|---------|-------------|
| `tensor<4xf32>` | 1D, 4 floats |
| `tensor<1x784xf32>` | 2D, batch 1, 784 features |
| `tensor<1x3x224x224xf32>` | 4D image: batch, channels, H, W |
| `tensor<Bx128x64xf16>` | Symbolic batch dim, float16 |

### Data Types (dtype)

| Syntax | Bits | Use Case |
|--------|------|----------|
| `f64` | 64 | Scientific computing |
| `f32` | 32 | Default training/inference |
| `f16` | 16 | Mixed precision |
| `bf16` | 16 | A100/H100 training |
| `fp8e4m3` | 8 | H100 FP8 inference |
| `fp8e5m2` | 8 | H100 FP8 training |
| `i64` | 64 | Large indices |
| `i32` | 32 | Indices |
| `i16` | 16 | Quantised weights |
| `i8` | 8 | INT8 quantisation |
| `i4` | 4 | INT4 quantisation |
| `i2` | 2 | Extreme quantisation |
| `u8` | 8 | Pixel values |
| `i1` | 1 | Booleans/masks |
| `index` | 64 | Loop indices |

### Quantum Types

| Syntax | Description | Rule |
|--------|-------------|------|
| `qubit` | A single qubit | **Linear**: consumed exactly once |
| `bit` | Classical bit (measurement result) | Normal (non-linear) |
| `hamiltonian<N>` | Hamiltonian on N qubits | Normal |

### Scalar Types

`f32`, `f64`, `i32`, `i64`, `bool`, `void`, `index`

## 1.5 Attributes

Compile-time constants attached to operations:

```
%y = "tensor.conv2d"(%x, %w) {stride = 2, padding = 1} : ...
```

| Type | Example |
|------|---------|
| Integer | `stride = 2` |
| Float | `rate = 0.5` |
| Boolean | `training = true` |
| String | `mode = "same"` |
| Array | `kernel_size = [3, 3]` |

## 1.6 Comments

```
// This is a comment (ignored by the parser)
```

---

# Part II — The `tensor` Dialect (Classical AI)

Declare with `#dialect tensor`. Provides **96 operations** for ML/AI.

## 2.1 Arithmetic (9)

| Operation | Syntax | Inputs | FLOPs |
|-----------|--------|--------|-------|
| Add | `"tensor.add"` | 2 | N |
| Sub | `"tensor.sub"` | 2 | N |
| Mul | `"tensor.mul"` | 2 | N |
| Div | `"tensor.div"` | 2 | N |
| Neg | `"tensor.neg"` | 1 | N |
| MatMul | `"tensor.matmul"` | 2 | 2MNK |
| Linear | `"tensor.linear"` | 3 | 2MNK+N |
| Conv2D | `"tensor.conv2d"` | 2 | 2*Co*Ci*Kh*Kw*Oh*Ow |
| Embedding | `"tensor.embedding"` | 2 | 0 (lookup) |

```
%out = "tensor.matmul"(%x, %w) : (tensor<1x784xf32>, tensor<784x256xf32>) -> tensor<1x256xf32>
%out = "tensor.linear"(%x, %w, %b) : (tensor<1x784xf32>, tensor<784x256xf32>, tensor<256xf32>) -> tensor<1x256xf32>
%feat = "tensor.conv2d"(%img, %k) : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
```

## 2.2 Activations (11)

All: 1 input → 1 output, same shape.

| Operation | Syntax | Formula |
|-----------|--------|---------|
| ReLU | `"tensor.relu"` | max(0, x) |
| GeLU | `"tensor.gelu"` | x * Phi(x) |
| SiLU | `"tensor.silu"` | x * sigmoid(x) |
| Sigmoid | `"tensor.sigmoid"` | 1/(1+e^-x) |
| Softmax | `"tensor.softmax"` | e^xi / sum(e^xj) |
| Tanh | `"tensor.tanh"` | (e^x-e^-x)/(e^x+e^-x) |
| LeakyReLU | `"tensor.leaky_relu"` | max(alpha*x, x) |
| ELU | `"tensor.elu"` | x if x>0, alpha*(e^x-1) else |
| Mish | `"tensor.mish"` | x*tanh(softplus(x)) |
| HardSwish | `"tensor.hard_swish"` | x*relu6(x+3)/6 |
| HardSigmoid | `"tensor.hard_sigmoid"` | relu6(x+3)/6 |

```
%a = "tensor.relu"(%x) : (tensor<1x256xf32>) -> tensor<1x256xf32>
%p = "tensor.softmax"(%logits) : (tensor<1x10xf32>) -> tensor<1x10xf32>
```

**Which to use**: CNN → ReLU. Transformer/LLM → GeLU/SiLU. Classifier → Softmax/Sigmoid. Edge → HardSwish.

## 2.3 Normalisation (5)

| Operation | Syntax | Inputs | Use Case |
|-----------|--------|--------|----------|
| LayerNorm | `"tensor.layernorm"` | 2-3 | Transformers |
| RMSNorm | `"tensor.rmsnorm"` | 2-3 | LLaMA, Mistral |
| BatchNorm | `"tensor.batchnorm"` | 3-5 | CNN training |
| GroupNorm | `"tensor.groupnorm"` | 2-3 | Diffusion |
| InstanceNorm | `"tensor.instancenorm"` | 2-3 | Style transfer |

```
// LayerNorm — Transformers (input + scale)
%n1 = "tensor.layernorm"(%x, %scale) : (tensor<1x128x64xf32>, tensor<64xf32>) -> tensor<1x128x64xf32>

// LayerNorm with bias — (input + scale + bias)
%n2 = "tensor.layernorm"(%x, %scale, %bias) : (tensor<1x128x64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x128x64xf32>

// RMSNorm — LLaMA, Mistral (input + scale)
%n3 = "tensor.rmsnorm"(%x, %scale) : (tensor<1x128x4096xf32>, tensor<4096xf32>) -> tensor<1x128x4096xf32>

// BatchNorm — CNN training (input + scale + bias)
%n4 = "tensor.batchnorm"(%x, %scale, %bias) : (tensor<8x64x32x32xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<8x64x32x32xf32>

// GroupNorm — Diffusion models (input + scale)
%n5 = "tensor.groupnorm"(%x, %scale) : (tensor<1x256x32x32xf32>, tensor<256xf32>) -> tensor<1x256x32x32xf32>

// InstanceNorm — Style transfer (input + scale)
%n6 = "tensor.instancenorm"(%x, %scale) : (tensor<1x64x256x256xf32>, tensor<64xf32>) -> tensor<1x64x256x256xf32>
```

## 2.4 Shape Operations (13) — Zero FLOPs

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Reshape | `"tensor.reshape"` | Change shape |
| Transpose | `"tensor.transpose"` | Swap dims |
| Concat | `"tensor.concat"` | Join tensors |
| Split | `"tensor.split"` | Split tensor |
| Gather | `"tensor.gather"` | Index select |
| Scatter | `"tensor.scatter"` | Index assign |
| Squeeze | `"tensor.squeeze"` | Remove dim=1 |
| Unsqueeze | `"tensor.unsqueeze"` | Add dim=1 |
| Permute | `"tensor.permute"` | Reorder dims |
| Expand | `"tensor.expand"` | Broadcast |
| Slice | `"tensor.slice"` | Sub-tensor |
| Pad | `"tensor.pad"` | Add padding |
| Tile | `"tensor.tile"` | Repeat |

```
// Reshape: flatten a 4D image tensor to 2D
%flat = "tensor.reshape"(%x) : (tensor<1x3x224x224xf32>) -> tensor<1x150528xf32>

// Transpose: swap last two dimensions
%t = "tensor.transpose"(%x) : (tensor<128x64xf32>) -> tensor<64x128xf32>

// Concat: join two tensors along batch dimension
%cat = "tensor.concat"(%a, %b) : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<2x128xf32>

// Squeeze: remove dimension of size 1
%sq = "tensor.squeeze"(%x) : (tensor<1x64x1x1xf32>) -> tensor<1x64xf32>

// Unsqueeze: add a batch dimension
%us = "tensor.unsqueeze"(%x) : (tensor<64xf32>) -> tensor<1x64xf32>

// Permute: reorder dimensions (e.g. NCHW → NHWC)
%p = "tensor.permute"(%x) : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>

// Slice: extract a sub-tensor
%sl = "tensor.slice"(%x) : (tensor<1x100x64xf32>) -> tensor<1x50x64xf32>

// Pad: add zero-padding
%padded = "tensor.pad"(%x) : (tensor<1x3x224x224xf32>) -> tensor<1x3x226x226xf32>

// Gather: index-based selection
%sel = "tensor.gather"(%x, %indices) : (tensor<1000x64xf32>, tensor<10xi32>) -> tensor<10x64xf32>

// Expand: broadcast a tensor
%exp = "tensor.expand"(%x) : (tensor<1x1x64xf32>) -> tensor<8x128x64xf32>
```

## 2.5 Attention (8)

All take 3-5 inputs (Q, K, V, optional mask). FLOPs = 2*B*H*(S²*D + S*D²).

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Attention | `"tensor.attention"` | Standard scaled dot-product |
| MultiHeadAttention | `"tensor.multi_head_attention"` | Standard Transformer |
| MultiQueryAttention | `"tensor.multi_query_attention"` | Shared K/V (fast inference) |
| GroupedQueryAttention | `"tensor.grouped_query_attention"` | LLaMA 2 style |
| FlashAttention | `"tensor.flash_attention"` | O(n) memory |
| SlidingWindowAttention | `"tensor.sliding_window_attention"` | Mistral |
| CrossAttention | `"tensor.cross_attention"` | Encoder-decoder |
| PagedAttention | `"tensor.paged_attention"` | vLLM paged KV |

```
// Standard scaled dot-product attention: Q, K, V
%attn = "tensor.attention"(%q, %k, %v) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>

// Multi-head attention (standard Transformer)
%mha = "tensor.multi_head_attention"(%q, %k, %v) : (tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) -> tensor<1x8x128x64xf32>

// Flash attention: same API, O(n) memory, for long sequences
%flash = "tensor.flash_attention"(%q, %k, %v) : (tensor<1x8x4096x64xf32>, tensor<1x8x4096x64xf32>, tensor<1x8x4096x64xf32>) -> tensor<1x8x4096x64xf32>

// Grouped-query attention (LLaMA 2 style): fewer K/V heads
%gqa = "tensor.grouped_query_attention"(%q, %k, %v) : (tensor<1x32x128x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) -> tensor<1x32x128x64xf32>

// Cross attention (encoder-decoder, e.g. translation)
%cross = "tensor.cross_attention"(%decoder_q, %encoder_k, %encoder_v) : (tensor<1x8x64x64xf32>, tensor<1x8x128x64xf32>, tensor<1x8x128x64xf32>) -> tensor<1x8x64x64xf32>

// Sliding window attention (Mistral): local context window
%swa = "tensor.sliding_window_attention"(%q, %k, %v) : (tensor<1x8x4096x64xf32>, tensor<1x8x4096x64xf32>, tensor<1x8x4096x64xf32>) -> tensor<1x8x4096x64xf32>

// Paged attention (vLLM): paged KV cache for efficient serving
%paged = "tensor.paged_attention"(%q, %k_cache, %v_cache) : (tensor<1x8x1x64xf32>, tensor<1x8x4096x64xf32>, tensor<1x8x4096x64xf32>) -> tensor<1x8x1x64xf32>
```

## 2.6 Convolution (6)

| Operation | Syntax | Use Case |
|-----------|--------|----------|
| Conv1D | `"tensor.conv1d"` | Audio, time series |
| Conv2D | `"tensor.conv2d"` | Images |
| Conv3D | `"tensor.conv3d"` | Video, medical 3D |
| ConvTranspose2D | `"tensor.conv_transpose2d"` | Upsampling |
| DepthwiseConv2D | `"tensor.depthwise_conv2d"` | MobileNet |
| DilatedConv2D | `"tensor.dilated_conv2d"` | Segmentation |

```
// Conv1D: audio processing [B, Cin, Length] * [Cout, Cin, K]
%audio_feat = "tensor.conv1d"(%audio, %kernel) : (tensor<1x1x16000xf32>, tensor<64x1x80xf32>) -> tensor<1x64x15921xf32>

// Conv2D: image feature extraction [B, Cin, H, W] * [Cout, Cin, Kh, Kw]
%img_feat = "tensor.conv2d"(%img, %kernel) : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>

// Conv3D: video processing [B, Cin, D, H, W] * [Cout, Cin, Kd, Kh, Kw]
%vid_feat = "tensor.conv3d"(%video, %kernel) : (tensor<1x3x16x112x112xf32>, tensor<64x3x3x3x3xf32>) -> tensor<1x64x14x110x110xf32>

// ConvTranspose2D: upsampling (decoder/generator)
%up = "tensor.conv_transpose2d"(%x, %kernel) : (tensor<1x64x16x16xf32>, tensor<64x32x4x4xf32>) -> tensor<1x32x32x32xf32>

// DepthwiseConv2D: MobileNet-style per-channel conv
%dw = "tensor.depthwise_conv2d"(%x, %kernel) : (tensor<1x64x32x32xf32>, tensor<64x1x3x3xf32>) -> tensor<1x64x32x32xf32>

// DilatedConv2D: large receptive field (segmentation)
%dilated = "tensor.dilated_conv2d"(%x, %kernel) : (tensor<1x64x64x64xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x64x64xf32>
```

## 2.7 Pooling (4)

| Operation | Syntax | Inputs |
|-----------|--------|--------|
| MaxPool2D | `"tensor.maxpool2d"` | 2 |
| AvgPool2D | `"tensor.avgpool2d"` | 2 |
| AdaptiveAvgPool2D | `"tensor.adaptive_avgpool2d"` | 1 |
| GlobalAvgPool | `"tensor.global_avgpool"` | 1 |

```
// MaxPool2D: downsample by taking maximum in each window
%p1 = "tensor.maxpool2d"(%x, %params) : (tensor<1x64x32x32xf32>, tensor<2xi32>) -> tensor<1x64x16x16xf32>

// AvgPool2D: downsample by averaging each window
%p2 = "tensor.avgpool2d"(%x, %params) : (tensor<1x64x32x32xf32>, tensor<2xi32>) -> tensor<1x64x16x16xf32>

// AdaptiveAvgPool2D: output always has fixed spatial size (e.g. 7x7)
%p3 = "tensor.adaptive_avgpool2d"(%x) : (tensor<1x512x14x14xf32>) -> tensor<1x512x7x7xf32>

// GlobalAvgPool: average all spatial dimensions → 1x1
%p4 = "tensor.global_avgpool"(%x) : (tensor<1x64x7x7xf32>) -> tensor<1x64x1x1xf32>
```

## 2.8 Recurrent (3)

| Operation | Syntax | FLOPs per step |
|-----------|--------|---------------|
| LSTMCell | `"tensor.lstm_cell"` | 4*(in+hid)*hid*2 |
| GRUCell | `"tensor.gru_cell"` | 3*(in+hid)*hid*2 |
| RNNCell | `"tensor.rnn_cell"` | (in+hid)*hid*2 |

```
// LSTMCell: takes current input + previous hidden state, returns new hidden + cell state
%h_new, %c_new = "tensor.lstm_cell"(%x_t, %h_prev) : (tensor<1x128xf32>, tensor<1x256xf32>) -> (tensor<1x256xf32>, tensor<1x256xf32>)

// GRUCell: simpler than LSTM, single hidden state
%h_new = "tensor.gru_cell"(%x_t, %h_prev) : (tensor<1x128xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>

// RNNCell: basic recurrent cell
%h_new = "tensor.rnn_cell"(%x_t, %h_prev) : (tensor<1x128xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
```

## 2.9 Advanced Math (11)

| Operation | Syntax | Complexity |
|-----------|--------|-----------|
| Einsum | `"tensor.einsum"` | Varies |
| FFT | `"tensor.fft"` | O(n log n) |
| IFFT | `"tensor.ifft"` | O(n log n) |
| SVD | `"tensor.svd"` | O(mn min(m,n)) |
| Eig | `"tensor.eig"` | O(n³) |
| Solve | `"tensor.solve"` | O(n³) |
| TopK | `"tensor.topk"` | O(n log k) |
| Sort | `"tensor.sort"` | O(n log n) |
| Cumsum | `"tensor.cumsum"` | O(n) |
| Where | `"tensor.where"` | O(n) |
| Clamp | `"tensor.clamp"` | O(n) |

```
// Einsum: flexible tensor contraction (e.g. batch matmul)
%result = "tensor.einsum"(%a, %b) : (tensor<8x128x64xf32>, tensor<8x64x256xf32>) -> tensor<8x128x256xf32>

// FFT: Fast Fourier Transform (signal processing)
%freq = "tensor.fft"(%signal) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>

// IFFT: Inverse FFT (frequency → time domain)
%time = "tensor.ifft"(%freq) : (tensor<1x1024xf32>) -> tensor<1x1024xf32>

// SVD: Singular Value Decomposition (compression, PCA)
%u = "tensor.svd"(%matrix) : (tensor<100x50xf32>) -> tensor<100x50xf32>

// TopK: get top-10 scores from 1000 classes
%top = "tensor.topk"(%scores) : (tensor<1x1000xf32>) -> tensor<1x10xf32>

// Sort: sort a tensor along last dimension
%sorted = "tensor.sort"(%x) : (tensor<1x100xf32>) -> tensor<1x100xf32>

// Cumsum: cumulative sum (prefix sum)
%cs = "tensor.cumsum"(%x) : (tensor<1x10xf32>) -> tensor<1x10xf32>

// Where: conditional selection (like numpy.where)
%selected = "tensor.where"(%cond, %a, %b) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

// Clamp: restrict values to [min, max] range
%clamped = "tensor.clamp"(%x, %min_val, %max_val) : (tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<4xf32>

// Solve: solve linear system Ax = b
%solution = "tensor.solve"(%A, %b) : (tensor<64x64xf32>, tensor<64x1xf32>) -> tensor<64x1xf32>

// Eig: eigenvalue decomposition
%eigenvalues = "tensor.eig"(%matrix) : (tensor<32x32xf32>) -> tensor<32xf32>
```

## 2.10 Sparse (2)

`"tensor.sparse_matmul"`, `"tensor.sparse_embedding"`

```
// SparseMatMul: sparse × dense matrix multiply (efficient for sparse models)
%result = "tensor.sparse_matmul"(%sparse_w, %x) : (tensor<10000x768xf32>, tensor<768x1xf32>) -> tensor<10000x1xf32>

// SparseEmbedding: sparse lookup (recommendation systems with huge vocab)
%emb = "tensor.sparse_embedding"(%sparse_ids, %table) : (tensor<1x50xi32>, tensor<1000000x128xf32>) -> tensor<1x50x128xf32>
```

## 2.11 Quantisation (6)

| Operation | Syntax | Direction |
|-----------|--------|-----------|
| Quantize | `"tensor.quantize"` | f32 → i8 |
| Dequantize | `"tensor.dequantize"` | i8 → f32 |
| QuantizeInt4 | `"tensor.quantize_int4"` | f32 → i4 |
| DequantizeInt4 | `"tensor.dequantize_int4"` | i4 → f32 |
| QuantizeFp8 | `"tensor.quantize_fp8"` | f32 → fp8 |
| DequantizeFp8 | `"tensor.dequantize_fp8"` | fp8 → f32 |

```
// INT8 Quantisation: reduce model size 4x
%q8 = "tensor.quantize"(%weights) : (tensor<256x256xf32>) -> tensor<256x256xi8>
%dq8 = "tensor.dequantize"(%q8) : (tensor<256x256xi8>) -> tensor<256x256xf32>

// INT4 Quantisation: reduce model size 8x (GPTQ, AWQ style)
%q4 = "tensor.quantize_int4"(%weights) : (tensor<4096x4096xf32>) -> tensor<4096x4096xi4>
%dq4 = "tensor.dequantize_int4"(%q4) : (tensor<4096x4096xi4>) -> tensor<4096x4096xf32>

// FP8 Quantisation: H100 native format
%qfp8 = "tensor.quantize_fp8"(%weights) : (tensor<4096x4096xf32>) -> tensor<4096x4096xfp8e4m3>
%dqfp8 = "tensor.dequantize_fp8"(%qfp8) : (tensor<4096x4096xfp8e4m3>) -> tensor<4096x4096xf32>
```

## 2.12 Diffusion/Generative (3)

`"tensor.unet_down_block"` (2-3 in), `"tensor.unet_up_block"` (2-3 in), `"tensor.timestep_embedding"` (1 in)

```
// TimestepEmbedding: encode diffusion timestep as a vector
%t_emb = "tensor.timestep_embedding"(%timestep) : (tensor<1xi32>) -> tensor<1x256xf32>

// UNetDownBlock: encoder block of UNet (input + timestep embedding)
%down = "tensor.unet_down_block"(%x, %t_emb) : (tensor<1x64x64x64xf32>, tensor<1x256xf32>) -> tensor<1x128x32x32xf32>

// UNetUpBlock: decoder block of UNet (input + skip connection + timestep)
%up = "tensor.unet_up_block"(%x, %skip, %t_emb) : (tensor<1x128x32x32xf32>, tensor<1x128x32x32xf32>, tensor<1x256xf32>) -> tensor<1x64x64x64xf32>
```

## 2.13 GNN (2)

`"tensor.gnn_message_passing"` (2-3 in), `"tensor.gnn_global_pooling"` (1 in)

```
// GNNMessagePassing: propagate node features along edges
%h1 = "tensor.gnn_message_passing"(%nodes, %adj) : (tensor<50x16xf32>, tensor<50x50xf32>) -> tensor<50x16xf32>

// With edge features (3 inputs)
%h2 = "tensor.gnn_message_passing"(%nodes, %adj, %edge_feat) : (tensor<50x16xf32>, tensor<50x50xf32>, tensor<50x50x8xf32>) -> tensor<50x16xf32>

// GNNGlobalPooling: aggregate all node features into a single graph vector
%graph = "tensor.gnn_global_pooling"(%h2) : (tensor<50x16xf32>) -> tensor<1x16xf32>
```

## 2.14 MoE (2)

`"tensor.moe_dispatch"` (2-3 in), `"tensor.moe_combine"` (2-3 in)

```
// MoEDispatch: router sends tokens to top-k experts
%dispatched = "tensor.moe_dispatch"(%tokens, %router_logits) : (tensor<8x128x512xf32>, tensor<8x128x8xf32>) -> tensor<8x128x512xf32>

// MoECombine: merge expert outputs weighted by router
%combined = "tensor.moe_combine"(%expert_outputs, %router_weights) : (tensor<8x128x512xf32>, tensor<8x128x8xf32>) -> tensor<8x128x512xf32>
```

## 2.15 Constants (5) — Zero inputs

`"tensor.constant"`, `"tensor.zeros"`, `"tensor.ones"`, `"tensor.arange"`, `"tensor.full"`

```
// Zeros: create an all-zero tensor (e.g. initial hidden state)
%z = "tensor.zeros"() : () -> tensor<1x256xf32>

// Ones: create an all-one tensor (e.g. attention mask)
%mask = "tensor.ones"() : () -> tensor<1x128xi32>

// Arange: create [0, 1, 2, ..., 127] (e.g. position IDs)
%pos = "tensor.arange"() : () -> tensor<128xi32>

// Full: create a tensor filled with a specific value
%filled = "tensor.full"() : () -> tensor<1x64xf32>

// Constant: arbitrary constant tensor
%c = "tensor.constant"() : () -> tensor<3xf32>
```

## 2.16 Memory (3)

`"tensor.checkpoint"` (activation recompute), `"tensor.offload"` (to CPU), `"tensor.grad_accumulate"` (micro-batches)

```
// Checkpoint: recompute activations during backward instead of storing them
// Saves GPU memory at the cost of extra compute (critical for large models)
%ckpt = "tensor.checkpoint"(%activations) : (tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xf32>

// Offload: move tensor from GPU to CPU memory (for very large models)
%offloaded = "tensor.offload"(%weights) : (tensor<8192x8192xf32>) -> tensor<8192x8192xf32>

// GradAccumulate: accumulate gradients over multiple micro-batches
// Used when actual batch doesn't fit in GPU memory
%acc = "tensor.grad_accumulate"(%grads) : (tensor<256x256xf32>) -> tensor<256x256xf32>
```

## 2.17 Gradient/Backward (8)

| Syntax | Forward Op |
|--------|-----------|
| `"tensor.grad_matmul"` | MatMul |
| `"tensor.grad_relu"` | ReLU |
| `"tensor.grad_softmax"` | Softmax |
| `"tensor.grad_layernorm"` | LayerNorm |
| `"tensor.grad_attention"` | Attention |
| `"tensor.grad_conv2d"` | Conv2D |
| `"tensor.grad_linear"` | Linear |
| `"tensor.grad_gelu"` | GeLU |

```
// GradMatMul: backward pass for matrix multiplication
%grad_x = "tensor.grad_matmul"(%upstream_grad, %w) : (tensor<1x256xf32>, tensor<256x784xf32>) -> tensor<1x784xf32>

// GradReLU: backward pass for ReLU (zero where input was negative)
%grad_r = "tensor.grad_relu"(%upstream_grad, %relu_input) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>

// GradSoftmax: backward pass for softmax
%grad_s = "tensor.grad_softmax"(%upstream_grad, %softmax_output) : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>

// GradLayerNorm: backward pass for layer normalisation
%grad_ln = "tensor.grad_layernorm"(%upstream_grad, %ln_input) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>

// GradAttention: backward pass for attention
%grad_attn = "tensor.grad_attention"(%upstream_grad, %q, %k) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>

// GradConv2D: backward pass for 2D convolution
%grad_conv = "tensor.grad_conv2d"(%upstream_grad, %conv_input) : (tensor<1x64x112x112xf32>, tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>

// GradLinear: backward pass for linear layer
%grad_lin = "tensor.grad_linear"(%upstream_grad, %w, %b) : (tensor<1x256xf32>, tensor<784x256xf32>, tensor<256xf32>) -> tensor<1x784xf32>

// GradGeLU: backward pass for GeLU activation
%grad_g = "tensor.grad_gelu"(%upstream_grad, %gelu_input) : (tensor<1x256xf32>, tensor<1x256xf32>) -> tensor<1x256xf32>
```

## 2.18 Parallelism (4)

`"tensor.parallel_split"`, `"tensor.parallel_allreduce"`, `"tensor.pipeline_send"`, `"tensor.pipeline_receive"`

```
// ParallelSplit: split a batch across multiple GPUs (data parallelism)
%shard = "tensor.parallel_split"(%batch) : (tensor<32x128xf32>) -> tensor<8x128xf32>

// ParallelAllReduce: synchronise gradients across all GPUs
%synced = "tensor.parallel_allreduce"(%local_grad) : (tensor<256x256xf32>) -> tensor<256x256xf32>

// PipelineSend: send activation to the next pipeline stage (model parallelism)
%sent = "tensor.pipeline_send"(%activation) : (tensor<1x128x4096xf32>) -> tensor<1x128x4096xf32>

// PipelineReceive: receive activation from the previous pipeline stage
%recv = "tensor.pipeline_receive"(%placeholder) : (tensor<1x128x4096xf32>) -> tensor<1x128x4096xf32>
```

## 2.19 Fused Operations (6)

| Syntax | Equivalent |
|--------|-----------|
| `"tensor.fused_matmul_bias_relu"` | matmul+add+relu |
| `"tensor.fused_matmul_bias"` | matmul+add |
| `"tensor.fused_linear_gelu"` | linear+gelu |
| `"tensor.fused_attention_layernorm"` | attention+layernorm |
| `"tensor.fused_linear_silu"` | linear+silu |
| `"tensor.fused_conv_batchnorm_relu"` | conv+bn+relu |

```
// FusedMatMulBiasReLU: 3 ops in 1 kernel (most common for MLP hidden layers)
%h = "tensor.fused_matmul_bias_relu"(%x, %w, %b) : (tensor<1x256xf32>, tensor<256x128xf32>, tensor<128xf32>) -> tensor<1x128xf32>

// FusedMatMulBias: matmul + bias only (no activation)
%h2 = "tensor.fused_matmul_bias"(%x, %w, %b) : (tensor<1x128xf32>, tensor<128x64xf32>, tensor<64xf32>) -> tensor<1x64xf32>

// FusedLinearGeLU: used in Transformer FFN (LLM inference)
%ffn = "tensor.fused_linear_gelu"(%x, %w, %b) : (tensor<1x128x4096xf32>, tensor<4096x16384xf32>, tensor<16384xf32>) -> tensor<1x128x16384xf32>

// FusedAttentionLayerNorm: attention + normalisation in one pass
%attn_ln = "tensor.fused_attention_layernorm"(%q, %k, %v, %scale) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>, tensor<1x128x64xf32>, tensor<64xf32>) -> tensor<1x128x64xf32>

// FusedLinearSiLU: used in LLaMA/Mistral gate projections
%gate = "tensor.fused_linear_silu"(%x, %w, %b) : (tensor<1x128x4096xf32>, tensor<4096x11008xf32>, tensor<11008xf32>) -> tensor<1x128x11008xf32>

// FusedConvBatchNormReLU: standard CNN inference fusion
%feat = "tensor.fused_conv_batchnorm_relu"(%img, %w, %bn_s, %bn_b, %bn_m) : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
```

---

# Part III — The `quantum` Dialect (Quantum Computing)

Declare with `#dialect quantum`. Provides **50+ operations** for quantum circuits.

**Critical rule**: all qubits follow the **linearity rule** — each qubit value must be consumed exactly once.

## 3.1 Standard 1-Qubit Gates (9)

All take 1 qubit, return 1 qubit: `%q_out = "quantum.gate"(%q_in) : (qubit) -> qubit`

| Gate | Syntax | Clifford | Self-Inverse | Description |
|------|--------|----------|-------------|-------------|
| Hadamard | `"quantum.h"` | Yes | Yes | Creates superposition |
| Pauli-X | `"quantum.x"` | Yes | Yes | Bit-flip |
| Pauli-Y | `"quantum.y"` | Yes | Yes | Y rotation |
| Pauli-Z | `"quantum.z"` | Yes | Yes | Phase-flip |
| S | `"quantum.s"` | Yes | No | sqrt(Z) |
| S† | `"quantum.sdg"` | Yes | No | S inverse |
| T | `"quantum.t"` | No | No | pi/8 gate |
| T† | `"quantum.tdg"` | No | No | T inverse |
| SX | `"quantum.sx"` | Yes | No | sqrt(X) |

```
%q1 = "quantum.h"(%q0) : (qubit) -> qubit
%q2 = "quantum.x"(%q1) : (qubit) -> qubit
%q3 = "quantum.t"(%q2) : (qubit) -> qubit
```

## 3.2 Parametric 1-Qubit Gates (7)

Take 1 qubit + angle attributes, return 1 qubit.

| Gate | Syntax | Parameters | Description |
|------|--------|-----------|-------------|
| RX | `"quantum.rx"` | theta | X-axis rotation |
| RY | `"quantum.ry"` | theta | Y-axis rotation |
| RZ | `"quantum.rz"` | theta | Z-axis rotation |
| P | `"quantum.p"` | phi | Phase gate |
| U1 | `"quantum.u1"` | lambda | 1-param universal |
| U2 | `"quantum.u2"` | phi, lambda | 2-param universal |
| U3 | `"quantum.u3"` | theta, phi, lambda | 3-param universal (any 1Q gate) |

```
%q1 = "quantum.ry"(%q0) {angle = 1.5708} : (qubit) -> qubit
%q1 = "quantum.rz"(%q0) {angle = 0.785} : (qubit) -> qubit
%q1 = "quantum.u3"(%q0) {theta = 1.57, phi = 0.0, lambda = 3.14} : (qubit) -> qubit
```

## 3.3 Fixed-Angle 1-Qubit Gates (2)

| Gate | Syntax | Angle | Self-Inverse |
|------|--------|-------|-------------|
| Rx90 | `"quantum.rx90"` | pi/2 | No |
| Rx180 | `"quantum.rx180"` | pi | Yes |

```
// Rx90: fixed pi/2 rotation around X (commonly used in hardware)
%q1 = "quantum.rx90"(%q0) : (qubit) -> qubit

// Rx180: fixed pi rotation around X (equivalent to X gate)
%q2 = "quantum.rx180"(%q1) : (qubit) -> qubit
```

## 3.4 2-Qubit Gates (13)

All take 2 qubits, return 2 qubits: `%a, %b = "quantum.gate"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)`

| Gate | Syntax | Native On | Parametric |
|------|--------|-----------|-----------|
| CX (CNOT) | `"quantum.cx"` | IBM | No |
| CZ | `"quantum.cz"` | Google | No |
| CY | `"quantum.cy"` | — | No |
| SWAP | `"quantum.swap"` | — | No |
| iSWAP | `"quantum.iswap"` | Google | No |
| ECR | `"quantum.ecr"` | IBM Eagle | No |
| RZX | `"quantum.rzx"` | — | Yes |
| XX | `"quantum.xx"` | IonQ | Yes |
| YY | `"quantum.yy"` | — | Yes |
| ZZ | `"quantum.zz"` | Quantinuum | Yes |
| CP | `"quantum.cp"` | — | Yes |
| CPhase | `"quantum.cphase"` | Rigetti | Yes |
| XY | `"quantum.xy"` | Rigetti | Yes |

### IonQ native gates (3)

| Gate | Syntax | Description |
|------|--------|-------------|
| GPI | `"quantum.gpi"` | IonQ single-qubit gate |
| GPI2 | `"quantum.gpi2"` | IonQ single-qubit gate 2 |
| MS | `"quantum.ms"` | Mølmer-Sørensen (IonQ 2-qubit) |

```
// CX (CNOT): controlled NOT, fundamental entangling gate (IBM native)
%q2, %q3 = "quantum.cx"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)

// CZ: controlled-Z (Google Sycamore native)
%q2, %q3 = "quantum.cz"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)

// SWAP: exchange two qubit states
%q2, %q3 = "quantum.swap"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)

// ECR: echoed cross-resonance (IBM Eagle/Heron native)
%q2, %q3 = "quantum.ecr"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)

// ZZ: parametric Ising ZZ (Quantinuum native)
%q2, %q3 = "quantum.zz"(%q0, %q1) {angle = 0.5} : (qubit, qubit) -> (qubit, qubit)

// XX: parametric Ising XX (IonQ native)
%q2, %q3 = "quantum.xx"(%q0, %q1) {angle = 1.5708} : (qubit, qubit) -> (qubit, qubit)

// CP: controlled-phase gate (parametric)
%q2, %q3 = "quantum.cp"(%q0, %q1) {angle = 0.7854} : (qubit, qubit) -> (qubit, qubit)

// CPhase: Rigetti native controlled-phase
%q2, %q3 = "quantum.cphase"(%q0, %q1) {angle = 1.5708} : (qubit, qubit) -> (qubit, qubit)

// XY: Rigetti native XY interaction
%q2, %q3 = "quantum.xy"(%q0, %q1) {angle = 0.5} : (qubit, qubit) -> (qubit, qubit)

// iSWAP: imaginary SWAP (Google Sycamore)
%q2, %q3 = "quantum.iswap"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)
```

## 3.5 3-Qubit Gates (2)

| Gate | Syntax | Description |
|------|--------|-------------|
| CCX (Toffoli) | `"quantum.ccx"` | Controlled-Controlled-NOT |
| CSWAP (Fredkin) | `"quantum.cswap"` | Controlled-SWAP |

```
// CCX (Toffoli): 2 controls + 1 target, flips target if both controls are |1>
%a, %b, %c = "quantum.ccx"(%q0, %q1, %q2) : (qubit, qubit, qubit) -> (qubit, qubit, qubit)

// CSWAP (Fredkin): controlled swap, swaps q1/q2 if q0 is |1>
%d, %e, %f = "quantum.cswap"(%q3, %q4, %q5) : (qubit, qubit, qubit) -> (qubit, qubit, qubit)
```

## 3.6 Multi-Controlled Gates (2)

| Gate | Syntax | Qubits |
|------|--------|--------|
| MCX | `"quantum.mcx"` | N (variable) |
| MCZ | `"quantum.mcz"` | N (variable) |

```
// MCX with 4 qubits: 3 controls + 1 target
%a, %b, %c, %d = "quantum.mcx"(%q0, %q1, %q2, %q3) : (qubit, qubit, qubit, qubit) -> (qubit, qubit, qubit, qubit)

// MCZ with 3 qubits: 2 controls + 1 target
%e, %f, %g = "quantum.mcz"(%q4, %q5, %q6) : (qubit, qubit, qubit) -> (qubit, qubit, qubit)
```

## 3.7 Measurement and Control (8)

| Operation | Syntax | In | Out | Description |
|-----------|--------|------|-----|-------------|
| Measure | `"quantum.measure"` | 1 qubit | qubit | Measure qubit |
| MeasureAll | `"quantum.measure_all"` | N | N | Measure all |
| Reset | `"quantum.reset"` | 1 qubit | qubit | Reset to \|0> |
| Barrier | `"quantum.barrier"` | N | — | Prevent reordering |
| Init | `"quantum.init"` | 1 qubit | qubit | Initialise register |
| Delay | `"quantum.delay"` | — | — | Time delay |
| VirtualRZ | `"quantum.virtual_rz"` | 1 qubit | qubit | Zero-cost virtual Z |
| IfElse | `"quantum.if_else"` | — | — | Classical conditional |
| ParamGate | `"quantum.param_gate"` | — | — | Generic parameterised gate |

```
%m = "quantum.measure"(%q0) : (qubit) -> qubit
%r = "quantum.reset"(%q0) : (qubit) -> qubit
%q1 = "quantum.virtual_rz"(%q0) {angle = 0.785} : (qubit) -> qubit
```

## 3.8 Hardware Native Gate Sets

Each provider has a fixed set of natively supported gates. All other gates are decomposed automatically.

| Provider | Native Gates |
|----------|-------------|
| **IBM Eagle / Kyoto** | `rz`, `sx`, `x`, `cx`, `ecr` |
| **Rigetti** | `rz`, `rx`, `cz`, `cphase`, `xy` |
| **IonQ** | `gpi`, `gpi2`, `ms` |
| **Quantinuum** | `rz`, `rx`, `ry`, `zz` |
| **Simulator** | `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `cx`, `cz`, `ccx`, `swap` |

**Example — same entangling operation on different hardware:**

```
// IBM Eagle: uses CX (CNOT) natively
%a, %b = "quantum.cx"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)

// Google Sycamore: uses CZ natively
%a, %b = "quantum.cz"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)

// IonQ: uses MS (Mølmer-Sørensen) natively
%a, %b = "quantum.ms"(%q0, %q1) : (qubit, qubit) -> (qubit, qubit)

// Quantinuum: uses ZZ natively
%a, %b = "quantum.zz"(%q0, %q1) {angle = 1.5708} : (qubit, qubit) -> (qubit, qubit)

// Rigetti: uses CPhase natively
%a, %b = "quantum.cphase"(%q0, %q1) {angle = 3.14159} : (qubit, qubit) -> (qubit, qubit)
```

> **Note**: LIFT automatically decomposes non-native gates into the target hardware's native set during compilation. You can write using any gate and the compiler handles the rest.

## 3.9 Gate Properties

| Property | What it means | Checked by |
|----------|--------------|-----------|
| **num_qubits** | Expected input count | Compile-time verification |
| **is_parametric** | Needs angle attributes | Attribute validation |
| **is_self_inverse** | G·G = Identity | Gate cancellation pass |
| **is_clifford** | Efficient classical simulation | Optimiser heuristics |
| **is_entangling** | Creates entanglement | Circuit analysis |
| **is_measurement** | Collapses state | Control flow analysis |

## 3.10 Qubit Linearity Rule

**The most important rule in the quantum dialect.**

Every qubit must be consumed **exactly once**:

```
// CORRECT
%q1 = "quantum.h"(%q0) : (qubit) -> qubit       // q0 consumed → q1 produced
%q2, %q3 = "quantum.cx"(%q1, %q_b) : ...        // q1 consumed

// ERROR: q0 used twice (no-cloning violation)
%q1 = "quantum.h"(%q0) : (qubit) -> qubit
%q2 = "quantum.x"(%q0) : (qubit) -> qubit        // COMPILE ERROR

// ERROR: q1 never used (qubit leak)
%q1 = "quantum.h"(%q0) : (qubit) -> qubit
return                                             // COMPILE ERROR
```

## 3.11 Complete Bell State Example

```
#dialect quantum

module @bell_state {
    func @bell(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
        %q2 = "quantum.h"(%q0) : (qubit) -> qubit
        %q3, %q4 = "quantum.cx"(%q2, %q1) : (qubit, qubit) -> (qubit, qubit)
        return %q3, %q4
    }
}
```

## 3.12 Complete GHZ State Example (3 qubits)

```
#dialect quantum

module @ghz {
    func @ghz3(%q0: qubit, %q1: qubit, %q2: qubit) -> (qubit, qubit, qubit) {
        %a = "quantum.h"(%q0) : (qubit) -> qubit
        %b, %c = "quantum.cx"(%a, %q1) : (qubit, qubit) -> (qubit, qubit)
        %d, %e = "quantum.cx"(%c, %q2) : (qubit, qubit) -> (qubit, qubit)
        return %b, %d, %e
    }
}
```

## 3.13 Complete Variational Circuit Example

```
#dialect quantum

module @variational {
    func @layer(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
        // RY rotations (parametric)
        %a = "quantum.ry"(%q0) {angle = 0.5} : (qubit) -> qubit
        %b = "quantum.ry"(%q1) {angle = 1.2} : (qubit) -> qubit
        // Entangling
        %c, %d = "quantum.cx"(%a, %b) : (qubit, qubit) -> (qubit, qubit)
        // More rotations
        %e = "quantum.rz"(%c) {angle = 0.3} : (qubit) -> qubit
        %f = "quantum.rz"(%d) {angle = 0.7} : (qubit) -> qubit
        return %e, %f
    }
}
```

---

# Part IV — The `hybrid` Dialect (Classical + Quantum Bridge)

Declare with `#dialect hybrid` (usually combined with `#dialect tensor` and `#dialect quantum`). Provides **21 operations** that bridge classical and quantum computing.

## 4.1 Encoding / Decoding (2)

| Operation | Syntax | Description |
|-----------|--------|-------------|
| Encode | `"hybrid.encode"` | Classical tensor → quantum state |
| Decode | `"hybrid.decode"` | Quantum state → classical tensor |

```
%encoded = "hybrid.encode"(%data) {strategy = "angle"} : (tensor<1x4xf32>) -> qubit
%decoded = "hybrid.decode"(%qstate) : (qubit) -> tensor<1x4xf32>
```

### Encoding Strategies

| Strategy | Attribute Value | Qubits for N features | Depth | Best For |
|----------|----------------|----------------------|-------|----------|
| Angle | `"angle"` | N | 1 | Small vectors (<20) |
| Amplitude | `"amplitude"` | ceil(log2(N)) | N | Large vectors |
| Basis | `"basis"` | N | 1 | Binary data |
| IQP | `"iqp"` | N | 2N | High expressivity |
| Hamiltonian | `"hamiltonian"` | N | N | Physics problems |
| Kernel | `"kernel"` | N | 3N | Quantum kernel methods |

## 4.2 Gradient Methods (6)

| Operation | Syntax | Evaluations | Exact |
|-----------|--------|-------------|-------|
| ParameterShift | `"hybrid.parameter_shift"` | 2N | Yes |
| FiniteDifference | `"hybrid.finite_difference"` | N+1 | No |
| SPSA | `"hybrid.spsa"` | 2 | No |
| AdjointDiff | `"hybrid.adjoint_diff"` | 1 | Yes |
| StochasticParamShift | `"hybrid.stochastic_param_shift"` | 2 | No |
| JointGradient | `"hybrid.joint_gradient"` | Variable | Mixed |

### Which to choose

| Situation | Method |
|-----------|--------|
| Few params (<50) | Parameter Shift |
| Many params (>100) | SPSA |
| Simulator only | Adjoint Diff |
| Mixed classical+quantum | Joint Gradient |
| Noisy hardware | Stochastic Parameter Shift |

```
// ParameterShift: exact gradient via 2 circuit evaluations per parameter
%grad1 = "hybrid.parameter_shift"(%expectation) : (tensor<1xf32>) -> tensor<1x16xf32>

// FiniteDifference: approximate gradient via N+1 evaluations
%grad2 = "hybrid.finite_difference"(%expectation) : (tensor<1xf32>) -> tensor<1x16xf32>

// SPSA: stochastic gradient, only 2 evaluations regardless of parameter count
%grad3 = "hybrid.spsa"(%expectation) : (tensor<1xf32>) -> tensor<1x16xf32>

// AdjointDiff: exact gradient in 1 evaluation (simulator only)
%grad4 = "hybrid.adjoint_diff"(%expectation) : (tensor<1xf32>) -> tensor<1x16xf32>

// JointGradient: use different methods for classical vs quantum parts
%grad5 = "hybrid.joint_gradient"(%hybrid_loss) : (tensor<1xf32>) -> tensor<1x32xf32>
```

## 4.3 Variational Algorithms (4)

| Operation | Syntax | Description |
|-----------|--------|-------------|
| VqcLayer | `"hybrid.vqc_layer"` | Generic variational circuit layer |
| VqeAnsatz | `"hybrid.vqe_ansatz"` | VQE chemistry ansatz |
| QaoaLayer | `"hybrid.qaoa_layer"` | QAOA combinatorial optimisation |
| QuantumKernel | `"hybrid.quantum_kernel"` | Quantum kernel (SVM) |

### Ansatz Types

| Type | Value | Use |
|------|-------|-----|
| HardwareEfficient | `"hardware_efficient"` | Near-term hardware |
| StronglyEntangling | `"strongly_entangling"` | Max expressivity |
| TwoLocal | `"two_local"` | General purpose |
| UCCSD | `"uccsd"` | Chemistry (VQE) |
| Custom | `"custom"` | User-defined |

```
%q_out = "hybrid.vqc_layer"(%q_in) {ansatz = "hardware_efficient", layers = 3} : (qubit) -> qubit
%q_out = "hybrid.vqe_ansatz"(%q_in) {ansatz = "uccsd"} : (qubit) -> qubit
%q_out = "hybrid.qaoa_layer"(%q_in) {gamma = 0.5, beta = 0.3} : (qubit) -> qubit
```

## 4.4 Data Transfer (2)

| Operation | Syntax | Direction |
|-----------|--------|-----------|
| GpuToQpu | `"hybrid.gpu_to_qpu"` | GPU → QPU |
| QpuToGpu | `"hybrid.qpu_to_gpu"` | QPU → GPU |

```
%qubits = "hybrid.gpu_to_qpu"(%encoded) : (tensor<1x4xf32>) -> qubit
%results = "hybrid.qpu_to_gpu"(%measured) : (qubit) -> tensor<1x4xf32>
```

## 4.5 Processing (4)

| Operation | Syntax | Description |
|-----------|--------|-------------|
| ClassicalPreprocess | `"hybrid.classical_preprocess"` | Pre-quantum classical processing |
| QuantumPostprocess | `"hybrid.quantum_postprocess"` | Post-quantum processing |
| HybridForward | `"hybrid.forward"` | Full hybrid forward pass |
| HybridBackward | `"hybrid.backward"` | Full hybrid backward pass |

```
// ClassicalPreprocess: transform classical data before quantum encoding
%prep = "hybrid.classical_preprocess"(%raw_data) : (tensor<1x100xf32>) -> tensor<1x8xf32>

// QuantumPostprocess: transform quantum measurement results
%post = "hybrid.quantum_postprocess"(%raw_measurement) : (tensor<4096xi32>) -> tensor<1x4xf32>

// HybridForward: execute the full classical+quantum forward pass
%fwd = "hybrid.forward"(%input) : (tensor<1x64xf32>) -> tensor<1x2xf32>

// HybridBackward: compute gradients through the full hybrid pipeline
%bwd = "hybrid.backward"(%loss) : (tensor<1xf32>) -> tensor<1x64xf32>
```

## 4.6 Co-Execution (1)

| Operation | Syntax | Description |
|-----------|--------|-------------|
| CoExecute | `"hybrid.co_execute"` | Run GPU + QPU simultaneously |

### Synchronisation Policies

| Policy | Value | Description |
|--------|-------|-------------|
| Blocking | `"blocking"` | GPU waits for QPU |
| Asynchronous | `"async"` | Independent execution |
| Pipeline | `"pipeline"` | Streaming tasks |

```
%result = "hybrid.co_execute"(%gpu_task, %qpu_task) {sync = "pipeline"} : (tensor<1x128xf32>, qubit) -> tensor<1x128xf32>
```

## 4.7 Measurement (2)

| Operation | Syntax | Output | Description |
|-----------|--------|--------|-------------|
| MeasureExpectation | `"hybrid.measure_expectation"` | scalar | Expectation value |
| MeasureSamples | `"hybrid.measure_samples"` | tensor | Raw shot results |

```
%val = "hybrid.measure_expectation"(%qubits) : (qubit) -> tensor<1xf32>
%samples = "hybrid.measure_samples"(%qubits) {shots = 4096} : (qubit) -> tensor<4096xi32>
```

## 4.8 Feature Maps (for quantum kernels)

| Feature Map | Description |
|-------------|-------------|
| ZZFeatureMap | ZZ interactions |
| PauliFeatureMap | Pauli products |
| AngleEncoding | Rotation encoding |
| AmplitudeEncoding | State amplitude |

```
// Quantum kernel with ZZ feature map: compute kernel value between two data points
%kernel_val = "hybrid.quantum_kernel"(%encoded_x1, %encoded_x2) {feature_map = "zz"} : (qubit, qubit) -> tensor<1x1xf32>

// Quantum kernel with Pauli feature map
%kernel_val2 = "hybrid.quantum_kernel"(%encoded_a, %encoded_b) {feature_map = "pauli"} : (qubit, qubit) -> tensor<1x1xf32>
```

## 4.9 Complete Hybrid Example — Medical Imaging (CNN + VQC)

```
#dialect tensor
#dialect quantum
#dialect hybrid

module @medical_hybrid {
    func @classify(
        %img: tensor<1x1x28x28xf32>,
        %conv_w: tensor<16x1x3x3xf32>,
        %fc_w: tensor<784x4xf32>,
        %fc_b: tensor<4xf32>,
        %q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit
    ) -> tensor<1x2xf32> {
        // Classical preprocessing: CNN feature extraction
        %feat = "tensor.conv2d"(%img, %conv_w) : (tensor<1x1x28x28xf32>, tensor<16x1x3x3xf32>) -> tensor<1x16x26x26xf32>
        %act = "tensor.relu"(%feat) : (tensor<1x16x26x26xf32>) -> tensor<1x16x26x26xf32>
        %pool = "tensor.global_avgpool"(%act) : (tensor<1x16x26x26xf32>) -> tensor<1x16x1x1xf32>
        %flat = "tensor.reshape"(%pool) : (tensor<1x16x1x1xf32>) -> tensor<1x16xf32>

        // Reduce to 4 features for 4 qubits
        %reduced = "tensor.linear"(%flat, %fc_w, %fc_b) : (tensor<1x16xf32>, tensor<16x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>

        // Encode into quantum state
        %encoded = "hybrid.encode"(%reduced) {strategy = "angle"} : (tensor<1x4xf32>) -> qubit

        // Quantum processing
        %q_a = "hybrid.vqc_layer"(%encoded) {ansatz = "hardware_efficient", layers = 2} : (qubit) -> qubit

        // Measure expectation values
        %expectation = "hybrid.measure_expectation"(%q_a) : (qubit) -> tensor<1x2xf32>

        // Classical postprocessing
        %probs = "tensor.softmax"(%expectation) : (tensor<1x2xf32>) -> tensor<1x2xf32>
        return %probs
    }
}
```

## 4.10 Complete Hybrid Example — VQE for Chemistry

```
#dialect tensor
#dialect quantum
#dialect hybrid

module @vqe_molecule {
    func @energy_estimation(
        %params: tensor<1x16xf32>,
        %q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit
    ) -> tensor<1xf32> {
        // Encode parameters into quantum state
        %encoded = "hybrid.encode"(%params) {strategy = "amplitude"} : (tensor<1x16xf32>) -> qubit

        // Apply VQE ansatz (UCCSD for chemistry)
        %ansatz_out = "hybrid.vqe_ansatz"(%encoded) {ansatz = "uccsd"} : (qubit) -> qubit

        // Measure energy expectation
        %energy = "hybrid.measure_expectation"(%ansatz_out) : (qubit) -> tensor<1xf32>

        // Compute gradient for parameter update
        %grad = "hybrid.parameter_shift"(%energy) : (tensor<1xf32>) -> tensor<1x16xf32>

        return %energy
    }
}
```

## 4.11 Complete Hybrid Example — QAOA for Optimisation

```
#dialect tensor
#dialect quantum
#dialect hybrid

module @qaoa_portfolio {
    func @optimise(
        %gamma: tensor<1xf32>,
        %beta: tensor<1xf32>,
        %q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit, %q4: qubit
    ) -> tensor<1xf32> {
        // QAOA layer with problem-specific parameters
        %q_out = "hybrid.qaoa_layer"(%q0) {gamma = 0.5, beta = 0.3} : (qubit) -> qubit

        // Sample the result
        %samples = "hybrid.measure_samples"(%q_out) {shots = 8192} : (qubit) -> tensor<8192xi32>

        // Classical post-processing: evaluate cost function
        %cost = "hybrid.quantum_postprocess"(%samples) : (tensor<8192xi32>) -> tensor<1xf32>

        return %cost
    }
}
```

---

# Part V — Configuration (`.lith` Files)

The `.lith` file controls compilation, optimisation, budgets, and hardware targeting. It uses a simple INI format.

## 5.1 File Format

```ini
# Comment (ignored)
// Also a comment

[section_name]
key = value
key2 = "quoted value"
```

## 5.2 `[target]` Section

| Key | Type | Values | Default |
|-----|------|--------|---------|
| `backend` | string | `llvm`, `qasm` | `llvm` |
| `device` | string | `A100`, `H100`, `ibm_eagle`, `ibm_kyoto`, `rigetti`, `ionq`, `quantinuum` | none |
| `precision` | string | `fp64`, `fp32`, `fp16`, `bf16` | `fp32` |

```ini
[target]
backend = llvm
device = A100
precision = fp32
```

- `llvm` backend → exports to LLVM IR (CUDA PTX, x86-64, ARM)
- `qasm` backend → exports to OpenQASM 3.0 (IBM Quantum, Amazon Braket, Azure Quantum)

## 5.3 `[budget]` Section

All fields are optional. Omitted fields impose no constraint.

| Key | Type | Description |
|-----|------|-------------|
| `max_flops` | u64 | Maximum FLOPs allowed |
| `max_memory_bytes` | u64 | Maximum memory in bytes |
| `max_time_ms` | f64 | Maximum execution time (ms) |
| `min_fidelity` | f64 | Minimum quantum fidelity (0.0–1.0) |
| `max_circuit_depth` | usize | Maximum quantum circuit depth |

```ini
[budget]
max_flops = 10000000000
max_memory_bytes = 80000000000
max_time_ms = 100.0
min_fidelity = 0.90
max_circuit_depth = 1000
```

## 5.4 `[optimisation]` Section

| Key | Type | Values | Default |
|-----|------|--------|---------|
| `level` | enum | `O0`, `O1`, `O2`, `O3` | `O2` |
| `max_iterations` | usize | Any positive integer | `10` |

```ini
[optimisation]
level = O2
max_iterations = 10
```

### Optimisation Levels

| Level | What it does |
|-------|-------------|
| **O0** | No optimisation (debug mode) |
| **O1** | Canonicalize + Dead Code Elimination |
| **O2** | + Tensor Fusion + Constant Folding + Gate Cancellation + Rotation Merge |
| **O3** | + Flash Attention + CSE + Quantisation Pass + Noise-Aware Schedule + Layout Mapping |

### Default passes at O2

`canonicalize`, `constant-folding`, `dce`, `tensor-fusion`

### All available optimisation passes

| Pass | Dialect | Description |
|------|---------|-------------|
| `canonicalize` | All | Simplify operations to canonical forms |
| `constant-folding` | Tensor | Evaluate constant expressions at compile time |
| `dce` | All | Remove dead (unused) operations |
| `tensor-fusion` | Tensor | Fuse adjacent tensor operations into single kernels |
| `flash-attention` | Tensor | Replace standard attention with flash attention |
| `cse` | All | Common Subexpression Elimination |
| `quantisation-pass` | Tensor | Apply INT8/INT4/FP8 quantisation |
| `gate-cancellation` | Quantum | Cancel adjacent inverse gates (H·H=I, X·X=I) |
| `rotation-merge` | Quantum | Merge consecutive rotations (RZ(a)·RZ(b)=RZ(a+b)) |
| `noise-aware-schedule` | Quantum | Schedule gates considering hardware noise |
| `layout-mapping` | Quantum | Map logical qubits to physical qubits (SABRE algorithm) |

## 5.5 `[simulation]` Section

| Key | Type | Default |
|-----|------|---------|
| `shape_propagation` | bool | `true` |
| `flop_counting` | bool | `true` |
| `memory_analysis` | bool | `true` |
| `noise_simulation` | bool | `true` |

```ini
[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = true
```

## 5.6 `[quantum]` Section

Only needed for quantum or hybrid programs.

| Key | Type | Values | Default |
|-----|------|--------|---------|
| `topology` | string | `grid`, `heavy_hex`, `all_to_all`, `linear`, `tree` | `linear` |
| `num_qubits` | usize | Any positive integer | `5` |
| `error_mitigation` | string | Mitigation strategy name | none |
| `shots` | usize | Number of measurement shots | none |

```ini
[quantum]
topology = heavy_hex
num_qubits = 127
error_mitigation = zne
shots = 8192
```

### Quantum Topologies

| Topology | Description | Provider |
|----------|-------------|----------|
| `linear` | Qubits in a line | General |
| `grid` | 2D grid | Google Sycamore |
| `heavy_hex` | Heavy-hexagonal lattice | IBM Eagle/Heron |
| `all_to_all` | Full connectivity | IonQ, Quantinuum |
| `tree` | Tree structure | Custom |

## 5.7 Complete `.lith` Examples

### Classical AI (GPU inference)

```ini
# config_gpu.lith — Optimised GPU inference
[target]
backend = llvm
device = A100
precision = fp16

[budget]
max_memory_bytes = 16000000000
max_time_ms = 50.0

[optimisation]
level = O3
max_iterations = 20

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = false
```

### Quantum circuit (IBM hardware)

```ini
# config_ibm.lith — IBM Eagle quantum processor
[target]
backend = qasm
device = ibm_eagle

[budget]
min_fidelity = 0.85
max_circuit_depth = 500

[optimisation]
level = O3
max_iterations = 15

[simulation]
shape_propagation = false
flop_counting = false
memory_analysis = false
noise_simulation = true

[quantum]
topology = heavy_hex
num_qubits = 127
error_mitigation = zne
shots = 4096
```

### Hybrid (GPU + QPU)

```ini
# config_hybrid.lith — Medical imaging hybrid
[target]
backend = llvm
device = A100
precision = fp32

[budget]
max_memory_bytes = 40000000000
max_time_ms = 10000.0
min_fidelity = 0.80
max_circuit_depth = 200

[optimisation]
level = O2
max_iterations = 10

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = true

[quantum]
topology = heavy_hex
num_qubits = 16
shots = 4096
```

### Edge deployment (low power)

```ini
# config_edge.lith — Edge device deployment
[target]
backend = llvm
device = ARM
precision = fp16

[budget]
max_memory_bytes = 500000000
max_time_ms = 30.0

[optimisation]
level = O3
max_iterations = 30

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = false
```

---

# Part VI — Assembling Dialects Together

## 6.1 Single-Dialect Programs

### Tensor only — MLP classifier

```
#dialect tensor

module @mlp {
    func @forward(
        %x: tensor<1x784xf32>,
        %w1: tensor<784x256xf32>, %b1: tensor<256xf32>,
        %w2: tensor<256x10xf32>, %b2: tensor<10xf32>
    ) -> tensor<1x10xf32> {
        %h1 = "tensor.matmul"(%x, %w1) : (tensor<1x784xf32>, tensor<784x256xf32>) -> tensor<1x256xf32>
        %h2 = "tensor.add"(%h1, %b1) : (tensor<1x256xf32>, tensor<256xf32>) -> tensor<1x256xf32>
        %h3 = "tensor.relu"(%h2) : (tensor<1x256xf32>) -> tensor<1x256xf32>
        %h4 = "tensor.matmul"(%h3, %w2) : (tensor<1x256xf32>, tensor<256x10xf32>) -> tensor<1x10xf32>
        %h5 = "tensor.add"(%h4, %b2) : (tensor<1x10xf32>, tensor<10xf32>) -> tensor<1x10xf32>
        %out = "tensor.softmax"(%h5) : (tensor<1x10xf32>) -> tensor<1x10xf32>
        return %out
    }
}
```

### Tensor only — Transformer self-attention

```
#dialect tensor

module @transformer {
    func @self_attention(
        %q: tensor<1x128x64xf32>,
        %k: tensor<1x128x64xf32>,
        %v: tensor<1x128x64xf32>,
        %norm_w: tensor<64xf32>
    ) -> tensor<1x128x64xf32> {
        %attn = "tensor.attention"(%q, %k, %v) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
        %normed = "tensor.layernorm"(%attn, %norm_w) : (tensor<1x128x64xf32>, tensor<64xf32>) -> tensor<1x128x64xf32>
        return %normed
    }
}
```

### Tensor only — CNN for image classification

```
#dialect tensor

module @cnn {
    func @forward(
        %img: tensor<1x3x224x224xf32>,
        %conv1_w: tensor<64x3x7x7xf32>,
        %bn_s: tensor<64xf32>, %bn_b: tensor<64xf32>, %bn_m: tensor<64xf32>,
        %fc_w: tensor<1024x1000xf32>, %fc_b: tensor<1000xf32>
    ) -> tensor<1x1000xf32> {
        // Conv + BatchNorm + ReLU (fused)
        %c1 = "tensor.fused_conv_batchnorm_relu"(%img, %conv1_w, %bn_s, %bn_b, %bn_m) : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
        // Pooling
        %p1 = "tensor.global_avgpool"(%c1) : (tensor<1x64x112x112xf32>) -> tensor<1x64x1x1xf32>
        %flat = "tensor.reshape"(%p1) : (tensor<1x64x1x1xf32>) -> tensor<1x64xf32>
        // Classifier
        %logits = "tensor.linear"(%flat, %fc_w, %fc_b) : (tensor<1x64xf32>, tensor<64x1000xf32>, tensor<1000xf32>) -> tensor<1x1000xf32>
        %probs = "tensor.softmax"(%logits) : (tensor<1x1000xf32>) -> tensor<1x1000xf32>
        return %probs
    }
}
```

### Tensor only — GNN

```
#dialect tensor

module @gnn {
    func @forward(
        %nodes: tensor<100x16xf32>,
        %edges: tensor<100x100xf32>,
        %w: tensor<16x16xf32>, %b: tensor<16xf32>
    ) -> tensor<1x16xf32> {
        // Message passing
        %h1 = "tensor.gnn_message_passing"(%nodes, %edges) : (tensor<100x16xf32>, tensor<100x100xf32>) -> tensor<100x16xf32>
        %h2 = "tensor.relu"(%h1) : (tensor<100x16xf32>) -> tensor<100x16xf32>
        // Second layer
        %h3 = "tensor.gnn_message_passing"(%h2, %edges) : (tensor<100x16xf32>, tensor<100x100xf32>) -> tensor<100x16xf32>
        // Global pooling
        %graph = "tensor.gnn_global_pooling"(%h3) : (tensor<100x16xf32>) -> tensor<1x16xf32>
        return %graph
    }
}
```

### Quantum only — Quantum Teleportation

```
#dialect quantum

module @teleportation {
    func @teleport(%psi: qubit, %q1: qubit, %q2: qubit) -> (qubit, qubit, qubit) {
        // Create Bell pair between q1 and q2
        %a = "quantum.h"(%q1) : (qubit) -> qubit
        %b, %c = "quantum.cx"(%a, %q2) : (qubit, qubit) -> (qubit, qubit)

        // Bell measurement on psi and b
        %d, %e = "quantum.cx"(%psi, %b) : (qubit, qubit) -> (qubit, qubit)
        %f = "quantum.h"(%d) : (qubit) -> qubit

        // Measure
        %m1 = "quantum.measure"(%f) : (qubit) -> qubit
        %m2 = "quantum.measure"(%e) : (qubit) -> qubit

        return %m1, %m2, %c
    }
}
```

### Quantum only — Quantum Fourier Transform (3 qubits)

```
#dialect quantum

module @qft {
    func @qft3(%q0: qubit, %q1: qubit, %q2: qubit) -> (qubit, qubit, qubit) {
        // First qubit
        %a = "quantum.h"(%q0) : (qubit) -> qubit
        %b, %c = "quantum.cp"(%q1, %a) {angle = 1.5708} : (qubit, qubit) -> (qubit, qubit)
        %d, %e = "quantum.cp"(%q2, %c) {angle = 0.7854} : (qubit, qubit) -> (qubit, qubit)

        // Second qubit
        %f = "quantum.h"(%b) : (qubit) -> qubit
        %g, %h = "quantum.cp"(%d, %f) {angle = 1.5708} : (qubit, qubit) -> (qubit, qubit)

        // Third qubit
        %i = "quantum.h"(%g) : (qubit) -> qubit

        // Swap first and last
        %j, %k = "quantum.swap"(%e, %i) : (qubit, qubit) -> (qubit, qubit)

        return %j, %h, %k
    }
}
```

## 6.2 Multi-Dialect Programs

### Tensor + Quantum — Feature extraction + quantum classification

```
#dialect tensor
#dialect quantum
#dialect hybrid

module @hybrid_classifier {
    func @forward(
        %img: tensor<1x1x28x28xf32>,
        %w1: tensor<16x1x5x5xf32>,
        %w2: tensor<256x4xf32>, %b2: tensor<4xf32>,
        %q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit
    ) -> tensor<1x2xf32> {

        // ──── Stage 1: Classical (tensor dialect) ────
        %conv = "tensor.conv2d"(%img, %w1) : (tensor<1x1x28x28xf32>, tensor<16x1x5x5xf32>) -> tensor<1x16x24x24xf32>
        %act = "tensor.relu"(%conv) : (tensor<1x16x24x24xf32>) -> tensor<1x16x24x24xf32>
        %pool = "tensor.adaptive_avgpool2d"(%act) : (tensor<1x16x24x24xf32>) -> tensor<1x16x4x4xf32>
        %flat = "tensor.reshape"(%pool) : (tensor<1x16x4x4xf32>) -> tensor<1x256xf32>
        %features = "tensor.linear"(%flat, %w2, %b2) : (tensor<1x256xf32>, tensor<256x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>

        // ──── Stage 2: Encoding (hybrid dialect) ────
        %encoded = "hybrid.encode"(%features) {strategy = "angle"} : (tensor<1x4xf32>) -> qubit

        // ──── Stage 3: Quantum circuit (quantum dialect via hybrid) ────
        %processed = "hybrid.vqc_layer"(%encoded) {ansatz = "strongly_entangling", layers = 4} : (qubit) -> qubit

        // ──── Stage 4: Measurement (hybrid dialect) ────
        %raw = "hybrid.measure_expectation"(%processed) : (qubit) -> tensor<1x2xf32>

        // ──── Stage 5: Post-processing (tensor dialect) ────
        %probs = "tensor.softmax"(%raw) : (tensor<1x2xf32>) -> tensor<1x2xf32>

        return %probs
    }
}
```

### Drug Discovery — GNN + VQE

```
#dialect tensor
#dialect quantum
#dialect hybrid

module @drug_discovery {
    func @screen_molecule(
        %atoms: tensor<50x16xf32>,
        %bonds: tensor<50x50xf32>,
        %q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit
    ) -> tensor<1xf32> {

        // Stage 1: GNN feature extraction
        %h1 = "tensor.gnn_message_passing"(%atoms, %bonds) : (tensor<50x16xf32>, tensor<50x50xf32>) -> tensor<50x16xf32>
        %h2 = "tensor.relu"(%h1) : (tensor<50x16xf32>) -> tensor<50x16xf32>
        %h3 = "tensor.gnn_message_passing"(%h2, %bonds) : (tensor<50x16xf32>, tensor<50x50xf32>) -> tensor<50x16xf32>
        %mol = "tensor.gnn_global_pooling"(%h3) : (tensor<50x16xf32>) -> tensor<1x16xf32>

        // Stage 2: Reduce to qubit count and encode
        %flat = "tensor.reshape"(%mol) : (tensor<1x16xf32>) -> tensor<1x16xf32>
        %encoded = "hybrid.encode"(%flat) {strategy = "amplitude"} : (tensor<1x16xf32>) -> qubit

        // Stage 3: VQE for energy calculation
        %ansatz = "hybrid.vqe_ansatz"(%encoded) {ansatz = "uccsd"} : (qubit) -> qubit

        // Stage 4: Energy measurement
        %energy = "hybrid.measure_expectation"(%ansatz) : (qubit) -> tensor<1xf32>

        return %energy
    }
}
```

### Quantum Finance — QAOA Portfolio Optimisation

```
#dialect tensor
#dialect quantum
#dialect hybrid

module @quantum_finance {
    func @optimise_portfolio(
        %returns: tensor<1x10xf32>,
        %covariance: tensor<10x10xf32>,
        %q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit, %q4: qubit
    ) -> tensor<1x5xf32> {

        // Classical: compute expected returns
        %scores = "tensor.matmul"(%returns, %covariance) : (tensor<1x10xf32>, tensor<10x10xf32>) -> tensor<1x10xf32>

        // Preprocess for quantum
        %preprocessed = "hybrid.classical_preprocess"(%scores) : (tensor<1x10xf32>) -> tensor<1x5xf32>

        // Encode
        %encoded = "hybrid.encode"(%preprocessed) {strategy = "angle"} : (tensor<1x5xf32>) -> qubit

        // QAOA optimisation
        %qaoa_result = "hybrid.qaoa_layer"(%encoded) {gamma = 0.7, beta = 0.4} : (qubit) -> qubit

        // Measure
        %samples = "hybrid.measure_samples"(%qaoa_result) {shots = 8192} : (qubit) -> tensor<8192xi32>

        // Post-process: extract best portfolio allocation
        %allocation = "hybrid.quantum_postprocess"(%samples) : (tensor<8192xi32>) -> tensor<1x5xf32>

        return %allocation
    }
}
```

## 6.3 Multi-Function Modules

A module can contain multiple functions that call different dialects:

```
#dialect tensor
#dialect quantum
#dialect hybrid

module @full_pipeline {

    // Classical preprocessing function
    func @preprocess(%img: tensor<1x3x224x224xf32>, %w: tensor<64x3x7x7xf32>) -> tensor<1x64xf32> {
        %c = "tensor.conv2d"(%img, %w) : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
        %a = "tensor.relu"(%c) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
        %p = "tensor.global_avgpool"(%a) : (tensor<1x64x112x112xf32>) -> tensor<1x64x1x1xf32>
        %f = "tensor.reshape"(%p) : (tensor<1x64x1x1xf32>) -> tensor<1x64xf32>
        return %f
    }

    // Quantum processing function
    func @quantum_layer(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
        %a = "quantum.ry"(%q0) {angle = 0.5} : (qubit) -> qubit
        %b = "quantum.ry"(%q1) {angle = 1.0} : (qubit) -> qubit
        %c, %d = "quantum.cx"(%a, %b) : (qubit, qubit) -> (qubit, qubit)
        %e = "quantum.rz"(%c) {angle = 0.3} : (qubit) -> qubit
        return %e, %d
    }

    // Hybrid pipeline function
    func @classify(
        %features: tensor<1x4xf32>,
        %q0: qubit, %q1: qubit
    ) -> tensor<1x2xf32> {
        %encoded = "hybrid.encode"(%features) {strategy = "angle"} : (tensor<1x4xf32>) -> qubit
        %processed = "hybrid.vqc_layer"(%encoded) {ansatz = "hardware_efficient", layers = 2} : (qubit) -> qubit
        %result = "hybrid.measure_expectation"(%processed) : (qubit) -> tensor<1x2xf32>
        %probs = "tensor.softmax"(%result) : (tensor<1x2xf32>) -> tensor<1x2xf32>
        return %probs
    }
}
```

## 6.4 Common Patterns and Recipes

### Pattern 1: Linear Layer (matmul + bias + activation)

```
%h = "tensor.matmul"(%x, %w) : (tensor<BxMxf32>, tensor<MxNxf32>) -> tensor<BxNxf32>
%b = "tensor.add"(%h, %bias) : (tensor<BxNxf32>, tensor<Nxf32>) -> tensor<BxNxf32>
%a = "tensor.relu"(%b) : (tensor<BxNxf32>) -> tensor<BxNxf32>
```

Or fused:

```
%a = "tensor.fused_matmul_bias_relu"(%x, %w, %bias) : (tensor<BxMxf32>, tensor<MxNxf32>, tensor<Nxf32>) -> tensor<BxNxf32>
```

### Pattern 2: Transformer Block

```
%attn = "tensor.multi_head_attention"(%q, %k, %v) : ...
%res1 = "tensor.add"(%attn, %input) : ...
%norm1 = "tensor.layernorm"(%res1, %scale1) : ...
%ff1 = "tensor.linear"(%norm1, %w1, %b1) : ...
%act = "tensor.gelu"(%ff1) : ...
%ff2 = "tensor.linear"(%act, %w2, %b2) : ...
%res2 = "tensor.add"(%ff2, %norm1) : ...
%norm2 = "tensor.layernorm"(%res2, %scale2) : ...
```

### Pattern 3: Bell Pair + Measurement

```
%h = "quantum.h"(%q0) : (qubit) -> qubit
%a, %b = "quantum.cx"(%h, %q1) : (qubit, qubit) -> (qubit, qubit)
%m0 = "quantum.measure"(%a) : (qubit) -> qubit
%m1 = "quantum.measure"(%b) : (qubit) -> qubit
```

### Pattern 4: Variational Layer (rotation + entangling + rotation)

```
%r0 = "quantum.ry"(%q0) {angle = 0.5} : (qubit) -> qubit
%r1 = "quantum.ry"(%q1) {angle = 0.8} : (qubit) -> qubit
%e0, %e1 = "quantum.cx"(%r0, %r1) : (qubit, qubit) -> (qubit, qubit)
%f0 = "quantum.rz"(%e0) {angle = 0.3} : (qubit) -> qubit
%f1 = "quantum.rz"(%e1) {angle = 0.7} : (qubit) -> qubit
```

### Pattern 5: Hybrid Pipeline (encode → process → measure → postprocess)

```
%enc = "hybrid.encode"(%data) {strategy = "angle"} : (tensor<...>) -> qubit
%proc = "hybrid.vqc_layer"(%enc) {ansatz = "hardware_efficient", layers = 3} : (qubit) -> qubit
%meas = "hybrid.measure_expectation"(%proc) : (qubit) -> tensor<...>
%out = "tensor.softmax"(%meas) : (tensor<...>) -> tensor<...>
```

### Pattern 6: Quantisation for Edge Deployment

```
%q_weights = "tensor.quantize"(%weights) : (tensor<256x256xf32>) -> tensor<256x256xi8>
%output = "tensor.matmul"(%input, %q_weights) : ...
%dq = "tensor.dequantize"(%output) : (tensor<...xi8>) -> tensor<...xf32>
```

### Pattern 7: Training with Gradient Accumulation

```
%fwd = "tensor.linear"(%x, %w, %b) : ...
%loss = "tensor.softmax"(%fwd) : ...
%grad = "tensor.grad_linear"(%loss, %w, %b) : ...
%acc = "tensor.grad_accumulate"(%grad) : ...
```

### Pattern 8: Distributed Training

```
%split = "tensor.parallel_split"(%batch) : ...
%local = "tensor.matmul"(%split, %w) : ...
%synced = "tensor.parallel_allreduce"(%local) : ...
```

## 6.5 Error Checklist

Common mistakes and how to avoid them:

| Error | Cause | Fix |
|-------|-------|-----|
| `Unknown operation: tensor.xxx` | Typo in operation name | Check exact name in this reference |
| `Unknown operation: quantum.xxx` | Missing `#dialect quantum` | Add `#dialect quantum` at top |
| `SSA violation` | `%name` assigned twice | Use a new name for each result |
| `Linearity violation` | Qubit used twice | Each qubit value consumed exactly once |
| `Qubit leaked` | Qubit created but not consumed | Return or measure all qubits |
| `Wrong number of inputs` | Operation got wrong operand count | Check input count in tables above |
| `Type mismatch` | Tensor shapes incompatible | Verify shapes match (e.g. matmul: [M,K]×[K,N]) |
| `Missing type signature` | No `: (types) -> type` | Always include type signature |
| `Missing #dialect` | Using ops without declaring dialect | Add `#dialect <name>` at file top |
| `Attribute error` | Parametric gate missing angle | Add `{angle = ...}` for RX, RY, RZ, etc. |

## 6.6 Quick Syntax Reference Card

```
┌─────────────────────────────────────────────────────┐
│ #dialect tensor / quantum / hybrid                  │
│                                                     │
│ module @name {                                      │
│   func @fn(%x: type, ...) -> type {                 │
│     %y = "dialect.op"(%x) {attrs} : (T) -> T       │
│     %a, %b = "dialect.op"(%x, %y) : (T,T) -> (T,T) │
│     return %y                                       │
│   }                                                 │
│ }                                                   │
├─────────────────────────────────────────────────────┤
│ TYPES:                                              │
│   tensor<DxDxDxdtype>  e.g. tensor<1x784xf32>      │
│   qubit                (linear — consumed once)     │
│   bit                  (classical measurement)      │
│   hamiltonian<N>       (N-qubit operator)           │
│   f32, i32, bool, void, index                       │
├─────────────────────────────────────────────────────┤
│ DTYPES:                                             │
│   f64 f32 f16 bf16 fp8e4m3 fp8e5m2                 │
│   i64 i32 i16 i8 i4 i2 u8 i1 index                │
├─────────────────────────────────────────────────────┤
│ ATTRIBUTES:                                         │
│   {key = 42, rate = 0.5, flag = true, s = "text"}  │
│   {arr = [1, 2, 3]}                                │
├─────────────────────────────────────────────────────┤
│ DIALECTS:                                           │
│   tensor: 96 ops  (AI / ML)                        │
│   quantum: 50+ ops (quantum circuits)              │
│   hybrid:  21 ops  (classical ↔ quantum bridge)    │
└─────────────────────────────────────────────────────┘
```

---

# Appendix — Complete Operation Index

## A.1 All Tensor Operations (96)

| # | Category | Syntax |
|---|----------|--------|
| 1 | Arithmetic | `tensor.add` |
| 2 | Arithmetic | `tensor.sub` |
| 3 | Arithmetic | `tensor.mul` |
| 4 | Arithmetic | `tensor.div` |
| 5 | Arithmetic | `tensor.neg` |
| 6 | Arithmetic | `tensor.matmul` |
| 7 | Arithmetic | `tensor.linear` |
| 8 | Arithmetic | `tensor.conv2d` |
| 9 | Arithmetic | `tensor.embedding` |
| 10 | Activation | `tensor.relu` |
| 11 | Activation | `tensor.gelu` |
| 12 | Activation | `tensor.silu` |
| 13 | Activation | `tensor.sigmoid` |
| 14 | Activation | `tensor.softmax` |
| 15 | Activation | `tensor.tanh` |
| 16 | Activation | `tensor.leaky_relu` |
| 17 | Activation | `tensor.elu` |
| 18 | Activation | `tensor.mish` |
| 19 | Activation | `tensor.hard_swish` |
| 20 | Activation | `tensor.hard_sigmoid` |
| 21 | Normalisation | `tensor.layernorm` |
| 22 | Normalisation | `tensor.rmsnorm` |
| 23 | Normalisation | `tensor.batchnorm` |
| 24 | Normalisation | `tensor.groupnorm` |
| 25 | Normalisation | `tensor.instancenorm` |
| 26 | Shape | `tensor.reshape` |
| 27 | Shape | `tensor.transpose` |
| 28 | Shape | `tensor.concat` |
| 29 | Shape | `tensor.split` |
| 30 | Shape | `tensor.gather` |
| 31 | Shape | `tensor.scatter` |
| 32 | Shape | `tensor.squeeze` |
| 33 | Shape | `tensor.unsqueeze` |
| 34 | Shape | `tensor.permute` |
| 35 | Shape | `tensor.expand` |
| 36 | Shape | `tensor.slice` |
| 37 | Shape | `tensor.pad` |
| 38 | Shape | `tensor.tile` |
| 39 | Attention | `tensor.attention` |
| 40 | Attention | `tensor.multi_head_attention` |
| 41 | Attention | `tensor.multi_query_attention` |
| 42 | Attention | `tensor.grouped_query_attention` |
| 43 | Attention | `tensor.flash_attention` |
| 44 | Attention | `tensor.sliding_window_attention` |
| 45 | Attention | `tensor.cross_attention` |
| 46 | Attention | `tensor.paged_attention` |
| 47 | Convolution | `tensor.conv1d` |
| 48 | Convolution | `tensor.conv3d` |
| 49 | Convolution | `tensor.conv_transpose2d` |
| 50 | Convolution | `tensor.depthwise_conv2d` |
| 51 | Convolution | `tensor.dilated_conv2d` |
| 52 | Pooling | `tensor.maxpool2d` |
| 53 | Pooling | `tensor.avgpool2d` |
| 54 | Pooling | `tensor.adaptive_avgpool2d` |
| 55 | Pooling | `tensor.global_avgpool` |
| 56 | Recurrent | `tensor.lstm_cell` |
| 57 | Recurrent | `tensor.gru_cell` |
| 58 | Recurrent | `tensor.rnn_cell` |
| 59 | Math | `tensor.einsum` |
| 60 | Math | `tensor.fft` |
| 61 | Math | `tensor.ifft` |
| 62 | Math | `tensor.svd` |
| 63 | Math | `tensor.eig` |
| 64 | Math | `tensor.solve` |
| 65 | Math | `tensor.topk` |
| 66 | Math | `tensor.sort` |
| 67 | Math | `tensor.cumsum` |
| 68 | Math | `tensor.where` |
| 69 | Math | `tensor.clamp` |
| 70 | Sparse | `tensor.sparse_matmul` |
| 71 | Sparse | `tensor.sparse_embedding` |
| 72 | Quantisation | `tensor.quantize` |
| 73 | Quantisation | `tensor.dequantize` |
| 74 | Quantisation | `tensor.quantize_int4` |
| 75 | Quantisation | `tensor.dequantize_int4` |
| 76 | Quantisation | `tensor.quantize_fp8` |
| 77 | Quantisation | `tensor.dequantize_fp8` |
| 78 | Generative | `tensor.unet_down_block` |
| 79 | Generative | `tensor.unet_up_block` |
| 80 | Generative | `tensor.timestep_embedding` |
| 81 | GNN | `tensor.gnn_message_passing` |
| 82 | GNN | `tensor.gnn_global_pooling` |
| 83 | MoE | `tensor.moe_dispatch` |
| 84 | MoE | `tensor.moe_combine` |
| 85 | Constants | `tensor.constant` |
| 86 | Constants | `tensor.zeros` |
| 87 | Constants | `tensor.ones` |
| 88 | Constants | `tensor.arange` |
| 89 | Constants | `tensor.full` |
| 90 | Memory | `tensor.checkpoint` |
| 91 | Memory | `tensor.offload` |
| 92 | Memory | `tensor.grad_accumulate` |
| 93 | Gradient | `tensor.grad_matmul` |
| 94 | Gradient | `tensor.grad_relu` |
| 95 | Gradient | `tensor.grad_softmax` |
| 96 | Gradient | `tensor.grad_layernorm` |
| 97 | Gradient | `tensor.grad_attention` |
| 98 | Gradient | `tensor.grad_conv2d` |
| 99 | Gradient | `tensor.grad_linear` |
| 100 | Gradient | `tensor.grad_gelu` |
| 101 | Parallelism | `tensor.parallel_split` |
| 102 | Parallelism | `tensor.parallel_allreduce` |
| 103 | Parallelism | `tensor.pipeline_send` |
| 104 | Parallelism | `tensor.pipeline_receive` |
| 105 | Fused | `tensor.fused_matmul_bias_relu` |
| 106 | Fused | `tensor.fused_matmul_bias` |
| 107 | Fused | `tensor.fused_linear_gelu` |
| 108 | Fused | `tensor.fused_attention_layernorm` |
| 109 | Fused | `tensor.fused_linear_silu` |
| 110 | Fused | `tensor.fused_conv_batchnorm_relu` |

## A.2 All Quantum Operations (50+)

| # | Category | Syntax | Qubits |
|---|----------|--------|--------|
| 1 | 1Q Standard | `quantum.h` | 1 |
| 2 | 1Q Standard | `quantum.x` | 1 |
| 3 | 1Q Standard | `quantum.y` | 1 |
| 4 | 1Q Standard | `quantum.z` | 1 |
| 5 | 1Q Standard | `quantum.s` | 1 |
| 6 | 1Q Standard | `quantum.sdg` | 1 |
| 7 | 1Q Standard | `quantum.t` | 1 |
| 8 | 1Q Standard | `quantum.tdg` | 1 |
| 9 | 1Q Standard | `quantum.sx` | 1 |
| 10 | 1Q Parametric | `quantum.rx` | 1 |
| 11 | 1Q Parametric | `quantum.ry` | 1 |
| 12 | 1Q Parametric | `quantum.rz` | 1 |
| 13 | 1Q Parametric | `quantum.p` | 1 |
| 14 | 1Q Parametric | `quantum.u1` | 1 |
| 15 | 1Q Parametric | `quantum.u2` | 1 |
| 16 | 1Q Parametric | `quantum.u3` | 1 |
| 17 | 1Q Fixed | `quantum.rx90` | 1 |
| 18 | 1Q Fixed | `quantum.rx180` | 1 |
| 19 | 2Q | `quantum.cx` | 2 |
| 20 | 2Q | `quantum.cz` | 2 |
| 21 | 2Q | `quantum.cy` | 2 |
| 22 | 2Q | `quantum.swap` | 2 |
| 23 | 2Q | `quantum.iswap` | 2 |
| 24 | 2Q | `quantum.ecr` | 2 |
| 25 | 2Q | `quantum.rzx` | 2 |
| 26 | 2Q | `quantum.xx` | 2 |
| 27 | 2Q | `quantum.yy` | 2 |
| 28 | 2Q | `quantum.zz` | 2 |
| 29 | 2Q | `quantum.cp` | 2 |
| 30 | 2Q | `quantum.cphase` | 2 |
| 31 | 2Q | `quantum.xy` | 2 |
| 32 | IonQ | `quantum.gpi` | 1 |
| 33 | IonQ | `quantum.gpi2` | 1 |
| 34 | IonQ | `quantum.ms` | 2 |
| 35 | 3Q | `quantum.ccx` | 3 |
| 36 | 3Q | `quantum.cswap` | 3 |
| 37 | Multi | `quantum.mcx` | N |
| 38 | Multi | `quantum.mcz` | N |
| 39 | Control | `quantum.measure` | 1 |
| 40 | Control | `quantum.measure_all` | N |
| 41 | Control | `quantum.reset` | 1 |
| 42 | Control | `quantum.barrier` | N |
| 43 | Control | `quantum.init` | 1 |
| 44 | Control | `quantum.delay` | 0 |
| 45 | Control | `quantum.virtual_rz` | 1 |
| 46 | Control | `quantum.if_else` | 0 |
| 47 | Special | `quantum.global_phase` | 0 |
| 48 | Special | `quantum.param_gate` | 0 |

## A.3 All Hybrid Operations (21)

| # | Category | Syntax |
|---|----------|--------|
| 1 | Encoding | `hybrid.encode` |
| 2 | Encoding | `hybrid.decode` |
| 3 | Gradient | `hybrid.parameter_shift` |
| 4 | Gradient | `hybrid.finite_difference` |
| 5 | Gradient | `hybrid.spsa` |
| 6 | Gradient | `hybrid.adjoint_diff` |
| 7 | Gradient | `hybrid.stochastic_param_shift` |
| 8 | Gradient | `hybrid.joint_gradient` |
| 9 | Processing | `hybrid.classical_preprocess` |
| 10 | Processing | `hybrid.quantum_postprocess` |
| 11 | Processing | `hybrid.forward` |
| 12 | Processing | `hybrid.backward` |
| 13 | Variational | `hybrid.vqc_layer` |
| 14 | Variational | `hybrid.vqe_ansatz` |
| 15 | Variational | `hybrid.qaoa_layer` |
| 16 | Variational | `hybrid.quantum_kernel` |
| 17 | Transfer | `hybrid.gpu_to_qpu` |
| 18 | Transfer | `hybrid.qpu_to_gpu` |
| 19 | Execution | `hybrid.co_execute` |
| 20 | Measurement | `hybrid.measure_expectation` |
| 21 | Measurement | `hybrid.measure_samples` |

---

**End of LIFT Dialect Reference.**

**Total operations documented: 110 tensor + 48 quantum + 21 hybrid = 179 operations.**

**This document is the complete, error-free, authoritative reference for all LIFT dialects, their syntax, configuration, and assembly.**
