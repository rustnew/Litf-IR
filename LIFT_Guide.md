# Guide complet du framework LIFT — Toutes les fonctionnalités

> **LIFT** — *Language for Intelligent Frameworks and Technologies*
> Représentation intermédiaire unifiée pour l'IA et le calcul quantique.

Ce document décrit **chaque fonctionnalité** du framework LIFT, numérotée et classée par crate. Pour chaque fonctionnalité : ce qu'elle fait, comment l'utiliser, et avec quelles autres fonctionnalités la combiner.

---

## Table des matières

1. [Architecture générale](#1-architecture-générale)
2. [lift-core — Noyau IR](#2-lift-core--noyau-ir)
3. [lift-ast — Parsing du langage .lif](#3-lift-ast--parsing-du-langage-lif)
4. [lift-tensor — Opérations tensorielles (90+ ops)](#4-lift-tensor--opérations-tensorielles-90-ops)
5. [lift-quantum — Portes quantiques et bruit (50+ portes)](#5-lift-quantum--portes-quantiques-et-bruit-50-portes)
6. [lift-hybrid — Calcul hybride classique-quantique](#6-lift-hybrid--calcul-hybride-classique-quantique)
7. [lift-opt — Passes d'optimisation (11 passes)](#7-lift-opt--passes-doptimisation-11-passes)
8. [lift-sim — Simulation et analyse de coût](#8-lift-sim--simulation-et-analyse-de-coût)
9. [lift-predict — Prédiction de performance](#9-lift-predict--prédiction-de-performance)
10. [lift-import — Importation de modèles](#10-lift-import--importation-de-modèles)
11. [lift-export — Exportation vers backends](#11-lift-export--exportation-vers-backends)
12. [lift-config — Configuration (.lith)](#12-lift-config--configuration-lith)
13. [lift-cli — Interface en ligne de commande](#13-lift-cli--interface-en-ligne-de-commande)
14. [Combinaisons et pipelines complets](#14-combinaisons-et-pipelines-complets)
15. [Exemples concrets](#15-exemples-concrets)

---

## 1. Architecture générale

LIFT est un compilateur modulaire composé de **13 crates** organisées en couches :

```
                    ┌──────────┐
                    │ lift-cli │  ← Interface utilisateur
                    └────┬─────┘
           ┌─────────────┼─────────────┐
           │             │             │
    ┌──────┴──────┐ ┌────┴────┐ ┌──────┴──────┐
    │ lift-import │ │lift-opt │ │ lift-export │
    └──────┬──────┘ └────┬────┘ └──────┬──────┘
           │             │             │
    ┌──────┴──────┐ ┌────┴────┐ ┌──────┴──────┐
    │  lift-ast   │ │lift-sim │ │lift-predict │
    └──────┬──────┘ └────┬────┘ └──────┬──────┘
           │             │             │
    ┌──────┴─────────────┴─────────────┴──────┐
    │              lift-core                    │
    ├──────────┬──────────┬───────────────────┤
    │lift-tensor│lift-quantum│  lift-hybrid    │
    └──────────┴──────────┴───────────────────┘
```

### 1.1 Le pipeline de compilation

Le flux de travail standard est :

```
Source (.lif) → Lexer → Parser → IR (SSA) → Vérification → Optimisation → Simulation → Export
```

### 1.2 Formats de fichiers

| Extension | Description |
|-----------|-------------|
| `.lif`    | Code source LIFT IR |
| `.lith`   | Configuration de compilation |

### 1.3 Ajouter LIFT comme dépendance

```toml
[dependencies]
lift-core     = "0.2.0"
lift-tensor   = "0.2.0"
lift-quantum  = "0.2.0"
lift-hybrid   = "0.2.0"
lift-opt      = "0.2.0"
lift-sim      = "0.2.0"
lift-predict  = "0.2.0"
lift-import   = "0.2.0"
lift-export   = "0.2.0"
lift-config   = "0.2.0"
```

---

## 2. lift-core — Noyau IR

Le cœur du framework. Fournit la représentation intermédiaire SSA (Static Single Assignment).

### 2.1 Context — Le conteneur central

```rust
use lift_core::Context;

let mut ctx = Context::new();
```

Le `Context` stocke **toutes** les données IR : valeurs, opérations, blocs, régions, fonctions, modules, chaînes internées et types.

| Champ | Description | Usage |
|-------|-------------|-------|
| `ctx.values` | Toutes les valeurs SSA | Chaque résultat d'opération est une valeur unique |
| `ctx.ops` | Toutes les opérations | Instructions du programme |
| `ctx.blocks` | Blocs de base | Contiennent des séquences d'opérations |
| `ctx.regions` | Régions | Contiennent des blocs (corps de fonctions) |
| `ctx.modules` | Modules | Unités de compilation |
| `ctx.strings` | Interning de chaînes | `ctx.strings.intern("nom")` |
| `ctx.type_interner` | Interning de types | Déduplication des types |

**Combiner avec** : Toutes les autres crates. Le `Context` est le point d'entrée de tout pipeline.

### 2.2 Types — Système de types

```rust
use lift_core::types::*;

// Types de données
let fp32 = DataType::FP32;
let fp16 = DataType::FP16;
let bf16 = DataType::BF16;
let int8 = DataType::INT8;
let fp64 = DataType::FP64;

// Dimensions (statiques ou symboliques)
let batch = Dimension::Constant(32);
let seq = Dimension::Symbolic("seq_len".to_string());

// Info type tenseur
let tensor_info = TensorTypeInfo {
    shape: vec![Dimension::Constant(1), Dimension::Constant(784)],
    dtype: DataType::FP32,
    layout: MemoryLayout::Contiguous,
};

// Taille en octets
let bytes = tensor_info.size_bytes(); // Some(3136) = 1*784*4
```

**Types de données disponibles** :

| Type | Taille | Usage |
|------|--------|-------|
| `FP64` | 8 octets | Haute précision scientifique |
| `FP32` | 4 octets | Entraînement standard |
| `FP16` | 2 octets | Inférence rapide |
| `BF16` | 2 octets | Entraînement mixte (Google Brain) |
| `INT8` | 1 octet | Quantisation post-entraînement |
| `INT32` | 4 octets | Indices, compteurs |
| `BOOL` | 1 octet | Masques |

**Layouts mémoire** : `Contiguous`, `Strided`.

### 2.3 Attributes — Métadonnées des opérations

```rust
use lift_core::attributes::{Attribute, Attributes};

let mut attrs = Attributes::new();

// Différents types d'attributs
attrs.set("num_heads", Attribute::Integer(8));
attrs.set("dropout", Attribute::Float(0.1));
attrs.set("causal", Attribute::Bool(true));

// Lecture
let heads = attrs.get_integer("num_heads"); // Some(8)
let drop = attrs.get_float("dropout");       // Some(0.1)
let causal = attrs.get_bool("causal");       // Some(true)

// Vérification
assert!(attrs.contains("num_heads"));
assert_eq!(attrs.len(), 3);

// Itération
for (key, val) in attrs.iter() {
    println!("{}: {:?}", key, val);
}
```

**Combiner avec** : `lift-opt` (les passes lisent/écrivent des attributs), `lift-export` (les exporteurs lisent les attributs).

### 2.4 Verifier — Vérification d'invariants

```rust
use lift_core::verifier;

let ctx = Context::new();
match verifier::verify(&ctx) {
    Ok(()) => println!("IR valide"),
    Err(errors) => {
        for e in &errors {
            eprintln!("Erreur: {}", e);
        }
    }
}
```

Vérifie :
- **SSA** : chaque valeur est définie exactement une fois
- **Linéarité des qubits** : chaque qubit est utilisé exactement une fois
- **Typage** : cohérence des types entre opérations
- **Structure** : blocs, régions, terminateurs corrects

**Combiner avec** : Toujours utiliser après l'importation et après chaque passe d'optimisation.

### 2.5 Printer — Affichage de l'IR

```rust
use lift_core::printer::print_ir;

let ctx = Context::new();
let output = print_ir(&ctx);
println!("{}", output);
```

Produit une représentation textuelle lisible de l'IR, utile pour le débogage.

### 2.6 Pass Manager — Gestionnaire de passes

```rust
use lift_core::pass::{PassManager, Pass, PassResult, AnalysisCache};

let mut pm = PassManager::new();
pm.add_pass(Box::new(lift_opt::Canonicalize));
pm.add_pass(Box::new(lift_opt::DeadCodeElimination));
pm.add_pass(Box::new(lift_opt::TensorFusion));

let results = pm.run_all(&mut ctx);
for (name, result) in &results {
    match result {
        PassResult::Changed => println!("{}: modifié", name),
        PassResult::Unchanged => println!("{}: inchangé", name),
        PassResult::Error(e) => println!("{}: erreur: {}", name, e),
        PassResult::RolledBack => println!("{}: annulé", name),
    }
}
```

**Combiner avec** : `lift-opt` (toutes les 11 passes), `lift-config` (sélection des passes par configuration).

### 2.7 Dialect — Système de dialectes

```rust
use lift_core::dialect::{DialectRegistry, Dialect};

let registry = DialectRegistry::new();
// Les dialectes tensor, quantum, hybrid sont enregistrés automatiquement
```

Les trois dialectes LIFT :
- **tensor** : opérations sur tenseurs (`tensor.matmul`, `tensor.relu`, etc.)
- **quantum** : portes quantiques (`quantum.h`, `quantum.cx`, etc.)
- **hybrid** : opérations hybrides (`hybrid.encode`, `hybrid.vqc_layer`, etc.)

---

## 3. lift-ast — Parsing du langage .lif

### 3.1 Lexer — Tokenisation

```rust
use lift_ast::Lexer;

let source = r#"
#dialect tensor
module @mlp {
    func @forward(%x: tensor<1x784xf32>) -> tensor<1x10xf32> {
        %out = "tensor.relu"(%x) : (tensor<1x784xf32>) -> tensor<1x784xf32>
        return %out
    }
}
"#;

let mut lexer = Lexer::new(source);
let tokens = lexer.tokenize().to_vec();
assert!(lexer.errors().is_empty(), "Erreurs de lexing: {:?}", lexer.errors());
```

### 3.2 Parser — Analyse syntaxique

```rust
use lift_ast::Parser;

let mut parser = Parser::new(tokens);
let program = parser.parse().expect("Erreurs de parsing");
```

### 3.3 IrBuilder — Construction de l'IR

```rust
use lift_ast::IrBuilder;
use lift_core::Context;

let mut ctx = Context::new();
let mut builder = IrBuilder::new();
builder.build_program(&mut ctx, &program).expect("Erreurs de construction IR");
```

### 3.4 Pipeline complet de parsing

```rust
fn load_lif_file(path: &str) -> Result<Context, String> {
    let source = std::fs::read_to_string(path)
        .map_err(|e| format!("Lecture échouée: {}", e))?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize().to_vec();
    if !lexer.errors().is_empty() {
        return Err(format!("Erreurs lexer: {:?}", lexer.errors()));
    }

    let mut parser = Parser::new(tokens);
    let program = parser.parse().map_err(|e| format!("Erreurs parser: {:?}", e))?;

    let mut ctx = Context::new();
    let mut builder = IrBuilder::new();
    builder.build_program(&mut ctx, &program)?;
    Ok(ctx)
}
```

**Combiner avec** : `lift-core` (Context), puis `lift-opt` (optimisation), `lift-sim` (analyse), `lift-export` (compilation).

---

## 4. lift-tensor — Opérations tensorielles (90+ ops)

### 4.1 Liste complète des opérations par catégorie

#### 4.1.1 Arithmétique de base (5 ops)

| # | Op | Nom IR | Entrées | Description |
|---|-----|---------|---------|-------------|
| 1 | `Add` | `tensor.add` | 2 | Addition élément par élément |
| 2 | `Sub` | `tensor.sub` | 2 | Soustraction |
| 3 | `Mul` | `tensor.mul` | 2 | Multiplication élément par élément |
| 4 | `Div` | `tensor.div` | 2 | Division |
| 5 | `Neg` | `tensor.neg` | 1 | Négation |

```rust
use lift_tensor::ops::TensorOp;

let op = TensorOp::MatMul;
println!("Nom: {}", op.name());           // "tensor.matmul"
println!("Inputs: {:?}", op.num_inputs()); // (2, 2)
println!("FLOPs: {}", op.flops_formula()); // "2*M*N*K"
```

#### 4.1.2 Algèbre linéaire (4 ops)

| # | Op | Entrées | Description |
|---|-----|---------|-------------|
| 6 | `MatMul` | 2 | Multiplication matricielle |
| 7 | `Linear` | 3 | Couche linéaire (matmul + bias) |
| 8 | `Embedding` | 2 | Table de plongement |
| 9 | `SparseMatMul` | 2 | MatMul creux |

#### 4.1.3 Activations (11 ops)

| # | Op | Description | Formule FLOPs |
|---|-----|-------------|---------------|
| 10 | `ReLU` | max(0, x) | N |
| 11 | `GeLU` | Gaussian Error Linear Unit | ~8N |
| 12 | `SiLU` | x * sigmoid(x) (Swish) | ~8N |
| 13 | `Sigmoid` | 1/(1+exp(-x)) | N |
| 14 | `Tanh` | Tangente hyperbolique | N |
| 15 | `Softmax` | exp(x)/sum(exp(x)) | 5N |
| 16 | `LeakyReLU` | max(αx, x) | N |
| 17 | `ELU` | Exponential Linear Unit | N |
| 18 | `Mish` | x * tanh(softplus(x)) | ~8N |
| 19 | `HardSwish` | Approximation de Swish | ~8N |
| 20 | `HardSigmoid` | Approximation de Sigmoid | N |

```rust
assert!(TensorOp::ReLU.is_activation());
assert!(!TensorOp::MatMul.is_activation());
```

#### 4.1.4 Normalisation (5 ops)

| # | Op | Entrées | Description |
|---|-----|---------|-------------|
| 21 | `LayerNorm` | 2-3 | Normalisation par couche |
| 22 | `RMSNorm` | 2-3 | Root Mean Square Norm (LLaMA) |
| 23 | `BatchNorm` | 3-5 | Normalisation par batch |
| 24 | `GroupNorm` | 2-3 | Normalisation par groupe |
| 25 | `InstanceNorm` | 2-3 | Normalisation par instance |

```rust
assert!(TensorOp::LayerNorm.is_normalisation());
```

#### 4.1.5 Attention (8 ops)

| # | Op | Entrées | Description |
|---|-----|---------|-------------|
| 26 | `Attention` | 3-4 | Attention standard (Q, K, V, [mask]) |
| 27 | `MultiHeadAttention` | 3-4 | Multi-tête |
| 28 | `MultiQueryAttention` | 3-4 | Multi-query (Llama) |
| 29 | `GroupedQueryAttention` | 3-4 | Grouped query (GQA) |
| 30 | `FlashAttention` | 3-4 | FlashAttention V2 (mémoire O(N)) |
| 31 | `SlidingWindowAttention` | 3-4 | Fenêtre glissante (Mistral) |
| 32 | `CrossAttention` | 3-4 | Cross-attention (encodeur-décodeur) |
| 33 | `PagedAttention` | 3-5 | Paged attention (vLLM) |

```rust
assert!(TensorOp::FlashAttention.is_attention());
```

#### 4.1.6 Convolutions (6 ops)

| # | Op | Description |
|---|-----|-------------|
| 34 | `Conv2D` | Convolution 2D standard |
| 35 | `Conv1D` | Convolution 1D (audio, séquences) |
| 36 | `Conv3D` | Convolution 3D (vidéo, volumétrique) |
| 37 | `ConvTranspose2D` | Convolution transposée (upsampling) |
| 38 | `DepthwiseConv2D` | Convolution en profondeur (MobileNet) |
| 39 | `DilatedConv2D` | Convolution dilatée (réceptif large) |

#### 4.1.7 Pooling (4 ops)

| # | Op | Description |
|---|-----|-------------|
| 40 | `MaxPool2D` | Max pooling 2D |
| 41 | `AvgPool2D` | Average pooling 2D |
| 42 | `AdaptiveAvgPool2D` | Adaptive average pooling |
| 43 | `GlobalAvgPool` | Global average pooling |

#### 4.1.8 Opérations de forme (13 ops)

| # | Op | Description | FLOPs |
|---|-----|-------------|-------|
| 44 | `Reshape` | Changer la forme | 0 |
| 45 | `Transpose` | Transposer | 0 |
| 46 | `Concat` | Concaténer | 0 |
| 47 | `Split` | Diviser | 0 |
| 48 | `Gather` | Indexation avancée | 0 |
| 49 | `Scatter` | Écriture indexée | 0 |
| 50 | `Squeeze` | Retirer dim=1 | 0 |
| 51 | `Unsqueeze` | Ajouter dim=1 | 0 |
| 52 | `Permute` | Permuter dimensions | 0 |
| 53 | `Expand` | Expansion broadcast | 0 |
| 54 | `Slice` | Tranche | 0 |
| 55 | `Pad` | Remplissage | 0 |
| 56 | `Tile` | Répétition | 0 |

```rust
assert!(TensorOp::Reshape.is_zero_flop());
```

#### 4.1.9 Constantes (5 ops)

| # | Op | Description |
|---|-----|-------------|
| 57 | `Constant` | Tenseur constant |
| 58 | `Zeros` | Tenseur de zéros |
| 59 | `Ones` | Tenseur de uns |
| 60 | `Arange` | Séquence [0, 1, ..., n-1] |
| 61 | `Full` | Tenseur rempli d'une valeur |

#### 4.1.10 Récurrent (3 ops)

| # | Op | Description |
|---|-----|-------------|
| 62 | `LSTMCell` | Cellule LSTM |
| 63 | `GRUCell` | Cellule GRU |
| 64 | `RNNCell` | Cellule RNN simple |

#### 4.1.11 Mathématiques avancées (9 ops)

| # | Op | Description |
|---|-----|-------------|
| 65 | `Einsum` | Notation Einstein |
| 66 | `FFT` | Transformée de Fourier rapide |
| 67 | `IFFT` | FFT inverse |
| 68 | `SVD` | Décomposition en valeurs singulières |
| 69 | `Eig` | Décomposition propre |
| 70 | `Solve` | Résolution de systèmes linéaires |
| 71 | `TopK` | K plus grandes valeurs |
| 72 | `Sort` | Tri |
| 73 | `Cumsum` | Somme cumulative |

#### 4.1.12 Quantisation (6 ops)

| # | Op | Description |
|---|-----|-------------|
| 74 | `Quantize` | FP → INT8 |
| 75 | `Dequantize` | INT8 → FP |
| 76 | `QuantizeInt4` | FP → INT4 |
| 77 | `DequantizeInt4` | INT4 → FP |
| 78 | `QuantizeFp8` | FP → FP8 |
| 79 | `DequantizeFp8` | FP8 → FP |

#### 4.1.13 Diffusion / Génératif (3 ops)

| # | Op | Description |
|---|-----|-------------|
| 80 | `UNetDownBlock` | Bloc descendant U-Net |
| 81 | `UNetUpBlock` | Bloc montant U-Net |
| 82 | `TimestepEmbedding` | Embedding temporel (Stable Diffusion) |

#### 4.1.14 GNN — Réseaux de neurones sur graphes (2 ops)

| # | Op | Description |
|---|-----|-------------|
| 83 | `GNNMessagePassing` | Passage de messages GNN |
| 84 | `GNNGlobalPooling` | Pooling global GNN |

#### 4.1.15 MoE — Mixture of Experts (2 ops)

| # | Op | Description |
|---|-----|-------------|
| 85 | `MoEDispatch` | Routage vers les experts |
| 86 | `MoECombine` | Combinaison des sorties experts |

#### 4.1.16 Mémoire et gradient (11 ops)

| # | Op | Description |
|---|-----|-------------|
| 87 | `Checkpoint` | Gradient checkpointing (économie mémoire) |
| 88 | `Offload` | Offload CPU (pour grands modèles) |
| 89 | `GradAccumulate` | Accumulation de gradients |
| 90 | `GradMatMul` | Gradient du MatMul |
| 91 | `GradReLU` | Gradient du ReLU |
| 92 | `GradSoftmax` | Gradient du Softmax |
| 93 | `GradLayerNorm` | Gradient du LayerNorm |
| 94 | `GradAttention` | Gradient de l'Attention |
| 95 | `GradConv2D` | Gradient du Conv2D |
| 96 | `GradLinear` | Gradient du Linear |
| 97 | `GradGeLU` | Gradient du GeLU |

#### 4.1.17 Parallélisme (4 ops)

| # | Op | Description |
|---|-----|-------------|
| 98 | `ParallelSplit` | Découpage pour data parallel |
| 99 | `ParallelAllReduce` | All-reduce entre GPUs |
| 100 | `PipelineSend` | Envoi pipeline parallel |
| 101 | `PipelineReceive` | Réception pipeline parallel |

#### 4.1.18 Opérations fusionnées (6 ops)

| # | Op | Description | Gain |
|---|-----|-------------|------|
| 102 | `FusedMatMulBiasReLU` | MatMul + Bias + ReLU | 1 kernel au lieu de 3 |
| 103 | `FusedMatMulBias` | MatMul + Bias | 1 kernel au lieu de 2 |
| 104 | `FusedLinearGeLU` | Linear + GeLU | Gain bandwidth |
| 105 | `FusedAttentionLayerNorm` | Attention + LayerNorm | Réduction mémoire |
| 106 | `FusedLinearSiLU` | Linear + SiLU | Gain bandwidth |
| 107 | `FusedConvBatchNormReLU` | Conv + BN + ReLU | Inférence rapide |

### 4.2 Shape Inference — Inférence de forme

```rust
use lift_core::types::*;
use lift_tensor::ops::TensorOp;
use lift_tensor::shape::ShapeInference;

fn mk(shape: Vec<usize>, dtype: DataType) -> TensorTypeInfo {
    TensorTypeInfo {
        shape: shape.into_iter().map(Dimension::Constant).collect(),
        dtype,
        layout: MemoryLayout::Contiguous,
    }
}

// Inférence de forme
let a = mk(vec![2, 3, 64], DataType::FP32);
let b = mk(vec![2, 64, 128], DataType::FP32);
let result = ShapeInference::infer_output_shape(&TensorOp::MatMul, &[&a, &b]).unwrap();
// result[0].shape = [2, 3, 128]

// Calcul de FLOPs
let flops = ShapeInference::compute_flops(&TensorOp::MatMul, &[&a, &b]);
println!("FLOPs: {:?}", flops); // Some(49152)

// Calcul de mémoire
let mem = ShapeInference::compute_memory_bytes(&TensorOp::MatMul, &[&a, &b]);
println!("Mémoire: {:?} bytes", mem);
```

**Combiner avec** : `lift-sim` (modèle de coût utilise les FLOPs), `lift-predict` (prédiction roofline).

### 4.3 Prédicats utiles

```rust
let op = TensorOp::FlashAttention;

op.is_attention();      // true — variante d'attention ?
op.is_convolution();    // false — convolution ?
op.is_normalisation();  // false — normalisation ?
op.is_activation();     // false — activation ?
op.is_fused();          // false — opération fusionnée ?
op.is_gradient();       // false — opération de gradient ?
op.is_zero_flop();      // false — zéro FLOPs (reshape, etc.) ?
op.num_inputs();        // (3, 4) — min/max nombre d'entrées
op.flops_formula();     // "2*B*H*(S^2*D + S*D^2)"
```

---

## 5. lift-quantum — Portes quantiques et bruit (50+ portes)

### 5.1 Portes quantiques

#### 5.1.1 Portes 1-qubit standard (9 portes)

| # | Porte | Nom IR | Type | Description |
|---|-------|--------|------|-------------|
| 1 | `H` | `quantum.h` | Clifford | Hadamard |
| 2 | `X` | `quantum.x` | Pauli | NOT quantique (bit-flip) |
| 3 | `Y` | `quantum.y` | Pauli | Rotation Y de π |
| 4 | `Z` | `quantum.z` | Pauli | Phase-flip |
| 5 | `S` | `quantum.s` | Clifford | Phase π/2 |
| 6 | `Sdg` | `quantum.sdg` | Clifford | S inverse |
| 7 | `T` | `quantum.t` | Non-Clifford | Phase π/4 (coûteuse pour QEC) |
| 8 | `Tdg` | `quantum.tdg` | Non-Clifford | T inverse |
| 9 | `SX` | `quantum.sx` | Clifford | Racine de X |

#### 5.1.2 Portes 1-qubit paramétriques (9 portes)

| # | Porte | Paramètres | Description |
|---|-------|-----------|-------------|
| 10 | `RX` | θ | Rotation autour de X |
| 11 | `RY` | θ | Rotation autour de Y |
| 12 | `RZ` | θ | Rotation autour de Z |
| 13 | `P` | φ | Phase |
| 14 | `U1` | λ | Porte unitaire U1 |
| 15 | `U2` | φ, λ | Porte unitaire U2 |
| 16 | `U3` | θ, φ, λ | Porte unitaire générale |
| 17 | `Rx90` | — | RX(π/2) fixe |
| 18 | `Rx180` | — | RX(π) fixe |

#### 5.1.3 Portes 2-qubits (14 portes)

| # | Porte | Description | Natif pour |
|---|-------|-------------|-----------|
| 19 | `CX` | CNOT | IBM |
| 20 | `CZ` | Controlled-Z | IBM, Rigetti |
| 21 | `CY` | Controlled-Y | — |
| 22 | `SWAP` | Échange de qubits | — |
| 23 | `ISWAP` | iSWAP | Rigetti |
| 24 | `ECR` | Echoed Cross-Resonance | IBM Eagle |
| 25 | `RZX` | ZX rotation | IBM |
| 26 | `XX` | Ising XX | IonQ |
| 27 | `YY` | Ising YY | IonQ |
| 28 | `ZZ` | Ising ZZ | IonQ |
| 29 | `CPhase` | Controlled Phase | Rigetti |
| 30 | `XY` | XY interaction | Rigetti |
| 31 | `CP` | Controlled Phase | — |
| 32 | `MS` | Mølmer–Sørensen | IonQ |

#### 5.1.4 Portes 3-qubits et multi-contrôle (4 portes)

| # | Porte | Description |
|---|-------|-------------|
| 33 | `CCX` | Toffoli (CCNOT) |
| 34 | `CSWAP` | Fredkin |
| 35 | `MCX` | Multi-controlled X |
| 36 | `MCZ` | Multi-controlled Z |

#### 5.1.5 Portes spéciales et contrôle (10 portes)

| # | Porte | Description |
|---|-------|-------------|
| 37 | `GlobalPhase` | Phase globale |
| 38 | `Delay` | Délai (décoherence) |
| 39 | `VirtualRZ` | RZ virtuel (sans coût physique) |
| 40 | `IfElse` | Contrôle conditionnel classique |
| 41 | `Measure` | Mesure 1 qubit |
| 42 | `MeasureAll` | Mesure tous les qubits |
| 43 | `Reset` | Réinitialisation |
| 44 | `Barrier` | Barrière (empêche l'optimisation) |
| 45 | `Init` | Initialisation |
| 46 | `ParamGate` | Porte paramétrée générique |

```rust
use lift_quantum::gates::QuantumGate;

let gate = QuantumGate::H;
println!("Nom: {}", gate.op_name());        // "quantum.h"
println!("Qubits: {}", gate.num_qubits());  // 1
println!("Clifford: {}", gate.is_clifford()); // true
println!("Parametric: {}", gate.is_parametric()); // false
println!("Self-inverse: {}", gate.is_self_inverse()); // true

// Retrouver une porte par son nom IR
let gate = QuantumGate::from_name("quantum.cx"); // Some(CX)
```

### 5.2 Hardware Providers — Jeux de portes natifs

```rust
use lift_quantum::gates::{QuantumGate, Provider};

// Portes natives par fournisseur
let ibm_basis = QuantumGate::native_basis(Provider::IbmEagle);
let rigetti_basis = QuantumGate::native_basis(Provider::Rigetti);
let ionq_basis = QuantumGate::native_basis(Provider::IonQ);
let quant_basis = QuantumGate::native_basis(Provider::Quantinuum);
```

| Provider | Portes natives |
|----------|---------------|
| `IbmEagle` | CX, RZ, SX, X |
| `IbmKyoto` | ECR, RZ, SX, X |
| `Rigetti` | CZ, RX, RZ |
| `IonQ` | GPI, GPI2, MS |
| `Quantinuum` | RZ, RX, ZZ |
| `Simulator` | Toutes les portes |

**Combiner avec** : `lift-opt::LayoutMapping` (transpilation vers matériel cible).

### 5.3 Device Topology — Topologie du matériel

```rust
use lift_quantum::topology::DeviceTopology;

// Topologies prédéfinies
let linear = DeviceTopology::linear(10);         // Chaîne linéaire
let grid = DeviceTopology::grid(3, 3);           // Grille 3x3
let hex = DeviceTopology::heavy_hex(27);         // Heavy-hex IBM
let ion = DeviceTopology::all_to_all(32);        // All-to-all (ions piégés)
let tree = DeviceTopology::tree(15);             // Arbre binaire

// Topologie personnalisée
let custom = DeviceTopology::custom("my_chip",
    &[(0,1), (1,2), (2,3), (0,3)], 0.99);

// Interrogation
linear.are_connected(0, 1);           // true
linear.shortest_path(0, 4);           // Some([0, 1, 2, 3, 4])
linear.swap_distance(0, 4);           // Some(3)
linear.avg_connectivity();             // connectivité moyenne
linear.diameter();                     // diamètre du graphe
grid.neighbors(4);                     // voisins du qubit 4
```

**Combiner avec** : `lift-opt::LayoutMapping`, `lift-opt::NoiseAwareSchedule`.

### 5.4 Noise Models — Modèles de bruit

```rust
use lift_quantum::noise::{NoiseModel, GateNoise, CircuitNoise};

// Modèles de bruit
let ideal = NoiseModel::Ideal;
let depol = NoiseModel::Depolarizing { p: 0.01 };
let bitflip = NoiseModel::BitFlip { p: 0.001 };
let phaseflip = NoiseModel::PhaseFlip { p: 0.001 };

// Fidélité du modèle
let fidelity = depol.fidelity(); // 0.99

// Bruit par porte
let gate_noise = GateNoise::with_depolarizing(0.999, 0.02);

// Analyse de circuit complet
let mut cn = CircuitNoise::new();
// ... accumulation des bruits
println!("Fidélité totale: {}", cn.total_fidelity);
println!("Portes 2-qubit: {}", cn.two_qubit_count);
```

### 5.5 Kraus Channels — Canaux de bruit quantique

```rust
use lift_quantum::kraus::{ComplexMatrix, KrausChannel};

// Canaux de bruit prédéfinis
let depol = KrausChannel::depolarizing(0.01, 1);     // Dépolarisant 1 qubit
let amp = KrausChannel::amplitude_damping(0.02);      // Amortissement d'amplitude
let phase = KrausChannel::phase_damping(0.01);         // Amortissement de phase

// Fidélité du canal
let fidelity = depol.average_gate_fidelity();
println!("Fidélité: {:.6}", fidelity);

// Matrices complexes
let mut m = ComplexMatrix::identity(2);
let dagger = m.dagger();    // Conjugué transposé
let product = m.mul(&dagger).unwrap();
let trace = m.trace().unwrap();
```

**Combiner avec** : `lift-sim::QuantumCostModel` (estimation fidélité circuit), `lift-opt::NoiseAwareSchedule`.

### 5.6 QEC — Correction d'erreurs quantiques

```rust
use lift_quantum::qec::{QecCode, QecAnalysis};

// Codes QEC disponibles
let surface = QecCode::SurfaceCode { distance: 5 };   // 25 qubits physiques/logique
let steane = QecCode::SteaneCode;                       // 7 qubits physiques/logique
let shor = QecCode::ShorCode;                            // 9 qubits physiques/logique
let rep = QecCode::RepetitionCode { distance: 7 };     // 7 qubits physiques
let ldpc = QecCode::LdpcCode { n: 100, k: 10 };       // Code LDPC

// Propriétés du code
println!("Physiques/logique: {}", surface.physical_per_logical()); // 25
println!("Distance: {}", surface.code_distance());                 // 5
println!("Profondeur syndrome: {}", surface.syndrome_circuit_depth()); // 5

// Analyse QEC complète
let analysis = QecAnalysis::analyse(
    10,     // qubits logiques
    100,    // profondeur circuit
    QecCode::SurfaceCode { distance: 5 },
    0.001,  // taux d'erreur physique
);
println!("Qubits physiques: {}", analysis.physical_qubits);
println!("Taux erreur logique: {:.2e}", analysis.logical_error_rate);
println!("Overhead: {}", analysis.overhead_qubits);
```

**Combiner avec** : `lift-sim::QuantumCostModel`, `lift-predict` (budget fidélité).

---

## 6. lift-hybrid — Calcul hybride classique-quantique

### 6.1 Opérations hybrides (21 ops)

#### 6.1.1 Encodage/Décodage (2 ops)

| # | Op | Nom IR | Description |
|---|-----|--------|-------------|
| 1 | `Encode` | `hybrid.encode` | Encoder données classiques → qubits |
| 2 | `Decode` | `hybrid.decode` | Décoder mesures quantiques → classique |

#### 6.1.2 Méthodes de gradient (6 ops)

| # | Op | Nom IR | Évaluations | Exact ? |
|---|-----|--------|-------------|---------|
| 3 | `ParameterShift` | `hybrid.parameter_shift` | 2N | Oui |
| 4 | `FiniteDifference` | `hybrid.finite_difference` | N+1 | Non |
| 5 | `SPSA` | `hybrid.spsa` | 2 | Non |
| 6 | `AdjointDifferentiation` | `hybrid.adjoint_diff` | 1 | Oui |
| 7 | `StochasticParameterShift` | `hybrid.stochastic_param_shift` | 2 | Non |
| 8 | `JointGradient` | `hybrid.joint_gradient` | Combiné | — |

```rust
use lift_hybrid::gradient::GradientMethod;

let method = GradientMethod::ParameterShift;
let evals = method.circuit_evaluations(100); // 200 évaluations pour 100 params
assert!(method.is_exact()); // true
```

#### 6.1.3 Traitement (4 ops)

| # | Op | Description |
|---|-----|-------------|
| 9 | `ClassicalPreprocess` | Pré-traitement classique |
| 10 | `QuantumPostprocess` | Post-traitement quantique |
| 11 | `HybridForward` | Passe forward hybride |
| 12 | `HybridBackward` | Passe backward hybride |

#### 6.1.4 Algorithmes variationnels (4 ops)

| # | Op | Description | Usage |
|---|-----|-------------|-------|
| 13 | `VqcLayer` | Couche de circuit variationnel | Classification quantique |
| 14 | `VqeAnsatz` | Ansatz pour VQE | Chimie quantique |
| 15 | `QaoaLayer` | Couche QAOA | Optimisation combinatoire |
| 16 | `QuantumKernel` | Noyau quantique | Machine learning quantique |

#### 6.1.5 Transfert de données (2 ops)

| # | Op | Description |
|---|-----|-------------|
| 17 | `GpuToQpu` | Transfert GPU → QPU |
| 18 | `QpuToGpu` | Transfert QPU → GPU |

#### 6.1.6 Co-exécution et mesure (3 ops)

| # | Op | Description |
|---|-----|-------------|
| 19 | `CoExecute` | Exécution simultanée classique+quantique |
| 20 | `MeasureExpectation` | Valeur d'espérance d'un observable |
| 21 | `MeasureSamples` | Échantillonnage de mesures |

```rust
use lift_hybrid::ops::HybridOp;

let op = HybridOp::VqcLayer;
assert!(op.is_variational());
assert!(!op.is_gradient());
```

### 6.2 Encoding Strategies — Stratégies d'encodage

```rust
use lift_hybrid::encoding::{EncodingStrategy, EncodingConfig};

let strategies = [
    EncodingStrategy::AngleEncoding,       // 1 qubit/feature, profondeur 1
    EncodingStrategy::AmplitudeEncoding,    // log2(n) qubits, profondeur n
    EncodingStrategy::BasisEncoding,        // 1 qubit/feature, profondeur 1
    EncodingStrategy::IQPEncoding,          // 1 qubit/feature, profondeur 2n
    EncodingStrategy::HamiltonianEncoding,  // 1 qubit/feature, profondeur n
    EncodingStrategy::KernelEncoding,       // 1 qubit/feature, profondeur 3n
];

// Configuration d'encodage
let config = EncodingConfig::new(EncodingStrategy::AmplitudeEncoding, 256);
println!("Qubits nécessaires: {}", config.num_qubits); // 8 = log2(256)
println!("Dimension classique: {}", config.classical_dim); // 256
```

| Stratégie | Qubits | Profondeur | Meilleur pour |
|-----------|--------|-----------|---------------|
| Angle | n | 1 | Peu de features |
| Amplitude | log₂(n) | n | Beaucoup de features |
| Basis | n | 1 | Données binaires |
| IQP | n | 2n | Avantage quantique |
| Hamiltonian | n | n | Simulation physique |
| Kernel | n | 3n | Quantum ML |

### 6.3 Gradient Configuration — Configuration du gradient joint

```rust
use lift_hybrid::gradient::{GradientMethod, JointGradientConfig};

let config = JointGradientConfig {
    classical_method: GradientMethod::Backprop,
    quantum_method: GradientMethod::ParameterShift,
    num_classical_params: 1000,
    num_quantum_params: 50,
};
println!("Évaluations totales: {}", config.total_evaluations());
// 1 (backprop) + 100 (2*50 parameter shift) = 101
```

### 6.4 Types auxiliaires

```rust
use lift_hybrid::ops::{AnsatzType, SyncPolicy, FeatureMap};

// Types d'ansatz pour VQC
let ansatz = AnsatzType::HardwareEfficient; // HardwareEfficient, StronglyEntangling, TwoLocal, UCCSD, Custom

// Politique de synchronisation
let sync = SyncPolicy::Blocking; // Blocking, Asynchronous, Pipeline

// Feature maps pour quantum kernels
let fm = FeatureMap::ZZFeatureMap; // ZZFeatureMap, PauliFeatureMap, AngleEncoding, AmplitudeEncoding
```

---

## 7. lift-opt — Passes d'optimisation (11 passes)

### 7.1 Passes classiques (5 passes)

#### 7.1.1 Canonicalize — Mise en forme canonique

```rust
use lift_opt::Canonicalize;
use lift_core::pass::Pass;

let pass = Canonicalize;
// Réordonne les opérations en forme canonique
// Normalise les patterns d'IR pour faciliter les optimisations suivantes
```

**Usage** : Toujours exécuter en premier dans le pipeline.

#### 7.1.2 ConstantFolding — Repli de constantes

```rust
use lift_opt::ConstantFolding;

let pass = ConstantFolding;
// Évalue les opérations dont tous les opérandes sont constants au compile-time
// Exemple: add(const(2), const(3)) → const(5)
```

#### 7.1.3 DeadCodeElimination — Élimination du code mort

```rust
use lift_opt::DeadCodeElimination;

let pass = DeadCodeElimination;
// Supprime les opérations dont les résultats ne sont jamais utilisés
// Respecte les opérations avec effets de bord (mesures, etc.)
```

#### 7.1.4 TensorFusion — Fusion de tenseurs

```rust
use lift_opt::TensorFusion;

let pass = TensorFusion;
// Fusionne les opérations consécutives en opérations fusionnées
// Exemple: MatMul + Bias + ReLU → FusedMatMulBiasReLU
// Réduit les accès mémoire et les lancements de kernels
```

**Combiner avec** : Exécuter après `Canonicalize` et `ConstantFolding`.

#### 7.1.5 CommonSubexprElimination — Élimination des sous-expressions communes

```rust
use lift_opt::CommonSubexprElimination;

let pass = CommonSubexprElimination;
// Détecte les opérations identiques (même op, mêmes opérandes)
// Remplace les duplicats par des références à la première occurrence
// Exclut les opérations avec effets de bord
```

### 7.2 Passes quantiques (3 passes)

#### 7.2.1 GateCancellation — Annulation de portes

```rust
use lift_opt::GateCancellation;

let pass = GateCancellation;
// Supprime les paires de portes qui s'annulent
// Exemple: H H → identité, X X → identité
// Respecte les invariants de linéarité des qubits
```

#### 7.2.2 RotationMerge — Fusion de rotations

```rust
use lift_opt::RotationMerge;

let pass = RotationMerge;
// Fusionne les rotations consécutives sur le même axe
// Exemple: RZ(0.3) RZ(0.5) → RZ(0.8)
// Supprime les rotations identité (angle ≈ 0)
```

#### 7.2.3 NoiseAwareSchedule — Ordonnancement conscient du bruit

```rust
use lift_opt::NoiseAwareSchedule;

let pass = NoiseAwareSchedule;
// Réordonne les portes quantiques pour minimiser la décohérence
// Priorise les portes rapides (1-qubit) avant les lentes (2-qubit)
// Respecte les dépendances SSA
```

**Combiner avec** : `lift-quantum::topology::DeviceTopology` pour la topologie cible.

### 7.3 Passes IA avancées (3 passes)

#### 7.3.1 FlashAttentionPass — Remplacement par FlashAttention

```rust
use lift_opt::FlashAttentionPass;

let pass = FlashAttentionPass::default(); // seuil = 512
let pass_custom = FlashAttentionPass { seq_len_threshold: 1024 };
// Remplace tensor.attention par tensor.flash_attention
// quand la longueur de séquence dépasse le seuil
// Réduit la complexité mémoire de O(N²) à O(N)
```

#### 7.3.2 QuantisationPass — Annotation de quantisation

```rust
use lift_opt::QuantisationPass;
use lift_opt::quantisation_pass::{QuantTarget, QuantMode};

let pass = QuantisationPass::default(); // INT8, Dynamic
let pass_custom = QuantisationPass {
    target_dtype: QuantTarget::Fp8E4M3,
    mode: QuantMode::Static,
};
// Annote les opérations lourdes (MatMul, Conv, Linear, Attention)
// avec des métadonnées de quantisation
// Insère des paires Quantize/Dequantize autour des ops annotées
```

| Cible | Taille | Usage |
|-------|--------|-------|
| `Int8` | 1 octet | Inférence standard |
| `Int4` | 0.5 octet | LLM compressés (GPTQ, AWQ) |
| `Fp8E4M3` | 1 octet | Entraînement H100 |
| `Fp8E5M2` | 1 octet | Inférence H100 |

#### 7.3.3 LayoutMapping — Mapping de qubits

```rust
use lift_opt::LayoutMapping;

let pass = LayoutMapping;
// Insère des portes SWAP pour mapper les qubits logiques aux physiques
// Basé sur la topologie du dispositif cible
// Marque les opérations nécessitant des swaps via attributs
```

**Combiner avec** : `lift-quantum::topology::DeviceTopology`.

### 7.4 Pipeline d'optimisation recommandé

```rust
use lift_core::PassManager;

let mut pm = PassManager::new();

// Phase 1: Nettoyage
pm.add_pass(Box::new(lift_opt::Canonicalize));
pm.add_pass(Box::new(lift_opt::ConstantFolding));
pm.add_pass(Box::new(lift_opt::DeadCodeElimination));
pm.add_pass(Box::new(lift_opt::CommonSubexprElimination));

// Phase 2: Fusion (IA)
pm.add_pass(Box::new(lift_opt::TensorFusion));
pm.add_pass(Box::new(lift_opt::FlashAttentionPass::default()));
pm.add_pass(Box::new(lift_opt::QuantisationPass::default()));

// Phase 3: Quantique
pm.add_pass(Box::new(lift_opt::GateCancellation));
pm.add_pass(Box::new(lift_opt::RotationMerge));
pm.add_pass(Box::new(lift_opt::NoiseAwareSchedule));
pm.add_pass(Box::new(lift_opt::LayoutMapping));

// Phase 4: Nettoyage final
pm.add_pass(Box::new(lift_opt::DeadCodeElimination));

let results = pm.run_all(&mut ctx);
```

---

## 8. lift-sim — Simulation et analyse de coût

### 8.1 CostModel — Modèle de coût classique

```rust
use lift_sim::cost::CostModel;

// Profils GPU prédéfinis
let a100 = CostModel::a100();  // 312 TFLOPS, 2039 GB/s
let h100 = CostModel::h100();  // 989 TFLOPS, 3350 GB/s

// Estimation du temps
let flops = 2 * 1024 * 1024 * 1024_u64;
let bytes = 4 * 1024 * 1024_u64;

let compute_ms = a100.compute_time_ms(flops);      // Temps compute
let memory_ms = a100.memory_time_ms(bytes);         // Temps mémoire
let roofline_ms = a100.roofline_time_ms(flops, bytes); // Modèle roofline

// Analyse
let ai = a100.arithmetic_intensity(flops, bytes);   // FLOPs/byte
let bound = a100.is_compute_bound(flops, bytes);    // true = compute-bound
let fits = a100.fits_in_memory(bytes);               // Tient en mémoire ?
let gpus = a100.num_gpus_needed(bytes);              // GPUs nécessaires
```

### 8.2 QuantumCostModel — Modèle de coût quantique

```rust
use lift_sim::cost::QuantumCostModel;

// Profils de processeurs quantiques
let sc = QuantumCostModel::superconducting_default(); // IBM-like: 127 qubits
let ion = QuantumCostModel::trapped_ion_default();     // IonQ-like: 32 qubits
let atom = QuantumCostModel::neutral_atom_default();   // Atom-like: 256 qubits

// Estimation de fidélité d'un circuit
let fidelity = sc.circuit_fidelity(50, 20); // 50 portes 1Q, 20 portes 2Q
println!("Fidélité: {:.6}", fidelity);

// Temps du circuit
let time_us = sc.circuit_time_us(50, 20, 5, 10); // 50 1Q, 20 2Q, 5 mesures, 10 profondeur
println!("Temps: {:.2} µs", time_us);

// Fidélité de décohérence
let decoherence = sc.decoherence_fidelity(time_us);
println!("Fidélité décohérence: {:.6}", decoherence);
```

| Paramètre | Supraconducteur | Ions piégés | Atomes neutres |
|-----------|----------------|-------------|----------------|
| Temps 1Q | 0.02 µs | 10 µs | 0.5 µs |
| Temps 2Q | 0.3 µs | 200 µs | 1.0 µs |
| Fidélité 1Q | 99.9% | 99.99% | 99.9% |
| Fidélité 2Q | 99% | 99.9% | 99.5% |
| T1 | 100 µs | 1 s | 5 ms |
| Qubits | 127 | 32 | 256 |

### 8.3 Budget — Contraintes de ressources

```rust
use lift_sim::cost::Budget;

let budget = Budget {
    max_flops: Some(1_000_000_000_000), // 1 TFLOP max
    max_memory_bytes: Some(80_000_000_000), // 80 GB
    max_time_ms: Some(100.0),           // 100 ms
    min_fidelity: Some(0.99),           // 99% fidélité min
    max_circuit_depth: Some(1000),      // 1000 couches max
};

budget.check_flops(500_000_000_000).unwrap();   // OK
budget.check_memory(40_000_000_000).unwrap();   // OK
budget.check_fidelity(0.995).unwrap();           // OK
```

### 8.4 EnergyModel — Estimation énergétique et carbone

```rust
use lift_sim::cost::EnergyModel;

let model = EnergyModel::a100();

// Énergie pour 1 seconde de calcul sur 4 GPUs
let joules = model.energy_joules(1000.0, 4);     // Joules
let kwh = model.energy_kwh(1000.0, 4);           // kWh
let carbon = model.carbon_grams(1000.0, 4);       // grammes CO₂

println!("Énergie: {:.2} J", joules);
println!("Carbone: {:.4} g CO₂", carbon);

// Énergie quantique (réfrigération cryogénique)
let q_joules = model.quantum_energy_joules(100.0, 127); // 100 µs, 127 qubits
```

### 8.5 ReactiveBudget — Budget dynamique

```rust
use lift_sim::cost::{Budget, ReactiveBudget};

let budget = Budget {
    max_flops: Some(1_000_000),
    max_memory_bytes: Some(1_000_000),
    max_time_ms: Some(50.0),
    min_fidelity: Some(0.9),
    max_circuit_depth: None,
};
let mut rb = ReactiveBudget::new(budget);

// Consommer des ressources au fur et à mesure
rb.consume(100_000, 50_000, 5.0, 0.99); // flops, mem, time, fidelity
rb.consume(200_000, 80_000, 10.0, 0.98);

// Vérifier le budget restant
rb.check_remaining().unwrap(); // OK si dans les limites

// Rapport d'utilisation
let util = rb.utilisation();
println!("FLOPs utilisés: {:.1}%", util.flop_ratio.unwrap() * 100.0);
println!("Temps utilisé: {:.1}%", util.time_ratio.unwrap() * 100.0);

// Budget restant
println!("FLOPs restants: {:?}", rb.remaining_flops());
println!("Temps restant: {:?} ms", rb.remaining_time_ms());
```

**Combiner avec** : `lift-opt` (arrêter l'optimisation si budget épuisé), `lift-predict` (vérifier que la prédiction respecte le budget).

### 8.6 Module Analysis — Analyse de module

```rust
use lift_sim::{analyze_module, analyze_quantum_ops};

let ctx = load_and_parse("model.lif").unwrap();

// Analyse classique
let report = analyze_module(&ctx);
println!("Total ops: {}", report.num_ops);
println!("Tensor ops: {}", report.num_tensor_ops);
println!("Quantum ops: {}", report.num_quantum_ops);
println!("Hybrid ops: {}", report.num_hybrid_ops);
println!("Total FLOPs: {}", report.total_flops);
println!("Total mémoire: {} bytes", report.total_memory_bytes);
println!("Pic mémoire: {} bytes", report.peak_memory_bytes);

// Analyse quantique
let quantum = analyze_quantum_ops(&ctx);
println!("Qubits: {}", quantum.num_qubits_used);
println!("Portes: {}", quantum.gate_count);
println!("1Q gates: {}", quantum.one_qubit_gates);
println!("2Q gates: {}", quantum.two_qubit_gates);
println!("Mesures: {}", quantum.measurements);
println!("Fidélité estimée: {:.6}", quantum.estimated_fidelity);
```

---

## 9. lift-predict — Prédiction de performance

```rust
use lift_predict::predict_performance;
use lift_sim::{analyze_module, cost::CostModel};

let report = analyze_module(&ctx);
let cost_model = CostModel::h100();
let prediction = predict_performance(&report, &cost_model);

println!("Temps compute: {:.4} ms", prediction.compute_time_ms);
println!("Temps mémoire: {:.4} ms", prediction.memory_time_ms);
println!("Temps prédit: {:.4} ms", prediction.predicted_time_ms);
println!("Intensité arithmétique: {:.2} FLOP/byte", prediction.arithmetic_intensity);
println!("Goulot: {}", prediction.bottleneck); // "compute" ou "memory"
```

**Combiner avec** : `lift-sim` (fournit le rapport d'analyse et le modèle de coût).

---

## 10. lift-import — Importation de modèles

### 10.1 Importation ONNX

```rust
use lift_import::OnnxImporter;

let importer = OnnxImporter::new();
let ctx = importer.import("model.onnx").expect("Importation ONNX échouée");
```

### 10.2 Importation PyTorch FX

```rust
use lift_import::PyTorchFxImporter;

let importer = PyTorchFxImporter::new();
let ctx = importer.import("model_fx.json").expect("Importation FX échouée");
```

### 10.3 Importation OpenQASM 3.0

```rust
use lift_import::OpenQasm3Importer;

let importer = OpenQasm3Importer::new();
let ctx = importer.import("circuit.qasm").expect("Importation QASM échouée");
```

**Combiner avec** : `lift-core::verifier` (vérifier l'IR importée), puis `lift-opt` (optimiser).

---

## 11. lift-export — Exportation vers backends

### 11.1 Export LLVM IR

```rust
use lift_export::LlvmExporter;

let exporter = LlvmExporter::new();
let llvm_ir = exporter.export(&ctx).expect("Export LLVM échoué");
std::fs::write("output.ll", &llvm_ir).unwrap();
```

Produit du LLVM IR compilable avec `clang` ou `llc`.

### 11.2 Export OpenQASM 3.0

```rust
use lift_export::QasmExporter;

let exporter = QasmExporter::new();
let qasm = exporter.export(&ctx).expect("Export QASM échoué");
std::fs::write("output.qasm", &qasm).unwrap();
```

Produit du OpenQASM 3.0 exécutable sur IBM Quantum, Rigetti, etc.

**Combiner avec** : `lift-opt` (optimiser avant export), `lift-quantum::Provider` (transpiler vers le jeu de portes natif).

---

## 12. lift-config — Configuration (.lith)

### 12.1 Format du fichier .lith

```ini
[target]
backend = "cuda"
device = "A100"
precision = "fp16"

[budget]
max_flops = 1000000000000
max_memory_bytes = 80000000000
max_time_ms = 100.0
min_fidelity = 0.99

[optimisation]
level = O2
max_iterations = 10

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = true

[quantum]
topology = "heavy_hex"
num_qubits = 127
shots = 4096
```

### 12.2 Chargement programmatique

```rust
use lift_config::{ConfigParser, LithConfig};

// Depuis un fichier
let source = std::fs::read_to_string("config.lith").unwrap();
let config = ConfigParser::new().parse(&source).unwrap();

// Configuration par défaut
let default = LithConfig::default();
// Backend: llvm, Niveau: O2, Passes: canonicalize, constant-folding, dce, tensor-fusion

// Avec quantique
let hybrid = LithConfig::default().with_quantum("heavy_hex", 127);
```

### 12.3 Niveaux d'optimisation

| Niveau | Passes | Usage |
|--------|--------|-------|
| `O0` | Aucune | Debug, vérification |
| `O1` | Canonicalize, DCE | Compilation rapide |
| `O2` | + ConstantFolding, TensorFusion | **Par défaut** — bon compromis |
| `O3` | + FlashAttention, Quantisation, CSE | Performance maximale |

---

## 13. lift-cli — Interface en ligne de commande

### 13.1 Commandes disponibles

#### 13.1.1 `lift verify` — Vérifier un fichier .lif

```bash
lift verify model.lif
lift verify --verbose model.lif
```

Vérifie les invariants SSA, la linéarité des qubits et le typage.

#### 13.1.2 `lift analyse` — Analyser un programme

```bash
lift analyse model.lif
lift analyse model.lif --format json
```

Produit un rapport : nombre d'ops, FLOPs, mémoire, analyse quantique.

#### 13.1.3 `lift print` — Afficher l'IR

```bash
lift print model.lif
```

Affiche l'IR en format lisible.

#### 13.1.4 `lift optimise` — Optimiser

```bash
lift optimise model.lif
lift optimise model.lif --config config.lith --output optimised.lif
```

Applique les passes d'optimisation configurées.

#### 13.1.5 `lift predict` — Prédire la performance

```bash
lift predict model.lif --device a100
lift predict model.lif --device h100
```

Prédit le temps d'exécution avec le modèle roofline.

#### 13.1.6 `lift export` — Exporter

```bash
lift export model.lif --backend llvm --output model.ll
lift export quantum.lif --backend qasm --output circuit.qasm
```

Exporte vers LLVM IR ou OpenQASM 3.0.

---

## 14. Combinaisons et pipelines complets

### 14.1 Pipeline IA complet (Transformer)

```rust
// 1. Importer un modèle ONNX
let ctx = OnnxImporter::new().import("bert.onnx")?;

// 2. Vérifier
verifier::verify(&ctx)?;

// 3. Analyser
let report = analyze_module(&ctx);

// 4. Optimiser
let mut pm = PassManager::new();
pm.add_pass(Box::new(Canonicalize));
pm.add_pass(Box::new(ConstantFolding));
pm.add_pass(Box::new(DeadCodeElimination));
pm.add_pass(Box::new(CommonSubexprElimination));
pm.add_pass(Box::new(TensorFusion));
pm.add_pass(Box::new(FlashAttentionPass { seq_len_threshold: 512 }));
pm.add_pass(Box::new(QuantisationPass {
    target_dtype: QuantTarget::Int8,
    mode: QuantMode::Dynamic,
}));
pm.add_pass(Box::new(DeadCodeElimination));
pm.run_all(&mut ctx);

// 5. Prédire la performance
let h100 = CostModel::h100();
let pred = predict_performance(&analyze_module(&ctx), &h100);

// 6. Exporter vers LLVM
let llvm = LlvmExporter::new().export(&ctx)?;
std::fs::write("bert_optimised.ll", llvm)?;
```

### 14.2 Pipeline quantique complet (Bell State)

```rust
// 1. Parser le circuit
let ctx = load_lif_file("quantum_bell.lif")?;

// 2. Analyser le bruit
let quantum = analyze_quantum_ops(&ctx);
let sc = QuantumCostModel::superconducting_default();
let fidelity = sc.circuit_fidelity(
    quantum.one_qubit_gates, quantum.two_qubit_gates
);

// 3. QEC si nécessaire
if fidelity < 0.99 {
    let analysis = QecAnalysis::analyse(2, 5,
        QecCode::SurfaceCode { distance: 3 }, 0.001);
    println!("Qubits physiques nécessaires: {}", analysis.physical_qubits);
}

// 4. Optimiser
let mut pm = PassManager::new();
pm.add_pass(Box::new(GateCancellation));
pm.add_pass(Box::new(RotationMerge));
pm.add_pass(Box::new(NoiseAwareSchedule));
pm.add_pass(Box::new(LayoutMapping));
pm.run_all(&mut ctx);

// 5. Exporter vers QASM
let qasm = QasmExporter::new().export(&ctx)?;
std::fs::write("bell_optimised.qasm", qasm)?;
```

### 14.3 Pipeline hybride complet (VQE)

```rust
// 1. Configurer l'encodage
let encoding = EncodingConfig::new(EncodingStrategy::AngleEncoding, 4);

// 2. Configurer le gradient
let grad_config = JointGradientConfig {
    classical_method: GradientMethod::Backprop,
    quantum_method: GradientMethod::ParameterShift,
    num_classical_params: 100,
    num_quantum_params: 20,
};

// 3. Budget réactif pour contrôler les ressources
let budget = Budget {
    max_flops: Some(1_000_000_000),
    max_memory_bytes: Some(8_000_000_000),
    max_time_ms: Some(60_000.0),
    min_fidelity: Some(0.95),
    max_circuit_depth: Some(500),
};
let mut rb = ReactiveBudget::new(budget);

// 4. Boucle d'optimisation VQE
for iteration in 0..100 {
    // Exécuter le circuit quantique
    rb.consume(10_000, 1_000, 0.5, 0.999);
    
    if rb.check_remaining().is_err() {
        println!("Budget épuisé à l'itération {}", iteration);
        break;
    }
    
    let util = rb.utilisation();
    println!("Itération {}: FLOP {:.1}%, Temps {:.1}%",
        iteration,
        util.flop_ratio.unwrap() * 100.0,
        util.time_ratio.unwrap() * 100.0
    );
}

// 5. Estimer l'empreinte carbone
let energy = EnergyModel::a100();
let carbon = energy.carbon_grams(rb.elapsed_ms, 1);
println!("Empreinte carbone: {:.4} g CO₂", carbon);
```

### 14.4 Pipeline CLI complet

```bash
# Vérifier, analyser, optimiser, prédire et exporter en une séquence
lift verify model.lif
lift analyse model.lif --format json > analysis.json
lift optimise model.lif --config production.lith --output optimised.lif
lift predict optimised.lif --device h100
lift export optimised.lif --backend llvm --output model.ll
```

---

## 15. Exemples concrets

### 15.1 MLP (Perceptron multi-couches)

Fichier `tensor_mlp.lif` :

```
#dialect tensor

module @mlp {
    func @forward(%x: tensor<1x784xf32>, %w1: tensor<784x256xf32>,
                  %b1: tensor<256xf32>, %w2: tensor<256x10xf32>,
                  %b2: tensor<10xf32>) -> tensor<1x10xf32> {
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

### 15.2 Self-Attention (Transformer)

Fichier `attention.lif` :

```
#dialect tensor

module @transformer {
    func @self_attention(%q: tensor<1x128x64xf32>, %k: tensor<1x128x64xf32>,
                         %v: tensor<1x128x64xf32>, %norm_w: tensor<64xf32>)
                         -> tensor<1x128x64xf32> {
        %attn = "tensor.attention"(%q, %k, %v) : (...) -> tensor<1x128x64xf32>
        %normed = "tensor.layernorm"(%attn, %norm_w) : (...) -> tensor<1x128x64xf32>
        return %normed
    }
}
```

### 15.3 État de Bell (Quantique)

Fichier `quantum_bell.lif` :

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

### 15.4 Configuration de production

Fichier `production.lith` :

```ini
[target]
backend = "cuda"
device = "H100"
precision = "fp16"

[budget]
max_flops = 1000000000000
max_memory_bytes = 80000000000
max_time_ms = 100.0

[optimisation]
level = O3
max_iterations = 20

[simulation]
shape_propagation = true
flop_counting = true
memory_analysis = true
noise_simulation = true

[quantum]
topology = "heavy_hex"
num_qubits = 127
shots = 4096
```

---

## Résumé des combinaisons par tâche

| Tâche | Crates à combiner |
|-------|-------------------|
| **Entraîner un LLM** | lift-tensor + lift-opt (TensorFusion, FlashAttention) + lift-sim (CostModel) + lift-export (LLVM) |
| **Inférence quantisée** | lift-tensor + lift-opt (QuantisationPass) + lift-predict + lift-export (LLVM) |
| **Circuit quantique** | lift-quantum + lift-opt (GateCancellation, RotationMerge, LayoutMapping) + lift-export (QASM) |
| **VQE / QAOA** | lift-hybrid + lift-quantum + lift-opt (NoiseAwareSchedule) + lift-sim (QuantumCostModel) |
| **Quantum ML** | lift-hybrid (QuantumKernel, encoding) + lift-tensor + lift-quantum |
| **Analyse de coût** | lift-sim (CostModel, EnergyModel) + lift-predict |
| **QEC planning** | lift-quantum (qec, topology) + lift-sim (QuantumCostModel) |
| **Import/Optimise/Export** | lift-import + lift-opt + lift-export |
| **Stable Diffusion** | lift-tensor (UNet ops) + lift-opt (TensorFusion) + lift-export |
| **GNN** | lift-tensor (GNNMessagePassing, GNNGlobalPooling) + lift-opt + lift-export |
