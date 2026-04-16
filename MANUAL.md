# LIFT — Manuel Complet des Problemes et Solutions

> **LIFT** (Layered Intermediate Framework for Tensors & Qubits) est un compilateur
> unifie qui transforme des descriptions de haut niveau en code executable optimise
> pour GPU, CPU et processeurs quantiques.

---

## Table des matieres

- [Partie I — Architecture de LIFT](#partie-i--architecture-de-lift)
- [Partie II — Problemes d'IA classique](#partie-ii--problemes-dia-classique)
- [Partie III — Problemes quantiques](#partie-iii--problemes-quantiques)
- [Partie IV — Problemes hybrides IA + Quantique](#partie-iv--problemes-hybrides-ia--quantique)
- [Partie V — Capacites transversales](#partie-v--capacites-transversales)
- [Partie VI — Reference rapide](#partie-vi--reference-rapide)

---

# Partie I — Architecture de LIFT

## 1.1 Les 12 crates du framework

| Crate | Role |
|-------|------|
| **lift-core** | IR SSA, types, verifieur, imprimante, registre de dialectes |
| **lift-ast** | Lexer, parser, constructeur d'IR depuis `.lif` |
| **lift-tensor** | 90+ operations IA/ML, inference de forme, FLOPs |
| **lift-quantum** | 50+ portes, bruit, topologie, QEC, Kraus |
| **lift-hybrid** | Encodage, gradients, VQC/VQE/QAOA, transfert GPU-QPU |
| **lift-opt** | 11 passes d'optimisation |
| **lift-sim** | Analyse statique, cout GPU/quantique, energie, budget |
| **lift-predict** | Prediction roofline GPU et fidelite/shots quantique |
| **lift-export** | Export LLVM IR et OpenQASM 3.0 |
| **lift-config** | Parsing des fichiers `.lith` |
| **lift-import** | Import depuis formats externes |
| **lift-cli** | Interface ligne de commande |

## 1.2 Le pipeline en 6 etapes

```
.lif → [1. PARSE & VERIFY] → [2. ANALYSE] → [3. OPTIMISE] → [4. PREDICT] → [5. EXPORT] → [6. FEEDBACK]
```

1. **Parse & Verify** : Lexer → Parser → SSA → verification types + linearite qubits
2. **Analyse** : FLOPs, memoire, profondeur circuit, fidelite estimee
3. **Optimise** : 11 passes classiques + quantiques + hybrides
4. **Predict** : Roofline GPU + modele de bruit QPU
5. **Export** : LLVM IR (GPU/CPU) + OpenQASM 3.0 (QPU)
6. **Feedback** : Budget, energie, CO2, comparaison predictions/reel

## 1.3 Les trois dialectes

| Dialecte | Prefixe | Domaine | Exemple |
|----------|---------|---------|---------|
| **tensor** | `tensor.` | IA / ML classique | `tensor.conv2d`, `tensor.attention` |
| **quantum** | `quantum.` | Calcul quantique | `quantum.h`, `quantum.cx` |
| **hybrid** | `hybrid.` | Liaison classique-quantique | `hybrid.encode`, `hybrid.vqc_layer` |

---

# Partie II — Problemes d'IA classique

## 2.1 Vision par ordinateur (CNN)

### Probleme

Classifier, detecter ou segmenter des images : radiographies medicales, photos satellite, controle qualite industriel, vehicules autonomes.

### Solution LIFT

```
func @image_classifier(%img: tensor<1x3x224x224xf32>,
                        %w1: tensor<64x3x7x7xf32>,
                        %w2: tensor<128x64x3x3xf32>,
                        %wfc: tensor<128x10xf32>,
                        %b: tensor<10xf32>) -> tensor<1x10xf32> {
  %c1 = tensor.conv2d %img, %w1
  %r1 = tensor.relu %c1
  %p1 = tensor.maxpool2d %r1
  %c2 = tensor.conv2d %p1, %w2
  %r2 = tensor.relu %c2
  %g  = tensor.global_avgpool %r2
  %fc = tensor.matmul %g, %wfc
  %out = tensor.add %fc, %b
  return %out
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Parse** | Construit le graphe SSA, verifie que `conv2d` recoit des tenseurs 4D compatibles |
| **Analyse** | Calcule les FLOPs exacts (`2 * N * C * H * W * K * K * F`), la VRAM, le pic memoire |
| **Optimise** | **Tensor fusion** : `conv2d + batchnorm + relu → fused_conv_bn_relu` (-30% memoire). **DCE** : elimine les poids inutilises |
| **Predict** | Roofline A100/H100 : determine si le CNN est compute-bound ou memory-bound |
| **Export** | LLVM IR avec appels cuDNN (convolutions) et cuBLAS (matmuls) |
| **Budget** | Verifie que la VRAM ne depasse pas 80 Go (A100) |

### Operations utilisees

`Conv2D`, `Conv1D`, `Conv3D`, `DepthwiseConv2D`, `DilatedConv2D`, `ConvTranspose2D`, `MaxPool2D`, `AvgPool2D`, `AdaptiveAvgPool2D`, `GlobalAvgPool`, `ReLU`, `BatchNorm`, `MatMul`, `Linear`, `Softmax`.

### Variantes

- **Classification** : CNN → softmax → etiquette
- **Detection d'objets** : backbone → region proposals → bounding boxes
- **Segmentation** : U-Net (`UNetDownBlock`, `UNetUpBlock`)
- **Super-resolution** : `ConvTranspose2D` pour l'upsampling

---

## 2.2 Traitement du langage naturel (Transformers)

### Probleme

Traduire, resumer, generer ou comprendre du texte : chatbots, moteurs de recherche, analyse de sentiment, generation de code.

### Solution LIFT

```
func @transformer_block(%x: tensor<1x512x768xf32>,
                         %wq: tensor<768x768xf32>,
                         %wk: tensor<768x768xf32>,
                         %wv: tensor<768x768xf32>) -> tensor<1x512x768xf32> {
  %q = tensor.matmul %x, %wq
  %k = tensor.matmul %x, %wk
  %v = tensor.matmul %x, %wv
  %attn = tensor.multi_head_attention %q, %k, %v
  %n1 = tensor.layernorm %attn
  %ff = tensor.linear %n1, %wff
  %act = tensor.gelu %ff
  %out = tensor.layernorm %act
  return %out
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Analyse** | Calcule le cout O(n^2) de l'attention pour la longueur de sequence |
| **Optimise** | **FlashAttention** : remplace `attention` par `flash_attention` (O(n) memoire). **CSE** : elimine les calculs redondants. **FusedAttentionLayerNorm** : un seul kernel |
| **Predict** | Inference LLM = memory-bound. Entrainement = compute-bound. LIFT identifie le regime |
| **Export** | Kernels fusionnes pour l'attention |

### 8 variantes d'attention

| Operation | Usage | Memoire |
|-----------|-------|---------|
| `Attention` | Standard | O(n^2) |
| `MultiHeadAttention` | GPT, BERT | O(n^2) |
| `MultiQueryAttention` | PaLM, Falcon (inference rapide) | O(n) par tete |
| `GroupedQueryAttention` | LLaMA 2 (compromis) | O(n * G/H) |
| `FlashAttention` | Entrainement et inference | O(n) |
| `SlidingWindowAttention` | Mistral (sequences longues) | O(n * w) |
| `CrossAttention` | Traduction, multimodal | O(n * m) |
| `PagedAttention` | Serving LLM (vLLM) | O(n) par page |

### Cas d'usage

- **Chatbot / LLM** : N blocs transformer + `PagedAttention` pour le serving
- **Traduction** : Encoder (`MultiHeadAttention`) + Decoder (`CrossAttention`)
- **Analyse de sentiment** : BERT + classification
- **Generation de code** : GPT + `SlidingWindowAttention` pour contexte long

---

## 2.3 Systemes de recommandation

### Probleme

Recommander produits, films, musique a des utilisateurs selon leur historique.

### Solution LIFT

```
func @recommender(%user_id: tensor<1xi64>,
                   %item_ids: tensor<100xi64>,
                   %user_emb: tensor<100000x64xf32>,
                   %item_emb: tensor<500000x64xf32>) -> tensor<100xf32> {
  %u = tensor.embedding %user_id, %user_emb
  %items = tensor.embedding %item_ids, %item_emb
  %cat = tensor.concat %u, %items
  %h = tensor.linear %cat, %w1
  %a = tensor.relu %h
  %scores = tensor.linear %a, %w2
  %probs = tensor.sigmoid %scores
  return %probs
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Analyse** | Taille des tables d'embedding (100k * 64 * 4 = 25 MiB utilisateurs) |
| **Optimise** | **SparseEmbedding** pour grandes tables. **FusedLinearReLU** pour les couches denses |
| **Predict** | Systemes de recommandation = memory-bound (gros embeddings) |
| **Budget** | Verifie que les tables tiennent en VRAM pour le serving temps reel |

### Operations cles

`Embedding`, `SparseEmbedding`, `Linear`, `ReLU`, `Sigmoid`, `Concat`, `Gather`, `Scatter`, `TopK`.

---

## 2.4 Series temporelles (LSTM / GRU / RNN)

### Probleme

Predire des valeurs futures a partir de sequences passees : prevision meteo, cours boursiers, maintenance predictive, ECG.

### Solution LIFT

```
func @forecast(%seq: tensor<1x100x16xf32>,
               %h0: tensor<1x64xf32>,
               %c0: tensor<1x64xf32>) -> tensor<1x1xf32> {
  %h, %c = tensor.lstm_cell %seq, %h0, %c0, %w
  %pred = tensor.matmul %h, %wfc
  return %pred
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Analyse** | FLOPs par pas de temps = 4 portes LSTM * matmul. 100 pas = 100x le cout |
| **Optimise** | **Constant folding** : biais constants. **Canonicalize** : expressions redondantes |
| **Predict** | Entrainement = compute-bound. Inference = memory-bound (etats caches) |

### Cellules recurrentes

| Operation | Description |
|-----------|-------------|
| `LSTMCell` | Long Short-Term Memory (4 portes) |
| `GRUCell` | Gated Recurrent Unit (3 portes) |
| `RNNCell` | RNN vanilla (tanh) |

---

## 2.5 Reseaux de neurones sur graphes (GNN)

### Probleme

Traiter des donnees en graphe : molecules, reseaux sociaux, systemes de transport, proteines.

### Solution LIFT

```
func @gnn_classify(%nodes: tensor<100x16xf32>,
                    %adj: tensor<100x100xf32>) -> tensor<1x2xf32> {
  %msg = tensor.gnn_message_passing %nodes, %adj, %w
  %act = tensor.relu %msg
  %graph = tensor.gnn_global_pooling %act
  %out = tensor.matmul %graph, %wfc
  %prob = tensor.softmax %out
  return %prob
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Analyse** | FLOPs du message passing : `N * N * D` (dense) ou `E * D` (sparse) |
| **Optimise** | **SparseMatMul** si adjacence < 10% non-zero |
| **Predict** | GNN = memory-bound (lookups irreguliers dans le graphe) |

### Types d'agregation GNN

Sum, Mean, Max, Min (via `AggregationType`).

### Cas d'usage

- **Proprietes moleculaires** : solubilite, toxicite
- **Decouverte de medicaments** : interaction proteine-ligand
- **Reseaux sociaux** : detection de communautes
- **Transport** : prediction de trafic

---

## 2.6 Modeles generatifs et diffusion

### Probleme

Generer des images, de l'audio ou de la video a partir d'un bruit ou d'une description textuelle : Stable Diffusion, DALL-E, synthese vocale.

### Solution LIFT

```
func @diffusion_step(%noisy: tensor<1x4x64x64xf32>,
                      %timestep: tensor<1xi64>,
                      %context: tensor<1x77x768xf32>) -> tensor<1x4x64x64xf32> {
  %t_emb = tensor.timestep_embedding %timestep
  %down = tensor.unet_down_block %noisy, %t_emb, %context
  %up = tensor.unet_up_block %down, %t_emb, %context
  return %up
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Analyse** | Cout total = N_etapes_debruitage * FLOPs_par_etape (20-50 passes U-Net) |
| **Optimise** | **FlashAttention** pour les blocs cross-attention. **Tensor fusion** pour les residuels. **Checkpoint** pour economiser la memoire |
| **Predict** | 1 pas SD 1.5 sur A100 ~ 50 ms. LIFT estime et identifie l'attention comme goulot |
| **Energie** | 50 pas * 50 ms = 2.5 s / image. LIFT calcule la conso pour 1M images/jour |

### Operations cles

`UNetDownBlock`, `UNetUpBlock`, `TimestepEmbedding`, `CrossAttention`, `GroupNorm`, `SiLU`, `Conv2D`, `ConvTranspose2D`.

---

## 2.7 Calcul scientifique (FFT, SVD, systemes lineaires)

### Probleme

Traitement du signal (FFT), decomposition de matrices (SVD, valeurs propres), resolution de systemes lineaires (Ax = b).

### Solution LIFT

```
func @signal_analysis(%signal: tensor<1x8192xf32>) -> tensor<1x8192xf32> {
  %freq = tensor.fft %signal
  %filtered = tensor.mul %freq, %mask
  %result = tensor.ifft %filtered
  return %result
}

func @solve_system(%A: tensor<1000x1000xf32>,
                    %b: tensor<1000xf32>) -> tensor<1000xf32> {
  %x = tensor.solve %A, %b
  return %x
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Analyse** | FFT : O(n log n). SVD : O(min(m,n)^2 * max(m,n)). Couts exacts |
| **Optimise** | **Constant folding** : pre-calcule les masques constants |
| **Export** | LLVM IR avec appels cuFFT (GPU) ou FFTW (CPU) |

### Operations

`FFT`, `IFFT`, `SVD`, `Eig`, `Solve`, `Einsum`, `Cumsum`, `Sort`, `TopK`, `Where`, `Clamp`.

### Cas d'usage

- **Traitement du signal** : filtrage, spectrogramme, compression audio
- **Analyse de donnees** : PCA via SVD, moindres carres
- **Simulation physique** : equations differentielles
- **Imagerie** : reconstruction Fourier (IRM, CT-scan)

---

## 2.8 IA embarquee et quantification

### Probleme

Deployer des modeles sur dispositifs a ressources limitees : smartphones, IoT, drones, cameras embarquees.

### Solution LIFT

```
func @quantized_model(%img: tensor<1x3x224x224xf32>,
                       %w: tensor<64x3x3x3xf32>) -> tensor<1x10xf32> {
  %w_q = tensor.quantize_int4 %w        // poids en INT4 (8x moins de memoire)
  %conv = tensor.conv2d %img, %w_q
  %out = tensor.dequantize_int4 %conv
  return %out
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Optimise** | **QuantisationPass** : annote les ops compatibles INT8/INT4/FP8, insere `quantize`/`dequantize` automatiquement |
| **Analyse** | Recalcule FLOPs et memoire apres quantification. INT4 divise la memoire par 8 |
| **Predict** | Estime le speedup sur le materiel cible |
| **Budget** | Verifie que le modele tient sur le dispositif (ex : 4 Go smartphone) |

### Niveaux de quantification

| Operation | Precision | Reduction memoire | Usage |
|-----------|-----------|-------------------|-------|
| `Quantize` / `Dequantize` | INT8 | 4x | Inference serveur |
| `QuantizeInt4` / `DequantizeInt4` | INT4 | 8x | Smartphones, edge |
| `QuantizeFp8` / `DequantizeFp8` | FP8 (E4M3/E5M2) | 2x | Entrainement H100 |

---

## 2.9 Mixture of Experts (MoE)

### Probleme

Construire des modeles massivement parametres (100B+) ou seule une fraction des parametres est activee par requete : Mixtral, Switch Transformer.

### Solution LIFT

```
func @moe_layer(%x: tensor<1x512x768xf32>,
                 %gate: tensor<768x8xf32>) -> tensor<1x512x768xf32> {
  %tokens, %weights = tensor.moe_dispatch %x, %gate   // routeur : top-2 / 8 experts
  %expert_out = tensor.linear %tokens, %experts
  %combined = tensor.moe_combine %expert_out, %weights
  return %combined
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Analyse** | Seuls 2/8 experts actifs → FLOPs effectifs = 25% du total |
| **Optimise** | **Tensor fusion** : fusionne dispatch + expert + combine si meme GPU |
| **Predict** | MoE = memory-bound (gros parametres) malgre des FLOPs moderes |
| **Parallelisme** | Sharding des experts via `ParallelSplit` + `ParallelAllReduce` |

---

## 2.10 Entrainement distribue et parallelisme

### Probleme

Entrainer des modeles trop grands pour un seul GPU sur des clusters multi-GPU.

### Solution LIFT

```
// Parallelisme de donnees
%grad_local = tensor.grad_matmul %x, %w
%grad_global = tensor.parallel_all_reduce %grad_local

// Parallelisme de pipeline
tensor.pipeline_send %activations, %device_1
%received = tensor.pipeline_receive %device_1

// Parallelisme de tenseur
%shards = tensor.parallel_split %weight, %num_gpus
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Analyse** | Volume de communication AllReduce : `2*(N-1)/N * taille_grad * N_GPU` |
| **Predict** | Ratio communication / calcul. Si > 30%, parallelisme inefficace |
| **Energie** | 8x A100, 24h = 92.4 kWh, 37 kg CO2 |

### Operations de parallelisme

`ParallelSplit`, `ParallelAllReduce`, `PipelineSend`, `PipelineReceive`, `Checkpoint`, `Offload`, `GradAccumulate`.

### Operations de gradient

`GradMatMul`, `GradReLU`, `GradSoftmax`, `GradLayerNorm`, `GradAttention`, `GradConv2D`, `GradLinear`, `GradGeLU`.

---

# Partie III — Problemes quantiques

## 3.1 Simulation quantique (Hamiltonien)

### Probleme

Simuler l'evolution d'un systeme quantique sous l'action d'un Hamiltonien : physique des materiaux, chimie quantique, physique des particules.

### Solution LIFT

```
func @hamiltonian_sim(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  // Trotterisation : e^{-iHt} approche par produit de portes
  %q0a = quantum.rx %q0       // terme X du Hamiltonien
  %q1a = quantum.rx %q1
  // Interaction ZZ
  %q0b, %q1b = quantum.cx %q0a, %q1a
  %q1c = quantum.rz %q1b      // e^{-i*J*t*ZZ}
  %q0c, %q1d = quantum.cx %q0b, %q1c
  %q2a = quantum.rx %q2
  %q3a = quantum.rx %q3
  %q2b, %q3b = quantum.cx %q2a, %q3a
  %q3c = quantum.rz %q3b
  %q2c, %q3d = quantum.cx %q2b, %q3c
  return %q0c, %q1d, %q2c, %q3d
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Verify** | Verifie la linearite : chaque qubit consomme exactement une fois (no-cloning) |
| **Analyse** | Compte portes 1Q/2Q, estime fidelite et profondeur |
| **Optimise** | **Rotation merge** : `Rz(a)*Rz(b) → Rz(a+b)`. **Gate cancellation** : `X*X → I`, `H*H → I` |
| **Predict** | Nombre de shots pour precision donnee sur l'observable |
| **Topologie** | Adapte le circuit au QPU (grid, heavy-hex) avec insertion de SWAP |

### Portes quantiques disponibles

**1-qubit standard** : H, X, Y, Z, S, S†, T, T†, SX

**1-qubit parametriques** : RX, RY, RZ, P, U1, U2, U3

**2-qubit** : CX (CNOT), CZ, CY, SWAP, iSWAP, ECR, RZX, XX, YY, ZZ, CP, CPhase, XY

**3-qubit** : CCX (Toffoli), CSWAP (Fredkin)

**Multi-controle** : MCX, MCZ

**Controle** : Measure, MeasureAll, Reset, Barrier, Init, Delay, VirtualRZ, IfElse

---

## 3.2 Correction d'erreurs quantiques (QEC)

### Probleme

Les qubits physiques sont bruites. Pour executer des algorithmes fiables, il faut encoder l'information dans des qubits logiques proteges par des codes correcteurs.

### Solution LIFT

LIFT integre un module QEC complet (`lift-quantum::qec`) :

```rust
// Analyse QEC : 10 qubits logiques, profondeur 100, surface code d=7
let analysis = QecAnalysis::analyse(
    10,                                    // qubits logiques
    100,                                   // profondeur circuit
    QecCode::SurfaceCode { distance: 7 },  // code surface
    0.001,                                 // taux d'erreur physique 0.1%
);
// Resultat : 490 qubits physiques, erreur logique ~10^-8
```

### Codes QEC supportes

| Code | Qubits phys. / logique | Distance | Seuil erreur | Usage |
|------|------------------------|----------|--------------|-------|
| **Surface Code** | d^2 | d | ~1% | Standard NISQ/FTQC |
| **Steane Code** | 7 | 3 | ~0.5% | Petits circuits |
| **Shor Code** | 9 | 3 | ~0.3% | Pedagogique |
| **Repetition Code** | d | d | ~3% | Bit-flip uniquement |
| **LDPC Code** | n/k | ~sqrt(n) | ~0.8% | Overhead reduit |

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Analyse** | Calcule qubits physiques, taux d'erreur logique, profondeur syndrome |
| **Suggest** | `suggest_distance()` recommande la distance minimale pour un taux d'erreur cible |
| **Budget** | Verifie que le QPU a assez de qubits physiques (ex : IBM Eagle = 127) |

---

## 3.3 Circuits variationnels (VQE)

### Probleme

Trouver l'etat fondamental d'un Hamiltonien moleculaire : energie de liaison, geometrie d'equilibre, proprietes electroniques.

### Solution LIFT

```
func @vqe(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  // Ansatz hardware-efficient
  %q0a = quantum.ry %q0    // theta_0
  %q1a = quantum.ry %q1    // theta_1
  %q2a = quantum.ry %q2    // theta_2
  %q3a = quantum.ry %q3    // theta_3
  // Entanglement
  %q0b, %q1b = quantum.cx %q0a, %q1a
  %q1c, %q2b = quantum.cx %q1b, %q2a
  %q2c, %q3b = quantum.cx %q2b, %q3a
  // Couche parametrisee
  %q0c = quantum.rz %q0b
  %q1d = quantum.rz %q1c
  %q2d = quantum.rz %q2c
  %q3c = quantum.rz %q3b
  return %q0c, %q1d, %q2d, %q3c
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Hybrid** | `hybrid.vqe_ansatz` gere la boucle classique-quantique |
| **Gradient** | **Parameter shift** : gradients exacts en 2N evaluations. **Adjoint** : 1 evaluation (simulateurs) |
| **Noise** | Fidelite porte par porte. Si < seuil, recommande un circuit plus court ou meilleur QPU |
| **Budget reactif** | Arrete l'optimisation VQE si le budget temps est ecoule |

### Types d'ansatz

| Type | Description | Usage |
|------|-------------|-------|
| `HardwareEfficient` | RY + CX alternatifs | NISQ, circuits courts |
| `StronglyEntangling` | Entanglement total | Expressivite maximale |
| `TwoLocal` | Couches locales + entanglement | Compromis |
| `UCCSD` | Unitary Coupled Cluster | Chimie quantique |
| `Custom` | Defini par l'utilisateur | Recherche |

---

## 3.4 Optimisation combinatoire (QAOA)

### Probleme

Resoudre des problemes NP-difficiles : coupe maximale (MaxCut), voyageur de commerce, coloration de graphe, allocation de ressources.

### Solution LIFT

```
func @qaoa_maxcut(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  // Superposition initiale
  %q0a = quantum.h %q0
  %q1a = quantum.h %q1
  %q2a = quantum.h %q2
  %q3a = quantum.h %q3
  // Couche probleme (ZZ interactions = aretes du graphe)
  %q0b, %q1b = quantum.cx %q0a, %q1a
  %q1c = quantum.rz %q1b          // gamma * poids arete (0,1)
  %q0c, %q1d = quantum.cx %q0b, %q1c
  %q2b, %q3b = quantum.cx %q2a, %q3a
  %q3c = quantum.rz %q3b          // gamma * poids arete (2,3)
  %q2c, %q3d = quantum.cx %q2b, %q3c
  // Couche mixer
  %q0d = quantum.rx %q0c          // beta
  %q1e = quantum.rx %q1d
  %q2d = quantum.rx %q2c
  %q3e = quantum.rx %q3d
  return %q0d, %q1e, %q2d, %q3e
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Hybrid** | `hybrid.qaoa_layer` empile P couches, optimise 2P parametres (gamma, beta) |
| **Topologie** | Insere des SWAP pour adapter le graphe probleme a la topologie QPU |
| **Noise-aware** | Reordonne les CX sur les paires de qubits ayant la meilleure fidelite |
| **Layout mapping** | SABRE mappe qubits logiques → physiques en minimisant les SWAP |
| **Predict** | Shots necessaires pour distinguer la meilleure solution |

---

## 3.5 Etats intriques et protocoles quantiques

### Probleme

Preparer des etats intriques pour la communication quantique, teleportation, distribution de cles (QKD), benchmarking de processeurs.

### Solution LIFT

```
// Etat de Bell |Phi+> = (|00> + |11>) / sqrt(2)
func @bell_state(%q0: qubit, %q1: qubit) -> (qubit, qubit) {
  %q0a = quantum.h %q0
  %q0b, %q1a = quantum.cx %q0a, %q1
  return %q0b, %q1a
}

// Etat GHZ a 4 qubits
func @ghz_state(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  %q0a = quantum.h %q0
  %q0b, %q1a = quantum.cx %q0a, %q1
  %q1b, %q2a = quantum.cx %q1a, %q2
  %q2b, %q3a = quantum.cx %q2a, %q3
  return %q0b, %q1b, %q2b, %q3a
}
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Verify** | Linearite stricte : chaque qubit consomme exactement une fois |
| **Analyse** | Bell = 2 portes, fidelite ~0.99. GHZ(4) = 4 portes, fidelite ~0.97 |
| **Export** | OpenQASM 3.0 executable sur IBM Quantum ou Amazon Braket |
| **QEC** | Calcule le code correcteur necessaire pour proteger les qubits EPR |

---

# Partie IV — Problemes hybrides IA + Quantique

## 4.1 Imagerie medicale hybride (CNN + VQC)

### Probleme

Classifier des radiographies thoraciques (pneumonie / normal) en combinant un CNN pour l'extraction de features et un circuit quantique variationnel pour la decision, potentiellement plus performant sur de petits jeux de donnees.

### Solution LIFT

```
// 1. CNN encoder (GPU) : image → vecteur 4D
func @cnn_encoder(%img: tensor<1x1x128x128xf32>) -> tensor<1x4xf32> {
  %c1 = tensor.conv2d %img, %w1
  %r1 = tensor.relu %c1
  %p1 = tensor.maxpool2d %r1
  %c2 = tensor.conv2d %p1, %w2
  %r2 = tensor.relu %c2
  %g  = tensor.global_avgpool %r2
  %features = tensor.linear %g, %wfc
  return %features
}

// 2. Transfert + encodage
%encoded = hybrid.encode %features           // angle encoding
%qubits = hybrid.gpu_to_qpu %encoded         // GPU → QPU

// 3. VQC classifier (QPU) : 4 qubits
func @vqc(%q0: qubit, %q1: qubit, %q2: qubit, %q3: qubit)
    -> (qubit, qubit, qubit, qubit) {
  %q0a = quantum.ry %q0
  %q1a = quantum.ry %q1
  %q2a = quantum.ry %q2
  %q3a = quantum.ry %q3
  %q0b, %q1b = quantum.cx %q0a, %q1a
  %q2b, %q3b = quantum.cx %q2a, %q3a
  %q0c = quantum.rz %q0b
  %q1c = quantum.rz %q1b
  %q2c = quantum.rz %q2b
  %q3c = quantum.rz %q3b
  return %q0c, %q1c, %q2c, %q3c
}

// 4. Mesure + post-traitement
%results = hybrid.measure_expectation %qubits
%class = hybrid.qpu_to_gpu %results
%probs = tensor.softmax %class
```

### Comment LIFT l'exploite (pipeline complet)

| Composant | Parse | Analyse | Optimise | Predict | Export |
|-----------|-------|---------|----------|---------|--------|
| **CNN** | Types tensoriels 4D | ~10 MFLOP, ~500 KiB | Fusion conv+relu, DCE | A100 : 0.001 ms | LLVM IR |
| **Interface** | Transfert GPU-QPU | Cout encoding O(4) | Fusion encode+linear | Latence ~1 ms | Script |
| **VQC** | Linearite qubit | 10 portes, fidelite ~0.97 | Rotation merge | IBM : 7.9 ms | OpenQASM |

### Strategies d'encodage

| Strategie | Qubits | Profondeur | Usage |
|-----------|--------|------------|-------|
| `AngleEncoding` | N | 1 | Vecteurs petits (< 20 features) |
| `AmplitudeEncoding` | log2(N) | N | Vecteurs grands (compression) |
| `BasisEncoding` | N | 1 | Donnees binaires |
| `IQPEncoding` | N | 2N | Expressivite elevee |
| `HamiltonianEncoding` | N | N | Physique-inspire |
| `KernelEncoding` | N | 3N | Quantum kernel methods |

---

## 4.2 Decouverte de medicaments (GNN + VQE)

### Probleme

Trouver de nouvelles molecules therapeutiques en combinant un GNN pour le screening rapide et un VQE pour le calcul precis de l'energie des meilleurs candidats.

### Solution LIFT

```
// Etape 1 : GNN screening (GPU) — filtre 10 000 molecules
func @molecule_screen(%atoms: tensor<50x16xf32>,
                       %bonds: tensor<50x50xf32>) -> tensor<1x1xf32> {
  %msg = tensor.gnn_message_passing %atoms, %bonds, %w
  %pool = tensor.gnn_global_pooling %msg
  %score = tensor.linear %pool, %wfc
  return %score
}

// Etape 2 : VQE energie (QPU) — top-K molecules
%circuit = hybrid.vqe_ansatz %qubits          // UCCSD ansatz
%energy = hybrid.measure_expectation %circuit  // energie de liaison
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **GNN** | Filtre 10 000 molecules en secondes. Optimise les lookups sparse |
| **VQE** | Energie des 10 meilleurs candidats. Budget reactif arrete si convergence |
| **Gradient** | Adjoint differentiation (1 evaluation = le plus efficace) |
| **Noise** | Fidelite suffisante pour precision chimique (1.6 mHartree) ? |
| **QEC** | Si erreurs trop grandes, recommande un code correcteur |

---

## 4.3 Finance quantique (QAOA + ML classique)

### Probleme

Optimiser un portefeuille d'investissement : modele classique pour predire les rendements + QAOA pour la selection discrete d'actifs sous contraintes.

### Solution LIFT

```
// Etape 1 : Prediction rendements (LSTM sur GPU)
func @returns(%prices: tensor<1x252x50xf32>) -> tensor<1x50xf32> {
  %h, %c = tensor.lstm_cell %prices, %h0, %c0, %w
  %returns = tensor.linear %h, %wfc
  return %returns
}

// Etape 2 : Selection portefeuille (QAOA sur QPU)
// Chaque qubit = decision d'inclure un actif
// Hamiltonien = max rendement - min risque - contrainte budget
%layer = hybrid.qaoa_layer %qubits, %gamma, %beta
%samples = hybrid.measure_samples %qubits, 4096
%best = tensor.topk %samples, 10
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **LSTM** | Predit rendements de 50 actifs. Identifie goulot memoire (etats caches) |
| **QAOA** | 50 qubits = 50 actifs, P=3 couches. Adapte a heavy-hex (127 qubits) |
| **Budget** | Total < 10 s (contrainte trading). Repartition : 1 ms LSTM + 9 s QAOA |
| **Energie** | Cout energetique de 4096 shots sur QPU supraconducteur |

---

## 4.4 Apprentissage automatique quantique (QML)

### Probleme

Utiliser des noyaux quantiques (quantum kernels) pour des taches de classification ou regression ou les espaces de features quantiques offrent un avantage.

### Solution LIFT

```
// Quantum Kernel : calcule la similarite dans l'espace de Hilbert
%encoded_x = hybrid.encode %x, "iqp"          // IQP encoding
%encoded_y = hybrid.encode %y, "iqp"
%kernel = hybrid.quantum_kernel %encoded_x, %encoded_y
%similarity = hybrid.measure_expectation %kernel

// Classification avec kernel quantique
%svm_result = tensor.matmul %kernel_matrix, %alpha
%class = tensor.sigmoid %svm_result
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Encoding** | IQP encoding pour expressivite maximale (profondeur 2N) |
| **Kernel** | `hybrid.quantum_kernel` calcule les elements de la matrice de Gram |
| **Predict** | Nombre de shots pour estimer chaque element avec precision epsilon |
| **Analyse** | Cout total = N^2 * shots * circuit_time (N = taille du dataset) |

### Feature Maps disponibles

| Feature Map | Description | Usage |
|-------------|-------------|-------|
| `ZZFeatureMap` | Interactions ZZ entre features | Classification standard |
| `PauliFeatureMap` | Produits de Pauli | Expressivite elevee |
| `AngleEncoding` | Rotation simple | Donnees continues |
| `AmplitudeEncoding` | Amplitudes d'etat | Compression de donnees |

---

## 4.5 Science des materiaux (VQE + Tenseur)

### Probleme

Predire les proprietes de nouveaux materiaux (supraconducteurs, batteries, catalyseurs) en combinant des simulations quantiques precises avec des modeles ML pour le screening.

### Solution LIFT

```
// ML rapide : prediction de proprietes (GPU)
func @material_screen(%composition: tensor<1x32xf32>) -> tensor<1x5xf32> {
  %h1 = tensor.linear %composition, %w1
  %a1 = tensor.gelu %h1
  %h2 = tensor.linear %a1, %w2
  %props = tensor.sigmoid %h2
  return %props                              // 5 proprietes predites
}

// VQE precis pour les candidats prometteurs (QPU)
%ansatz = hybrid.vqe_ansatz %qubits          // UCCSD pour chimie
%energy = hybrid.measure_expectation %ansatz  // energie du materiau
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Screening** | MLP rapide sur GPU, filtre des milliers de compositions |
| **VQE** | Calcul quantique precis des meilleurs candidats |
| **Co-execution** | `hybrid.co_execute` pour paralleliser screening et VQE |
| **Sync** | `SyncPolicy::Pipeline` pour traiter les candidats en flux |

---

## 4.6 Detection d'anomalies hybride

### Probleme

Detecter des anomalies dans des donnees complexes (fraude financiere, cyberattaque, defauts industriels) en combinant un autoencodeur classique et un circuit quantique pour la detection dans un espace de features quantique.

### Solution LIFT

```
// Autoencodeur classique (GPU) : compression des features
func @encoder(%data: tensor<1x100xf32>) -> tensor<1x8xf32> {
  %h1 = tensor.linear %data, %w1
  %a1 = tensor.relu %h1
  %latent = tensor.linear %a1, %w2
  return %latent
}

// Detection quantique : mesure de distance dans l'espace de Hilbert
%encoded = hybrid.encode %latent, "angle"
%circuit = hybrid.vqc_layer %encoded
%anomaly_score = hybrid.measure_expectation %circuit
```

### Comment LIFT l'exploite

| Etape | Action de LIFT |
|-------|----------------|
| **Encoder** | Compresse 100 features → 8 dimensions. Analyse FLOPs et memoire |
| **VQC** | Mesure la distance dans l'espace quantique. Les anomalies ont un score different |
| **Co-execute** | `hybrid.co_execute` + `SyncPolicy::Asynchronous` pour le temps reel |
| **Budget** | Latence < 10 ms pour detection en temps reel |

---

# Partie V — Capacites transversales

Ces capacites s'appliquent a **tous** les problemes decrits dans les parties II, III et IV.

## 5.1 Pipeline d'optimisation (11 passes)

LIFT dispose de 11 passes d'optimisation organisees en trois familles :

### Passes classiques (IA)

| Passe | Description | Gain |
|-------|-------------|------|
| **Canonicalize** | Simplifie `x+0 → x`, `reshape(reshape(x)) → reshape(x)`, `mul(x,1) → x` | Reduction du graphe |
| **Constant Folding** | Evalue les expressions constantes a la compilation | Moins d'ops au runtime |
| **Dead Code Elimination** | Supprime les operations dont les resultats ne sont jamais utilises | -10 a -30% d'ops |
| **Tensor Fusion** | Fusionne `conv2d + batchnorm + relu` en un seul kernel | -30% memoire, -20% latence |
| **Flash Attention** | Remplace l'attention standard O(n^2) par FlashAttention O(n) | -90% memoire pour seq longues |
| **Common Subexpr Elim** | Detecte et elimine les calculs redondants | Variable |
| **Quantisation Pass** | Annote les ops compatibles INT8/INT4/FP8 | 2-8x moins de memoire |

### Passes quantiques

| Passe | Description | Gain |
|-------|-------------|------|
| **Gate Cancellation** | Supprime les paires auto-inverses : `H*H → I`, `X*X → I` | Moins de portes |
| **Rotation Merge** | Fusionne les rotations consecutives : `Rz(a)*Rz(b) → Rz(a+b)` | Profondeur reduite |
| **Noise-Aware Schedule** | Reordonne les portes 2Q sur les paires de qubits haute fidelite | Meilleure fidelite |
| **Layout Mapping** | Algorithme SABRE : mappe qubits logiques → physiques, minimise les SWAP | Circuit executable |

### Utilisation CLI

```bash
lift optimise model.lif --config config.lith -o optimised.lif
```

### Utilisation programmatique (Rust)

```rust
let mut pm = PassManager::new();
pm.add_pass(Box::new(Canonicalize));
pm.add_pass(Box::new(TensorFusion));
pm.add_pass(Box::new(DeadCodeElimination));
pm.add_pass(Box::new(GateCancellation));
pm.add_pass(Box::new(RotationMerge));
let results = pm.run_all(&mut ctx);
// results: Vec<(String, PassResult)>  — Changed / Unchanged / Error
```

---

## 5.2 Prediction de performance

### Modele roofline (GPU classique)

LIFT modelise deux GPU NVIDIA avec des parametres precis :

| GPU | TFLOPS FP16 | Bande passante | VRAM |
|-----|-------------|----------------|------|
| **A100** | 312 TFLOPS | 2039 GB/s | 80 Go |
| **H100** | 989 TFLOPS | 3350 GB/s | 80 Go |

Pour chaque modele, LIFT calcule :

- **Temps compute** = FLOPs / TFLOPS
- **Temps memoire** = Bytes / Bande passante
- **Temps predit** = max(compute, memoire) → identifie le goulot
- **Intensite arithmetique** = FLOPs / Bytes → compare au point de croisement
- **Nombre de GPU** = ceil(memoire_totale / VRAM_par_GPU)

### Modele quantique

LIFT modelise trois types de QPU :

| QPU | Temps 1Q | Temps 2Q | Fidelite 1Q | Fidelite 2Q | Qubits |
|-----|----------|----------|-------------|-------------|--------|
| **Supraconducteur** (IBM) | 0.02 us | 0.3 us | 99.9% | 99.0% | 127 |
| **Ions pieges** (IonQ) | 10 us | 200 us | 99.97% | 99.5% | 32 |
| **Atomes neutres** | 1 us | 5 us | 99.5% | 98.0% | 256 |

Pour chaque circuit, LIFT calcule :

- **Fidelite estimee** = prod(fidelite_1Q^n1Q * fidelite_2Q^n2Q)
- **Temps de circuit** = n1Q * t1Q + n2Q * t2Q + nMeas * tMeas
- **Shots necessaires** = 1 / (precision^2 * fidelite^2)
- **Temps total** = shots * temps_circuit

### Utilisation CLI

```bash
lift predict model.lif --device a100 --quantum-device ibm_kyoto
```

---

## 5.3 Modelisation du bruit quantique

LIFT modelise 8 types de bruit quantique :

| Modele de bruit | Formule de fidelite | Usage |
|-----------------|---------------------|-------|
| **Ideal** | F = 1.0 | Reference |
| **Depolarisant** | F = 1 - p | Bruit generique |
| **Amortissement amplitude** | F = 1 - gamma/2 | Decroissance T1 |
| **Amortissement phase** | F = 1 - gamma/2 | Dephasing T2 |
| **Bit-flip** | F = 1 - p | Erreur classique |
| **Phase-flip** | F = 1 - p | Erreur de phase |
| **Relaxation thermique** | F = (1 + e^{-t/T1} + 2*e^{-t/T2}) / 4 | Realiste (IBM) |
| **Kraus** | F ≈ 0.99 (approx.) | Canal quantique general |

### Canaux de Kraus

LIFT dispose d'une algebre complete de canaux de Kraus (`lift-quantum::kraus`) :

- **Depolarisant** : `KrausChannel::depolarizing(p, n_qubits)`
- **Amortissement amplitude** : `KrausChannel::amplitude_damping(gamma)`
- **Amortissement phase** : `KrausChannel::phase_damping(lambda)`
- **Canal de Pauli** : `KrausChannel::pauli(px, py, pz)`
- **Composition** : `channel1.compose(&channel2)`
- **Fidelite moyenne** : `channel.average_gate_fidelity()`

### Suivi du bruit porte par porte

```rust
let mut circuit = CircuitNoise::new();
let g1q = GateNoise::with_depolarizing(0.999, 0.02);  // 1Q gate
let g2q = GateNoise::with_depolarizing(0.99, 0.3);    // 2Q gate

circuit.add_gate(&g1q, false);  // RY
circuit.add_gate(&g2q, true);   // CX — source dominante d'erreur
// circuit.total_fidelity, circuit.gate_count, circuit.meets_threshold(0.90)
```

---

## 5.4 Topologie des processeurs quantiques

LIFT modelise 5 topologies de QPU :

| Topologie | Constructeur | Qubits | Connectivite |
|-----------|-------------|--------|-------------|
| **Grid** (n x m) | Google Sycamore | n*m | 4 voisins max |
| **Heavy-hex** | IBM Eagle/Osprey | 127 | 2-3 voisins |
| **All-to-all** | IonQ | variable | Tous connectes |
| **Linear** | Chain simple | variable | 2 voisins |
| **Tree** | Hierarchique | variable | log(n) profondeur |

### Fonctionnalites

Pour chaque topologie, LIFT fournit :

- **Connectivite** : `are_connected(q0, q1)` — deux qubits sont-ils voisins ?
- **Plus court chemin** : `shortest_path(from, to)` — BFS sur le graphe
- **Distance SWAP** : `swap_distance(from, to)` — nombre de SWAP necessaires
- **Voisins** : `neighbors(q)` — qubits adjacents
- **Diametre** : `diameter()` — plus long plus court chemin
- **Connectivite moyenne** : `avg_connectivity()` — degre moyen

### Impact sur la compilation

La topologie determine le cout du **layout mapping** (passe SABRE) :

| Topologie | SWAP q0→q3 (4 qubits) | Impact fidelite |
|-----------|----------------------|-----------------|
| All-to-all | 0 SWAP | Aucun |
| Grid 2x2 | 1 SWAP | 3 portes CX supplementaires |
| Linear | 2 SWAP | 6 portes CX supplementaires |
| Heavy-hex | Variable | Depend du placement |

---

## 5.5 Estimation energetique et empreinte carbone

LIFT estime la consommation energetique a trois niveaux :

### Inference GPU

```
Energie (J) = TDP (W) * PUE * temps (s) * n_GPU
CO2 (g) = Energie (kWh) * facteur_emission (g/kWh)
```

| GPU | TDP | PUE | Facteur emission |
|-----|-----|-----|-----------------|
| **A100** | 400 W | 1.1 | 400 g CO2/kWh |
| **H100** | 700 W | 1.1 | 400 g CO2/kWh |

### Entrainement (cluster)

Exemple calcule par LIFT : **8x A100 pendant 24 heures** :
- Energie = 8 * 400 W * 1.1 * 86400 s = 92.4 kWh
- CO2 = 92.4 * 400 = 36 960 g = **37 kg CO2**

### Execution quantique

Le cout energetique d'un QPU supraconducteur est domine par la **cryogenie** :
- Puissance cryostat : ~25 kW (pour maintenir 15 mK)
- Puissance par qubit : negligeable vs. cryostat
- `energy.quantum_energy_joules(circuit_time_us, n_qubits)` : inclut le cout cryogenique

---

## 5.6 Budget et contraintes de deploiement

### Budget statique

Verifie des contraintes hard avant la compilation :

```rust
let budget = Budget {
    max_flops: Some(10_000_000_000),       // 10 GFLOP max
    max_memory_bytes: Some(80_000_000_000), // 80 Go VRAM
    max_time_ms: Some(100.0),              // 100 ms latence max
    min_fidelity: Some(0.90),              // 90% fidelite min
    max_circuit_depth: None,
};

budget.check_flops(report.total_flops)?;      // Ok ou Err
budget.check_memory(report.total_memory)?;     // Ok ou Err
budget.check_fidelity(analysis.fidelity)?;     // Ok ou Err
```

### Budget reactif

Suit la consommation en temps reel lors de boucles iteratives (VQE, QAOA) :

```rust
let mut tracker = ReactiveBudget::new(budget);

for i in 0..max_iterations {
    tracker.consume(flops, memory, time_ms, fidelity_decay);
    if tracker.check_remaining().is_err() {
        println!("Budget epuise a l'iteration {}", i);
        break;
    }
}

let util = tracker.utilisation();
// util.time_ratio, util.flops_ratio, util.memory_ratio
```

---

## 5.7 Export vers les backends

### LLVM IR (GPU / CPU)

```bash
lift export model.lif --backend llvm -o model.ll
```

LIFT genere du LLVM IR qui peut etre compile par `llc` vers :
- **CUDA PTX** : execution sur GPU NVIDIA
- **x86-64** : execution sur CPU (avec AVX-512 si disponible)
- **ARM** : deploiement embarque

### OpenQASM 3.0 (QPU)

```bash
lift export model.lif --backend qasm -o circuit.qasm
```

LIFT genere du OpenQASM 3.0 compatible avec :
- **IBM Quantum** (Qiskit Runtime)
- **Amazon Braket**
- **Azure Quantum**
- **Simulateurs** (Qiskit Aer, Cirq)

### Utilisation programmatique

```rust
// Export LLVM
let llvm_exporter = LlvmExporter::new();
let llvm_ir = llvm_exporter.export(&ctx)?;

// Export OpenQASM
let qasm_exporter = QasmExporter::new();
let qasm_code = qasm_exporter.export(&ctx)?;
```

---

## 5.8 Configuration (.lith)

Le fichier `.lith` configure le pipeline LIFT avec un format INI simple :

```ini
[target]
backend = llvm
device = A100

[budget]
max_flops = 10000000000
max_memory_bytes = 80000000000
min_fidelity = 0.90

[optimisation]
level = O3
passes = canonicalize,tensor_fusion,dce,gate_cancellation,rotation_merge

[simulation]
enabled = true
noise_model = depolarizing

[quantum]
topology = grid
num_qubits = 4
shots = 4096
error_rate = 0.001
```

### Niveaux d'optimisation

| Niveau | Passes activees |
|--------|----------------|
| **O0** | Aucune (IR brut) |
| **O1** | Canonicalize, DCE |
| **O2** | + Tensor fusion, Gate cancellation, Rotation merge |
| **O3** | + Flash attention, CSE, Noise-aware schedule, Layout mapping, Quantisation |

---

# Partie VI — Reference rapide

## 6.1 Catalogue des 90+ operations tensorielles

### Arithmetique

| Operation | Syntaxe `.lif` | Description |
|-----------|---------------|-------------|
| `Add` | `tensor.add` | Addition element par element |
| `Sub` | `tensor.sub` | Soustraction |
| `Mul` | `tensor.mul` | Multiplication element par element |
| `Div` | `tensor.div` | Division |
| `Neg` | `tensor.neg` | Negation |
| `MatMul` | `tensor.matmul` | Produit matriciel |
| `Linear` | `tensor.linear` | Couche lineaire (matmul + biais) |
| `Conv2D` | `tensor.conv2d` | Convolution 2D |
| `Embedding` | `tensor.embedding` | Lookup dans table d'embeddings |

### Activations

| Operation | Syntaxe `.lif` | Formule |
|-----------|---------------|---------|
| `ReLU` | `tensor.relu` | max(0, x) |
| `GeLU` | `tensor.gelu` | x * Phi(x) |
| `SiLU` | `tensor.silu` | x * sigmoid(x) |
| `Sigmoid` | `tensor.sigmoid` | 1 / (1 + e^-x) |
| `Softmax` | `tensor.softmax` | e^x_i / sum(e^x_j) |
| `Tanh` | `tensor.tanh` | (e^x - e^-x) / (e^x + e^-x) |
| `LeakyReLU` | `tensor.leaky_relu` | max(alpha*x, x) |
| `ELU` | `tensor.elu` | x si x>0, alpha*(e^x-1) sinon |
| `Mish` | `tensor.mish` | x * tanh(softplus(x)) |
| `HardSwish` | `tensor.hard_swish` | x * relu6(x+3) / 6 |
| `HardSigmoid` | `tensor.hard_sigmoid` | relu6(x+3) / 6 |

### Normalisation

| Operation | Syntaxe `.lif` | Usage |
|-----------|---------------|-------|
| `LayerNorm` | `tensor.layernorm` | Transformers |
| `RMSNorm` | `tensor.rmsnorm` | LLaMA, Mistral |
| `BatchNorm` | `tensor.batchnorm` | CNN (entrainement) |
| `GroupNorm` | `tensor.groupnorm` | Diffusion models |
| `InstanceNorm` | `tensor.instancenorm` | Style transfer |

### Forme

| Operation | Syntaxe `.lif` | Zero-FLOP |
|-----------|---------------|-----------|
| `Reshape` | `tensor.reshape` | Oui |
| `Transpose` | `tensor.transpose` | Oui |
| `Concat` | `tensor.concat` | Oui |
| `Split` | `tensor.split` | Oui |
| `Gather` | `tensor.gather` | Oui |
| `Scatter` | `tensor.scatter` | Oui |
| `Squeeze` | `tensor.squeeze` | Oui |
| `Unsqueeze` | `tensor.unsqueeze` | Oui |
| `Permute` | `tensor.permute` | Oui |
| `Expand` | `tensor.expand` | Oui |
| `Slice` | `tensor.slice` | Oui |
| `Pad` | `tensor.pad` | Oui |
| `Tile` | `tensor.tile` | Oui |

### Attention (8 variantes)

| Operation | Syntaxe `.lif` |
|-----------|---------------|
| `Attention` | `tensor.attention` |
| `MultiHeadAttention` | `tensor.multi_head_attention` |
| `MultiQueryAttention` | `tensor.multi_query_attention` |
| `GroupedQueryAttention` | `tensor.grouped_query_attention` |
| `FlashAttention` | `tensor.flash_attention` |
| `SlidingWindowAttention` | `tensor.sliding_window_attention` |
| `CrossAttention` | `tensor.cross_attention` |
| `PagedAttention` | `tensor.paged_attention` |

### Convolution (6 variantes)

| Operation | Syntaxe `.lif` |
|-----------|---------------|
| `Conv1D` | `tensor.conv1d` |
| `Conv2D` | `tensor.conv2d` |
| `Conv3D` | `tensor.conv3d` |
| `ConvTranspose2D` | `tensor.conv_transpose2d` |
| `DepthwiseConv2D` | `tensor.depthwise_conv2d` |
| `DilatedConv2D` | `tensor.dilated_conv2d` |

### Pooling

| Operation | Syntaxe `.lif` |
|-----------|---------------|
| `MaxPool2D` | `tensor.maxpool2d` |
| `AvgPool2D` | `tensor.avgpool2d` |
| `AdaptiveAvgPool2D` | `tensor.adaptive_avgpool2d` |
| `GlobalAvgPool` | `tensor.global_avgpool` |

### Recurrent

| Operation | Syntaxe `.lif` |
|-----------|---------------|
| `LSTMCell` | `tensor.lstm_cell` |
| `GRUCell` | `tensor.gru_cell` |
| `RNNCell` | `tensor.rnn_cell` |

### Mathematiques avancees

| Operation | Syntaxe `.lif` | Complexite |
|-----------|---------------|-----------|
| `Einsum` | `tensor.einsum` | Variable |
| `FFT` | `tensor.fft` | O(n log n) |
| `IFFT` | `tensor.ifft` | O(n log n) |
| `SVD` | `tensor.svd` | O(mn min(m,n)) |
| `Eig` | `tensor.eig` | O(n^3) |
| `Solve` | `tensor.solve` | O(n^3) |
| `TopK` | `tensor.topk` | O(n log k) |
| `Sort` | `tensor.sort` | O(n log n) |
| `Cumsum` | `tensor.cumsum` | O(n) |
| `Where` | `tensor.where` | O(n) |
| `Clamp` | `tensor.clamp` | O(n) |

### Sparse, Quantification, Diffusion, GNN, MoE

| Operation | Syntaxe `.lif` |
|-----------|---------------|
| `SparseMatMul` | `tensor.sparse_matmul` |
| `SparseEmbedding` | `tensor.sparse_embedding` |
| `Quantize` | `tensor.quantize` |
| `Dequantize` | `tensor.dequantize` |
| `QuantizeInt4` | `tensor.quantize_int4` |
| `DequantizeInt4` | `tensor.dequantize_int4` |
| `QuantizeFp8` | `tensor.quantize_fp8` |
| `DequantizeFp8` | `tensor.dequantize_fp8` |
| `UNetDownBlock` | `tensor.unet_down_block` |
| `UNetUpBlock` | `tensor.unet_up_block` |
| `TimestepEmbedding` | `tensor.timestep_embedding` |
| `GNNMessagePassing` | `tensor.gnn_message_passing` |
| `GNNGlobalPooling` | `tensor.gnn_global_pooling` |
| `MoEDispatch` | `tensor.moe_dispatch` |
| `MoECombine` | `tensor.moe_combine` |

### Memoire, Gradient, Parallelisme, Fused

| Operation | Syntaxe `.lif` |
|-----------|---------------|
| `Checkpoint` | `tensor.checkpoint` |
| `Offload` | `tensor.offload` |
| `GradAccumulate` | `tensor.grad_accumulate` |
| `GradMatMul` | `tensor.grad_matmul` |
| `GradReLU` | `tensor.grad_relu` |
| `GradSoftmax` | `tensor.grad_softmax` |
| `GradLayerNorm` | `tensor.grad_layernorm` |
| `GradAttention` | `tensor.grad_attention` |
| `GradConv2D` | `tensor.grad_conv2d` |
| `GradLinear` | `tensor.grad_linear` |
| `GradGeLU` | `tensor.grad_gelu` |
| `ParallelSplit` | `tensor.parallel_split` |
| `ParallelAllReduce` | `tensor.parallel_all_reduce` |
| `PipelineSend` | `tensor.pipeline_send` |
| `PipelineReceive` | `tensor.pipeline_receive` |
| `FusedMatMulBiasReLU` | `tensor.fused_matmul_bias_relu` |
| `FusedMatMulBias` | `tensor.fused_matmul_bias` |
| `FusedLinearGeLU` | `tensor.fused_linear_gelu` |
| `FusedAttentionLayerNorm` | `tensor.fused_attention_layernorm` |
| `FusedLinearSiLU` | `tensor.fused_linear_silu` |
| `FusedConvBatchNormReLU` | `tensor.fused_conv_batchnorm_relu` |

### Constantes

| Operation | Syntaxe `.lif` |
|-----------|---------------|
| `Constant` | `tensor.constant` |
| `Zeros` | `tensor.zeros` |
| `Ones` | `tensor.ones` |
| `Arange` | `tensor.arange` |
| `Full` | `tensor.full` |

---

## 6.2 Catalogue des 50+ portes quantiques

### 1-qubit standard

| Porte | Syntaxe `.lif` | Clifford | Auto-inverse | Parametrique |
|-------|---------------|----------|-------------|-------------|
| H | `quantum.h` | Oui | Oui | Non |
| X | `quantum.x` | Oui | Oui | Non |
| Y | `quantum.y` | Oui | Oui | Non |
| Z | `quantum.z` | Oui | Oui | Non |
| S | `quantum.s` | Oui | Non | Non |
| S† | `quantum.sdg` | Oui | Non | Non |
| T | `quantum.t` | Non | Non | Non |
| T† | `quantum.tdg` | Non | Non | Non |
| SX | `quantum.sx` | Oui | Non | Non |

### 1-qubit parametriques

| Porte | Syntaxe `.lif` | Parametres |
|-------|---------------|-----------|
| RX | `quantum.rx` | theta |
| RY | `quantum.ry` | theta |
| RZ | `quantum.rz` | theta |
| P | `quantum.p` | phi |
| U1 | `quantum.u1` | lambda |
| U2 | `quantum.u2` | phi, lambda |
| U3 | `quantum.u3` | theta, phi, lambda |

### 2-qubit

| Porte | Syntaxe `.lif` | Entangling | Natif sur |
|-------|---------------|-----------|-----------|
| CX (CNOT) | `quantum.cx` | Oui | IBM |
| CZ | `quantum.cz` | Oui | Google |
| CY | `quantum.cy` | Oui | — |
| SWAP | `quantum.swap` | Oui | — |
| iSWAP | `quantum.iswap` | Oui | Google |
| ECR | `quantum.ecr` | Oui | IBM Eagle |
| RZX | `quantum.rzx` | Oui | — |
| XX | `quantum.xx` | Oui | IonQ |
| YY | `quantum.yy` | Oui | — |
| ZZ | `quantum.zz` | Oui | — |
| CP | `quantum.cp` | Oui | — |
| CPhase | `quantum.cphase` | Oui | Rigetti |
| XY | `quantum.xy` | Oui | Rigetti |

### 3-qubit et multi-controle

| Porte | Syntaxe `.lif` | Qubits |
|-------|---------------|--------|
| CCX (Toffoli) | `quantum.ccx` | 3 |
| CSWAP (Fredkin) | `quantum.cswap` | 3 |
| MCX | `quantum.mcx` | N |
| MCZ | `quantum.mcz` | N |

### Controle et mesure

| Porte | Syntaxe `.lif` | Description |
|-------|---------------|-------------|
| Measure | `quantum.measure` | Mesure un qubit |
| MeasureAll | `quantum.measure_all` | Mesure tous les qubits |
| Reset | `quantum.reset` | Reinitialise un qubit a \|0> |
| Barrier | `quantum.barrier` | Empeche la reordonnance |
| Init | `quantum.init` | Initialise un registre |
| Delay | `quantum.delay` | Delai temporel |
| VirtualRZ | `quantum.virtual_rz` | Rotation virtuelle (cout zero) |
| IfElse | `quantum.if_else` | Branchement conditionnel classique |

### Jeux de portes natifs par constructeur

| Constructeur | Portes natives |
|-------------|---------------|
| **IBM Eagle** | RZ, SX, X, CX, ECR |
| **IBM Kyoto** | RZ, SX, X, CX, ECR |
| **Rigetti** | RX, RZ, CPhase, XY |
| **IonQ** | GPI, GPI2, MS |
| **Quantinuum** | RZ, RX, ZZ |

---

## 6.3 Catalogue des operations hybrides

### Encodage / Decodage

| Operation | Syntaxe `.lif` | Description |
|-----------|---------------|-------------|
| `Encode` | `hybrid.encode` | Encode des donnees classiques dans un etat quantique |
| `Decode` | `hybrid.decode` | Extrait des donnees classiques d'un etat quantique |

### Methodes de gradient

| Operation | Syntaxe `.lif` | Evaluations | Exact |
|-----------|---------------|-------------|-------|
| `ParameterShift` | `hybrid.parameter_shift` | 2N | Oui |
| `FiniteDifference` | `hybrid.finite_difference` | N+1 | Non |
| `SPSA` | `hybrid.spsa` | 2 | Non |
| `AdjointDiff` | `hybrid.adjoint_diff` | 1 | Oui |
| `StochasticParamShift` | `hybrid.stochastic_param_shift` | 2 | Non |
| `JointGradient` | `hybrid.joint_gradient` | Variable | Mixte |

### Algorithmes variationnels

| Operation | Syntaxe `.lif` | Usage |
|-----------|---------------|-------|
| `VqcLayer` | `hybrid.vqc_layer` | Circuit variationnel generique |
| `VqeAnsatz` | `hybrid.vqe_ansatz` | Eigensolver variationnel (chimie) |
| `QaoaLayer` | `hybrid.qaoa_layer` | Optimisation combinatoire |
| `QuantumKernel` | `hybrid.quantum_kernel` | Noyau quantique (SVM) |

### Transfert de donnees

| Operation | Syntaxe `.lif` | Direction |
|-----------|---------------|-----------|
| `GpuToQpu` | `hybrid.gpu_to_qpu` | GPU → QPU |
| `QpuToGpu` | `hybrid.qpu_to_gpu` | QPU → GPU |

### Traitement et mesure

| Operation | Syntaxe `.lif` | Description |
|-----------|---------------|-------------|
| `ClassicalPreprocess` | `hybrid.classical_preprocess` | Pre-traitement classique |
| `QuantumPostprocess` | `hybrid.quantum_postprocess` | Post-traitement quantique |
| `HybridForward` | `hybrid.forward` | Passe forward hybride |
| `HybridBackward` | `hybrid.backward` | Passe backward hybride |
| `CoExecute` | `hybrid.co_execute` | Co-execution GPU + QPU |
| `MeasureExpectation` | `hybrid.measure_expectation` | Valeur moyenne d'observable |
| `MeasureSamples` | `hybrid.measure_samples` | Echantillons de mesure |

---

# Resume — La valeur de LIFT

| Dimension | Sans LIFT | Avec LIFT |
|-----------|-----------|-----------|
| **Langages** | Python (IA) + Qiskit (quantum) + CUDA (GPU) | Un seul fichier `.lif` |
| **Optimisation** | Manuelle, specifique a chaque framework | 11 passes automatiques |
| **Verification** | Tests manuels, erreurs au runtime | SSA + types + linearite a la compilation |
| **Performance** | Benchmarks empiriques | Prediction roofline + modele quantique |
| **Bruit** | Ignore ou modele separement | Integre dans le pipeline de compilation |
| **Energie** | Inconnue | Estimation automatique (GPU + QPU + CO2) |
| **Budget** | Pas de contrainte | Statique + reactif, arret automatique |
| **Export** | Conversion manuelle entre formats | LLVM IR + OpenQASM 3.0 en une commande |
| **Topologie** | Adaptation manuelle au QPU | Layout mapping automatique (SABRE) |

**LIFT transforme une description hybride en un systeme de production fiable, optimise et previsible, en un seul outil.**
