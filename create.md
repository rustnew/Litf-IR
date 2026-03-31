# Guide complet pour publier LIFT sur crates.io sans erreur

Ce guide détaille toutes les étapes, règles et conditions à respecter pour que l’ensemble des crates LIFT soient publiés avec succès sur [crates.io](https://crates.io). Il couvre la structuration du workspace, les métadonnées de chaque crate, la gestion des dépendances, l’ordre de publication, les tests, et les vérifications finales.

---

## 1. Prérequis

Avant de commencer, assurez-vous d’avoir :

- Un compte sur [crates.io](https://crates.io) (connectez‑vous avec GitHub).
- Un token API crates.io. Dans votre profil, générez un token et connectez‑le en local :

  ```bash
  cargo login <votre-token>
  ```

- Rust stable à jour :

  ```bash
  rustup update stable
  ```

- Tous les tests passent localement :

  ```bash
  cargo test --workspace
  ```

- Une licence choisie (MIT pour LIFT) et un fichier `LICENSE` à la racine.

---

## 2. Organisation du workspace

LIFT est un **workspace Cargo** contenant 13 crates. Le `Cargo.toml` racine doit ressembler à ceci :

```toml
[workspace]
resolver = "2"
members = [
    "crates/lift-core",
    "crates/lift-ast",
    "crates/lift-tensor",
    "crates/lift-quantum",
    "crates/lift-hybrid",
    "crates/lift-sim",
    "crates/lift-predict",
    "crates/lift-opt",
    "crates/lift-import",
    "crates/lift-export",
    "crates/lift-config",
    "crates/lift-cli",
    "crates/lift-python",
]

[workspace.package]
version = "0.2.0"
edition = "2021"
authors = ["Votre nom <email>"]
license = "MIT"
repository = "https://github.com/lift-framework/lift"
```

**Règles importantes :**
- Toutes les crates partagent la même version (0.2.0) pour simplifier.
- Les champs communs (`authors`, `license`, `repository`, etc.) peuvent être définis dans `[workspace.package]` et hérités par chaque crate via `workspace = true`.

---

## 3. Configuration de chaque crate

Chaque crate doit avoir un `Cargo.toml` valide. Voici le modèle pour `lift-core` (les autres suivent le même schéma, avec leurs dépendances spécifiques).

### 3.1 Exemple pour `lift-core/Cargo.toml`

```toml
[package]
name = "lift-core"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
description = "LIFT core: SSA IR, types, verifier, pass manager"
readme = "../../README.md"   # ou un README propre à la crate
keywords = ["compiler", "ir", "ssa", "type-system"]
categories = ["compilers", "science"]

[dependencies]
slotmap = "1.0"
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
# ... autres dépendances

[dev-dependencies]
proptest = "1.4"

[features]
default = []
```

**Règles :**
- `name` doit être unique sur crates.io.
- `version`, `edition`, `authors`, `license`, `repository` utilisent `workspace = true` pour hériter du workspace.
- `description` est **obligatoire** et doit être une courte phrase.
- `readme` pointe vers un fichier Markdown présent dans le dépôt.
- `keywords` (max 5) et `categories` aident à la découverte.
- Toutes les dépendances doivent avoir des versions explicites (pas `*`).
- Si une crate a des fonctionnalités optionnelles, définissez `[features]`.

### 3.2 Gestion des dépendances internes

Pour une crate qui dépend d’une autre crate du workspace, utilisez **à la fois un chemin relatif et une version** :

```toml
[dependencies]
lift-core = { path = "../lift-core", version = "0.2.0" }
```

Cela permet à Cargo de travailler en développement et de savoir quelle version publier sur crates.io.

---

## 4. Exclusion des fichiers inutiles

Pour éviter d’inclure des fichiers de test, exemples, etc., dans le package publié, ajoutez un fichier `.cargo/config.toml` à la racine du projet :

```toml
[package]
exclude = [
    "examples/",
    "tests/",
    "benches/",
    "target/",
    ".github/",
    ".gitignore",
    "rustfmt.toml"
]
```

Cela réduit la taille du package et évite d’exposer du code de test.

---

## 5. Documentation

Chaque crate doit avoir une documentation de base dans son `lib.rs` :

```rust
//! LIFT core: SSA-based intermediate representation.
//!
//! This crate provides the foundational types and algorithms for
//! the LIFT compiler framework.
```

De plus, assurez‑vous que tous les items publics sont documentés (ou au minimum que `#![warn(missing_docs)]` est activé pour les crates de bibliothèque). Cela n’est pas obligatoire pour la publication, mais fortement recommandé pour une bonne adoption.

---

## 6. Tests et validation

Avant toute publication, exécutez ces commandes pour vous assurer que tout est correct :

```bash
# Compilation de tout le workspace
cargo build --workspace

# Exécution de tous les tests
cargo test --workspace

# Vérification des warnings
cargo clippy --workspace -- -D warnings

# Vérification du formatage
cargo fmt --all -- --check

# Vérification des dépendances pour les vulnérabilités
cargo audit

# Test de publication à blanc pour chaque crate
cargo publish --dry-run --manifest-path crates/lift-core/Cargo.toml
cargo publish --dry-run --manifest-path crates/lift-ast/Cargo.toml
# ... pour toutes les crates
```

Si le `--dry-run` passe, la publication réelle devrait réussir.

---

## 7. Ordre de publication

Les crates doivent être publiées dans l’ordre des dépendances, de la plus basse à la plus haute. Pour LIFT :

1. `lift-core`
2. `lift-ast`
3. `lift-tensor`
4. `lift-quantum`
5. `lift-hybrid`
6. `lift-sim`
7. `lift-predict`
8. `lift-opt`
9. `lift-import`
10. `lift-export`
11. `lift-config`
12. `lift-cli`
13. `lift-python` (optionnel, nécessite la feature `pyo3`)

Pour chaque crate, utilisez :

```bash
cargo publish --manifest-path crates/lift-core/Cargo.toml
```

Après chaque publication, attendez quelques secondes que crates.io traite la crate avant de publier la suivante (sinon, la dépendance peut ne pas être encore disponible). Un script automatisé peut attendre 10 secondes entre chaque.

---

## 8. Gestion des features flags

Si certaines fonctionnalités sont optionnelles (par exemple `cuda`, `llvm`, `pyo3`), définissez‑les comme features dans les crates concernées. Par exemple, dans `lift-export` :

```toml
[features]
default = []
cuda = ["dep:cudarc"]
llvm = ["dep:inkwell"]
```

Assurez‑vous que ces features ne sont pas activées par défaut pour la publication. Les utilisateurs pourront choisir ce dont ils ont besoin.

---

## 9. Vérifications finales avant publication

- **Version** : incrémentez correctement (0.2.0 pour l’instant). La version doit être cohérente entre toutes les crates.
- **Dépendances internes** : toutes utilisent `version = "0.2.0"` avec le chemin relatif.
- **Licence** : le champ `license` est renseigné (MIT).
- **Description** : chaque crate a une description unique et pertinente.
- **Documentation** : au moins un commentaire de module dans `lib.rs`.
- **Absence de `*` dans les versions** : toutes les dépendances externes ont des versions fixes.
- **Aucun `unwrap()`** dans le code des crates de bibliothèque (il y a déjà été corrigé).
- **Fichiers exclus** : vérifiez avec `cargo package --list` pour chaque crate que seuls les fichiers nécessaires sont inclus.

---

## 10. Publication pas à pas (exemple pour lift-core)

```bash
cd crates/lift-core
cargo publish --dry-run   # vérification finale
# Si tout est correct
cargo publish
```

Répétez pour chaque crate dans l’ordre.

---

## 11. Après publication

- **Tag git** : créez un tag pour la version publiée :

  ```bash
  git tag v0.2.0
  git push origin v0.2.0
  ```

- **Documentation** : crates.io génère automatiquement la documentation sur `docs.rs`. Vérifiez qu’elle s’affiche correctement quelques minutes après publication.

- **Annonce** : informez la communauté (Discord, Reddit, etc.) que LIFT est maintenant disponible.

---

## 12. Pièges courants et solutions

| Problème | Solution |
|----------|----------|
| **Version déjà existante** | Incrémentez la version (patch, minor) et republiez. |
| **Dépendance interne non publiée** | Publiez la dépendance avant. |
| **Token invalide** | Régénérez le token sur crates.io et refaites `cargo login`. |
| **Description manquante** | Ajoutez `description` dans chaque `Cargo.toml`. |
| **Licence manquante** | Ajoutez `license = "MIT"`. |
| **Un `*` dans une version** | Remplacez par une version exacte. |
| **Fichiers de test inclus** | Utilisez `.cargo/config.toml` pour exclure les dossiers. |
| **`unwrap()` dans le code** | Remplacez par `?` ou une gestion d’erreur appropriée. |

---

## 13. Structure finale du dépôt

```
lift/
├── Cargo.toml                     # workspace root
├── .cargo/
│   └── config.toml                # exclusion de fichiers
├── README.md
├── LICENSE
├── crates/
│   ├── lift-core/
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── lift-ast/
│   │   ├── Cargo.toml
│   │   └── src/
│   └── ... (toutes les autres crates)
├── examples/
├── tests/
└── benches/
```

---

## 14. Statut actuel de préparation (v0.2.0)

Toutes les conditions du guide ci-dessus ont été appliquées. Voici le résumé :

| Condition | Statut |
|-----------|--------|
| `[workspace.package]` version, edition, license, authors, repository | DONE |
| Internal deps have `version = "0.2.0"` in workspace | DONE |
| Chaque crate a `description`, `keywords`, `categories`, `readme` | DONE (12 crates) |
| `lift-tests` a `publish = false` | DONE |
| `LICENSE` (MIT) a la racine | DONE |
| `README.md` a la racine | DONE |
| Module-level `//!` docs dans chaque `lib.rs` | DONE |
| `cargo build --workspace` | DONE - Passe |
| `cargo test --workspace` | DONE - 505 tests, 0 echecs |
| `cargo publish --dry-run -p lift-core` | DONE - Passe |
| `cargo publish --dry-run -p lift-config` | DONE - Passe |

### Commandes de publication prêtes à exécuter

```bash
# 1. Commit + tag
git add -A && git commit -m "chore: prepare v0.2.0 for crates.io" && git tag v0.2.0

# 2. Login
cargo login <TOKEN>

# 3. Publier dans l'ordre
cargo publish -p lift-core    && sleep 60
cargo publish -p lift-config  && sleep 60
cargo publish -p lift-ast     && sleep 60
cargo publish -p lift-tensor  && sleep 60
cargo publish -p lift-quantum && sleep 60
cargo publish -p lift-hybrid  && sleep 60
cargo publish -p lift-sim     && sleep 60
cargo publish -p lift-export  && sleep 60
cargo publish -p lift-import  && sleep 60
cargo publish -p lift-opt     && sleep 60
cargo publish -p lift-predict && sleep 60
cargo publish -p lift-cli

# 4. Push tag
git push origin main --tags