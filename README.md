# 🧬 DNABERT-Style Transformer From Scratch (PyTorch)

A from-scratch educational implementation of a **DNABERT-style Transformer** for DNA sequence modeling using **PyTorch**.

The primary goal of this project is **to understand how DNA foundation models work internally**, rather than simply using pretrained models from Hugging Face.

This project implements the complete pipeline required to train a BERT-style model on genomic sequences, including tokenization, embeddings, Transformer encoder blocks, Masked Language Modeling (MLM), and the training loop.

---

# 🎯 Project Goal

Modern genomics models such as DNABERT are based on the Transformer architecture.

Instead of treating DNA as plain text, these models learn contextual representations of nucleotide sequences that can later be fine-tuned for downstream tasks such as:

* Disease prediction
* Promoter prediction
* Enhancer prediction
* Variant effect prediction
* DNA sequence classification
* Other genomics tasks

This repository focuses on understanding **how these models are built**, not just how they are used.

---

# 🚀 Features

* DNA sequence preprocessing
* DNA tokenization
* Embedding layer
* Multi-Head Self-Attention
* ALiBi positional bias
* Transformer Encoder Blocks
* Feed Forward Network (FFN)
* Layer Normalization
* Residual Connections
* Masked Language Modeling (MLM)
* Complete training pipeline
* PyTorch implementation

---

# 🏗 Project Architecture

```text
Genome FASTA
      │
      ▼
Dataset Preparation
      │
      ▼
DNA Tokenization
      │
      ▼
Input IDs
      │
      ▼
Embedding Layer
      │
      ▼
Transformer Encoder
      │
      ├── Multi-Head Self Attention
      ├── ALiBi Bias
      ├── LayerNorm
      ├── Feed Forward Network
      └── Residual Connections
      │
      ▼
MLM Prediction Head
      │
      ▼
Loss
      │
      ▼
Backpropagation
      │
      ▼
Updated Model
```

---

# 📂 Project Structure

```text
Project
│
├── Dataset Preparation
├── Tokenizer
├── Configuration
├── DNA Embedding
├── MultiHeadSelfAttention
├── Transformer Encoder Block
├── DNABERT Model
├── MLM Label Generator
├── DataLoader
└── Training Loop
```

---

# 🧠 Model Components

### 1. DNA Tokenizer

Converts DNA sequences into numerical token IDs that can be processed by neural networks.

---

### 2. Embedding Layer

Maps every DNA token into a dense vector representation.

---

### 3. Multi-Head Self-Attention

Allows every DNA token to attend to other relevant positions within the sequence, enabling the model to learn long-range biological dependencies.

---

### 4. ALiBi Positional Bias

Uses Attention with Linear Biases (ALiBi) to encode positional information without traditional positional embeddings.

---

### 5. Transformer Encoder Block

Each encoder block contains:

* Multi-Head Attention
* Residual Connection
* Layer Normalization
* Feed Forward Network
* Residual Connection

---

### 6. MLM Head

Predicts randomly masked DNA tokens during pretraining.

This enables the model to learn contextual representations of genomic sequences.

---

# 🧬 Training Objective

This project uses **Masked Language Modeling (MLM)**.

Example:

Input

```text
A C [MASK] T G
```

Target

```text
A C G T G
```

The model learns by predicting the masked nucleotide using its surrounding context.

---

# 🛠 Technologies Used

* Python
* PyTorch
* Transformers
* NumPy
* Biopython
* Google Colab

---

# 📚 Learning Outcomes

Through this project I gained hands-on understanding of:

* DNA sequence preprocessing
* Tokenization
* Embedding layers
* Self-Attention
* Multi-Head Attention
* ALiBi positional encoding
* Transformer encoder architecture
* Feed Forward Networks
* Layer Normalization
* Residual Connections
* Masked Language Modeling
* Backpropagation and training workflow

---

# 🔮 Future Work

Planned improvements include:

* Fine-tuning for disease prediction
* Promoter prediction
* Enhancer prediction
* DNABERT-2 implementation
* HyenaDNA implementation
* Model benchmarking
* Architecture improvements

---

# 📌 Disclaimer

This repository is an **educational implementation** created to understand the internal workings of DNABERT-style Transformer models.

It is **not an official implementation of DNABERT or DNABERT-2**, and it is not intended to reproduce the exact training procedures or pretrained weights from the original research papers.

AI-assisted tools were used during development for learning, debugging, and code exploration. The implementation has been studied and documented as part of my learning process.

---

# 📖 References

* DNABERT: Pre-trained Bidirectional Encoder Representations from Transformers for DNA-Language in Genome
* DNABERT-2
* Attention Is All You Need
* ALiBi: Train Short, Test Long

---

# ⭐ Acknowledgements

Thanks to the open-source research community for publishing foundational work on Transformer architectures and genomic language models that inspired this educational implementation.

