# 【NeurIPS' 2025】K-DECORE: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning via Knowledge Decoupling

## Overview

K-DECORE (Knowledge DEcoupling for COntinual REasoning) is a novel framework for continual structured knowledge reasoning (CSKR) that enables large language models to effectively handle sequential SKR tasks while maintaining a fixed parameter count. 

![image](https://github.com/user-attachments/assets/d13a6aa5-bcb5-4049-a354-3c10e7574872)

The framework introduces:
- 🔄 Knowledge decoupling mechanism for task-specific and task-agnostic reasoning
- 🧠 Dual-perspective memory consolidation
- 🔍 Structure-guided pseudo-data synthesis
- 📈 State-of-the-art performance across multiple SKR benchmarks

## Features

- **Knowledge Decoupling**: Separates reasoning into task-specific schema filtering and task-agnostic query building
- **Dual Memory Mechanism**: 
  - Schema memory for comprehensive task coverage
  - Query memory for preserving diverse query structures
- **Pseudo Query Synthesis**: Generates novel structured queries to enhance generalization
- **PEFT-based Architecture**: Uses lightweight parameter-efficient fine-tuning modules
- **Multi-Task Support**: Handles diverse SKR tasks with heterogeneous schemas

## Requirements

```bash
# install LLaMA-Factory
git clone --depth 1 http://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
bash K-DECORE_order1_llama3.sh
```

## Directory Structure

```
K-DECORE/
├── data/                   # Dataset storage
├── model/                  # Model implementation
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── K-DECORE.sh           # Main training pipeline
└── requirements.txt       # Dependencies
```

## Data

The framework is evaluated on four benchmark datasets:
- GrailQA
- MTOP 
- Spider
- CompWebQ

## Data Preparation

### 1. Evaluation Directory Structure
```
K-DECORE/
├── data/                      # Training dataset storage
└── evaluate/                  # Evaluation scripts and assets
    └── database/
        └── spider/            # <-- Setup needed for Spider
            ├── database/      # Folder containing multiple .sqlite files
            ├── dev_gold.sql
            ├── dev.json
            ├── tables.json
            └── ...
```

### 2. Dataset Setup Guidelines
#### 📊 Spider Dataset Setup
The evaluation scripts require the complete Spider SQL schemas. You can quickly download the pre-packaged environment from the [StructLM](https://github.com/TIGER-AI-Lab/StructLM/tree/main/third_party/spider) repository.

1. Download the spider folder from the link above.
2. Unzip and place the entire contents into evaluate/database/spider/.
3. Crucial: Ensure that the final path containing all the SQLite databases is exactly evaluate/database/spider/database/.

#### 🌐 CompWebQ (ComplexWebQuestions) Knowledge Graph Setup
Evaluating the CompWebQ task requires interacting with a local deployment of the Freebase Knowledge Graph.

1. Graph Database: Ensure you have installed and populated a Virtuoso instance with the official Freebase RDF triples (or the specific subset required for CompWebQ evaluation).
2. SPARQL Endpoint: Start your local Virtuoso server before triggering the evaluation script.
3. Configuration: Double-check your SPARQL endpoint connection string (typically http://localhost:8890/sparql) inside the evaluate/configure/ configuration files to match your local port setup.

## Results

K-DECORE achieves state-of-the-art performance on continual SKR tasks, demonstrating:
- Effective knowledge transfer across diverse tasks
- Reduced catastrophic forgetting
- Enhanced generalization to novel query structures

## Citation

```bibtex
@article{chen2025k,
  title={K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning via Knowledge Decoupling},
  author={Chen, Yongrui and Huang, Yi and Liu, Yunchang and Zhang, Shenyu and He, Junhao and Wu, Tongtong and Qi, Guilin and Wu, Tianxing},
  journal={arXiv preprint arXiv:2509.16929},
  year={2025}
}
```
