# K-DECORE: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning via Knowledge Decoupling

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

## Data

The framework is evaluated on four benchmark datasets:
- GrailQA
- MTOP 
- Spider
- CompWebQ

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

## Results

K-DECORE achieves state-of-the-art performance on continual SKR tasks, demonstrating:
- Effective knowledge transfer across diverse tasks
- Reduced catastrophic forgetting
- Enhanced generalization to novel query structures

## Citation

```bibtex
[Citation will be added after publication]
```
