# Scalable Hybrid Big Data IDS: Distributed Graph-Enhanced ML on Apache Spark

![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.4.0-E25A1C?style=flat-square&logo=apache-spark)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![GraphFrames](https://img.shields.io/badge/GraphFrames-0.8.3-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A scalable, distributed Network Intrusion Detection System (NIDS) built entirely on the Apache Spark ecosystem (PySpark, Spark MLlib, Spark Structured Streaming). This project processes massive network flow data (CICIDS2017) and integrates graph-based feature engineering (PageRank centrality) with ensemble machine learning and authenticated encryption to provide robust, real-time threat detection.

## 🌟 Key Features

1. **Distributed Big Data Processing**: Built on PySpark to handle millions of network flows in parallel, eliminating the memory and compute bottlenecks of single-node libraries like scikit-learn.
2. **Graph-Enhanced Feature Engineering**: Utilizes **GraphFrames** to construct network communication graphs, calculating PageRank centrality to identify critical victim/attacker nodes. This topology-aware feature significantly improves detection accuracy.
3. **Multi-Model Parallel Classification**: Trains and evaluates multiple MLlib models (Random Forest, Decision Tree, Logistic Regression, Naive Bayes) using 5-fold cross-validation distributed across Spark executors.
4. **Authenticated Encryption Defense Layer**: Implements purely authenticated encryption (ChaCha20-Poly1305 / AES-256-GCM) prior to ML inference, blocking Replay and Man-in-the-Middle (MITM) attacks with 100% accuracy and zero false positives.
5. **Cross-Domain Generalization**: Includes simulations for concept drift (e.g., modern Edge-IIoTset and CIC-IoT-2023 traffic patterns) demonstrating the zero-shot robustness of the pre-trained Random Forest model.

## 🏗 Architecture

The framework consists of five layers:
1. **Ingestion & Cleaning**: Parallel CSV reading, sanitization, and type-casting.
2. **Graph Analytics**: Graph construction and PageRank score computation.
3. **ML Pipeline**: Feature VectorAssembler, StandardScaler, and Random Forest classification.
4. **Encryption Pre-filter**: AEAD packet validation.
5. **Streaming**: Real-time Kafka integration (1,000+ flows/second throughput).

*(Insert architectural diagram here if desired: e.g., `![Architecture](paper_figures/fig2_system_architecture.png)`)*

## 🚀 Getting Started

### Prerequisites

- **Java**: JDK 11 or 17 (Required by Apache Spark)
- **Apache Spark**: Version 3.4.0+
- **Python**: Version 3.9+
- **GraphFrames**: `graphframes:graphframes:0.8.3-spark3.4-s_2.12`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BigData-Spark-IDS.git
   cd BigData-Spark-IDS
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Dataset:
   - Download the **CICIDS2017** dataset.
   - Place the CSV files in the project root directory or update the paths in the scripts.

### Execution

To run the pipeline scripts, you must include the GraphFrames package in your `spark-submit` command:

```bash
# 1. Run the core ML evaluation and comparison pipeline
spark-submit --packages graphframes:graphframes:0.8.3-spark3.4-s_2.12 ids_compare_advanced.py

# 2. Extract and compute network graph features (PageRank)
spark-submit --packages graphframes:graphframes:0.8.3-spark3.4-s_2.12 ids_graph.py

# 3. Run cross-domain generalization (concept drift simulation)
spark-submit ids_cross_data.py

# 4. Generate the generalization plots
python plot_generalization.py
```

## 📊 Results Summary

Performance on the CICIDS2017 dataset (2.8M flows):
- **Random Forest**: 99.44% Weighted F1-Score
- **Speedup**: 6.6$\times$ faster than single-node configurations
- **Generalization**: Maintains 92.5\% F1 under moderate concept drift (unseen IoT edge data).

## 📄 Publications

This repository contains the codebase corresponding to the research paper submitted to the *Cluster Computing* journal. The LaTeX source code and figures for the paper can be found in the `overleaf_submission/` directory.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
