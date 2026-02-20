# 🧬 LexiBenchmark: The Evolution of Text Classification

**LexiBenchmark** is a comprehensive NLP evaluation suite that traces the historical evolution of text classification. It provides a standardized environment to compare classical statistical linguistics against modern transformer-based architectures using a unified dataset interface.

---

## 🏗 Three-Tiered Architecture

This project implements and evaluates three distinct generations of Natural Language Processing:

### 1. The Statistical Era (TF-IDF + Logistic Regression)

* **Mechanism:** Mathematical importance based on word frequency.
* **Advantage:** Extremely fast, interpretable, and requires no GPU.
* **Limitation:** Completely ignores word order and semantic context.

### 2. The Semantic Era (Word2Vec + Average Pooling)

* **Mechanism:** Mapping words into a continuous 300D vector space where similar meanings cluster together.
* **Advantage:** Captures analogies (e.g., King - Man + Woman = Queen).
* **Limitation:** Cannot distinguish between homonyms (e.g., "Bank" of a river vs. "Bank" for money).

### 3. The Contextual Era (BERT - Bidirectional Transformers)

* **Mechanism:** Leveraging Multi-Head Self-Attention to understand word meaning based on all surrounding words.
* **Advantage:** SOTA accuracy; understands nuance, sarcasm, and complex syntax.
* **Requirement:** High computational overhead (CUDA required).

---

## 🚀 Technical Implementation

### 🛠 Tech Stack

* **Core:** `PyTorch` & `TensorFlow`
* **Data Orchestration:** `HuggingFace Datasets`
* **Embeddings:** `Gensim` (Word2Vec)
* **Modeling:** `Transformers` (BERT), `Scikit-Learn` (Logistic Regression)

### 📊 Comparative Metrics

| Model | Feature Type | Contextual? | Training Cost |
| --- | --- | --- | --- |
| **LogReg** | Sparse (TF-IDF) | ❌ No | Low |
| **Word2Vec** | Dense (Static) | ❌ No | Medium |
| **BERT** | Dense (Dynamic) | ✅ Yes | High |

---

## ⚙️ Installation & Workflow

### Environment Setup

```bash
pip install datasets transformers gensim scikit-learn torch tensorflow tqdm

```

### Execution Pipeline

1. **Data Ingestion:** Loads benchmarks from the `datasets` library.
2. **Preprocessing:** Cleans text using `re` and `string` modules.
3. **Tokenization:** Dual-path (Simple whitespace vs. BERT WordPiece).
4. **Optimization:** AdamW optimizer with a linear learning rate scheduler.
5. **Evaluation:** Generates a full precision-recall-f1 matrix for side-by-side comparison.

---

## 📂 Repository Structure

```text
LexiBenchmark/
├── models/
│   ├── classical_ml.py    # TF-IDF & Logistic Regression
│   ├── embeddings.py      # Word2Vec implementation
│   └── transformers.py    # BERT Fine-tuning logic
├── utils/
│   └── preprocessing.py   # Regex & Cleaning scripts
├── benchmark_results/     # Output logs and metrics
└── main.py                # Pipeline orchestrator

