# Fake Review Detection using Temporal Graph Networks, XLM-RoBERTa, and GNNExplainer

An Explainable Temporal Graph-Based Framework for Detecting Coordinated Fake Reviews in Multilingual E-Commerce Platforms.

---

## Overview

Online marketplaces are increasingly exposed to coordinated fake review campaigns that manipulate product reputation and influence customer purchasing decisions. Unlike traditional approaches that focus on individual reviews, this project models relationships among reviews, shops, and products as a heterogeneous graph to detect coordinated review behavior.

The framework combines:

* **XLM-RoBERTa** for multilingual review representation learning
* **Temporal Graph Networks (TGN)** for modeling temporal interactions
* **Graph Neural Networks (GNNs)** for graph-based classification
* **GNNExplainer** for model interpretability

The system is designed to identify suspicious review groups based on semantic similarity, temporal proximity, and relational structures within e-commerce ecosystems.

---

## Features

### Multilingual Review Representation

Review texts are encoded using:

* XLM-RoBERTa Base
* mBERT (Baseline)

to support multilingual review analysis.

### Temporal Graph Modeling

Reviews are represented as nodes in a heterogeneous graph and connected through:

* Review → Shop
* Review → Product

relationships.

Temporal information is incorporated using Temporal Graph Networks to capture evolving review behaviors.

### Coordinated Review Detection

The model identifies potentially coordinated review activities by learning patterns from:

* Text similarity
* Temporal proximity
* Rating consistency
* Shared shop interactions

### Explainable AI

GNNExplainer is used to highlight:

* Important nodes
* Influential edges
* Graph structures responsible for model predictions

This improves model transparency and interpretability.

---

## Project Structure

```text
project/
│
├── data/
│   ├── raw_reviews.csv
│   ├── processed_reviews.csv
│   ├── graph_data_xlmr.pt
│   └── graph_data_mbert.pt
│
├── outputs/
│   ├── embeddings/
│   ├── models/
│   ├── results/
│   └── explanations/
│
├── preprocessing.py
├── feature_extraction.py
├── graph_construction.py
├── model.py
├── train.py
├── evaluate.py
├── explain.py
└── requirements.txt
```

---

## Dataset

The dataset consists of Tokopedia product reviews collected from multiple product categories.

### Original Attributes

* review_text
* review_date
* review_id
* product_name
* product_category
* product_variant
* product_price
* product_url
* product_id
* rating
* sold_count
* shop_id
* sentiment_label

### Added Label

* coordinated

The coordinated label is generated using rule-based annotation based on:

* Temporal proximity (≤ 48 hours)
* Text similarity threshold
* Rating consistency
* Shared shop interactions

---

## Methodology

### 1. Data Preprocessing

* Remove missing values
* Normalize review texts
* Convert timestamps
* Generate review-level features

### 2. Feature Extraction

Multilingual embeddings are generated using:

* XLM-RoBERTa Base
* mBERT (Baseline)

Output:

```text
Review Embeddings (768 dimensions)
```

### 3. Graph Construction

A heterogeneous graph is created with:

Node Types:

* Review
* Shop
* Product

Edge Types:

* Review → Shop
* Review → Product

### 4. Model Training

Four configurations are evaluated:

| Model          | Description                      |
| -------------- | -------------------------------- |
| Baseline 1     | Static GNN                       |
| Baseline 2     | TGN + mBERT                      |
| Baseline 3     | TGN without Explainability       |
| Proposed Model | TGN + XLM-RoBERTa + GNNExplainer |

### 5. Evaluation

Performance metrics:

* Precision
* Recall
* F1-Score
* ROC-AUC

### 6. Explainability

GNNExplainer identifies:

* Critical graph structures
* Important neighboring reviews
* Influential review-product-shop relationships

for each prediction.

---

## Running on Google Colab

This project was developed and executed using Google Colab with GPU acceleration.

### 1. Upload Project Files

Upload the project folder to Google Drive and mount it in Google Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Navigate to the project directory:

```python
%cd /content/drive/MyDrive/rM
```

---

### 2. Install Dependencies

```python
!pip install -r requirements.txt
```

---

### 3. Data Preprocessing

```python
!python preprocessing.py
```

Output:

```text
processed_reviews.csv
```

---

### 4. Feature Extraction

Generate multilingual review embeddings using XLM-RoBERTa or mBERT:

```python
!python feature_extraction.py
```

Output:

```text
outputs/embeddings/
```

---

### 5. Graph Construction

Build the heterogeneous graph consisting of review, shop, and product nodes:

```python
!python graph_construction.py
```

Output:

```text
data/graph_data_xlmr.pt
data/graph_data_mbert.pt
```

---

### 6. Model Training

Train all baseline and proposed models:

```python
!python train.py
```

Output:

```text
outputs/models/
```

---

### 7. Model Evaluation

Evaluate trained models and generate performance metrics:

```python
!python evaluate.py
```

Output:

```text
outputs/results/metrics.csv
outputs/results/metrics.json
```

---

### 8. Explainability Analysis

Generate graph explanations using GNNExplainer:

```python
!python explain.py
```

Output:

```text
outputs/explanations/
```

---

### Hardware Recommendation

For optimal performance:

* Google Colab GPU (T4, L4, or A100)
* RAM ≥ 12 GB
* CUDA-enabled runtime

To enable GPU:

```text
Runtime → Change Runtime Type → GPU
```

---

## Experimental Results

| Model                                       | F1 Score | AUC-ROC |
| ------------------------------------------- | -------- | ------- |
| Baseline 1 (Static GNN)                     | 0.368    | 0.979   |
| Baseline 2 (TGN + mBERT)                    | 0.416    | 0.985   |
| Baseline 3 (TGN No XAI)                     | 0.378    | 0.978   |
| Proposed (TGN + XLM-RoBERTa + GNNExplainer) | 0.372    | 0.978   |

---

## Requirements

Main dependencies:

* Python 3.10+
* PyTorch
* PyTorch Geometric
* Transformers
* Scikit-learn
* Pandas
* NumPy
* NetworkX
* Matplotlib
* tqdm

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Future Work

Potential improvements include:

* Dynamic graph augmentation
* Contrastive graph learning
* Focal Loss for class imbalance
* Advanced temporal edge modeling
* Cross-platform review fraud detection

---

## Authors

Naila Riani
Sabrina Salma Almira
Alvina Aulia
Kenny Jingga

Research Project:
**Fake Review Detection Using Temporal Graph Networks, XLM-RoBERTa, and Explainable AI**
