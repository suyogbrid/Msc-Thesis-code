# Msc-Thesis-code

# MSc Thesis — A Study About the Stability of Explainable AI 
for Full Deep Learning Models and Individual Hidden Layers

**Author:** Suyog Ramdas Brid  
**Student ID:** 24205214  
**Programme:** MSc Data Analytics  
**Institution:** National College of Ireland  
**Supervisor:** Dr Giovani Estrada  

---

## Project Overview

This repository contains the complete implementation code 
for the MSc thesis research project. The study investigates 
the stability of XAI (Explainable AI) explanations at two 
levels — the full model output and individual hidden layers 
— across three deep learning architectures trained on credit 
risk data.

**Research Question:**  
How does the stability of XAI predictions change at 
different levels of deep learning models, and how does 
the number of hidden layers affect internal explainability?

---

## Dataset

**File:** `credit_risk_dataset.csv`  
**Source:** Kaggle Credit Risk Dataset  
**Records:** 32,581 borrowers  
**Features:** 11 input features + 1 binary target (loan_status)  
**Class distribution:** 78% non-default, 22% default  

---

## Code Files

All experiments are implemented as Jupyter notebooks.  
Run each notebook **from top to bottom** in order.

### Notebook Naming Convention

---

### File 1 — `FH_1l_1h.ipynb`
**1 hidden layer — SHAP on hidden layer 1 (Dense 32)**

- Models: GRU, Transformer, CapsuleNet
- Architecture: single Dense(32) hidden layer
- SHAP target: `last_hidden` = Dense(32)
- Steps: Data preprocessing → Model training → 
  Output SHAP → Hidden SHAP → Permutation → 
  Spearman correlation → R(x) → S(x)

---

### File 2 — `FH_2l_1h.ipynb`
**2 hidden layers — SHAP on hidden layer 1 (Dense 64)**

- Models: GRU, Transformer, CapsuleNet  
- Architecture: Dense(64) → Dense(32)
- SHAP target: `hidden_2` = Dense(64) 
  *(second to last hidden layer)*
- Purpose: Tests whether an intermediate hidden layer 
  gives different internal explanations than the 
  final hidden layer in a 2-layer architecture

---

### File 3 — `FH_2l_2h.ipynb`
**2 hidden layers — SHAP on hidden layer 2 (Dense 32)**

- Models: GRU, Transformer, CapsuleNet  
- Architecture: Dense(64) → Dense(32)
- SHAP target: `last_hidden` = Dense(32) 
  *(final hidden layer)*
- Purpose: Companion to FH_2l_1h — compares whether 
  the final vs intermediate hidden layer produces 
  more consistent internal explanations at 2-layer depth

---

### File 4 — `FH_3l_2h.ipynb`
**3 hidden layers — SHAP on hidden layer 2 (Dense 64)**

- Models: GRU, Transformer, CapsuleNet  
- Architecture: Dense(128) → Dense(64) → Dense(32)
- SHAP target: `hidden_2` = Dense(64) 
  *(middle hidden layer)*
- Purpose: Tests internal explanations at the middle 
  layer of a 3-layer architecture

---

### File 5 — `FH_3l_3h - Copy.ipynb` ⭐ MAIN RESULT FILE
**3 hidden layers — SHAP on hidden layer 3 (Dense 32)**

- Models: GRU, Transformer, CapsuleNet  
- Architecture: Dense(128) → Dense(64) → Dense(32)
- SHAP target: `last_hidden` = Dense(32) 
  *(final hidden layer)*
- **Best performance:** CapsuleNet 90.15% accuracy, 
  AUC 0.9077
- **Key finding:** 1-layer Transformer internal shift 
  detected (Spearman r = 0.65)
- Contains: Full evaluation including R(x), S(x) 
  stability metrics, all Spearman correlations, 
  rank heatmap, and auto-generated LaTeX tables

---

### File 6 — `FH_4l_3h.ipynb`
**4 hidden layers — SHAP on hidden layer 3 (Dense 64)**

- Models: GRU, Transformer, CapsuleNet  
- Architecture: Dense(256) → Dense(128) → Dense(64) 
  → Dense(32)
- SHAP target: `hidden_3` = Dense(64) 
  *(third hidden layer)*
- Purpose: Tests internal explanations at an 
  intermediate layer of a 4-layer architecture

---

### File 7 — `FH_4l_4h.ipynb`
**4 hidden layers — SHAP on hidden layer 4 (Dense 32)**

- Models: GRU, Transformer, CapsuleNet  
- Architecture: Dense(256) → Dense(128) → Dense(64) 
  → Dense(32)
- SHAP target: `last_hidden` = Dense(32) 
  *(final hidden layer)*
- Purpose: Confirms that 4-layer depth causes 
  overfitting — performance drops vs 3-layer models

---

### Summary Table

| File | Layers | SHAP Target | Key Purpose |
|------|--------|-------------|-------------|
| FH_1l_1h | 1 | Dense(32) | Baseline — 1 layer |
| FH_2l_1h | 2 | Dense(64) | Middle layer comparison |
| FH_2l_2h | 2 | Dense(32) | Final layer at 2-depth |
| FH_3l_2h | 3 | Dense(64) | Middle layer at 3-depth |
| **FH_3l_3h** | **3** | **Dense(32)** | **⭐ Best results** |
| FH_4l_3h | 4 | Dense(64) | Middle layer at 4-depth |
| FH_4l_4h | 4 | Dense(32) | Overfitting confirmation |

---

## Steps in Each Notebook

Every notebook follows the same 15-step KDD pipeline:

| Step | Description |
|------|-------------|
| Step 1 | Load and inspect dataset |
| Step 2 | Handle missing values (median imputation) |
| Step 3 | Encode categorical features (LabelEncoder) |
| Step 4 | Scale numerical features (StandardScaler) |
| Step 5 | Separate features and target |
| Step 6 | Train/test split (80/20 stratified) |
| Step 7 | Reshape for sequence models |
| Step 8 | Build and train GRU model |
| Step 9 | Build and train Transformer model |
| Step 10 | Build and train CapsuleNet model |
| Step 11 | Full model SHAP (output level) |
| Step 12 | Hidden layer SHAP (internal level) |
| Step 13 | Permutation feature importance |
| Step 14 | Spearman rank correlation |
| Step 15 | R(x) and S(x) stability metrics |

---

## Requirements

```bash
pip install tensorflow==2.x
pip install shap==0.42
pip install scikit-learn==1.3
pip install pandas numpy scipy matplotlib seaborn
```

**Python version:** 3.10  
**Environment:** CPU (no GPU required)

---

## Key Results

| Model | Layers | Accuracy | AUC | Spearman r |
|-------|--------|----------|-----|------------|
| GRU | 1 | 86.54% | 0.9017 | 0.90 |
| Transformer | 1 | 85.36% | 0.8947 | **0.65** ← shift |
| CapsuleNet | 1 | 86.51% | 0.9080 | 0.85 |
| GRU | 3 | 87.25% | 0.9016 | 0.92 |
| Transformer | 3 | 86.39% | 0.9003 | 0.88 |
| **CapsuleNet** | **3** | **90.15%** | **0.9077** | **0.91** |
| GRU | 4 | 87.05% | 0.8987 | 0.95 |
| Transformer | 4 | 85.78% | 0.8929 | 0.93 |
| CapsuleNet | 4 | 87.54% | 0.8963 | 0.90 |

**Main finding:** The 1-layer Transformer detected an 
internal feature shift (r = 0.65) where `home_ownership` 
dominated the hidden layer but not the output explanation. 
This shift resolved with deeper architectures.

