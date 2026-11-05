# Intrusion Storyline Builder

A network intrusion detection and incident storyline generation system built on the **KDD Cup 1999** dataset.

## What It Does

1. **Anomaly Detection** — Isolation Forest scores each network connection for suspiciousness
2. **Attack Family Classification** — Logistic Regression predicts 5 attack families (Normal, DoS, Probe, R2L, U2R)
3. **Storyline Generation** — Groups suspicious events into episodes and writes auto-generated incident narratives

## Project Structure

```
intrusion_storyline_builder/
├── app.py                  # Streamlit app (6 tabs)
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/                # Original dataset files
│   └── processed/          # Cleaned train/test CSVs
├── models/
│   ├── anomaly_model.pkl   # Isolation Forest
│   ├── family_classifier.pkl # Logistic Regression
│   ├── preprocessor.pkl    # ColumnTransformer
│   ├── label_mapper.pkl    # Attack name mappings
│   └── eval_results.pkl    # Evaluation metrics
└── src/
    ├── data_prep.py        # Loading, cleaning, label mapping, sampling
    ├── eda.py              # Charts and summary tables
    ├── train.py            # Model training pipeline
    └── storyline.py        # Episode builder + narrative generation
```

## Setup and Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data (downloads KDD dataset via sklearn)

```bash
python src/data_prep.py
```

### 3. Train models

```bash
python src/train.py
```

### 4. Launch the app

```bash
streamlit run app.py
```

## App Features

The Streamlit app has 6 interactive tabs:

| Tab | Features |
|-----|----------|
| **Overview** | Problem statement, model flow, dataset summary, metrics |
| **Data Explorer** | Filterable data table, distribution charts, correlation heatmap, PCA 2D, boxplots |
| **Suspicion Scanner** | Demo samples, manual entry, CSV upload, single/batch prediction |
| **Storyline Builder** | Suspicion timeline, episode grouping, auto-generated narratives, downloadable report |
| **Model Performance** | Anomaly ROC/AUC, classifier confusion matrix, per-class recall, sample predictions |
| **Explainability** | Feature importance, per-class coefficients, normal vs suspicious comparison |

## Dataset

- **Source:** [KDD Cup 1999](https://archive.ics.uci.edu/dataset/130/kdd+cup+1999+data) (UCI ML Repository)
- **Size:** ~494K records (10% subset), sampled to ~100K after deduplication
- **Features:** 41 network connection attributes
- **Attack Families:** Normal, DoS, Probe, R2L, U2R

## Models

| Model | Task | Key Metric |
|-------|------|------------|
| Isolation Forest | Anomaly detection | ROC-AUC: 0.68 |
| Logistic Regression | 5-class classification | Macro F1: 0.53, Accuracy: 92% |

## Deployment

### Streamlit Community Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo, branch, and `app.py`
4. Deploy

**Note:** For deployment, run `data_prep.py` and `train.py` locally first, then commit the `data/processed/` and `models/` folders.
