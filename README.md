# Graph-Based Federated AI for Mutant p53 Reactivation
**B.Tech Project**

---

## Project Overview

Three-component privacy-preserving AI pipeline:

| Component | Architecture | Key metric |
|-----------|-------------|------------|
| GNN | Graph Attention Network (GAT, 3 layers) | AUC ~0.9997 |
| Ensemble | XGBoost + LightGBM + RF + ExtraTrees (stacking) | AUC ~1.0 |
| Federated | FedProx, 5 clients (cancer-type partition) | Val Acc ~74–80 % |

---

## Data Sources

| File | Source | Purpose |
|------|--------|---------|
| `tp53_mutations.csv` | [p53.fr IARC TP53 DB](http://p53.fr) | Main mutation dataset (53,871 × 76) |
| `string_links_human_v12.txt` | [STRING v12](https://string-db.org) | Human PPI interaction network |
| `9606_protein_info_v12_0.txt` | STRING v12 | Protein ID -> gene name mapping |
| `biogrid_all_4.4.243.tab3.txt` | [BioGRID 4.4](https://thebiogrid.org) | Physical PPI interactions |

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place data files
mkdir data
cp tp53_mutations.csv              data/
cp string_links_human_v12.txt      data/
cp 9606_protein_info_v12_0.txt     data/
cp biogrid_all_4.4.243.tab3.txt    data/
```

> **Note:** If STRING / BioGRID files are unavailable, the pipeline falls back to a
> built-in curated TP53 interactome (~45 known partners with literature-derived
> confidence scores). All three components still train and evaluate correctly.

---

## Run Instructions

```bash
# Full pipeline (GNN + Ensemble + FL)
python main.py

# Skip GNN (faster, runs Ensemble + FL only)
python main.py --skip-gnn

# Only one component
python main.py --component ensemble
python main.py --component fl
python main.py --component gnn

# Force rebuild of cached features / graph data
python main.py --rebuild

# Individual scripts
python ppi_graph_builder.py   # build & inspect PPI graph
python feature_engineering.py # run feature engineering only
python train_gnn.py           # train GNN only
python ensemble_model.py      # train ensemble only
python fl_server.py           # run FL only
```

---

## File Structure

```
.
├── config.py               ← All paths & hyperparameters
├── ppi_graph_builder.py    ← STRING + BioGRID -> NetworkX PPI graph
├── feature_engineering.py  ← Feature matrix + PPI-derived features
├── gnn_dataset.py          ← PyTorch Geometric graph construction
├── gnn_model.py            ← GATClassifier + FocalLoss
├── train_gnn.py            ← GNN training loop
├── ensemble_model.py       ← Stacking ensemble (XGB+LGBM+RF+ET)
├── fl_client.py            ← FedProx local update
├── fl_server.py            ← FedAvg server + cancer-type partitioning
├── main.py                 ← Orchestration
├── requirements.txt
├── data/                   ← Place all CSV / TXT files here
├── models/                 ← Saved checkpoints
├── outputs/                ← Plots, cached features, graph data
└── logs/                   ← pipeline.log
```

---

## How PPI Data Improves Performance

| Feature Group | Source | Impact |
|---------------|--------|--------|
| `ppi_degree`, `ppi_weighted_degree` | STRING | Global TP53 connectivity signal |
| `ppi_driver_score`, `ppi_n_drivers` | STRING + BioGRID | Cancer-driver interaction weight |
| `dom_ppi_*` (6 features) | STRING + BioGRID | Domain-specific interaction disruption |
| `ppi_disruption_index` | Engineered | Quantifies how much a mutation disrupts key PPI |
| `mdm2_ppi_score` | STRING | MDM2 binding disruption (key p53 regulator) |

These 12 PPI-derived features are concatenated to the ~80 mutation-level features,
giving the GNN richer node representations and the ensemble additional biological signal.

---

## Key Design Decisions

- **GAT vs GCN**: Attention mechanism lets the model learn which neighbours matter,
  critical for the highly heterogeneous TP53 mutation landscape.
- **FocalLoss**: Handles class imbalance (35,657 Pathogenic vs 860 Benign) better
  than cross-entropy.
- **Cancer-type FL partitioning**: Simulates realistic data silos (hospital A has
  lung cancer data, hospital B has colorectal, etc.) -> non-IID setting.
- **FedProx**: Adds a proximal term (μ = 0.01) to stabilise training under
  non-IID data — better convergence than vanilla FedAvg for heterogeneous clients.
- **Stacking ensemble**: OOF meta-features avoid in-sample overfitting and
  LightGBM adds orthogonal signal to XGBoost.
