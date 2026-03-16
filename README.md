# Sink Hole Node Detection in AODV MANETs

A hybrid two-stage intrusion detection system for detecting **Sink hole node** in Mobile Ad-hoc Networks (MANETs) using the AODV routing protocol. The system combines **XGBoost** (supervised, primary filter) with an **Autoencoder** (unsupervised, zero-day net) in a sequential cascade architecture.

---

## 📌 Problem Statement

Sink hole attacks are a critical security threat in MANETs. A malicious node advertises itself as having the shortest path to a destination, attracts traffic, and then drops packets — silently disrupting the network. Detecting such nodes is challenging because:

- Attacks may be **low-intensity** (few malicious nodes)
- **Zero-day variants** may not match known attack signatures
- Node behavior is **time-varying** and context-dependent

---

## 🏗️ Architecture

```
Raw AODV Logs
     │
     ▼
┌─────────────────────────────────────────────┐
│  Time-Window Aggregation (5-second bins)    │
│  Feature Engineering (Drop_Ratio, RREP/RREQ)│
└─────────────────────┬───────────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   ┌─────────────┐        ┌──────────────────┐
   │  XGBoost    │        │  Autoencoder     │
   │ (Supervised)│        │ (Unsupervised)   │
   │ + SMOTE     │        │ Trained on normal│
   └──────┬──────┘        └────────┬─────────┘
          │    CASCADE LOGIC       │
          └───────────┬────────────┘
                      │  OR (max vote)
                      ▼
             Final Node-Level Prediction
```

**Cascade Logic:** If XGBoost flags a node → malicious. If XGBoost misses it but the Autoencoder detects abnormal reconstruction error → malicious.

---

## 📊 Dataset

Simulated AODV network traces with varying numbers of black hole nodes:

| Split    | Scenarios                        |
|----------|----------------------------------|
| Train    | `0_nodes`, `3_nodes`, `7_nodes`  |
| Test     | `5_nodes`, `10_nodes`            |

**Features used:**

| Feature            | Description                                |
|--------------------|--------------------------------------------|
| `RREQ_Sent`        | Route Request packets sent                 |
| `RREQ_Recv`        | Route Request packets received             |
| `RREP_Sent`        | Route Reply packets sent                   |
| `RREP_Recv`        | Route Reply packets received               |
| `Pkts_Forwarded`   | Data packets forwarded                     |
| `Pkts_Dropped`     | Data packets dropped                       |
| `Drop_Ratio`       | Engineered: Dropped / (Forwarded + Dropped)|
| `RREP_to_RREQ_Ratio`| Engineered: RREP_Sent / RREQ_Recv        |

---

## 📈 Results (Node-Level)

### Scenario: 5 Malicious Nodes (out of 100)

| Model              | Precision (Attack) | Recall (Attack) | F1 (Attack) |
|--------------------|--------------------|-----------------|-------------|
| Pure XGBoost       | 0.00               | 0.00            | 0.00        |
| **Cascade (Ours)** | **0.71**           | **1.00**        | **0.83**    |

### Scenario: 10 Malicious Nodes (out of 100)

| Model              | Precision (Attack) | Recall (Attack) | F1 (Attack) |
|--------------------|--------------------|-----------------|-------------|
| Pure XGBoost       | 0.77               | 1.00            | 0.87        |
| **Cascade (Ours)** | **0.71**           | **1.00**        | **0.83**    |

> **Key insight:** The cascade achieves **100% recall on attack class in both scenarios** — zero missed black hole nodes.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Detection

```bash
# Update FILE_PATH in model.py to point to your dataset
python model.py
```

---

## 📁 Project Structure

```
blackhole-manet-detection/
├── model.py                  # Main detection pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── notebooks/
│   └── exploration.ipynb     # EDA and visualization notebook
└── results/
    └── classification_reports.txt
```

---

## 🔬 Key Design Decisions

1. **SMOTE (0.25 ratio):** Handles severe class imbalance without over-generating minority samples.
2. **Critical AE Threshold (2× max train MSE):** Intentionally conservative — only flags extreme outliers, minimizing false positives from the AE while catching zero-day patterns.
3. **OR-fusion cascade:** XGBoost handles known attack signatures; AE catches novel behaviors that slip past the supervised model.
4. **Scenario-based train/test split:** Ensures the model generalizes to unseen attack intensities.

---

## 📚 References

- Perkins, C., Belding-Royer, E., & Das, S. (2003). *Ad hoc On-Demand Distance Vector (AODV) Routing.* RFC 3561.
- Sen, J. (2010). *A Survey on Wireless Sensor Network Security.* IJCNS.
- SMOTE: Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.

---

## 📄 License

MIT License
