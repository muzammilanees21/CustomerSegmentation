# CustomerSegmentation
A machine learning project that performs customer segmentation using the K-Means clustering algorithm. Built in Jupyter Notebook using a Kaggle dataset, including data preprocessing, visualization, and cluster analysis.

# Customer Segmentation — Unsupervised Learning (K-Means + PCA)

Segments mall customers into distinct, business-interpretable personas using K-Means clustering, validated mathematically (not guessed) and translated back into real-world units for stakeholder use.

---

## 🎯 Goal

Use distance-based algorithms to discover hidden mathematical groupings in unlabeled retail/customer data, then turn those groupings into actionable business personas.

## 🧩 Pipeline (IPO Architecture)

| Phase | Step | Technique |
|---|---|---|
| 1. **Scale** (Input) | Standardize features | `StandardScaler` |
| 2. **Compress** (Process) | Reduce dimensionality | `PCA` (95% variance rule) |
| 3. **Cluster** (Process) | Find optimal K, fit model | Elbow Method + Silhouette Score + `KMeans` |
| 4. **Translate** (Output) | Recover business meaning | Inverse PCA + inverse scaling → Persona Matrix |

## 📊 Dataset

**Mall Customer Segmentation Dataset** ([Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)) — 200 customers, 5 columns: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`.

## 🛠️ Methodology

1. **EDA** — distribution plots, correlation heatmap, missing-value and duplicate checks.
2. **Feature engineering** — drop `CustomerID`, label-encode `Gender`.
3. **Scaling** — z-score standardization (`z = (x - μ) / σ`) so income doesn't dominate spending score.
4. **PCA** — fit on all components, keep enough to explain 95% cumulative variance (capped at 3 for visualization on this low-dimensional dataset).
5. **Optimal K selection** — swept K = 2 to 10, cross-checked with both the Elbow Method (WCSS) and Silhouette Score; picked the K with the highest Silhouette Score rather than eyeballing the elbow.
6. **Final K-Means fit** — clustered in PCA space.
7. **Centroid translation** — inverse-transformed centroids from PCA space → scaled space → original units (`C_original = (C_scaled ⊙ σ) + μ`), cross-checked against actual per-cluster means to confirm correctness.
8. **Persona matrix** — each cluster labeled by an income/spending-score quadrant rule and paired with a recommended business action.

## 📈 Results

**Optimal K = 7** (Silhouette Score = 0.425)

| Cluster | Persona | Size | Avg Age | Avg Income | Avg Spending Score | % Female | Recommended Action |
|---|---|---|---|---|---|---|---|
| 0 | Conservative Minimizers | 44 | 46.0 | $46.3k | 36.8 | 100% | Minimize spend, clear price-value messaging, basic utility. |
| 1 | High-Value Trendsetters | 28 | 29.9 | $81.7k | 66.8 | 0% | Exclusive perks, early access, experiential marketing. |
| 2 | Affluent Conservatives | 44 | 50.6 | $60.9k | 28.7 | 0% | High-touch support, warranties, loyalty programs. |
| 3 | High-Value Trendsetters | 35 | 29.0 | $77.0k | 68.0 | 100% | Exclusive perks, early access, experiential marketing. |
| 4 | Affluent Conservatives | 15 | 49.3 | $92.4k | 30.3 | 100% | High-touch support, warranties, loyalty programs. |
| 5 | Budget-Conscious Explorers | 16 | 27.6 | $31.9k | 71.7 | 0% | Influencer campaigns, flash sales, buy-now-pay-later. |
| 6 | Budget-Conscious Explorers | 18 | 26.8 | $28.8k | 72.6 | 100% | Influencer campaigns, flash sales, buy-now-pay-later. |

**Persona logic:** clusters are grouped along two axes — income (above/below median) and spending score (above/below median) — into four base personas: *High-Value Trendsetters* (high income, high spend), *Affluent Conservatives* (high income, low spend), *Budget-Conscious Explorers* (low income, high spend), and *Conservative Minimizers* (low income, low spend).

## 📁 Repository Structure

```
.
├── Project3_Customer_Segmentation.ipynb   # full pipeline notebook
├── Mall_Customers.csv                     # raw dataset
├── customer_segments_output.csv           # dataset with cluster labels (generated)
├── persona_matrix.csv                     # final persona summary (generated)
└── README.md
```

## ▶️ How to Run

1. Clone the repo and make sure `Mall_Customers.csv` is in the same folder as the notebook.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run all cells top to bottom in Jupyter / Colab. Outputs (`customer_segments_output.csv`, `persona_matrix.csv`) are written to the working directory.

## 🧰 Tech Stack

`pandas` · `numpy` · `scikit-learn` (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`) · `matplotlib` · `seaborn`

## 🔑 Key Takeaways

- K was chosen mathematically (Silhouette Score) rather than guessed from the elbow plot alone.
- Splitting Gender alongside Age/Income/Spending Score effectively duplicated demographic personas (e.g. two "High-Value Trendsetters" clusters by gender), which is useful for gender-targeted campaigns but could be collapsed to 4 clusters if that split isn't needed.
- Clustering was done in PCA space for noise reduction and visualization, then centroids were mapped back to original units — abstract PCA coordinates are meaningless to a business stakeholder, actual Age/Income/Spending Score values are not.

---
*Part of the DecodeLabs Data Science Industrial Training Kit (Batch 2026).*
