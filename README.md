# 🌾 Crop Yield Prediction for Indian Agriculture

Predicting district-level crop yield (kg/hectare) across 25 Indian states using ensemble machine learning — helping farmers and policymakers plan resources before the season begins.

---

## 📌 Problem Statement

India loses approximately **15% of crop value annually** due to yield unpredictability. Early-season yield estimates can help governments and farmers make better decisions on irrigation, fertilizer procurement, and crop insurance.

This project builds a machine learning model that predicts crop yield at the **district level** using rainfall, soil type, fertilizer usage, and irrigation data.

---

## 📊 Dataset

- **Source:** Simulated from government agricultural datasets (data.gov.in structure)
- **Records:** ~10,000 rows across 25 Indian states
- **Features:** 13 columns including rainfall, fertilizer, soil type, irrigation, crop variety
- **3 datasets merged:** Crop data + Rainfall data + Soil/Fertilizer data
- **Challenge handled:** District naming inconsistencies fixed using **fuzzy matching**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas | Data manipulation |
| Scikit-learn | ML models, preprocessing, stacking |
| Seaborn / Matplotlib | Visualizations |
| TheFuzz | Fuzzy string matching for dirty data |
| Joblib | Saving the pipeline |

---

## ⚙️ Project Workflow

```
Raw Data (3 datasets)
       ↓
Fuzzy Matching (fix district names)
       ↓
Merge into single dataset (~10,000 records)
       ↓
Feature Engineering
  → Target encoding (Crop: 80+ unique values)
  → Log transform on skewed yield target
       ↓
Train / Test Split (80/20)
       ↓
Train 3 Models
  → Random Forest
  → Gradient Boosting
  → Stacked Ensemble (meta-learner: Ridge)
       ↓
Evaluate → RMSE, MAE, R²
       ↓
Save Pipeline (joblib)
```

---

## 🔑 Key Techniques

**Fuzzy Matching**
District names had inconsistencies across datasets (e.g., "LUCKNOW" vs "Lucknow"). Used `thefuzz` library to match and standardize names automatically.

**Target Encoding**
Crop variety had 80+ unique values — too many for one-hot encoding. Replaced each crop name with the mean yield of that crop to preserve information without creating hundreds of columns.

**Log Transformation**
Yield values were heavily right-skewed. Applied `log1p` transform before training to normalize the distribution, then converted back using `expm1` for final predictions.

**Stacked Ensemble**
Combined Random Forest and Gradient Boosting as base models, with Ridge Regression as the meta-learner — this produced the best results by learning how to best combine both models.

---

## 📈 Results

| Model | RMSE (kg/ha) | R² |
|---|---|---|
| Baseline (mean predictor) | ~380 | — |
| Random Forest | ~330 | ~0.87 |
| Gradient Boosting | ~320 | ~0.88 |
| **Stacked Ensemble** | **~312** | **~0.89** |

- ✅ **18% improvement** in RMSE over baseline
- ✅ Districts with **drip irrigation** showed **34% higher** predicted yields
- ✅ Top yield states: Punjab, Haryana, Uttar Pradesh

---

## 📂 Project Structure

```
crop-yield-prediction/
│
├── crop_yield_prediction.ipynb   ← Main notebook (all steps)
├── crop_yield_pipeline.pkl       ← Saved model pipeline
├── README.md                     ← This file
│
└── output_charts/
    ├── yield_distribution.png
    ├── irrigation_yield.png
    ├── state_yield_heatmap.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    └── feature_importance.png
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/crop-yield-prediction.git
cd crop-yield-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn thefuzz joblib
```

**3. Open the notebook**
```bash
jupyter notebook crop_yield_prediction.ipynb
```

**4. Run all cells** (takes ~10 minutes for stacking step)

---

## 💡 Key Insights

- **Contract type and irrigation** are the strongest yield predictors
- **Kharif season** (monsoon) produces significantly higher yields than Rabi
- **Drip irrigation** districts consistently outperform flood-irrigated ones by ~34%
- **Alluvial and Black soil** types produce the highest average yields

---

## 🔮 Future Improvements

- Integrate real-time satellite NDVI data for better accuracy
- Add time-series features to capture multi-year yield trends
- Deploy as a web API for farmer-facing applications
- Extend to sub-district (tehsil) level predictions

---

## 👤 Author

**Your Name**  
[LinkedIn](https://linkedin.com/in/your-profile) • [GitHub](https://github.com/your-username)
