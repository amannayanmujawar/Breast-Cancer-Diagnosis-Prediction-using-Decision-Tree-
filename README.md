# 🩺 Breast Cancer Diagnosis Prediction

A machine learning project that predicts whether a breast tumor is **Malignant (M)** or **Benign (B)** using cell nuclei measurements from the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## 📌 Problem Statement
Given lab-measured characteristics of white blood cell/tumor samples (`wbc.csv`), build a classification model that predicts the presence of cancer (Malignant vs Benign).

## 📊 Dataset
- **File:** `wbc.csv`
- **Samples:** 569
- **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension — each as `_mean`, `_se`, `_worst`)
- **Target:** `diagnosis` → `M` (Malignant) = 1, `B` (Benign) = 0
- **Class balance:** 357 Benign, 212 Malignant
- No missing values, no duplicates.

## 🔧 Workflow
1. **EDA** – shape, dtypes, nulls, duplicates, class distribution
2. **Cleaning** – dropped `id` column (not predictive)
3. **Encoding** – mapped `diagnosis`: M → 1, B → 0
4. **Split** – 75% train / 25% test (`random_state=7`)
5. **Scaling** – `StandardScaler` on features
6. **Modeling** – `DecisionTreeClassifier`
7. **Hyperparameter tuning** – used **10-fold cross-validation** to find the optimal `max_depth`
8. **Feature importance** – identified top predictive features

## 📈 Results

| Model | Train Accuracy | Test Accuracy | F1 Score |
|---|---|---|---|
| Decision Tree (default, unrestricted depth) | 100% | 93.01% | 88.89% |
| Decision Tree (tuned via 10-fold CV, `max_depth=5`) | 99.30% | 92.31% | 87.36% |

**Best cross-validated score:** 92.93% (avg. across 10 folds at `max_depth=5`)

> The tuned model has a marginally lower single-split test score, but it's the more trustworthy one — see "Why Cross-Validation" below.

### 🔑 Top predictive features
`perimeter_worst`, `concave points_mean`, `concave points_worst`, `concave points_se`, `texture_worst`

## 🧠 Why Cross-Validation?
The default (unrestricted) tree memorized the training data (100% train accuracy = overfitting). Its 93% test score looks good, but it's a **single lucky split** — no guarantee it generalizes.

10-fold cross-validation was used to test `max_depth` values from 1–30 on the training data *before* touching the test set. This gives a stable, averaged estimate of generalization performance instead of relying on one split. `max_depth=5` gave the best average CV score (92.93%) and reduced overfitting (train accuracy dropped from 100% → 99.3%), making the model more reliable on unseen data — even though the one-off test accuracy is nearly the same.

## 🛠 Tech Stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## 🚀 How to Run
```bash
git clone <your-repo-url>
cd breast-cancer-diagnosis-prediction
pip install -r requirements.txt
jupyter notebook Breast_Cancer_Dignosis_Prediction.ipynb
```

## 📂 Repo Structure
```
├── Breast_Cancer_Dignosis_Prediction.ipynb
├── wbc.csv
├── README.md
└── requirements.txt
```

## 🔮 Future Improvements
- Try ensemble models (Random Forest, XGBoost) for better generalization
- Use `GridSearchCV`/`RandomizedSearchCV` for multi-parameter tuning
- Add ROC-AUC and confusion matrix visualization
- Handle class imbalance with SMOTE if extending to a larger dataset
