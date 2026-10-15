# Loan Approval Prediction

Exploratory data analysis and a Logistic Regression model that predicts whether a personal loan application is approved (`Loan_Approved`), using 500 applications in Egyptian pounds.

## Key Results

| Metric | Test set (100 applications) |
|---|---|
| Accuracy | 0.610 |
| Precision | 0.652 |
| Recall | 0.729 |
| F1-Score | 0.688 |
| ROC-AUC | 0.622 |
| Baseline (always predict "Yes") | 0.586 |

- **Credit Score is the only variable with a clear signal:** the approval rate rises steadily from 45% (score 450-550) to 76% (score 750-850).
- Income, loan amount, age, education, employment type and housing have weak or no relationship with the decision.
- The model beats the baseline only by a small margin. Decision Tree, Random Forest and KNN were also tested and gave similar or lower accuracy (0.51 to 0.61), which suggests the limit comes from the data, not the method.
- The data looks synthetic (uniform distributions, almost no correlation between features), so the results show the workflow rather than real lending behavior.
- The model leans toward approving (recall higher than precision). In a real bank the decision threshold should be raised above 0.5.

## Pipeline
1. **Data check:** no duplicates, 8 missing values (1.6%) in each of 4 columns
2. **Cleaning:** median for numeric and mode for categorical columns
3. **Feature engineering:** 4 financial ratios (`Debt_To_Income`, `Monthly_Installment`, `Total_DTI`, `Loan_To_Income`)
4. **Encoding:** target Yes/No to 1/0, one-hot encoding with `drop_first=True`
5. **EDA:** 10 charts (missing values, target, distributions, approval rate by credit score and income, correlation matrix, and more)
6. **Modeling:** 80/20 stratified split, `StandardScaler` inside a `Pipeline` (no leakage), `C=0.05` chosen with 5-fold cross-validation (CV accuracy 0.618)
7. **Evaluation:** confusion matrix, ROC curve, coefficients

## Project Structure
```
Loan_Approval_Prediction/
├── data/loan_data.xlsx               original data
├── loan_approval_analysis.ipynb      full notebook with outputs
├── loan_approval_analysis.py         same analysis as a script
├── figures/                          10 PNG charts
├── outputs/                          result tables (CSV) and insights
├── REPORT.md                         detailed report (Arabic)
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt

# Option 1: script
python loan_approval_analysis.py

# Option 2: notebook
jupyter notebook loan_approval_analysis.ipynb
```

## Tech Stack
Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, Jupyter

## Author
Ahmed Tharwat Muhammad Ibrahim — [LinkedIn](https://www.linkedin.com/in/ahmed-tharwat-at111)
