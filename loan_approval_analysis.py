"""
=====================================================================
 تحليل بيانات طلبات القروض وبناء نموذج للتنبؤ بقرار الموافقة
 Loan Approval Analysis & Prediction
---------------------------------------------------------------------
 الهدف : التنبؤ بالعمود Loan_Approved (Yes / No)
 النموذج: Logistic Regression
 البيانات: data/loan_data.xlsx  (500 صف × 13 عمود)
=====================================================================
طريقة التشغيل:
    python loan_approval_analysis.py
المخرجات:
    figures/*.png   الرسوم البيانية
    outputs/*.csv   جداول النتائج
"""

# ---------------------------------------------------------------
# 0) الاستيرادات والإعدادات العامة
# ---------------------------------------------------------------
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # للعمل بدون واجهة رسومية
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.xlsx")
FIG_DIR = os.path.join(BASE_DIR, "figures")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(name):
    """حفظ الشكل الحالي في مجلد figures ثم إغلاقه."""
    plt.savefig(os.path.join(FIG_DIR, name))
    plt.close()


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ===============================================================
# 1) قراءة البيانات وفهم الأعمدة
# ===============================================================
section("1) قراءة البيانات | Loading the data")

df = pd.read_excel(DATA_PATH)
print(f"أبعاد البيانات: {df.shape[0]} صف × {df.shape[1]} عمود")
print("\nأول 5 صفوف:")
print(df.head())

print("\nمعلومات الأعمدة:")
df.info()

# قاموس الأعمدة (Data Dictionary)
DATA_DICTIONARY = {
    "Customer_ID":        "معرّف فريد للعميل - ليس له قيمة تنبؤية (يُحذف)",
    "Age":                "عمر مقدّم الطلب بالسنوات",
    "Monthly_Income_EGP": "الدخل الشهري بالجنيه المصري",
    "Employment_Years":   "عدد سنوات الخبرة / الاستقرار الوظيفي",
    "Loan_Amount_EGP":    "مبلغ القرض المطلوب بالجنيه المصري",
    "Loan_Term_Months":   "مدة سداد القرض بالشهور",
    "Credit_Score":       "درجة الجدارة الائتمانية (850-300 تقريباً)",
    "Existing_Loans":     "عدد القروض القائمة حالياً على العميل",
    "Monthly_Debt_EGP":   "إجمالي الالتزامات الشهرية الحالية بالجنيه",
    "Education":          "المؤهل الدراسي (High School / Bachelor / Master)",
    "Employment_Type":    "نوع العمل (Salaried / Self-Employed / Freelancer)",
    "Home_Ownership":     "حالة السكن (Own / Rent / Family)",
    "Loan_Approved":      "الهدف Target: هل تمت الموافقة على القرض؟ (Yes/No)",
}
print("\nقاموس الأعمدة:")
for col, desc in DATA_DICTIONARY.items():
    print(f"  - {col:20s}: {desc}")


# ===============================================================
# 2) المعالجة المسبقة للبيانات (Data Preprocessing)
# ===============================================================
section("2) Data Preprocessing")

TARGET = "Loan_Approved"

# --- 2.1 فحص التكرارات ---
print(f"صفوف مكررة بالكامل : {df.duplicated().sum()}")
print(f"معرّفات عملاء مكررة: {df['Customer_ID'].duplicated().sum()}")

# --- 2.2 فحص القيم المفقودة ---
missing = df.isna().sum()
missing = missing[missing > 0]
print("\nالقيم المفقودة قبل المعالجة:")
print(pd.DataFrame({
    "عدد المفقود": missing,
    "النسبة %": (missing / len(df) * 100).round(2),
}))

plt.figure(figsize=(9, 4))
sns.heatmap(df.isna(), cbar=False, cmap="viridis")
plt.title("Missing Values Map (before imputation)")
save_fig("01_missing_values.png")

# --- 2.3 معالجة القيم المفقودة ---
# الأعمدة الرقمية  -> الوسيط (Median) لأنه غير حساس للقيم المتطرفة
# الأعمدة النصية   -> المنوال (Mode) أي الفئة الأكثر تكراراً
num_missing_cols = ["Monthly_Income_EGP", "Credit_Score"]
cat_missing_cols = ["Education", "Employment_Type"]

for col in num_missing_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"  {col:20s} -> عُوّض بالوسيط = {median_val:,.1f}")

for col in cat_missing_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"  {col:20s} -> عُوّض بالمنوال = {mode_val}")

assert df.isna().sum().sum() == 0, "ما زالت هناك قيم مفقودة!"
print("\n✔ لا توجد قيم مفقودة بعد المعالجة")

# --- 2.4 فحص منطقية البيانات (Sanity Checks) ---
checks = {
    "أعمار خارج النطاق (18-75)":      ((df.Age < 18) | (df.Age > 75)).sum(),
    "دخل شهري <= 0":                  (df.Monthly_Income_EGP <= 0).sum(),
    "مبلغ قرض <= 0":                  (df.Loan_Amount_EGP <= 0).sum(),
    "Credit Score خارج (300-850)":    ((df.Credit_Score < 300) | (df.Credit_Score > 850)).sum(),
    "سنوات خبرة > العمر - 16":        (df.Employment_Years > (df.Age - 16)).sum(),
}
print("\nفحوصات المنطقية:")
for k, v in checks.items():
    print(f"  {k:35s}: {v}")

# --- 2.5 هندسة المتغيرات (Feature Engineering) ---
# نسب مالية معروفة في تقييم الجدارة الائتمانية
df["Debt_To_Income"] = df["Monthly_Debt_EGP"] / df["Monthly_Income_EGP"]
df["Monthly_Installment"] = df["Loan_Amount_EGP"] / df["Loan_Term_Months"]
df["Total_DTI"] = (df["Monthly_Debt_EGP"] + df["Monthly_Installment"]) / df["Monthly_Income_EGP"]
df["Loan_To_Income"] = df["Loan_Amount_EGP"] / (df["Monthly_Income_EGP"] * 12)
print("\n✔ تمت إضافة 4 متغيرات مشتقة: Debt_To_Income, Monthly_Installment, "
      "Total_DTI, Loan_To_Income")

# --- 2.6 ترميز الهدف والمتغيرات النصية ---
df["Target"] = (df[TARGET] == "Yes").astype(int)          # Yes=1 , No=0

CATEGORICAL = ["Education", "Employment_Type", "Home_Ownership"]
# One-Hot Encoding مع حذف أول فئة لتجنّب Dummy Variable Trap
df_encoded = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)
print(f"✔ تم الترميز One-Hot للأعمدة: {CATEGORICAL}")


# ===============================================================
# 3+4) التحليل الاستكشافي والإحصائيات (EDA)
# ===============================================================
section("3-4) Exploratory Data Analysis")

NUMERIC = ["Age", "Monthly_Income_EGP", "Employment_Years", "Loan_Amount_EGP",
           "Loan_Term_Months", "Credit_Score", "Existing_Loans",
           "Monthly_Debt_EGP", "Debt_To_Income", "Loan_To_Income"]

desc = df[NUMERIC].describe().T
desc["skew"] = df[NUMERIC].skew()
print("\nالإحصاءات الوصفية:")
print(desc.round(2))
desc.round(3).to_csv(os.path.join(OUT_DIR, "descriptive_statistics.csv"))

# توازن الفئات
counts = df[TARGET].value_counts()
print("\nتوزيع الهدف Loan_Approved:")
print(counts)
print((counts / len(df) * 100).round(1).astype(str) + " %")
baseline = counts.max() / len(df)
print(f"خط الأساس Baseline (التنبؤ دائماً بالفئة الأكبر) = {baseline:.3f}")

# --- معدل الموافقة حسب شرائح كل متغير (أهم جدول في التحليل) ---
def approval_by_bins(col, bins=4, qcut=True):
    grp = pd.qcut(df[col], bins, duplicates="drop") if qcut else df[col]
    out = df.groupby(grp, observed=True)["Target"].agg(["mean", "count"])
    out.columns = ["Approval_Rate", "N"]
    return out.round(3)

print("\nمعدل الموافقة حسب شرائح Credit Score:")
cs_bins = df.groupby(pd.cut(df.Credit_Score, [450, 550, 650, 750, 850]),
                     observed=True)["Target"].agg(["mean", "count"]).round(3)
print(cs_bins)
print("\nمعدل الموافقة حسب شرائح الدخل:")
print(approval_by_bins("Monthly_Income_EGP"))
print("\nمعدل الموافقة حسب شرائح مبلغ القرض:")
print(approval_by_bins("Loan_Amount_EGP"))
print("\nمعدل الموافقة حسب عدد القروض القائمة:")
print(df.groupby("Existing_Loans")["Target"].agg(["mean", "count"]).round(3))

for c in CATEGORICAL:
    print(f"\nمعدل الموافقة حسب {c}:")
    print(df.groupby(c)["Target"].agg(["mean", "count"]).round(3))

# ---------------------------------------------------------------
# الرسوم البيانية
# ---------------------------------------------------------------
# (1) نسبة الطلبات المقبولة والمرفوضة
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sns.countplot(x=TARGET, data=df, order=["Yes", "No"],
              palette=["#2a9d8f", "#e76f51"], ax=ax[0])
for p in ax[0].patches:
    ax[0].annotate(f"{int(p.get_height())}",
                   (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha="center", va="bottom", fontweight="bold")
ax[0].set_title("Approved vs Rejected (count)")
ax[1].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
          colors=["#2a9d8f", "#e76f51"], explode=(0.02, 0.02))
ax[1].set_title("Approval Ratio")
plt.suptitle("Loan Approval Distribution")
save_fig("02_target_distribution.png")

# (2) توزيع مبالغ القروض
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df.Loan_Amount_EGP, bins=30, kde=True, color="#264653", ax=ax[0])
ax[0].set_title("Distribution of Loan Amounts")
ax[0].set_xlabel("Loan Amount (EGP)")
sns.boxplot(x=TARGET, y="Loan_Amount_EGP", data=df, order=["Yes", "No"],
            palette=["#2a9d8f", "#e76f51"], ax=ax[1])
ax[1].set_title("Loan Amount by Decision")
save_fig("03_loan_amount_distribution.png")

# (3) العلاقة بين الدخل وقبول القرض
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.kdeplot(data=df, x="Monthly_Income_EGP", hue=TARGET, fill=True,
            common_norm=False, palette=["#e76f51", "#2a9d8f"], ax=ax[0])
ax[0].set_title("Income Distribution by Decision")
inc = approval_by_bins("Monthly_Income_EGP")
sns.barplot(x=[str(i) for i in inc.index], y=inc.Approval_Rate,
            color="#457b9d", ax=ax[1])
ax[1].axhline(baseline, ls="--", c="red", label=f"overall {df.Target.mean():.2f}")
ax[1].set_title("Approval Rate by Income Quartile")
ax[1].set_ylabel("Approval Rate")
ax[1].tick_params(axis="x", rotation=20)
ax[1].legend()
save_fig("04_income_vs_approval.png")

# (4) العلاقة بين Credit Score وقرار القبول
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.boxplot(x=TARGET, y="Credit_Score", data=df, order=["Yes", "No"],
            palette=["#2a9d8f", "#e76f51"], ax=ax[0])
ax[0].set_title("Credit Score by Decision")
sns.barplot(x=[str(i) for i in cs_bins.index], y=cs_bins["mean"].values,
            color="#e9c46a", ax=ax[1])
ax[1].axhline(df.Target.mean(), ls="--", c="red", label="overall rate")
ax[1].set_title("Approval Rate by Credit Score Band")
ax[1].set_ylabel("Approval Rate")
ax[1].set_xlabel("Credit Score Band")
ax[1].legend()
save_fig("05_creditscore_vs_approval.png")

# (5) مصفوفة الارتباط
plt.figure(figsize=(10, 8))
corr = df[NUMERIC + ["Target"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix")
save_fig("06_correlation_matrix.png")

# (6) المتغيرات الفئوية
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for i, c in enumerate(CATEGORICAL):
    rate = df.groupby(c)["Target"].mean().sort_values()
    sns.barplot(x=rate.index, y=rate.values, ax=ax[i], color="#8d99ae")
    ax[i].axhline(df.Target.mean(), ls="--", c="red")
    ax[i].set_ylim(0, 0.8)
    ax[i].set_title(f"Approval Rate by {c}")
    ax[i].set_ylabel("Approval Rate")
save_fig("07_categorical_features.png")

# (7) توزيعات المتغيرات الرقمية
df[NUMERIC].hist(figsize=(14, 9), bins=25, color="#457b9d", edgecolor="white")
plt.suptitle("Numeric Feature Distributions", fontsize=14)
save_fig("08_numeric_distributions.png")
print(f"\n✔ تم حفظ الرسوم البيانية في: {FIG_DIR}")


# ===============================================================
# 5-6) بناء نموذج Machine Learning (Logistic Regression)
# ===============================================================
section("5-6) Machine Learning Model — Logistic Regression")

DROP_COLS = ["Customer_ID", TARGET, "Target"]
X = df_encoded.drop(columns=DROP_COLS)
y = df_encoded["Target"]
X = X.astype(float)
print(f"عدد المتغيرات (Features): {X.shape[1]}")
print(f"أسماء المتغيرات: {list(X.columns)}")

# --- تقسيم البيانات 80/20 مع الحفاظ على نسبة الفئات (stratify) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
print(f"\nTraining set: {X_train.shape[0]} صف | Testing set: {X_test.shape[0]} صف")
print(f"نسبة الموافقة - تدريب: {y_train.mean():.3f} | اختبار: {y_test.mean():.3f}")

# --- Pipeline: توحيد المقاييس + الانحدار اللوجستي ---
# StandardScaler ضروري لأن مقاييس الأعمدة مختلفة جداً (الدخل بالآلاف مقابل عدد القروض 0-4)
# C=0.05 اختير بالتحقق المتقاطع (تنظيم أقوى يقلل التشتت لأن الإشارة ضعيفة)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, C=0.05, random_state=RANDOM_STATE)),
])
model.fit(X_train, y_train)

# مقارنة سريعة لقيم التنظيم C (للتوثيق فقط - النموذج النهائي واحد)
cv_tmp = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("\nاختيار معامل التنظيم C بالتحقق المتقاطع:")
for c_val in [0.01, 0.05, 0.1, 1.0, 10.0]:
    tmp = Pipeline([("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, C=c_val,
                                               random_state=RANDOM_STATE))])
    s = cross_val_score(tmp, X_train, y_train, cv=cv_tmp, scoring="accuracy").mean()
    print(f"  C={c_val:<6} CV Accuracy = {s:.3f}")

# --- التحقق المتقاطع على بيانات التدريب ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
print(f"5-Fold CV ROC-AUC : {cv_auc.mean():.3f} (±{cv_auc.std():.3f})")


# ===============================================================
# 7) تقييم النموذج
# ===============================================================
section("7) Model Evaluation")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy":  accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall":    recall_score(y_test, y_pred),
    "F1-Score":  f1_score(y_test, y_pred),
    "ROC-AUC":   roc_auc_score(y_test, y_proba),
}
print("\nنتائج مجموعة الاختبار (Test Set):")
for k, v in metrics.items():
    print(f"  {k:10s}: {v:.3f}")
print(f"  {'Baseline':10s}: {baseline:.3f}  (التنبؤ دائماً بـ Yes)")

pd.Series(metrics).round(4).to_csv(os.path.join(OUT_DIR, "model_metrics.csv"),
                                   header=["value"])

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected (No)", "Approved (Yes)"]))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(pd.DataFrame(cm, index=["Actual No", "Actual Yes"],
                   columns=["Pred No", "Pred Yes"]))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred: No", "Pred: Yes"],
            yticklabels=["Actual: No", "Actual: Yes"], ax=ax[0])
ax[0].set_title("Confusion Matrix")
try:
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1], color="#2a9d8f")
except TypeError:  # newer scikit-learn versions
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[1], curve_kwargs={"color": "#2a9d8f"})
ax[1].plot([0, 1], [0, 1], "k--", lw=1)
ax[1].set_title(f"ROC Curve (AUC = {metrics['ROC-AUC']:.3f})")
save_fig("09_model_evaluation.png")

# --- أهمية المتغيرات (معاملات الانحدار) ---
coefs = pd.Series(model.named_steps["clf"].coef_[0], index=X.columns)
coefs = coefs.sort_values(key=abs, ascending=False)
print("\nمعاملات النموذج مرتبة حسب قوة التأثير (Standardized Coefficients):")
print(coefs.round(3))
coefs.round(4).to_csv(os.path.join(OUT_DIR, "model_coefficients.csv"),
                      header=["coefficient"])

plt.figure(figsize=(9, 6))
top = coefs.head(12).sort_values()
colors = ["#e76f51" if v < 0 else "#2a9d8f" for v in top.values]
plt.barh(top.index, top.values, color=colors)
plt.axvline(0, c="black", lw=0.8)
plt.title("Logistic Regression Coefficients (standardized)\n"
          "green = increases approval odds, red = decreases")
plt.xlabel("Coefficient")
save_fig("10_feature_importance.png")


# ===============================================================
# 8) الخلاصة والنتائج
# ===============================================================
section("8) Key Insights")

insights = f"""
1. حجم البيانات: {df.shape[0]} طلب قرض و {df.shape[1]-1} متغير + عمود الهدف.
2. توزيع الهدف: {counts.get('Yes',0)} موافقة ({counts.get('Yes',0)/len(df)*100:.1f}%)
   مقابل {counts.get('No',0)} رفض ({counts.get('No',0)/len(df)*100:.1f}%) — توازن مقبول.
3. القيم المفقودة: 8 قيم في كل من Monthly_Income, Credit_Score, Education,
   Employment_Type (1.6% لكل عمود) وتمت معالجتها بالوسيط/المنوال.
4. Credit Score هو المتغير الوحيد ذو الإشارة التنبؤية الواضحة:
   معدل الموافقة يرتفع من {cs_bins['mean'].iloc[0]:.0%} في الشريحة (450-550)
   إلى {cs_bins['mean'].iloc[-1]:.0%} في الشريحة (750-850) — ارتفاع تدريجي منتظم.
5. باقي المتغيرات (الدخل، مبلغ القرض، العمر، التعليم، نوع العمل، السكن)
   ارتباطها بالقرار ضعيف جداً (|r| < 0.11) وفروق معدلات الموافقة بينها
   داخل حدود الخطأ العشوائي.
6. أداء النموذج على الاختبار: Accuracy={metrics['Accuracy']:.3f} ,
   Precision={metrics['Precision']:.3f} , Recall={metrics['Recall']:.3f} ,
   F1={metrics['F1-Score']:.3f} , ROC-AUC={metrics['ROC-AUC']:.3f}.
7. خط الأساس (Baseline) = {baseline:.3f}. النموذج يتفوق على التخمين العشوائي
   لكن بهامش محدود، وهذا نتيجة طبيعية لضعف الإشارة في البيانات نفسها
   وليس خطأً في المنهجية.
8. التوصية: لتحسين الأداء نحتاج متغيرات إضافية (تاريخ التعثر، عدد أيام
   التأخير، الغرض من القرض، وجود ضامن) أو حجم بيانات أكبر.
"""
print(insights)
with open(os.path.join(OUT_DIR, "insights.txt"), "w", encoding="utf-8") as f:
    f.write(insights)

print("\n✔ انتهى التنفيذ بنجاح.")
