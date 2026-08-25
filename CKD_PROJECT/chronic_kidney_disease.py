from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import shap
import dice_ml
from dice_ml import Dice
import joblib

# 1. Veri setini tanımak ve incelemek
chronic_kidney_disease = fetch_ucirepo(id=336) 
X = chronic_kidney_disease.data.features 
y = chronic_kidney_disease.data.targets 
df = pd.concat([X, y], axis=1)
  
kategorik_sutunlar = df.select_dtypes(include=[object]).columns.tolist()
for col in kategorik_sutunlar:
    df[col] = df[col].str.strip()

# 2. Ön İşleme 
X = df.drop(columns=["class", "class_num"], errors='ignore')
y = df["class"].map({"ckd": 1, "notckd": 0})

# İkili değişkenleri sayıya çevirme
binary_map = {
    "rbc": {"normal":1, "abnormal":0}, "pc":  {"normal":1, "abnormal":0},
    "pcc": {"present":1, "notpresent":0}, "ba":  {"present":1, "notpresent":0},
    "htn": {"yes":1, "no":0}, "dm":  {"yes":1, "no":0}, "cad": {"yes":1, "no":0},
    "appet": {"good":1, "poor":0}, "pe":  {"yes":1, "no":0}, "ane": {"yes":1, "no":0}
}
for col, mapping in binary_map.items():
    if col in X.columns:
        X[col] = X[col].map(mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

sayisal_sutunlar = X_train.select_dtypes(include=[np.number]).columns.tolist()
for col in sayisal_sutunlar:
    median_val = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# 3. Model Eğitimi (XGBoost)
model = XGBClassifier(random_state=42, eval_metric="logloss")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

# 4. SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 5. DiCE
gercek_surekli_degiskenler = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc', 'sg', 'al', 'su']

train_df = X_train.copy()
train_df["class"] = y_train.values

d = dice_ml.Data(dataframe=train_df, continuous_features=gercek_surekli_degiskenler, outcome_name="class")
m = dice_ml.Model(model=model, backend="sklearn")
exp = Dice(d, m, method="genetic")

dar_permitted_range = {
    'sc': [0.4, 6.0], 'hemo': [7.0, 17.0], 'pcv': [22, 52],        
    'bu': [10, 90], 'al': [0, 4], 'su': [0, 5],
    'bgr': [70, 140], 'sod': [135, 145], 'pot': [3.5, 5.2],
    'wbcc': [4000, 11000], 'bp': [60, 90]
}

degistirilemez_ozellikler = ['age', 'htn', 'dm', 'cad']  
features_to_vary = [col for col in X_train.columns if col not in degistirilemez_ozellikler]

# 6. Dışa Aktarım (Streamlit için)
joblib.dump(model, "ckd_model.pkl")
joblib.dump(X_train.columns.tolist(), "columns.pkl")
joblib.dump(dar_permitted_range, "permitted_range.pkl")
joblib.dump(features_to_vary, "features_to_vary.pkl")
joblib.dump(explainer, 'shap_explainer.pkl')
train_df.to_csv("train_data.csv", index=False)
X_test.to_csv("test_data.csv", index=False)
print("Sızıntısız Model Kaydedildi!")