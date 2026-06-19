from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc
)
import json

app = FastAPI(title="Campus Placement API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files ────────────────────────────────────────────────────
@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.get("/styles.css")
async def read_css():
    return FileResponse("styles.css")

@app.get("/app.js")
async def read_js():
    return FileResponse("app.js")

# ── Data Loading & Preprocessing ────────────────────────────────────
def load_and_preprocess():
    df = pd.read_csv("placement.csv")
    df_raw = df.copy()
    
    # Processed copy
    df_proc = df.copy()
    df_proc.drop(columns=["sl_no"], inplace=True)
    df_proc["salary"] = df_proc["salary"].fillna(0)
    
    le = LabelEncoder()
    categorical_cols = ["ssc_b", "hsc_b", "hsc_s", "degree_t", "specialisation"]
    for col in categorical_cols:
        df_proc[col] = le.fit_transform(df_proc[col])
        
    df_proc["gender"] = df_proc["gender"].map({"M": 0, "F": 1})
    df_proc["workex"] = df_proc["workex"].map({"No": 0, "Yes": 1})
    df_proc["status"] = df_proc["status"].map({"Not Placed": 0, "Placed": 1})
    
    FEATURES = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p",
                "gender", "workex", "specialisation", "hsc_s", "degree_t"]
    
    return df_raw, df_proc, FEATURES

def train_all_models(df_proc, FEATURES):
    X = df_proc[FEATURES].values
    y = df_proc["status"].values
    
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr)
    Xte_sc = sc.transform(Xte)
    
    clfs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=7),
    }
    
    results = {}
    for name, clf in clfs.items():
        clf.fit(Xtr_sc, ytr)
        yp = clf.predict(Xte_sc)
        yprob = clf.predict_proba(Xte_sc)[:, 1]
        
        # ROC Curve data
        fpr, tpr, _ = roc_curve(yte, yprob)
        
        results[name] = {
            "model": clf,
            "metrics": {
                "accuracy": float(accuracy_score(yte, yp)),
                "precision": float(precision_score(yte, yp)),
                "recall": float(recall_score(yte, yp)),
                "f1": float(f1_score(yte, yp)),
                "roc_auc": float(roc_auc_score(yte, yprob)),
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc(fpr, tpr))
            },
            "confusion_matrix": confusion_matrix(yte, yp).tolist()
        }
    
    return results, sc

# Global State (Initialized on startup)
data_raw, data_proc, FEATURES = load_and_preprocess()
model_results, global_scaler = train_all_models(data_proc, FEATURES)
best_model_name = max(model_results, key=lambda n: model_results[n]["metrics"]["f1"])

# ── Pydantic Models ──────────────────────────────────────────────────
class PredictInput(BaseModel):
    ssc_p: float
    hsc_p: float
    degree_p: float
    etest_p: float
    mba_p: float
    gender: str  # "Male" / "Female"
    workex: str  # "Yes" / "No"
    specialisation: str  # "Mkt&HR" / "Mkt&Fin"
    hsc_s: str  # "Commerce" / "Science" / "Arts"
    degree_t: str  # "Comm&Mgmt" / "Sci&Tech" / "Others"

# ── API Endpoints ────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    total = len(data_raw)
    placed = int(data_raw["status"].value_counts().get("Placed", 0))
    return {
        "total_students": total,
        "placed_count": placed,
        "not_placed_count": total - placed,
        "placement_rate": round(placed / total * 100, 1),
        "best_model": {
            "name": best_model_name,
            "accuracy": round(model_results[best_model_name]["metrics"]["accuracy"] * 100, 1),
            "f1": round(model_results[best_model_name]["metrics"]["f1"], 3)
        }
    }

@app.get("/api/charts")
async def get_charts():
    # 1. Placement Distribution
    dist = data_raw["status"].value_counts().to_dict()
    
    # 2. Placement by Gender
    gender_placed = data_raw.groupby(["gender", "status"]).size().unstack(fill_value=0).to_dict('index')
    
    # 3. Placement by Work Experience
    workex_placed = data_raw.groupby(["workex", "status"]).size().unstack(fill_value=0).to_dict('index')
    
    # 4. Feature Importance (Random Forest)
    rf_model = model_results["Random Forest"]["model"]
    importances = rf_model.feature_importances_.tolist()
    feat_importance = [{"feature": f, "importance": i} for f, i in zip(FEATURES, importances)]
    feat_importance = sorted(feat_importance, key=lambda x: x["importance"], reverse=True)
    
    return {
        "placement_distribution": dist,
        "gender_distribution": gender_placed,
        "workex_distribution": workex_placed,
        "feature_importance": feat_importance
    }

@app.get("/api/models")
async def get_models():
    output = {}
    for name, data in model_results.items():
        output[name] = {
            "metrics": data["metrics"],
            "roc_curve": data["roc_curve"],
            "confusion_matrix": data["confusion_matrix"]
        }
    return output

@app.post("/api/predict")
async def predict(input_data: PredictInput):
    hsc_s_map = {"Arts": 0, "Commerce": 1, "Science": 2}
    degree_t_map = {"Comm&Mgmt": 0, "Others": 1, "Sci&Tech": 2}
    spec_map = {"Mkt&Fin": 0, "Mkt&HR": 1}
    
    # Map input to numeric
    try:
        vec = [
            input_data.ssc_p,
            input_data.hsc_p,
            input_data.degree_p,
            input_data.etest_p,
            input_data.mba_p,
            1 if input_data.gender == "Female" else 0,
            1 if input_data.workex == "Yes" else 0,
            spec_map[input_data.specialisation],
            hsc_s_map[input_data.hsc_s],
            degree_t_map[input_data.degree_t]
        ]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid categorical value: {str(e)}")
    
    # Scale
    inp_sc = global_scaler.transform([vec])
    
    # Predict with best model
    clf = model_results[best_model_name]["model"]
    pred = int(clf.predict(inp_sc)[0])
    probs = clf.predict_proba(inp_sc)[0].tolist()
    
    return {
        "prediction": "Placed" if pred == 1 else "Not Placed",
        "confidence": round(probs[pred] * 100, 1),
        "probabilities": {
            "Not Placed": round(probs[0] * 100, 1),
            "Placed": round(probs[1] * 100, 1)
        },
        "model_used": best_model_name
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
