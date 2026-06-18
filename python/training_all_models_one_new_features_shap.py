import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
import lightgbm as lgb

# Params
from parameter import PARAMS
from validation import evaluate_model

import shap
import matplotlib.pyplot as plt
import os


# =====================================================
# SHAP EXPLAINER SELECTOR
# =====================================================
def get_shap_explainer(model, X_sample):

    model_name = str(type(model))

    if any(x in model_name for x in ["XGB", "LGB", "Forest", "Tree"]):
        return shap.TreeExplainer(model)

    elif "LogisticRegression" in model_name:
        return shap.LinearExplainer(model, X_sample)

    else:
        return shap.KernelExplainer(model.predict, X_sample)


# =====================================================
# SAFE SHAP PIPELINE
# =====================================================
def run_shap_analysis(model, X_test, model_name, output_dir):

    print(f"\n[SHAP] Starting analysis for model: {model_name}")

    shap_dir = os.path.join(output_dir, "shap")
    os.makedirs(shap_dir, exist_ok=True)

    X_sample = X_test.sample(min(100, len(X_test)), random_state=42)

    explainer = get_shap_explainer(model, X_sample)

    try:
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    except Exception as e:
        print(f"[SHAP ERROR] skipped {model_name}: {e}")
        return None

    shap_importance = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": X_test.columns,
        "shap_importance": shap_importance
    }).sort_values("shap_importance", ascending=False)

    shap_df.to_csv(
        os.path.join(shap_dir, f"shap_importance_{model_name}.csv"),
        index=False
    )

    # Summary plot
    try:
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(os.path.join(shap_dir, f"shap_summary_{model_name}.png"),
                    bbox_inches="tight")
        plt.close()
    except:
        pass

    # Bar plot
    try:
        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig(os.path.join(shap_dir, f"shap_bar_{model_name}.png"),
                    bbox_inches="tight")
        plt.close()
    except:
        pass

    # Dependence plots
    try:
        top_features = shap_df["feature"].head(3).tolist()

        for feat in top_features:
            if feat in X_test.columns:
                plt.figure()
                shap.dependence_plot(feat, shap_values, X_test, show=False)
                plt.savefig(os.path.join(
                    shap_dir,
                    f"shap_dependence_{model_name}_{feat}.png"
                ), bbox_inches="tight")
                plt.close()
    except:
        pass

    print(f"[SHAP] Completed: {model_name}")

    return shap_df


# =====================================================
# MAIN TRAINING FUNCTION
# =====================================================
def custom_train_model(data):

    print("Custom training started")

    df = data.copy()

    # =====================================================
    # FEATURE ENGINEERING (UNCHANGED)
    # =====================================================

    df["diff_TO"] = abs(df["TO_sup"] - df["TO_buyer"])
    df["diff_NPE"] = abs(df["NPE_sup"] - df["NPE_buyer"])
    df["same_sector"] = (df["NACE_SEC_sup"] == df["NACE_SEC_buyer"]).astype(int)
    df["same_region"] = (df["NUTS3_2024_sup"] == df["NUTS3_2024_buyer"]).astype(int)
    df["diff_WAGES"] = abs(df["WAGES_sup"] - df["WAGES_buyer"])

    df["LINKED"] = pd.to_numeric(df["LINKED"].fillna(0)).clip(0, 1)

    df["diff_ACT"] = abs(df["ACT_sup"] - df["ACT_buyer"])
    df["diff_PURCH"] = abs(df["PURCH_sup"] - df["PURCH_buyer"])

    df["supplier_larger_TO"] = (df["TO_sup"] > df["TO_buyer"]).astype(int)
    df["supplier_larger_NPE"] = (df["NPE_sup"] > df["NPE_buyer"]).astype(int)
    df["supplier_larger_ACT"] = (df["ACT_sup"] > df["ACT_buyer"]).astype(int)

    df["same_DIM"] = (df["DIM_sup"] == df["DIM_buyer"]).astype(int)

    df["log_TO_sup"] = np.log1p(df["TO_sup"])
    df["log_TO_buyer"] = np.log1p(df["TO_buyer"])

    df["log_NPE_sup"] = np.log1p(df["NPE_sup"])
    df["log_NPE_buyer"] = np.log1p(df["NPE_buyer"])

    df["log_ACT_sup"] = np.log1p(df["ACT_sup"])
    df["log_ACT_buyer"] = np.log1p(df["ACT_buyer"])

    df["log_PURCH_sup"] = np.log1p(df["PURCH_sup"])
    df["log_PURCH_buyer"] = np.log1p(df["PURCH_buyer"])

    df["purch_to_to_sup"] = df["PURCH_sup"] / (df["TO_sup"] + 1)
    df["purch_to_to_buyer"] = df["PURCH_buyer"] / (df["TO_buyer"] + 1)

    df["wages_to_to_sup"] = df["WAGES_sup"] / (df["TO_sup"] + 1)
    df["wages_to_to_buyer"] = df["WAGES_buyer"] / (df["TO_buyer"] + 1)

    df["to_per_emp_sup"] = df["TO_sup"] / (df["NPE_sup"] + 1)
    df["to_per_emp_buyer"] = df["TO_buyer"] / (df["NPE_buyer"] + 1)

    df["act_per_emp_sup"] = df["ACT_sup"] / (df["NPE_sup"] + 1)
    df["act_per_emp_buyer"] = df["ACT_buyer"] / (df["NPE_buyer"] + 1)

    df["trade_potential"] = np.log1p(df["TO_sup"]) * np.log1p(df["PURCH_buyer"])

    df["economic_similarity"] = abs(
        np.log1p(df["TO_sup"]) - np.log1p(df["TO_buyer"])
    )

    df["TO_ratio"] = df["TO_sup"] / (df["TO_buyer"] + 1)
    df["NPE_ratio"] = df["NPE_sup"] / (df["NPE_buyer"] + 1)
    df["ACT_ratio"] = df["ACT_sup"] / (df["ACT_buyer"] + 1)

    # =====================================================
    # FEATURES
    # =====================================================

    features = [
        "diff_TO", "diff_NPE", "same_sector", "same_region", "diff_WAGES",
        "diff_ACT", "diff_PURCH",
        "supplier_larger_TO", "supplier_larger_NPE", "supplier_larger_ACT",
        "same_DIM",
        "log_TO_sup", "log_TO_buyer",
        "log_NPE_sup", "log_NPE_buyer",
        "log_ACT_sup", "log_ACT_buyer",
        "log_PURCH_sup", "log_PURCH_buyer",
        "purch_to_to_sup", "purch_to_to_buyer",
        "wages_to_to_sup", "wages_to_to_buyer",
        "to_per_emp_sup", "to_per_emp_buyer",
        "act_per_emp_sup", "act_per_emp_buyer",
        "trade_potential",
        "economic_similarity",
        "TO_ratio", "NPE_ratio", "ACT_ratio"
    ]

    X = df[features].copy()
    y = df["LINKED"]

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PARAMS["test_size"],
        random_state=PARAMS["random_seed"]
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = lgb.LGBMClassifier(**PARAMS["model_params"]["LightGBM"])
    model.fit(X_train, y_train)

    # evaluation
    evaluate_model(
        model, X_test, y_test, "LightGBM",
        models_folder=PARAMS["models_folder"],
        roc_folder=PARAMS["roc_folder"]
    )

    # =====================================================
    # SHAP (SAFE)
    # =====================================================

    shap_df = run_shap_analysis(
        model=model,
        X_test=X_test,
        model_name="LightGBM",
        output_dir=PARAMS["models_folder"]
    )

    # =====================================================
    # FINAL SAFETY RETURN (FIX FOR NONE ERROR)
    # =====================================================

    return y_test, model.predict(X_test)
