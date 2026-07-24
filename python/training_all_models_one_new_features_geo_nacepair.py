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

# Import parameters
from parameter import PARAMS
# Import evaluation module
from validation import evaluate_model


def main():
    print("You are in training module")


def custom_train_model(data):
    print("Custom training started on data:")
    print(data.head())

    df = data.copy()

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    # -----------------------------------------------------
    # Baseline features
    # -----------------------------------------------------

    df["diff_TO"] = abs(df["TO_sup"] - df["TO_buyer"])
    df["diff_NPE"] = abs(df["NPE_sup"] - df["NPE_buyer"])
    df["same_sector"] = (df["NACE_SEC_sup"] == df["NACE_SEC_buyer"]).astype(int)
    df["same_region"] = (df["NUTS3_2024_sup"] == df["NUTS3_2024_buyer"]).astype(int)
    df["diff_WAGES"] = abs(df["WAGES_sup"] - df["WAGES_buyer"])

    # -----------------------------------------------------
    # Target variable
    # -----------------------------------------------------

    df["LINKED"] = pd.to_numeric(df["LINKED"].fillna(0)).clip(0, 1)




    # -----------------------------------------------------
    # Geographical features (gravity model)
    # NEW - added after introducing municipality distances
    # -----------------------------------------------------

    df["log_ROAD_DST"] = np.log1p(df["ROAD_DST"].fillna(df["ROAD_DST"].median()))

    df["log_LINE_DST"] = np.log1p(df["LINE_DST"].fillna(df["LINE_DST"].median()))

    df["same_municipality"] = (
        df["MN_ORIGIN"] == df["MN_DESTINY"]
    ).astype(int)

    df["distance_ratio"] = (
        df["ROAD_DST"].fillna(df["ROAD_DST"].median()) /
        (df["LINE_DST"].fillna(df["LINE_DST"].median()) + 1)
    )




    # -----------------------------------------------------
    # NEW - NACE SECTOR PAIR ENCODING
    # Supplier sector + Buyer sector interaction
    # Example: C_G, C_M, A_F
    # -----------------------------------------------------

    df["NACE_PAIR"] = (
        df["NACE_SEC_S"].astype(str)
        + "_"
        + df["NACE_SEC_B"].astype(str)
    )


    # Numeric encoding of sector pair
    df["NACE_PAIR_encoded"] = (
        df["NACE_PAIR"]
        .astype("category")
        .cat.codes
    )


    # Frequency of sector pair
    df["NACE_PAIR_frequency"] = (
        df["NACE_PAIR"]
        .map(df["NACE_PAIR"].value_counts(normalize=True))
    )









    # -----------------------------------------------------
    # Structural differences
    # -----------------------------------------------------

    df["diff_ACT"] = abs(df["ACT_sup"] - df["ACT_buyer"])
    df["diff_PURCH"] = abs(df["PURCH_sup"] - df["PURCH_buyer"])

    # -----------------------------------------------------
    # Firm size asymmetry
    # -----------------------------------------------------

    df["supplier_larger_TO"] = (df["TO_sup"] > df["TO_buyer"]).astype(int)
    df["supplier_larger_NPE"] = (df["NPE_sup"] > df["NPE_buyer"]).astype(int)
    df["supplier_larger_ACT"] = (df["ACT_sup"] > df["ACT_buyer"]).astype(int)

    # -----------------------------------------------------
    # DIM similarity
    # -----------------------------------------------------

    df["same_DIM"] = (df["DIM_sup"] == df["DIM_buyer"]).astype(int)


	

    # -----------------------------------------------------
    # Log transformations
    # -----------------------------------------------------

    df["log_TO_sup"] = np.log1p(df["TO_sup"])
    df["log_TO_buyer"] = np.log1p(df["TO_buyer"])

    df["log_NPE_sup"] = np.log1p(df["NPE_sup"])
    df["log_NPE_buyer"] = np.log1p(df["NPE_buyer"])

    df["log_ACT_sup"] = np.log1p(df["ACT_sup"])
    df["log_ACT_buyer"] = np.log1p(df["ACT_buyer"])

    df["log_PURCH_sup"] = np.log1p(df["PURCH_sup"])
    df["log_PURCH_buyer"] = np.log1p(df["PURCH_buyer"])

    # -----------------------------------------------------
    # Financial intensity
    # -----------------------------------------------------

    df["purch_to_to_sup"] = df["PURCH_sup"] / (df["TO_sup"] + 1)
    df["purch_to_to_buyer"] = df["PURCH_buyer"] / (df["TO_buyer"] + 1)

    df["wages_to_to_sup"] = df["WAGES_sup"] / (df["TO_sup"] + 1)
    df["wages_to_to_buyer"] = df["WAGES_buyer"] / (df["TO_buyer"] + 1)

    # -----------------------------------------------------
    # Productivity
    # -----------------------------------------------------

    df["to_per_emp_sup"] = df["TO_sup"] / (df["NPE_sup"] + 1)
    df["to_per_emp_buyer"] = df["TO_buyer"] / (df["NPE_buyer"] + 1)

    df["act_per_emp_sup"] = df["ACT_sup"] / (df["NPE_sup"] + 1)
    df["act_per_emp_buyer"] = df["ACT_buyer"] / (df["NPE_buyer"] + 1)

    # -----------------------------------------------------
    # Economic matching features
    # -----------------------------------------------------

    df["trade_potential"] = (
        np.log1p(df["TO_sup"]) *
        np.log1p(df["PURCH_buyer"])
    )

    df["economic_similarity"] = abs(
        np.log1p(df["TO_sup"]) - np.log1p(df["TO_buyer"])
    )

    # -----------------------------------------------------
    # Ratios
    # -----------------------------------------------------

    df["TO_ratio"] = df["TO_sup"] / (df["TO_buyer"] + 1)
    df["NPE_ratio"] = df["NPE_sup"] / (df["NPE_buyer"] + 1)
    df["ACT_ratio"] = df["ACT_sup"] / (df["ACT_buyer"] + 1)

    # -----------------------------------------------------
    # Sector interaction
    # -----------------------------------------------------

    ###df["S2S_feature"] = df["S2S"]

    # =====================================================
    # FEATURE LIST
    # =====================================================

    features = [
        # baseline
        "diff_TO",
        "diff_NPE",
        "same_sector",
        "same_region",
        "diff_WAGES",



	# geography
        # NEW - municipality distance features
        "log_ROAD_DST",
        "log_LINE_DST",
        "same_municipality",
        "distance_ratio",


        # -------------------------------------------------
        # NEW - NACE sector pair features
        # -------------------------------------------------

        "NACE_PAIR_encoded",
        "NACE_PAIR_frequency",








        # structure
        "diff_ACT",
        "diff_PURCH",

        # asymmetry
        "supplier_larger_TO",
        "supplier_larger_NPE",
        "supplier_larger_ACT",
        "same_DIM",

        # logs
        "log_TO_sup",
        "log_TO_buyer",
        "log_NPE_sup",
        "log_NPE_buyer",
        "log_ACT_sup",
        "log_ACT_buyer",
        "log_PURCH_sup",
        "log_PURCH_buyer",

        # intensity
        "purch_to_to_sup",
        "purch_to_to_buyer",
        "wages_to_to_sup",
        "wages_to_to_buyer",

        # productivity
        "to_per_emp_sup",
        "to_per_emp_buyer",
        "act_per_emp_sup",
        "act_per_emp_buyer",

        # economic matching
        "trade_potential",
        "economic_similarity",

        # ratios
        "TO_ratio",
        "NPE_ratio",
        "ACT_ratio",

        # sector
        ###"S2S_feature"
    ]

    if "diff_DIM" in df.columns:
        features.append("diff_DIM")

    X = df[features].copy()
    y = df["LINKED"]

    # Convert categorical columns to numeric
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category").cat.codes

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PARAMS["test_size"],
        random_state=PARAMS["random_seed"]
    )

# Definicja modeli dynamicznie z PARAMS
    model_instances = {
        "K-Nearest Neighbors": KNeighborsClassifier(**PARAMS["model_params"]["K-Nearest Neighbors"]),
        "Naive Bayes": GaussianNB(**PARAMS["model_params"]["Naive Bayes"]),
        "Decision Tree": DecisionTreeClassifier(**PARAMS["model_params"]["Decision Tree"]),
        "Extra Trees": ExtraTreesClassifier(**PARAMS["model_params"]["Extra Trees"]),
        "Logistic Regression": LogisticRegression(**PARAMS["model_params"]["Logistic Regression"]),
        "MLP Classifier": MLPClassifier(**PARAMS["model_params"]["MLP Classifier"]),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(**PARAMS["model_params"]["Quadratic Discriminant Analysis"]),
        "Histogram-Based Gradient Boosting": HistGradientBoostingClassifier(**PARAMS["model_params"]["Histogram-Based Gradient Boosting"]),
        "XGBoost Classifier": xgb.XGBClassifier(**PARAMS["model_params"]["XGBoost Classifier"]),
        "LightGBM": lgb.LGBMClassifier(**PARAMS["model_params"]["LightGBM"]),
        "AdaBoost": AdaBoostClassifier(**PARAMS["model_params"]["AdaBoost"]),
        "Random Forest": RandomForestClassifier(**PARAMS["model_params"]["Random Forest"]),
    }

    all_results = []

    for name in PARAMS["models_to_train"]:
        model = model_instances[name]
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)

        # Walidacja i zapis wyników
        model_results = evaluate_model(
            model, X_test, y_test, name,
            models_folder=PARAMS["models_folder"],
            roc_folder=PARAMS["roc_folder"]
        )
        all_results.extend(model_results)

    # Zapis wszystkich wyników do CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(PARAMS["results_csv_path"], index=False)
    print(f"All results saved to {PARAMS['results_csv_path']}")

    print("Custom training done.")
    return y_test, model.predict(X_test)

# -------------------------------
# Uruchomienie modułu
# -------------------------------
if __name__ == "__main__":
    main()
    print("Module training started")

