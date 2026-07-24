import pandas as pd
from sklearn.model_selection import train_test_split

# Modele
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
import lightgbm as lgb

# Import parametrów
from parameter import PARAMS
# Import modułu walidacji
from validation import evaluate_model

def main():
    print("You are in training module")

def custom_train_model(data):
    print("Custom training started on data:")
    print(data.head())

    df = data.copy()

    # Feature engineering
    df["diff_TO"] = abs(df["TO_sup"] - df["TO_buyer"])
    df["diff_NPE"] = abs(df["NPE_sup"] - df["NPE_buyer"])
    df["same_sector"] = (df["NACE_SEC_sup"] == df["NACE_SEC_buyer"]).astype(int)
    df["same_region"] = (df["NUTS3_2024_sup"] == df["NUTS3_2024_buyer"]).astype(int)
    df["diff_WAGES"] = abs(df["WAGES_sup"] - df["WAGES_buyer"])

    df['LINKED'] = pd.to_numeric(df['LINKED'].fillna(0)).clip(0, 1)

    features = ["diff_TO", "diff_NPE", "same_sector", "same_region", "diff_WAGES"]
    X = df[features].copy()
    y = df["LINKED"]

    # Convert object columns in X to numeric
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PARAMS["test_size"], random_state=PARAMS["random_seed"]
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

