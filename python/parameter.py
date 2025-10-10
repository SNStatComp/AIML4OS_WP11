# parameter.py

import os

print("Module 'parameter' started")
print("Initializing parameters ...")

PARAMS = {
    # -------------------------------
    # general settings
    # -------------------------------
    "random_seed": 42,
    "test_size": 0.2,

    # -------------------------------
    # input_data
    # -------------------------------
    "data_path_companies": "wp11_companies_synthetic_data.parquet",
    "data_path_links": "wp11_b2b_syntetic_data_linked_1_only_rows.parquet",
    "balance_strategy": "undersample_positives",  # lub "oversample_negatives"

    # -------------------------------
    # models
    # -------------------------------
    "models_to_train": [
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Decision Tree",
        "Extra Trees",
        "Logistic Regression",
        "MLP Classifier",
        "Quadratic Discriminant Analysis",
        "Histogram-Based Gradient Boosting",
        "XGBoost Classifier",
        "LightGBM",
        "AdaBoost",
        "Random Forest"
    ],

    "model_params": {
        "K-Nearest Neighbors": {"n_neighbors": 9, "weights": "uniform", "metric": "manhattan"},
        "Naive Bayes": {},
        "Decision Tree": {"max_depth": 3, "min_samples_leaf": 1, "min_samples_split": 2, "random_state": 42},
        "Extra Trees": {"max_depth": 10, "min_samples_split": 2, "n_estimators": 50, "random_state": 42},
        "Logistic Regression": {"C": 0.01, "solver": "liblinear", "max_iter": 1000, "random_state": 42},
        "MLP Classifier": {"activation": "relu", "alpha": 0.0001, "hidden_layer_sizes": (50,), "max_iter": 500, "random_state": 42},
        "Quadratic Discriminant Analysis": {"reg_param": 0.0},
        "Histogram-Based Gradient Boosting": {"learning_rate": 0.1, "max_iter": 200, "max_depth": None, "random_state": 42},
        "XGBoost Classifier": {"learning_rate": 0.2, "max_depth": 7, "n_estimators": 100, "subsample": 0.7,
                               "use_label_encoder": False, "eval_metric": "logloss", "random_state": 42},
        "LightGBM": {"learning_rate": 0.1, "max_depth": 10, "n_estimators": 200, "num_leaves": 50, "random_state": 42},
        "AdaBoost": {"learning_rate": 0.01, "n_estimators": 50, "random_state": 42},
        "Random Forest": {"max_depth": 5, "min_samples_split": 2, "n_estimators": 50, "random_state": 42}
    },

    # -------------------------------
    # saving_paths
    # -------------------------------
    "models_folder": "models/",
    "roc_folder": "roc_curves/",
    "results_csv_path": "training_results.csv",

    # -------------------------------
    # validation
    # -------------------------------
    "validation_metrics": ["accuracy", "precision", "recall", "f1-score"],
    "detailed_report": True,
    "save_validation_report": True,
    "validation_report_path": "results/validation_report.txt"
}

# -------------------------------
# initialize folder if they dont exist
# -------------------------------
def initialize_folders():
    folders = [
        PARAMS["models_folder"],
        PARAMS["roc_folder"],
        os.path.dirname(PARAMS["validation_report_path"])
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Folder checked/created: {folder}")

# folder init
initialize_folders()


print (PARAMS)
print("Parameters initialized successfully.")
print("Module 'parameter' done")

def main():
    print("You are in parameter module")
