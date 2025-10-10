import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import pickle

def evaluate_model(model, X_test, y_test, model_name, models_folder="models", roc_folder="roc_curves"):
    """
    Oblicza metryki, zapisuje model i generuje krzywą ROC.
    Zwraca listę wyników do CSV.
    """
    # Predykcja
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    # Save model
    save_model_pickle(model, model_name, folder=models_folder)

    # Save classification results for CSV
    all_results = []
    for label, metrics in report_dict.items():
        if label == 'accuracy':
            all_results.append({
                "model": model_name,
                "class": "accuracy",
                "precision": "",
                "recall": "",
                "f1-score": "",
                "support": "",
                "value": accuracy
            })
        else:
            all_results.append({
                "model": model_name,
                "class": label,
                "precision": metrics.get('precision', ''),
                "recall": metrics.get('recall', ''),
                "f1-score": metrics.get('f1-score', ''),
                "support": metrics.get('support', '')
            })

    # Krzywa ROC
    try:
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test)
        else:
            print(f"Model {model_name} does not support probability prediction, skipping ROC curve.")
            return all_results

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        save_roc_curve(fpr, tpr, roc_auc, model_name, folder=roc_folder)

    except Exception as e:
        print(f"Could not plot/save ROC for {model_name} due to error: {e}")

    return all_results


def save_model_pickle(model, model_name, folder="models"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"model_{model_name.replace(' ', '_')}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model '{model_name}' saved to {path}")


def save_roc_curve(fpr, tpr, roc_auc, model_name, folder="roc_curves"):
    os.makedirs(folder, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    path = os.path.join(folder, f"roc_{model_name.replace(' ', '_')}.png")
    plt.savefig(path)
    plt.close()
    print(f"ROC curve saved to {path}")
