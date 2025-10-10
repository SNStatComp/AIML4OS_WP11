# main_sampling2.py

from parameter import PARAMS
import data_sampling
import training_all_models_one

def main():
    print("=== MAIN PIPELINE STARTED ===")

    # 1️⃣ Parameters
    print("\n[1/4] Loading parameters...")
    print("Parameters loaded successfully.")

    # 2️⃣ Data
    print("\n[2/4] Loading data...")
    b2b_merged = data_sampling.load_data()
    print("Data loaded. Shape:", b2b_merged.shape)

    # 3️⃣ Training and validation of all models
    print("\n[3/4] Training and validating models...")
    # The function custom_train_model handles training all models and validation
    y_test, y_pred = training_all_models_one.custom_train_model(b2b_merged)
    print("Training and validation completed")

    # 4️⃣ Optional: additional analysis
    print("\n[4/4] Optional analysis...")
    # For example, preview the CSV results saved by custom_train_model
    import pandas as pd
    results_df = pd.read_csv(PARAMS["results_csv_path"])
    print(f"Validation results loaded. Total rows: {len(results_df)}")
    #print(results_df.head())
    print(results_df)
    

    print("\n=== MAIN PIPELINE DONE ===")

if __name__ == "__main__":
    main()

