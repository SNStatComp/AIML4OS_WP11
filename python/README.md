
# 📅 Project Summary — 31.10.2025


## Introduction of geographic data into the synthetic dataset: proposed procedure




Using the original synthetic files available on the Onyxia portal:


Companies table: wp11_companies_synthetic_data.parquet  (Includes postal codes CP3, CP4 )

Transactions table: wp11_b2b_synthetic_data.parquet (Includes postal codes CP7)

Geo distances table : municipio_distances_v2.csv  (Includes  municipalities names ONLY ) 


and an additional  file

cod_post_freg_matched.csv from  :

https://github.com/dssg-pt/mp-mapeamento-cp7  ( Includes postal codes CP7 AND municipalities names )

repository of the external  Mini-Project: Mapping of Postal Codes to locations in Portugal, 


we can add information on distances between companies for each row of the final synthetic merged dataset:

LINE_DST – linear distance

ROAD_DST – road distance

Proposed procedure:

Using the file cod_post_freg_matched.csv, it is possible to map CP7 postal codes (CodigoPostal) to municipality names (Concelho).

In the AIML project, the companies table contains CP3 and CP4 codes. By combining them, we obtain the CP7 code.

After merging the companies table with the file cod_post_freg_matched.csv, the companies table includes the municipality name (Concelho) for each company.

Next, we merge the companies table with the transactions table from Onyxia (according to our pipeline). As a result, each transaction gets columns with the municipality names of the supplier and the buyer:

sup_Concelho – supplier municipality

buyer_Concelho – buyer municipality

The next step is to join the data with the file municipio_distances_v2 (from Onyxia), which contains linear and road distances for each pair of municipalities labeled as MN_ORIGIN_DSG and MN_DESTINY_DSG.

Assuming:

Concelho_sup = MN_ORIGIN_DSG

Concelho_buyer = MN_DESTINY_DSG


we merge the data for the same combinations of Concelho_sup – Concelho_buyer and MN_ORIGIN_DSG – MN_DESTINY_DSG.

As a result, we obtain 4,159,560 rows with geo distances. linear distance and road distance

Now we are ready for next steps : 

It will be possible to incorporate distances into the model and perform geo normalization.




---




# 📅 Project Summary — 10.10.2025

## 🔄 Recent Changes


- **Added file**:  console_logs_10_10_2025 
- **Modified file**: main_sampling.py, parameter.py, training_all_model_one.py, validation.py




## 🎯 Purpose of the Changes

➡️ Structuring the pipeline by relocating code to the proper modules: main, parameter, data, training, validation


✅ Final outcome:


All parameters are initialized in a single place (parameter.py).

The main_sampling.py module passes them to all other modules (data_sampling, training, validation).

The pipeline is now modular, easily extendable, and fully configurable via PARAMS.



Actual flow of the pipeline 


main_sampling > parameter > data_sampling > training_all_models_one > validation 



## ▶️ How to Run the Code 


To execute code type :

python main_sampling.py





---












# 📅 Project Summary — 02.10.2025

## 🔄 Recent Changes


- **Added file**:  main_sampling.py , data_sampling.py
- **Modified file**: training_all_model_one.py




## 🎯 Purpose of the Changes

➡️ Implementation of negative sampling method (instead of the Cartesian product)




The code in new data module (data_sampling.py)  works as follows:

The input contains only positives (LINKED = 1).

The code generates negatives (LINKED = 0) automatically through negative sampling:

It takes supplier–buyer pairs from the same NACE sections,

Randomly selects pairs that do not appear in the positive data,

Creates a set of negatives from them (so that positives and negatives are 50/50).

Next, it combines positives and negatives into combined_df.

Then, it applies a balancing strategy (if needed):

"oversample_negatives" → duplicates the generated negatives to match the number of positives,

"undersample_positives" → randomly trims the positives to match the number of negatives.

Using the new method, we have:

 LINKED
 
1    206736 (loaded from file) 

0    206736 (sampled) 

number of all rows combined  413472



## ▶️ How to Run the Code 


To execute code type :

python main_sampling.py


## ▶️ Changes effects :


🔑 Key Observations

Accuracy

Old results: most models had accuracy ~0.92–0.95

New results: accuracy dropped to ~0.50–0.60

➡️ There is a huge drop in performance – old models almost always predicted class 0 (high accuracy due to imbalanced data), while the new ones have to recognize a more balanced dataset.

Class 1 (minority) – F1

Old results: F1 for class 1.0 ≈ 0.0–0.16 (almost zero, classifiers ignored the minority class).

New results: F1 for class 1.0 increased to 0.47–0.65 (depending on the model).


➡️ The new models capture class 1 much better, even though overall accuracy has decreased.

Models particularly improved in the new results

MLP Classifier: previously ignored class 1, now F1(1.0) = 0.6577

Quadratic Discriminant Analysis: jump from ~0.005 to 0.654 F1 for class 1

Histogram-Based Gradient Boosting, XGBoost, LightGBM: F1(1.0) ≈ 0.54–0.56 – very stable

Decision Tree, Random Forest, Extra Trees: F1(1.0) ≈ 0.47–0.48 (better than old results, which were 0.16)

Overall balance

Old trainings : models have artificially high accuracy, but completely useless for class 1

New trainings : models have lower accuracy (0.58–0.60), but meaningfully distinguish both classes

📊 Summary

Old results were a consequence of imbalanced data → models almost always predicted class 0
.
New results show that models were trained/tested on a balanced dataset (50/50) or with a different metric → accuracy dropped, but F1 for class 1 increased significantly.

Now the models are much more practically useful, as they can detect class 1.





---











# 📅 Project Summary — 19.09.2025

## 🔄 Recent Changes


- **Added file**:

- AI_ML_WP_11_PIPELINE_19_09_2025_PYTHON.ZIP,

- AI_ML_WP_11_PIPELINE_19_09_2025_PYTHON_CODE_DESCRIPTION.DOC


## 🎯 Purpose of the Changes

Archive of pipeline that will be sent via email 




---




# 📅 Project Summary — 18.09.2025

## 🔄 Recent Changes


- **Added file**:  main_one.py , main_one_linked.py , training_all_models_one.py, requirements.txt


## 🎯 Purpose of the Changes


Preparation of  pipeline for all models with use of the best parameters after tuning.

Update of requirements.txt file that lists all udes python packages as stated for 18.09.2025


Actual flow of the pipeline 

main_one_linked > parameters > data_linked > training_all_models_one > validation 

output of pipeline - training results for each model are saved as:

- ✅ Trained models: `.pkl` files (in the `models` folder)
- 📉 ROC curves: `.png` files (in the `roc_curves` folder)
- 📊 Classification metrics: `training_results.csv`


You can easily choose models used or switch between them by commenting/uncommenting code lines in the file training_all_models_one.py.
You can also run all models at once.

Importatnt : This code (with some description) will be sent to Sónia for analysis and for verifying the conditions and safety of execution before the visit to Portugal.



## ▶️ How to Run the Code 


To execute code type :

python main_one_linked.py

Note: This will train only one model, for demonstration purposes. 





---





# 📅 Project Summary — 12.09.2025

## 🔄 Recent Changes


- **Added file**:  main_linked.py , data_linked.py , training_model_1_linked.py 


## 🎯 Purpose of the Changes


The goal of the updates is to: Simulates how the Portuguese VAT register would be processed in Data Module of pipeline

Assumming real portugal vat register will look as stated in document : Deliverable	11.0 – Report describing the training and test sets, 
Annex 2  - Variables of Portuguese electronic invoices  E-Invoices (V_TF_EFAT_ENCRIPTADA_AAAA )


Assumming that you can load all register into dataframe and then  change name of columns in that dataframe

(some kind of mapping data structure e-invoices vs wp11_b2b_syntetic_data) :

'ANO': 'YYYY'

'NIF_EMITENTE': 'SUPPLIER'

'NIF_ADQUIRENTE_NAC_COL': 'BUYER'

after that you can add column LINKED with values set as 1 


In this code we use file "wp11_b2b_syntetic_data_linked_1_only_rows.parquet" 
This file contains only rows with real transactions taken from file "wp11_b2b_synthetic_data.parquet" all with linked=1. 

This is kind of simulation of loading real portugal VAT register as it will contains only real transaction wchich are classified as linked=1 by nature. 


With utilization of this code you can easly replace wp11_b2b_syntetic_data_linked_1_only_rows.parquet with real VAT register in data module when times comes



## ▶️ How to Run the Code 


To execute code type :

python main_linked.py

Note: This will train only one model, for demonstration purposes




---

### 1. Program start

* Displays a message that the data preparation module has started.

---

### 2. Loading company data

* Loads the file **`wp11_companies_synthetic_data.parquet`** into the DataFrame `df_companies`.
* Prints the first rows (`head`) and the total number of companies.

---

### 3. Loading transaction data (only existing ones)

* Loads the file **`wp11_b2b_syntetic_data_linked_1_only_rows.parquet`** into the DataFrame `df`.

  * This file contains only **real transactions** between companies, marked as `LINKED = 1`.
  * It simulates what the Portuguese VAT register would look like (each invoice = real transaction).
* Prints the number of rows.

---

### 4. Extracting unique companies

* Collects all unique company IDs from the **SUPPLIER** and **BUYER** columns.
* Creates a sorted list `unique_companies`.

---

### 5. Generating all possible company pairs

* Uses the **Cartesian product** (`itertools.product`) to generate all possible `SUPPLIER-BUYER` pairs.
* Stores the result in the DataFrame `all_pairs_df`.
* Prints the size of this DataFrame (the number of all possible relations).

---

### 6. Selecting existing pairs

* Extracts existing `SUPPLIER-BUYER` pairs from the transaction data (`df`).
* Stores them in `existing_pairs_df`.

---

### 7. Creating non-existing pairs (LINKED=0)

* Merges `all_pairs_df` with `existing_pairs_df` (left join).
* Pairs that **do not occur in the transaction data** are treated as **non-existent transactions**.
* Adds a column **LINKED=0** to those.
* The result is `zeros_df`.

---

### 8. Combining data

* Adds a **LINKED=1** column to the existing pairs (`existing_pairs_df`).
* Concatenates existing pairs (`LINKED=1`) and non-existing pairs (`LINKED=0`) into **combined\_df**.
* Resets the index.
* Now, `combined_df` contains all possible company pairs with labels:

  * `1` → real transactions exist,
  * `0` → transactions do not exist.

---

### 9. Final datasets is ready for training

* `b2b_df = combined_df` → the full dataset of company relationships (ready for modeling).
* `business_df` = company data (`wp11_companies_synthetic_data.parquet`).

---

### 10. Summary

The whole code:

* Simulates how the Portuguese VAT register would be processed.
* From real transactions (`LINKED=1`) and the complete list of companies, it generates a **full matrix of possible inter-company relationships**.
* Creates binary labels (`LINKED=0/1`), which can be used to train ML models for predicting B2B links.

---



# 📅 Project Summary — 10.09.2025

## 🔄 Recent Changes

- **Added file**: `training_all_models_hypertuning_gridsearch.py` 'main_hypertuning.py'
 
- **Added folder**: `tuning results`

## 🎯 Purpose of the Changes

The goal of the updates is to: Hyperparameter Tuning

file `training_all_models_hypertuning_gridsearch.py`  contains additional function :


tune_all_models(X_train, y_train)

For most models, it defines a hyperparameter grid (param_grid).

For each model:

Runs GridSearchCV and RandomizedSearchCV to find the best parameters.

Saves the results to CSV files in the tuning_results directory.

Creates a summary of the best parameters (best_params_summary.csv).


## ▶️ How to Run the Code 

To execute the training process (with hyperparameter tuning) :

python main_hypertuning.py


> **Note:** Hyperparameter tuning, once started could last very long time  (even 12 hours and more.)
>           Naive Bayes": GaussianNB() was excluded from hyperparameter tuning as it has no parameters for tuning.

---



# 📅 Project Summary — 05.06.2025

## 🔄 Recent Changes

- **Added file**: `training_all_models.py`  
- **Modified file**: `main.py`

## 🎯 Purpose of the Changes

The goal of the updates is to:

- Enable **simultaneous training** of all 12 existing models.
- Eliminate **code redundancy** and simplify execution.
- Save training results to files, allowing trained models to be reused **without retraining** (via loading `.pkl` files).

> **Note:** Not all trained models were saved in the `models` folder, as some files exceeded GitHub's hard limit of **100 MB** for file uploads (`git push`).

---

## ⚙️ What the Code Does

For each of the 12 models, the code:

- 📊 **Generates new features** based on differences between companies  
  (e.g., turnover, fixed assets, wages).
- 🔁 **Transforms categorical variables** and **splits** the data into training and test sets.
- 🧠 **Trains classification models**, including:
  - KNN
  - Naive Bayes
  - Decision Tree
  - XGBoost
  - LightGBM
  - ...and others
- 📈 **Calculates accuracy** and generates a **classification report**.

---

## 💾 Output

Training results for each model are saved as:

- ✅ Trained models: `.pkl` files (in the `models` folder)
- 📉 ROC curves: `.png` files (in the `roc_curves` folder)
- 📊 Classification metrics: `training_results.csv`

For status messages and print logs, see:  
📄 `console_log_prints.docx`

---

## ▶️ How to Run the Code

To execute the training process:

python main.py

No need to modify the code to switch between models.
Running main.py once will train all 12 models.


---


# 📅 Project Summary — 30.04.2025

A search for the most appropriate model was conducted.

The training, testing, and evaluation process covered the following 12 models:

- K-Nearest Neighbors  
- Naive Bayes  
- Decision Tree  
- Extra Trees  
- Logistic Regression  
- MLP Classifier  
- Quadratic Discriminant Analysis  
- Histogram-Based Gradient Boosting  
- XGBoost Classifier  
- LightGBM  
- AdaBoost  
- Random Forest

For each model, a supervised machine learning approach was implemented to classify the existence of network connections (denoted as **LINKED**) between pairs of companies based on relational features.

---

## ⚙️ How to Run a Specific Model

The `main.py` script is designed to evaluate various machine learning models. To run a specific model, follow these steps:

### 1. Select a Model

At the top of `main.py`, locate the list of available model imports. Uncomment the line corresponding to the model you wish to run. For example, to use **K-Nearest Neighbors**, uncomment:

```python
import training_model_1  # K-Nearest Neighbors
````

Make sure all other `training_model_X` imports remain commented out.

### 2. Enable the Model Execution

Scroll down to the training section in the `__main__` block. Uncomment the line that calls the selected model's `custom_train_model` function:

```python
y_test, y_pred = training_model_1.custom_train_model(b2b_merged)
```

Again, only one `training_model_X` function call should be active at a time.

### 3. Run the Script

Execute the script using Python:

```bash
python main.py
```

This will:

* Load parameters and data
* Train the selected model
* Generate evaluation metrics
* Display and/or save the corresponding ROC/AUC plot

---

## 📚 Available Models

| Model                           | Module              |
| ------------------------------- | ------------------- |
| K-Nearest Neighbors             | `training_model_1`  |
| Naive Bayes                     | `training_model_2`  |
| Decision Tree                   | `training_model_3`  |
| Extra Trees                     | `training_model_4`  |
| Logistic Regression             | `training_model_5`  |
| MLP Classifier                  | `training_model_6`  |
| Quadratic Discriminant Analysis | `training_model_7`  |
| HistGradientBoosting            | `training_model_8`  |
| XGBoost (XGBClassifier)         | `training_model_9`  |
| LightGBM                        | `training_model_10` |
| AdaBoost                        | `training_model_11` |
| Random Forest                   | `training_model_12` |



---


# 📅 Project Summary — 09.04.2025



Description of the Program’s Operation, Its Modules, Inputs and Outputs, and Data Flow:

#### **🧩 Program Modules**

1. **parameter**

   * **Input**: None (no input data from other modules)

   * **Output**: No explicit data, but most likely sets or loads configuration parameters

   * **Description**: Initializes model parameters or the entire pipeline. It may set global variables, paths, model configurations, etc.

2. **data**

   * **Main Function**: load\_data()

   * **Input**: No direct input (but may use parameters set in parameter)

     Loads data files in parquet format. The files named wp11\_b2b\_synthetic\_data.parquet and wp11\_companies\_synthetic\_data.parquet should be copied to the main project folder (the files are available on the Onyxia portal).

   * **Output**: b2b\_merged — a large DataFrame containing transactional data, including supplier data, buyer data, distance, wages, sectors, etc.

   * **Description**: Loads data, performs merging, validation (e.g., checking for duplicates, negative values, missing data), and returns the processed DataFrame to the training module.

3. **training**

   * **Functions**:

     * log\_train\_model(b2b\_merged) — trains a logistic regression model

     * lgbm\_train\_model(b2b\_merged) — trains a LightGBM model

   * **Input**: b2b\_merged DataFrame from data

   * **Output**:

     * y\_test, y\_pred — data required for model validation

   * **Description**: Performs two types of model training (logistic regression and LightGBM). The LightGBM training returns data needed for validation. Additionally, it prints detailed information on model fitting (e.g., precision, coefficients).

4. **validation**

   * **Functions**:

     * validate\_model(y\_test, y\_pred) — model metrics for LightGBM

     * main() — additional operations, e.g., visualizations, logic checks, etc.

   * **Input**:

     * y\_test, y\_pred — predicted and actual data from training

   * **Output**: None (results are printed to the console)

   * **Description**: Calculates classification metrics (accuracy, precision, recall, F1). Informs about the model's effectiveness and may perform additional analysis.

---

### **🔄 Program Execution Order**

1. Execute main.py

2. parameter.main()

   * Initialize parameters

3. data.load\_data()

   * Load and prepare data

   * Validate data quality

4. training.log\_train\_model()

   * Train the logistic regression model

   * Print results to the console

5. training.lgbm\_train\_model()

   * Train the LightGBM model

   * Return y\_test and y\_pred

6. validation.validate\_model()

   * Evaluate the effectiveness of the LightGBM model

7. validation.main()

   * Perform additional validation steps

8. The program ends

---

### **🔁 Data Flow Between Modules**

parameter \-\> \[sets global parameters\]  
data \-\> b2b\_merged \-\> training \-\> y\_test, y\_pred \-\> validation

---

### **📎 Additional Notes**

* Input data is a large transactional dataset (over 4 million records).

* The logistic and LGBM models are trained on the same dataset.

* The LightGBM model achieves high accuracy (\~95%) but very low effectiveness for class 1.0 — likely due to imbalanced data.

* The control flow is procedural, with print statements providing messages.

---

