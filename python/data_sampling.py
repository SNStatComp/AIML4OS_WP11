def main():
    print("you are in  data preparation module")

print("modul data started")




import pandas as pd



#######################negative_sampling_02_10_2025_start


import pandas as pd
import numpy as np

np.random.seed(42)

# --- parameters ---
balance_strategy = "undersample_positives"  # or "oversample_negatives"

# --- input data ---
df_companies = pd.read_parquet("wp11_companies_synthetic_data.parquet")
df = pd.read_parquet("wp11_b2b_syntetic_data_linked_1_only_rows.parquet")

# add NACE codes
df = df.merge(df_companies[["ID", "NACE_SEC"]], left_on="SUPPLIER", right_on="ID", how="left") \
       .rename(columns={"NACE_SEC": "NACE_SEC_S"}).drop(columns="ID")
df = df.merge(df_companies[["ID", "NACE_SEC"]], left_on="BUYER", right_on="ID", how="left") \
       .rename(columns={"NACE_SEC": "NACE_SEC_B"}).drop(columns="ID")
df = df.loc[:, ~df.columns.duplicated()]

# --- aggregate NACE pairs ---
nace_pairs = df.groupby(["NACE_SEC_S", "NACE_SEC_B"]).size().reset_index(name="count")

# --- dictionary of companies by NACE ---
suppliers_dict = df_companies.groupby("NACE_SEC")["ID"].apply(np.array).to_dict()
buyers_dict = df_companies.groupby("NACE_SEC")["ID"].apply(np.array).to_dict()

# --- memory-efficient negative sampling ---
negatives_list = []
existing_pairs = set(zip(df["SUPPLIER"], df["BUYER"]))

for _, row in nace_pairs.iterrows():
    nace_s, nace_b, true_count = row["NACE_SEC_S"], row["NACE_SEC_B"], row["count"]

    suppliers = suppliers_dict.get(nace_s, np.array([]))
    buyers = buyers_dict.get(nace_b, np.array([]))

    if len(suppliers) == 0 or len(buyers) == 0:
        continue

    max_negatives = len(suppliers) * len(buyers) - true_count
    n_samples = min(true_count, max_negatives)
    if n_samples <= 0:
        continue

    sampled_pairs = set()
    attempts = 0
    max_attempts = n_samples * 10  # prevent infinite loop

    while len(sampled_pairs) < n_samples and attempts < max_attempts:
        s = np.random.choice(suppliers, n_samples - len(sampled_pairs), replace=True)
        b = np.random.choice(buyers, n_samples - len(sampled_pairs), replace=True)
        for pair in zip(s, b):
            if pair not in existing_pairs and pair not in sampled_pairs:
                sampled_pairs.add(pair)
        attempts += 1

    if len(sampled_pairs) == 0:
        continue

    sampled_pairs = np.array(list(sampled_pairs))
    nace_s_array = np.full(len(sampled_pairs), nace_s)
    nace_b_array = np.full(len(sampled_pairs), nace_b)
    link_array = np.zeros(len(sampled_pairs), dtype=int)

    negatives_list.append(pd.DataFrame({
        "SUPPLIER": sampled_pairs[:,0],
        "BUYER": sampled_pairs[:,1],
        "LINKED": link_array,
        "NACE_SEC_S": nace_s_array,
        "NACE_SEC_B": nace_b_array
    }))

# --- combine negatives ---
negatives_df = pd.concat(negatives_list, ignore_index=True)

# --- positives ---
positives_df = df[["SUPPLIER", "BUYER", "NACE_SEC_S", "NACE_SEC_B"]].copy()
positives_df["LINKED"] = 1

# --- combine ---
combined_df = pd.concat([positives_df, negatives_df], ignore_index=True)

# --- balancing ---
counts = combined_df["LINKED"].value_counts()
print("Before balancing:\n", counts)

if balance_strategy == "oversample_negatives" and counts[0] < counts[1]:
    deficit = counts[1] - counts[0]
    print(f"Oversampling {deficit} negatives...")
    sampled_negatives = negatives_df.sample(n=deficit, replace=True, random_state=42)
    combined_df = pd.concat([combined_df, sampled_negatives], ignore_index=True)

elif balance_strategy == "undersample_positives" and counts[1] > counts[0]:
    surplus = counts[1] - counts[0]
    print(f"Undersampling {surplus} positives...")
    positives_sampled = positives_df.sample(n=counts[0], random_state=42)
    combined_df = pd.concat([positives_sampled, negatives_df], ignore_index=True)

# --- shuffle ---
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- final check ---
print("After balancing:\n", combined_df["LINKED"].value_counts())



#######################negative_sampling_02_10_2025_end




#Reasoning: Load file "wp11_b2b_syntetic_data_linked_1_only_rows.parquet" into df dataframe. This file contains only rows with real transactions 
# taken from file "wp11_b2b_synthetic_data.parquet" all with linked=1. This is kind of simulation of loading real portugal VAT reistry as it will contains only 
# real transaction wchich are classified as linked=1 by nature. 


#Reasoning: This is only first attempt. 
#
#Assumming real portugal vat register will look as stated in document : Deliverable	11.0 – Report describing the training and test sets, 
#Annex 2  - Variables of Portuguese electronic invoices  E-Invoices (V_TF_EFAT_ENCRIPTADA_AAAA )
#
#You could load all registery into dataframe df_invoices. 
#Change name of key column:
#df_invoices = df_invoices.rename(columns={'ANO': 'YYYY'})
#df_invoices = df_invoices.rename(columns={'NIF_EMITENTE': 'SUPPLIER'})
#df_invoices = df_invoices.rename(columns={'NIF_ADQUIRENTE_NAC_COL': 'BUYER'})
#and after that add column LINKED with vaues set as 1 :
#df_invoices['LINKED'] = 1
#You can use naw df_invoices as replacement of file "wp11_b2b_syntetic_data_linked_1_only_rows.parquet" !!! in this code
#
#Note : This is only an assumption because we did not know the actual VAT register that we would be working with during planned visit to Portugal.







#######################################combined_df as result of negative sampling 


print(combined_df.head())
print(combined_df.tail())

print("number of all rows combined  ")
print(len(combined_df))


#################################################################################

    
"""WCZYTANIE DANYCH / LOAD DATA"""


import pandas as pd



#Reasoning: Load combinded dataframe created above as b2b_df, instead of loading file "wp11_b2b_synthetic_data.parquet"


#b2b_df = pd.read_parquet("wp11_b2b_synthetic_data.parquet", engine="pyarrow")
b2b_df = combined_df

print("number of all rows combined loaded as b2b_df  ")
print(len(b2b_df))





#Reasoning: Load file "wp11_companies_synthetic_data.parquet" as business_df. No change here!


business_df = pd.read_parquet("wp11_companies_synthetic_data.parquet", engine="pyarrow")

print("load data")



#Reasoning: Excecute all code below as usual.





"""ŁACZENIE DANYCH / MERGE DATA"""

import pandas as pd

#Joining tables: we are adding information about the supplier and the recipient.
b2b_merged = b2b_df.merge(business_df, left_on="SUPPLIER", right_on="ID", suffixes=("_sup", "_buyer"))
b2b_merged = b2b_merged.merge(business_df, left_on="BUYER", right_on="ID", suffixes=("_sup", "_buyer"))

print("merge data")

#We are removing unnecessary ID columns.
b2b_merged.drop(columns=["ID_sup", "ID_buyer"], inplace=True)

#Data verification.
print(b2b_merged.head())

"""KONTROLA DANYCH PO POLACZENIU / DATA VALIDATION AFTER MERGING"""

import pandas as pd

#b2b_merged.head(100).to_excel('first_100_rows.xlsx', index=False)

print("data validation")

"""SPRAWDZENIE TYPÓW DANYCH / DATA TYPE CHECKING"""

#b2b_merged.dtypes

"""SPRAWDZENIE ISTNIENIA DUPLIKATOW / CHECKING FOR DUPLICATES"""

#Checking if the DataFrame contains duplicates

duplicate_rows = b2b_merged[b2b_merged.duplicated()]

if duplicate_rows.empty:
   print("The DataFrame does not contain duplicates.")


else:
  print("The DataFrame contains duplicates:")
duplicate_rows


#Optionally: Removing duplicates
#b2b_merged.drop_duplicates(inplace=True)
#print("Number of rows after removing duplicates:", len(b2b_merged))

"""SPRAWDZENIE ISTNIENIA UJEMNYCH WARTOSCI / CHECKING FOR NEGATIVE VALUES"""

#Checking if the DataFrame contains negative values in numeric columns

numeric_cols = b2b_merged.select_dtypes(include=['number'])
negative_values = numeric_cols[numeric_cols < 0].any()

if negative_values.any():
    print("The DataFrame contains negative values in the following columns:")
    print(negative_values[negative_values])

else:
    print("The DataFrame does not contain negative values.")

"""SPRAWDZENIE ISTNIENIA PUSTYCH WARTOSCI / CHECKING FOR MISSING VALUES"""

#Checking if the DataFrame contains missing values (NaN)

missing_values = b2b_merged.isnull().sum()

if missing_values.any():
    print("The DataFrame contains missing values (NaN) in the following columns:")
    print(missing_values[missing_values > 0])
else:
    print("The DataFrame does not contain missing values (NaN).")

"""WYPELNIENIE ZERAMI W MIEJSCE PUSTYCH WARTOSCI - TYLKO KOLUMNA LINKED / FILLING BLANK  VALUES WITH ZEROS – ONLY IN THE LINKED COLUMN"""

b2b_merged['LINKED'] = b2b_merged['LINKED'].fillna(0)

# Ensure 'LINKED' column is numeric and within [0, 1]
b2b_merged['LINKED'] = pd.to_numeric(b2b_merged['LINKED'])  # Convert to numeric if necessary
b2b_merged['LINKED'] = b2b_merged['LINKED'].clip(0, 1)     # Clip values to 0-1 range

"""SPRAWDZENIE ISTNIENIA ZEROWYCH WARTOSCI / CHECKING FOR ZERO VALUES"""

# Check for zero values in the DataFrame and show the columns where they exist.
zero_values = b2b_merged.isin([0]).sum()

if (zero_values > 0).any():
    print("DataFrame contains zero values in the following columns:")
    print(zero_values[zero_values > 0])
else:
    print("DataFrame does not contain any zero values.")

"""SPRAWDZENIE CZY WARTOŚCI W KOLUMNACH NACE_SEC_S I NACE_SEC_B MAJĄ PO JEDNYM ZNAKU LITEROWYM / CHECK IF THE VALUES IN THE COLUMNS NACE_SEC_S AND NACE_SEC_B CONTAIN EXACTLY ONE LETTER CHARACTER."""


"""

# Check if 'NACE_SEC_S' and 'NACE_SEC_B' columns have exactly one letter
b2b_merged['NACE_SEC_S_check'] = b2b_merged['NACE_SEC_S'].astype(str).str.len() == 1
b2b_merged['NACE_SEC_B_check'] = b2b_merged['NACE_SEC_B'].astype(str).str.len() == 1

# Display rows where the check fails
incorrect_nace_rows = b2b_merged[(b2b_merged['NACE_SEC_S_check'] == False) | (b2b_merged['NACE_SEC_B_check'] == False)]
print(incorrect_nace_rows[['NACE_SEC_S', 'NACE_SEC_B', 'NACE_SEC_S_check', 'NACE_SEC_B_check']])

# Print summary
if len(incorrect_nace_rows) > 0:
    print(f"Found {len(incorrect_nace_rows)} rows where 'NACE_SEC_S' or 'NACE_SEC_B' do not have exactly one character.")
else:
  print("All values in 'NACE_SEC_S' and 'NACE_SEC_B' have exactly one character.")

"""



import pandas as pd

def load_data():
    global b2b_merged

    return b2b_merged


print("transfer data to training module")


print("modul data done")
