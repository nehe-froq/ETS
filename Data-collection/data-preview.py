import pandas as pd

# ==========================
# 🔹 Load the CSV file
# ==========================
csv_path = "/Users/froquser/Desktop/Nehe/ETS /Source code/dam-main/embeddings/json_table_output.csv"
tab = pd.read_csv(csv_path)

# ==========================
# 🔹 Basic Info
# ==========================
print("📊 Basic Information\n")
print(f"Total rows: {tab.shape[0]}")
print(f"Total columns: {tab.shape[1]}")
print("\nColumn Names:")
print(tab.columns.tolist())

print("\nData Types:")
print(tab.dtypes)

# ==========================
# 🔹 Missing Values
# ==========================
print("\n❗ Missing Values per Column:")
missing = tab.isna().sum()
print(missing[missing > 0] if missing.any() else "No missing values!")

# ==========================
# 🔹 Unique Values per Column
# ==========================
print("\n🔹 Unique Values per Column:")
for col in tab.columns:
    print(f"{col}: {tab[col].nunique()} unique values")

# ==========================
# 🔹 Summary Statistics
# ==========================
print("\n📈 Summary Statistics (Numeric Columns):")
numeric_cols = tab.select_dtypes(include=['number'])
if not numeric_cols.empty:
    print(numeric_cols.describe())

else:
    print("No numeric columns found.")

# ==========================
# 🔹 Top Frequent Values for Object Columns
# ==========================
print("\n📊 Top Frequent Values (Object/Text Columns):")
object_cols = tab.select_dtypes(include=['object'])
for col in object_cols:
    print(f"\nColumn: {col}")
    print(tab[col].value_counts().head(5))

# ==========================
# 🔹 Optional: Save the data profile to CSV
# ==========================
profile_summary = []

for col in tab.columns:
    profile_summary.append({
        "Column": col,
        "DataType": tab[col].dtype,
        "NonMissingCount": tab[col].notna().sum(),
        "MissingCount": tab[col].isna().sum(),
        "UniqueValues": tab[col].nunique(),
        "TopValue": tab[col].mode().iloc[0] if not tab[col].mode().empty else None
    })

profile_df = pd.DataFrame(profile_summary)
profile_df.to_csv("data_profile_summary.csv", index=False)
print("\n✅ Detailed data profile saved as data_profile_summary.csv")
