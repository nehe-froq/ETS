import json
import pandas as pd
import os

# ==============================================================
#  CONFIGURATION
# ==============================================================

# Path to your JSON file
JSON_FILE = "/Users/froquser/Desktop/Nehe/ETS /Source code/dam-main/data.json"

# Output file name (choose CSV or Excel)
OUTPUT_FILE = "json_table_output.csv"

# ==============================================================
#  HELPER FUNCTION: Flatten Nested JSON
# ==============================================================

def flatten_json(y, parent_key='', sep='.'):
    """
    Recursively flattens a nested JSON object into a single-level dict.
    """
    items = []
    if isinstance(y, dict):
        for k, v in y.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_json(v, new_key, sep=sep).items())
    elif isinstance(y, list):
        # Convert list of simple values to comma-separated string
        if all(not isinstance(i, (dict, list)) for i in y):
            items.append((parent_key, ', '.join(map(str, y))))
        else:
            for i, v in enumerate(y):
                new_key = f"{parent_key}[{i}]"
                items.extend(flatten_json(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, y))
    return dict(items)

# ==============================================================
# MAIN EXECUTION
# ==============================================================

def main():
    # 1 Load JSON
    if not os.path.exists(JSON_FILE):
        print(f" File not found: {JSON_FILE}")
        return

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2 Handle single object or list
    if isinstance(data, list):
        print(f" Detected {len(data)} JSON records.")
        flat_data = [flatten_json(d) for d in data]
    else:
        print(" Detected a single JSON object.")
        flat_data = [flatten_json(data)]

    # 3 Convert to DataFrame
    df = pd.DataFrame(flat_data)

    # 4 Fill missing columns with blank values for consistency
    df = df.fillna("")

    # 5 Display formatted table preview
    print("\n JSON Extracted Table (Preview):\n")
    print(df.head().to_string(index=False))
    print(f"\nTotal columns: {len(df.columns)} | Total rows: {len(df)}")

    # 6 Save output
    if OUTPUT_FILE.endswith(".csv"):
        df.to_csv(OUTPUT_FILE, index=False)
    elif OUTPUT_FILE.endswith(".xlsx"):
        df.to_excel(OUTPUT_FILE, index=False)
    else:
        print(" Unsupported format. Please use .csv or .xlsx")
        return

    print(f"\n File successfully saved as: {OUTPUT_FILE}")

# ==============================================================
#  ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()
