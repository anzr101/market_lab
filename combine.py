import pandas as pd
import os

INPUT_FOLDER = "data/stocks"
OUTPUT_FILE = "data/raw.csv"


def combine_csvs():
    all_dfs = []

    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".csv"):
            path = os.path.join(INPUT_FOLDER, file)

            try:
                df = pd.read_csv(path)

                ticker = file.replace(".csv", "")
                df["Ticker"] = ticker

                all_dfs.append(df)

            except Exception as e:
                print(f"Error reading {file}: {e}")

    if not all_dfs:
        raise ValueError("No CSV files found.")

    combined = pd.concat(all_dfs, ignore_index=True)

    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"])
        combined = combined.sort_values(["Date", "Ticker"])

    combined.to_csv(OUTPUT_FILE, index=False)

    print("Combined file saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    combine_csvs()
