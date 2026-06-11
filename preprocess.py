import pandas as pd
import re
import os

INPUT_PATH  = "data/tokopedia_labeled.csv"   
OUTPUT_PATH = "data/processed_reviews.csv"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_dates(df: pd.DataFrame, date_col: str = "review_date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["timestamp"] = df[date_col].astype("int64") // 10**9  # Unix timestamp (seconds)
    n_invalid = df[date_col].isna().sum()
    if n_invalid:
        print(f"  Warning: {n_invalid} rows with unparseable dates — dropped")
        df = df.dropna(subset=[date_col])
    return df

def validate_rating(df: pd.DataFrame, rating_col: str = "rating") -> pd.DataFrame:
    before = len(df)
    df = df[df[rating_col].between(1, 5)]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with invalid rating (outside 1-5)")
    return df

def preprocess(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    print("Preprocessing")

    df = load_data(input_path)

    print("Validating ratings...")
    df = validate_rating(df)

    print("Parsing dates -> adding Unix timestamp...")
    df = parse_dates(df)

    print("Cleaning review text...")
    df["review_text_clean"] = df["review_text"].apply(clean_text)

    before = len(df)
    df = df[df["review_text_clean"].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with empty text after cleaning")

    df = df.reset_index(drop=True)

    # veruf label
    n_coord = df["coordinated"].sum()
    print(f"\nLabel check -- coordinated: {n_coord:,} | normal: {len(df)-n_coord:,}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows -> {output_path}")
    print("Done")

    return df

if __name__ == "__main__":
    preprocess()
