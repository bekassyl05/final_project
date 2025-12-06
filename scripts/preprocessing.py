import os, re
import pandas as pd
import numpy as np
from datetime import datetime

INFILE = "data/raw_news_data.csv"
OUTFILE = "data/clean_news_data.csv"

def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_timestamp(val, source):
    try:
        if pd.isna(val) or val == "":
            return pd.NaT
        if source == "hackernews":
            return pd.to_datetime(int(val), unit="s", errors="coerce")
        if source == "reddit":
            return pd.to_datetime(float(val), unit="s", errors="coerce")
        return pd.to_datetime(val, errors="coerce")
    except Exception:
        return pd.NaT

def pipeline(infile=INFILE, outfile=OUTFILE):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df = pd.read_csv(infile, low_memory=False)

    df['title'] = df['title'].astype(str).fillna("").apply(lambda x: x.strip())
    df = df[df['title'] != ""].copy()

    df['title_clean'] = df['title'].apply(clean_text).str.lower()
    df['body_clean'] = df.get('body', "").apply(clean_text).str.lower()

    df['timestamp_parsed'] = df.apply(lambda row: normalize_timestamp(row.get('timestamp'), row.get('source')), axis=1)

    if 'score' in df.columns:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        score_median = df['score'].median(skipna=True)
        df['score'] = df['score'].fillna(score_median)
    else:
        df['score'] = 0

    if 'num_comments' in df.columns:
        df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce').fillna(0)
    else:
        df['num_comments'] = 0

    df['word_count'] = df['title_clean'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['title_clean'].apply(lambda x: len(str(x)))

    df.drop_duplicates(subset=['title_clean','source'], inplace=True)

    df.to_csv(outfile, index=False)
    print("Saved cleaned data to", outfile)
    return df

if __name__ == "__main__":
    pipeline()
