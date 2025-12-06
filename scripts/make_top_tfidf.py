# scripts/make_top_tfidf.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

# CONFIG: change input CSV path if needed
INPUT_CSV = "data/clean_news_data.csv"   # <- өзгертуге болады
OUT_DIR = "../figures"
OUT_FILE = os.path.join(OUT_DIR, "top_tfidf_terms.png")
TOP_N = 30

os.makedirs(OUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(INPUT_CSV, low_memory=False)

# choose column: prefer title_clean, then title, then description
for col in ("title_clean","title","description","body","content","text"):
    if col in df.columns:
        texts = df[col].astype(str).fillna("").tolist()
        print("Using column:", col)
        break
else:
    raise SystemExit("No text column found in CSV. Rename a column to 'title_clean' or 'title' or use the synthetic script.")

# optionally limit to first 10000 rows
texts = texts[:10000]

# TF-IDF
vect = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
X = vect.fit_transform(texts)
scores = np.array(X.sum(axis=0)).flatten()
terms = vect.get_feature_names_out()
idx = np.argsort(scores)[::-1][:TOP_N]
top_terms = terms[idx]
top_scores = scores[idx]

# Plot (matplotlib, simple)
plt.figure(figsize=(10, max(6, TOP_N*0.18)))
plt.barh(range(TOP_N)[::-1], top_scores[::-1])
plt.yticks(range(TOP_N)[::-1], top_terms[::-1])
plt.xlabel("Global TF-IDF score (sum across documents)")
plt.title("Top TF-IDF terms")
plt.tight_layout()
plt.savefig(OUT_FILE)
plt.close()
print("Saved", OUT_FILE)
