import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

IN = "data/features_news.csv"
OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

def posts_over_time(df):
    if 'timestamp_parsed' not in df.columns:
        return
    s = pd.to_datetime(df['timestamp_parsed'], errors='coerce').dropna()
    if s.empty:
        return
    per_day = s.dt.date.value_counts().sort_index()
    plt.figure(figsize=(10,4))
    per_day.plot()
    plt.title("Posts per day")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "posts_per_day.png"))
    plt.close()

def posts_by_source(df):
    plt.figure(figsize=(6,4))
    sns.countplot(y="source", data=df, order=df['source'].value_counts().index)
    plt.title("Posts by source")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "posts_by_source.png"))
    plt.close()

def wordcount_hist(df):
    if 'word_count' in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(df['word_count'].dropna(), bins=30)
        plt.title("Title word count distribution")
        plt.xlabel("Words in title")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "wordcount_hist.png"))
        plt.close()

def cluster_pca(df):
    svd_cols = [c for c in df.columns if c.startswith("svd_")]
    if len(svd_cols) < 2 or 'cluster' not in df.columns:
        return
    X = df[svd_cols].fillna(0).values
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X)
    plotdf = pd.DataFrame({"pca1":proj[:,0], "pca2":proj[:,1], "cluster":df['cluster'].astype(str)})
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=plotdf, x="pca1", y="pca2", hue="cluster", palette="tab10", s=20, legend='full')
    plt.title("PCA projection of SVD features colored by cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "cluster_pca.png"))
    plt.close()

def main():
    df = pd.read_csv(IN, low_memory=False)
    posts_over_time(df)
    posts_by_source(df)
    wordcount_hist(df)
    cluster_pca(df)
    print("Saved figures to", OUTDIR)

if __name__ == "__main__":
    main()
