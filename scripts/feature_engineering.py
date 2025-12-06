import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CLEAN_IN = "data/clean_news_data.csv"
FEATURES_OUT = "data/features_news.csv"
MODELS_DIR = "../models"
os.makedirs(MODELS_DIR, exist_ok=True)

try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    nlp_spacy = None
    HAS_SPACY = False

def tfidf_reduce(texts, max_features=4000, n_components=100):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words="english")
    X = vect.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xred = svd.fit_transform(X)
    return Xred, vect, svd

def lda_topics(texts, n_topics=8, max_features=4000):
    cv = CountVectorizer(max_features=max_features, stop_words="english")
    X = cv.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    lda.fit(X)
    feature_names = cv.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        terms = [feature_names[idx] for idx in comp.argsort()[-12:][::-1]]
        topics.append((i, terms))
    return lda, cv, topics

def run():
    df = pd.read_csv(CLEAN_IN, low_memory=False)
    texts = (df['title_clean'].fillna("") + " " + df['body_clean'].fillna("")).astype(str)
    print("TF-IDF + SVD ...")
    Xred, tfidf_vect, svd = tfidf_reduce(texts, max_features=4000, n_components=100)
    print("KMeans clustering ...")
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xred)
    df['cluster'] = labels

    print("LDA topics ...")
    lda, cv, topics = lda_topics(texts, n_topics=8, max_features=4000)

    topics_df = pd.DataFrame([(t[0], ", ".join(t[1])) for t in topics], columns=["topic_id", "top_words"])
    topics_df.to_csv("reports/lda_topics.csv", index=False)

    if HAS_SPACY:
        print("Running spaCy NER (may be slow)...")
        ents_list = []
        for doc in nlp_spacy.pipe(df['title_clean'].astype(str).tolist(), n_process=1, disable=[]):
            counts = {}
            for ent in doc.ents:
                counts[ent.label_] = counts.get(ent.label_, 0) + 1
            ents_list.append(counts)
        ents_df = pd.DataFrame(ents_list).fillna(0).astype(int)
        df = pd.concat([df.reset_index(drop=True), ents_df.reset_index(drop=True)], axis=1)

    svd_cols = [f"svd_{i}" for i in range(Xred.shape[1])]
    svd_df = pd.DataFrame(Xred, columns=svd_cols, index=df.index)
    df = pd.concat([df, svd_df], axis=1)

    os.makedirs("../reports", exist_ok=True)
    df.to_csv(FEATURES_OUT, index=False)
    joblib.dump({"tfidf_vect":tfidf_vect, "svd":svd, "kmeans":kmeans, "lda":lda, "lda_cv":cv}, os.path.join(MODELS_DIR, "news_models.joblib"))
    topics_df.to_csv("reports/topics_readable.csv", index=False)
    print("Saved features to", FEATURES_OUT)
    print("Saved models to", MODELS_DIR)
    return df

if __name__ == "__main__":
    run()
