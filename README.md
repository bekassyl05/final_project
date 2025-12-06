# Social Media News Trends Analysis

A compact, reproducible pipeline for collecting, preprocessing, and analyzing news & discussion posts from multiple online sources (Hacker News, Reddit `r/news`, BBC). The project uses classical NLP and machine learning techniques (TF-IDF, SVD, LDA, K-Means) to surface topics, temporal trends, and clusters — **no LLMs or large language models are used**.

---

## Project summary
- **Goal:** Gather ≥3,000 posts from multiple sources and discover topical trends, temporal patterns, and meaningful clusters.  
- **Sources:** Hacker News (JSON API), Reddit `/r/news` (JSON), BBC News (HTML scraping).  
- **Outputs:** cleaned datasets, feature matrices, topic lists, model artifacts, static visualizations, and a Streamlit explorer.

---

## Quick start (run pipeline)
Install dependencies:
```bash
pip install -r requirements.txt

# Run pipeline steps (recommended order):
python scripts/data_acquisition.py      
python scripts/preprocessing.py       
python scripts/feature_engineering.py   
python scripts/eda_visualization.py

# Optional: launch the interactive dashboard:
streamlit run scripts/dashboard.py

.
├── data/               # raw and cleaned CSVs
├── figures/            # generated plots (PNG)
├── models/             # saved model artifacts (joblib)
├── reports/            # LDA topics, short reports
├── scripts/            # acquisition → preprocess → features → viz → dashboard
├── requirements.txt
