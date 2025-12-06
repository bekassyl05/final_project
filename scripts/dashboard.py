import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="News Trends EDA", layout="wide")
st.title("News Trends — Explorer")

uploaded = st.file_uploader("Upload features CSV (data/features_news.csv recommended)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded, low_memory=False)
else:
    if os.path.exists("../data/features_news.csv"):
        df = pd.read_csv("../data/features_news.csv", low_memory=False)
        st.info("Loaded data/features_news.csv automatically.")
    else:
        st.info("Upload data/features_news.csv to explore.")
        st.stop()

st.markdown("### Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))
col2.metric("Sources", df['source'].nunique() if 'source' in df.columns else 0)
col3.metric("Clusters", df['cluster'].nunique() if 'cluster' in df.columns else 0)

# Filters
with st.sidebar:
    st.header("Filters")
    sources = st.multiselect("Source", options=sorted(df['source'].unique()), default=sorted(df['source'].unique()))
    clusters = None
    if 'cluster' in df.columns:
        clusters = st.multiselect("Clusters", options=sorted(df['cluster'].unique()), default=sorted(df['cluster'].unique()))
    min_score = st.slider("Min score", int(float(df['score'].min() if 'score' in df.columns else 0)), int(float(df['score'].max() if 'score' in df.columns else 1000)), int(float(df['score'].min() if 'score' in df.columns else 0)))

dff = df[df['source'].isin(sources)]
if 'cluster' in dff.columns and clusters is not None:
    dff = dff = dff = dff = dff = None
    dff = dff[dff['cluster'].isin(clusters)]
else:
    dff = dff
if 'score' in dff.columns:
    dff = dff[dff['score'] >= min_score]

st.subheader("Posts by source")
fig1 = px.histogram(dff, x="source", title="Posts by source", labels={"source":"source"})
st.plotly_chart(fig1, use_container_width=True)

if 'timestamp_parsed' in dff.columns:
    st.subheader("Posts over time")
    dff['date'] = pd.to_datetime(dff['timestamp_parsed'], errors='coerce').dt.date
    counts = dff.groupby('date').size().reset_index(name='count')
    fig2 = px.line(counts, x='date', y='count', title='Posts per day')
    st.plotly_chart(fig2, use_container_width=True)

if 'cluster' in dff.columns:
    st.subheader("Cluster counts")
    ctab = dff['cluster'].value_counts().reset_index()
    ctab.columns = ['cluster','count']
    fig3 = px.bar(ctab, x='cluster', y='count', title='Cluster sizes', text='count')
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Sample posts (first 50)")
st.dataframe(dff[['source','title','url','score','num_comments']].head(50))
