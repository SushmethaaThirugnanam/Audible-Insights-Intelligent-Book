"""
Audible Insights — Streamlit App
=================================
Run:
    pip install streamlit pandas scikit-learn plotly
    streamlit run app.py
"""

import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Audible Insights",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #13151a; }
    .metric-card {
        background: #1a1d24; border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 20px; text-align: center;
    }
    .book-card {
        background: #1a1d24; border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 16px; margin-bottom: 10px;
    }
    h1, h2, h3 { font-family: Georgia, serif; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE (cached)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_duration(text) -> int:
    if pd.isna(text):
        return 0
    h = re.search(r'(\d+)\s*hour', str(text))
    m = re.search(r'(\d+)\s*min',  str(text))
    return (int(h.group(1)) if h else 0) * 60 + (int(m.group(1)) if m else 0)


def _extract_genres(raw: str) -> list:
    genres = []
    for part in str(raw).split(','):
        match = re.match(r'#\d+\s+in\s+(.+)', part.strip())
        if match:
            g = match.group(1).strip()
            if 'Audible Audiobooks' not in g and 'See Top' not in g:
                genres.append(g)
    return genres


@st.cache_data(show_spinner="Loading & cleaning data…")
def load_and_clean(path1: str, path2: str) -> pd.DataFrame:
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df  = pd.merge(df2, df1[['Book Name', 'Author', 'Number of Reviews']],
                   on=['Book Name', 'Author'], how='left', suffixes=('', '_df1'))

    df = df[df['Rating'] >= 0].copy()
    df['Description']     = df['Description'].fillna('').str.strip()
    df['Ranks and Genre'] = df['Ranks and Genre'].fillna('')
    df['duration_mins']   = df['Listening Time'].apply(_parse_duration)
    df['genres_list']     = df['Ranks and Genre'].apply(_extract_genres)
    df['primary_genre']   = df['genres_list'].apply(lambda x: x[0] if x else 'Other')
    df['Price']           = pd.to_numeric(df['Price'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('Number of Reviews', ascending=False)
    df = df.drop_duplicates(subset='Book Name', keep='first').reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Building recommendation model…")
def build_model(df: pd.DataFrame):
    df_top = df.nlargest(600, 'Number of Reviews').reset_index(drop=True)
    df_top['text_features'] = (
        df_top['Description'].str[:400]
        + ' ' + df_top['primary_genre']
        + ' ' + df_top['Author']
    )
    vec = TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1, 2))
    mat = vec.fit_transform(df_top['text_features'].fillna(''))
    sim = cosine_similarity(mat)
    return df_top, sim


def get_recommendations(title: str, df_top: pd.DataFrame, sim: np.ndarray, n: int = 5):
    matches = df_top[df_top['Book Name'].str.contains(title, case=False, na=False)]
    if matches.empty:
        return pd.DataFrame()
    idx    = matches.index[0]
    scores = sorted(enumerate(sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    result = df_top.iloc[[i for i, _ in scores]][
        ['Book Name', 'Author', 'Rating', 'Price', 'primary_genre', 'duration_mins']
    ].copy()
    result['Similarity'] = [round(s, 3) for _, s in scores]
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎧 Audible Insights")
    st.markdown("---")

    st.markdown("### Upload Datasets")
    file1 = st.file_uploader("Audible_Catlog.csv",          type="csv", key="f1")
    file2 = st.file_uploader("Audible_Catlog_Advanced_Features.csv", type="csv", key="f2")

    st.markdown("---")
    page = st.radio("Navigate", ["📊 Overview", "🔍 EDA", "🤖 Recommender", "📚 Browse Genre"])
    st.markdown("---")
    st.caption("Built with TF-IDF · Cosine Similarity · Streamlit")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

if not file1 or not file2:
    st.title("🎧 Audible Insights")
    st.markdown("### Intelligent Book Recommendation System")
    st.info("👈  Upload both CSV files in the sidebar to get started.", icon="📂")
    st.markdown("""
    **What this app does:**
    - Cleans and merges your two Audible datasets
    - Runs exploratory data analysis with interactive charts
    - Builds a TF-IDF + cosine similarity recommendation engine
    - Lets you search for any book and get 5 similar recommendations
    """)
    st.stop()

df         = load_and_clean(file1, file2)
df_top, sim = build_model(df)

# Shared genre counts
all_genres  = [g for gl in df['genres_list'] for g in gl]
genre_counts = Counter(all_genres).most_common(10)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Audible Catalog — Overview")
    st.markdown("High-level statistics from the cleaned dataset.")
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Books",    f"{len(df):,}")
    c2.metric("Average Rating", f"{df['Rating'].mean():.2f} ★")
    c3.metric("Unique Genres",  df['primary_genre'].nunique())
    c4.metric("Unique Authors", f"{df['Author'].nunique():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Top genres bar
        gdf = pd.DataFrame(genre_counts, columns=['Genre', 'Count'])
        fig = px.bar(gdf, x='Count', y='Genre', orientation='h',
                     title='Top 10 Genres', color='Count',
                     color_continuous_scale='YlOrBr')
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis={'categoryorder': 'total ascending'},
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Rating donut
        bins   = [0, 3, 3.5, 4, 4.5, 5.01]
        labels = ['< 3', '3–3.5', '3.5–4', '4–4.5', '4.5–5']
        df['rating_bucket'] = pd.cut(df['Rating'], bins=bins, labels=labels, right=False)
        rd = df['rating_bucket'].value_counts().sort_index()
        fig2 = px.pie(values=rd.values, names=rd.index, hole=0.5,
                      title='Rating Distribution',
                      color_discrete_sequence=px.colors.sequential.YlOrBr)
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    # Top authors
    author_stats = (
        df.groupby('Author')
          .agg(avg_rating=('Rating', 'mean'), count=('Rating', 'count'))
          .query('count >= 3')
          .nlargest(8, 'avg_rating')
          .reset_index()
    )
    fig3 = px.bar(author_stats, x='avg_rating', y='Author', orientation='h',
                  title='Top Authors by Average Rating (min 3 books)',
                  color='avg_rating', color_continuous_scale='YlOrBr',
                  range_x=[4.5, 5.0])
    fig3.update_layout(showlegend=False, coloraxis_showscale=False,
                       yaxis={'categoryorder': 'total ascending'},
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: EDA
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Deep-dive into patterns and trends in the Audible catalog.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Price distribution
        df_price = df[df['Price'] > 0]
        fig = px.histogram(df_price, x='Price', nbins=50, range_x=[0, 5000],
                           title='Price Distribution (INR, capped ₹5,000)',
                           color_discrete_sequence=['#c9a84c'])
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Reviews vs Rating scatter
        sample = df[df['Number of Reviews'] > 0].sample(min(500, len(df)), random_state=42)
        fig2 = px.scatter(sample, x='Number of Reviews', y='Rating',
                          color='primary_genre', hover_data=['Book Name', 'Author'],
                          title='Ratings vs Number of Reviews',
                          log_x=True, opacity=0.6)
        fig2.update_layout(showlegend=False,
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Duration distribution
        df_dur = df[df['duration_mins'] > 0]
        fig3 = px.histogram(df_dur, x='duration_mins', nbins=40, range_x=[0, 1200],
                            title='Listening Duration (minutes)',
                            color_discrete_sequence=['#3b82f6'])
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Genre treemap
        gdf = pd.DataFrame(genre_counts, columns=['Genre', 'Count'])
        fig4 = px.treemap(gdf, path=['Genre'], values='Count',
                          title='Genre Treemap (Top 10)',
                          color='Count', color_continuous_scale='YlOrBr')
        fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

    # Key findings
    st.markdown("---")
    st.markdown("### 📌 Key Findings")
    f1, f2, f3 = st.columns(3)
    f1.success(f"**{(df['Rating'] >= 4.5).mean()*100:.0f}%** of books rated 4.5★ or above")
    f2.info(   f"Median price is **₹{df[df['Price']>0]['Price'].median():,.0f}**")
    f3.warning(f"Average listening time: **{df[df['duration_mins']>0]['duration_mins'].mean()/60:.1f} hours**")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 Recommender":
    st.title("🤖 Book Recommender")
    st.markdown("Select a book you love — we'll find 5 similar titles using **TF-IDF + Cosine Similarity**.")
    st.markdown("---")

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        query = st.text_input("🔍 Search for a book", placeholder="e.g. Think Like a Monk, Atomic Habits…")
    with col2:
        genre_filter = st.selectbox("Filter by genre",
                                    ["All"] + sorted(df_top['primary_genre'].unique()))
    with col3:
        n_recs = st.slider("Recommendations", 3, 10, 5)

    # Filter browseable books
    display_df = df_top.copy()
    if genre_filter != "All":
        display_df = display_df[display_df['primary_genre'] == genre_filter]

    if query:
        matches = display_df[display_df['Book Name'].str.contains(query, case=False, na=False)]
    else:
        matches = display_df.head(12)

    st.markdown(f"**{len(matches)} books found** — click a title to get recommendations")
    st.markdown("---")

    # Book grid
    cols = st.columns(3)
    for i, (_, row) in enumerate(matches.head(12).iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{row['Book Name'][:60]}**")
                st.caption(f"{row['Author']}  ·  {row['primary_genre'][:30]}")
                st.markdown(f"⭐ {row['Rating']}  ·  ₹{row['Price']:,}")
                if st.button("Get Recommendations", key=f"rec_{i}"):
                    st.session_state['selected_book'] = row['Book Name']

    # Show recommendations
    if 'selected_book' in st.session_state:
        st.markdown("---")
        title = st.session_state['selected_book']
        st.markdown(f"### 📖 Because you liked *{title[:60]}*…")

        recs = get_recommendations(title, df_top, sim, n=n_recs)
        if recs.empty:
            st.warning("No recommendations found. Try a different book.")
        else:
            for _, r in recs.iterrows():
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns([4, 2, 1, 1])
                    c1.markdown(f"**{r['Book Name'][:70]}**  \n{r['Author']}")
                    c2.caption(r['primary_genre'][:35])
                    c3.markdown(f"⭐ {r['Rating']}")
                    c4.markdown(f"₹{r['Price']:,}")

    # Hidden gems section
    st.markdown("---")
    st.markdown("### 💎 Hidden Gems")
    st.caption("Highly rated books (≥ 4.5★) with fewer than 500 reviews")
    gems = df[
        (df['Rating'] >= 4.5) &
        (df['Number of Reviews'] <= 500) &
        (df['Number of Reviews'] > 0)
    ].nlargest(5, 'Rating')[['Book Name', 'Author', 'Rating', 'Number of Reviews', 'primary_genre']]

    st.dataframe(gems, use_container_width=True, hide_index=True,
                 column_config={
                     'Book Name':        st.column_config.TextColumn(width='large'),
                     'Rating':           st.column_config.NumberColumn(format="⭐ %.1f"),
                     'Number of Reviews':st.column_config.NumberColumn("Reviews"),
                 })


#elif page == "🤖 Recommender":
    """st.title("🤖 Book Recommender")
    query = st.text_input("Enter a book you love:")
    if query:
        matches = df[df['Book Name'].str.contains(query, case=False)]
        if not matches.empty:
            idx = matches.index[0]
            scores = sorted(enumerate(sim[idx]), key=lambda x:x[1], reverse=True)[1:6]
            recs = df.iloc[[i for i,_ in scores]][['Book Name','Author','Rating','primary_genre']]
            st.table(recs)
        else:
            st.warning("No matches found")"""


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: BROWSE BY GENRE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📚 Browse Genre":
    st.title("📚 Browse by Genre")
    st.markdown("Explore top-rated audiobooks in each category.")
    st.markdown("---")

    genres_available = [g for g, _ in genre_counts]
    selected_genre   = st.selectbox("Choose a genre", genres_available)

    min_rating = st.slider("Minimum rating", 3.0, 5.0, 4.0, 0.1)

    subset = df[
        (df['primary_genre'] == selected_genre) &
        (df['Rating'] >= min_rating)
    ].nlargest(20, 'Rating')

    st.markdown(f"**{len(subset)} books** in *{selected_genre}* with rating ≥ {min_rating}★")
    st.markdown("---")

    for rank, (_, row) in enumerate(subset.iterrows(), 1):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"**#{rank}  {row['Book Name']}**")
                st.caption(f"by {row['Author']}")
                if row['Description']:
                    st.markdown(f"<small>{str(row['Description'])[:180]}…</small>",
                                unsafe_allow_html=True)
            with c2:
                st.metric("Rating", f"{row['Rating']} ★")
                st.caption(f"₹{row['Price']:,}")
                dur = row['duration_mins']
                if dur:
                    st.caption(f"⏱ {dur//60}h {dur%60}m")