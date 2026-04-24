import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from collections import Counter

# Load data + model
df = pd.read_csv("data/books_clustered.csv")
vec = pickle.load(open("models/tfidf_vectorizer.pkl","rb"))
sim = pickle.load(open("models/similarity_matrix.pkl","rb"))

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["📊 Overview","🔍 EDA","🤖 Recommender","📚 Browse Genre"])

all_genres  = [g for gl in df['genres_list'] for g in gl]
genre_counts = Counter(all_genres).most_common(10)
df_top = df.nlargest(600, 'Number of Reviews').reset_index(drop=True)

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

# Overview page
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
    st.subheader("📌 Key Insights")
    st.markdown("""
    - Most books are rated above 4★, showing strong overall quality.
    - Median price is around ₹{:.0f}, making audiobooks affordable.
    - Average listening time is {:.1f} hours, suggesting long-form content dominates.
    - Genres like Self-Help, Fiction, and Business are among the most popular.
    """.format(
        df[df['Price']>0]['Price'].median(),
        df[df['duration_mins']>0]['duration_mins'].mean()/60
    ))

# EDA page
elif page == "🔍 EDA":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Deep-dive into patterns and trends in the Audible catalog.")
    st.markdown("---")

    # Row 1: Price distribution & Reviews vs Rating
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

    # Row 2: Duration distribution
    col3 = st.columns(1)[0]
    with col3:
        # Duration distribution
        df_dur = df[df['duration_mins'] > 0]
        fig3 = px.histogram(df_dur, x='duration_mins', nbins=40, range_x=[0, 1200],
                            title='Listening Duration (minutes)',
                            color_discrete_sequence=['#3b82f6'])
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)


    # Key findings
    st.markdown("---")
    st.markdown("### 📌 Key Findings")
    f1, f2, f3 = st.columns(3)
    f1.success(f"**{(df['Rating'] >= 4.5).mean()*100:.0f}%** of books rated 4.5★ or above")
    f2.info(   f"Median price is **₹{df[df['Price']>0]['Price'].median():,.0f}**")
    f3.warning(f"Average listening time: **{df[df['duration_mins']>0]['duration_mins'].mean()/60:.1f} hours**")


# Recommender page          
elif page == "🤖 Recommender":
    st.title("🤖 Book Recommender")
    st.markdown("Select a book you love — we'll find 5 similar titles using **TF-IDF + Cosine Similarity**.")
    st.markdown("---")

    col1, col2, col3 = st.columns([3, 3, 1])

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

# Browse Genre page
elif page == "📚 Browse Genre":
    st.title("📚 Browse by Genre")
    st.markdown("Explore top-rated audiobooks in each category.")
    st.markdown("---")

    selected_genre = st.selectbox(
        "Choose a genre",
        sorted(df['primary_genre'].unique()),   
        key="browse_genre"                    
    )

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

