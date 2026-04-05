import streamlit as st

st.title("📚 Audible Insights: Intelligent Book Recommendations")

user_input = st.text_input("Enter a book title you like:")
if user_input:
    recommendations = recommend_books(user_input, n=5)
    st.write("Top Recommendations:")
    for rec in recommendations:
        st.write(f"- {rec}")

# Add EDA visualizations
st.subheader("Genre Distribution")
st.bar_chart(genre_counts[:10])