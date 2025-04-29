import streamlit as st
import matplotlib.pyplot as plt
from utils.UTS_A_Jawa_Ngoko_visualization import plot_label_distribution
from utils.UTS_A_Jawa_Ngoko_sidebar import beautify_sidebar
import importlib

def run():
    st.title("📊 Dataset Explorer")
    if "df" not in st.session_state:
        st.session_state.df = load_data()
    df = st.session_state.df

    # Basic dataset info
    st.markdown("## 🗂️ Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Number of Samples", value=df.shape[0])

    with col2:
        st.metric(label="Number of Features", value=df.shape[1])

    # Sample data preview
    st.markdown("## 🧩 Sample Data")
    st.dataframe(df.head(), use_container_width=True)

    # Label distribution
    st.markdown("## 📈 Label Distribution")

    labels = ['fuel', 'machine', 'others', 'part', 'price', 'service']
    for i in range(0, len(labels), 3):
        cols = st.columns(3)
        for j, column in enumerate(labels[i:i+3]):
            with cols[j]:
                st.markdown(f"#### {column.capitalize()}")
                fig = plot_label_distribution(df, column)
                st.pyplot(fig, use_container_width=True)

    # Explore samples by sentiment
    st.markdown("## 💬 Sample Sentences by Sentiment")

    sentiment_to_explore = st.selectbox(
        "🔍 Choose aspect to explore:",
        labels,
        index=0
    )

    st.markdown(f"### ✨ {sentiment_to_explore.capitalize()} Sentiment Examples")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 😡 Negative")
        negatives = df[df[sentiment_to_explore] == 'negative'].head(3)
        for _, row in negatives.iterrows():
            st.markdown(f"> {row['translated']}")

    with col2:
        st.markdown("#### 😐 Neutral")
        neutrals = df[df[sentiment_to_explore] == 'neutral'].head(3)
        for _, row in neutrals.iterrows():
            st.markdown(f"> {row['translated']}")

    with col3:
        st.markdown("#### 😍 Positive")
        positives = df[df[sentiment_to_explore] == 'positive'].head(3)
        for _, row in positives.iterrows():
            st.markdown(f"> {row['translated']}")
