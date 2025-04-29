from utils.Code_Projek_UTS_A_Jawa_Ngoko_data_loader import load_data
import streamlit as st
from utils.Code_Projek_UTS_A_Jawa_Ngoko_sidebar import beautify_sidebar
import importlib

st.set_page_config(
    page_title="Multi-label Text Classification",
    layout="wide",
    page_icon="🚗"
)

# Hide default page navigation sidebar
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
        <style>
            .sidebar-title {
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 10px;
                color: #4CAF50;
            }
            .sidebar-section {
                padding: 15px;
                background-color: #f0f2f6;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .sidebar-radio label {
                display: block;
                padding: 8px 16px;
                border-radius: 5px;
                transition: all 0.3s ease;
            }
            .sidebar-radio label:hover {
                background-color: #d2e3fc;
                cursor: pointer;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🧠 Klasifikasi Teks<br>Jawa Ngoko</div>', unsafe_allow_html=True)
    
    with st.container():
        selected_page = st.radio(
            "📚 Navigation",
            ["🏠 Main", "📊 Dataset Explorer", "🛠️ Model Training", "🔮 Prediction"],
            index=0,
            key="sidebar_navigation"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.info("Use the sidebar to navigate between app pages 🚀", icon="ℹ️")
    st.markdown('</div>', unsafe_allow_html=True)


if selected_page == "🏠 Main":
    st.markdown('<p class="big-font center-text">🚗 Automotive Reviews Multi-label Text Classification</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">Multi-label classification for automotive reviews across different aspects: fuel, machine, other, part, price, and service.</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown("### 🎯 Welcome to the Multi-label Text Classification App")
        st.write("""
        This application demonstrates text classification that can predict multiple labels simultaneously. 
        Use this tool to explore, train, and make predictions on automotive review texts.
        """)

        st.markdown("#### 📋 Available Pages:")
        st.markdown("""
        - **Dataset Explorer**: Explore and understand the dataset  
        - **Model Training**: Train and evaluate multi-label classification models  
        - **Prediction**: Make predictions on new text inputs
        """)
        st.info("👉 Use the sidebar to navigate between pages.", icon="🧭")

    
    st.markdown("### 🗂️ Dataset Overview")
    if "df" not in st.session_state:
        st.session_state.df = load_data()
    df = st.session_state.df
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Number of Samples", value=df.shape[0])
    with col2:
        st.metric(label="Number of Features", value=df.shape[1])

    st.dataframe(df.head(5), use_container_width=True)

else:
    
    if selected_page == "📊 Dataset Explorer":
        page = importlib.import_module("app_pages.Code_Projek_UTS_A_Jawa_Ngoko_dataset_explorer")
        page.run()
    elif selected_page == "🛠️ Model Training":
        page = importlib.import_module("app_pages.Code_Projek_UTS_A_Jawa_Ngoko_Model_Training")
        page.run()
    elif selected_page == "🔮 Prediction":
        page = importlib.import_module("app_pages.Code_Projek_UTS_A_Jawa_Ngoko_Prediction")
        page.run()
