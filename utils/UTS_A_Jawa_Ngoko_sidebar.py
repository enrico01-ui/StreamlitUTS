import streamlit as st

def beautify_sidebar():
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                border-right: 1px solid #d3d3d3;
                border-radius: 0px 10px 10px 0px;
                font-family: 'Roboto', sans-serif;
                color: #333;
            }
            [data-testid="stSidebar"] .css-1d391kg {
                font-size: 18px;
                font-weight: 600;
                color: #444;
            }
            [data-testid="stSidebar"] .css-1d391kg:hover {
                color: #007bff;
                transition: all 0.3s ease;
            }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

