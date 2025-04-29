import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from models.UTS_A_Jawa_Ngoko_multi_label_classifiers import get_multilabel_classifier, create_vectorizer, evaluate_multilabel_model, create_multilabel_target
from utils.UTS_A_Jawa_Ngoko_visualization import plot_multilabel_confusion_matrix
from utils.UTS_A_Jawa_Ngoko_sidebar import beautify_sidebar

def run():
    st.title("🛠️ Model Training and Evaluation")
    beautify_sidebar()
    if "df" not in st.session_state:
        st.session_state.df = load_data()
    df = st.session_state.df

    # --- Model Selection ---
    st.markdown("## 🎯 Model & Training Settings")

    model_option = st.selectbox(
        "Select Model", ["KKN", "SVM"], index=1
    )

    # --- Vectorization Settings ---
    st.markdown("### 🧹 Text Vectorization Parameters")

    max_features = st.slider(
        "Max Features for TF-IDF Vectorizer",
        min_value=1000, max_value=10000, value=5000, step=1000
    )

    # --- Train-Test Split ---
    st.markdown("### 🧪 Train-Test Split")

    test_size = st.slider(
        "Test Set Size (%)", min_value=0.1, max_value=0.5, value=0.2, step=0.05
    )

    # --- Model-Specific Parameters ---
    st.markdown("### ⚙️ Model Hyperparameters")

    model_params = {}

    if model_option == "KKN":
        model_params['n_neighbors'] = st.slider(
            "Number of Neighbors (k)", 1, 20, value=5
        )
        model_params['p'] = st.slider(
            "Distance Metric (p)", 1, 5, value=2
        )
        model_params['weights'] = st.selectbox(
            "Weight Function", options=['uniform', 'distance'], index=0
        )

    elif model_option == "SVM":
        model_params['C'] = st.slider(
            "Regularization Parameter (C)", 0.01, 10.0, value=1.0, step=0.01
        )
        model_params['kernel'] = st.selectbox(
            "Kernel Function", ['linear', 'poly', 'rbf', 'sigmoid'], index=2
        )
        model_params['gamma'] = st.selectbox(
            "Kernel Coefficient (gamma)", ['scale', 'auto'], index=0
        )


    st.markdown("---")

    # --- Training Process ---
    if st.button("🚀 Train Model"):
        with st.spinner('Training in progress... Please wait.'):
            # Preparing Data
            X = df['sentence']
            label_columns = [
                'fuel_negative', 'fuel_neutral', 'fuel_positive',
                'machine_negative', 'machine_neutral', 'machine_positive',
                'others_negative', 'others_neutral', 'others_positive',
                'part_negative', 'part_neutral', 'part_positive',
                'price_negative', 'price_neutral', 'price_positive',
                'service_negative', 'service_neutral', 'service_positive'
            ]

            y_multilabel = create_multilabel_target(df)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_multilabel, test_size=test_size, random_state=42
            )

            # Vectorize text
            vectorizer = create_vectorizer(max_features)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            # Get model and train
            model = get_multilabel_classifier(model_option, **model_params)
            model.fit(X_train_tfidf, y_train)

            # Save artifacts to session
            st.session_state.trained_model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.model_name = model_option
            st.session_state.label_columns = label_columns

            # Evaluate
            accuracy, f1_micro, hamming, mcm, comparison_df, y_pred = evaluate_multilabel_model(
                model, X_test_tfidf, y_test, label_columns
            )

        # --- Results ---
        st.success("🎉 Training Complete!")

        st.markdown("## 📊 Model Performance")
        st.metric(label="Overall Accuracy", value=f"{accuracy:.4f}")
        st.metric(label="F1-Score (Micro Average)", value=f"{f1_micro:.4f}")
        st.metric(label="Hamming Loss", value=f"{hamming:.4f}")

        st.markdown("---")

        # --- Sample Predictions ---
        st.markdown("## 🔎 Sample Predictions")
        comparison_df['Text'] = X_test.reset_index(drop=True)
        st.dataframe(comparison_df.head(10), use_container_width=True)

        st.markdown("---")

        # --- Confusion Matrices ---
        st.markdown("## 🔥 Confusion Matrices")

        for row in range(3):
            cols = st.columns(3)
            for col in range(3):
                label_idx = row * 3 + col
                if label_idx < len(label_columns):
                    with cols[col]:
                        st.markdown(f"**{label_columns[label_idx]}**")
                        fig = plot_multilabel_confusion_matrix(
                            mcm[label_idx], label_columns[label_idx]
                        )
                        st.pyplot(fig)
                        plt.close(fig)
