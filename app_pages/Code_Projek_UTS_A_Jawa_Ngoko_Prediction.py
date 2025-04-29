import streamlit as st
import pandas as pd
from utils.Code_Projek_UTS_A_Jawa_Ngoko_data_loader import load_data
from utils.Code_Projek_UTS_A_Jawa_Ngoko_preprocessing import preprocess_text
from models.Code_Projek_UTS_A_Jawa_Ngoko_multi_label_classifiers import get_multilabel_classifier, create_vectorizer, create_multilabel_target
from sklearn.model_selection import train_test_split
from utils.Code_Projek_UTS_A_Jawa_Ngoko_sidebar import beautify_sidebar

def run():

    st.title("🔮 Make Predictions")
    beautify_sidebar()
    # Access dataset
    if "df" not in st.session_state:
        st.session_state.df = load_data()
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = None
    if "label_columns" not in st.session_state:
        st.session_state.label_columns = None
    df = st.session_state.df

    # --- Input Text ---
    st.markdown("## ✏️ Input Review")

    user_input = st.text_area(
        "Tuliskan review (bebas):",
        "Avanza bahan bakar nya boros banget"
    )

    # --- Check Model ---
    if st.session_state.trained_model is None:
        st.warning("⚠️ No trained model found. Please train a model in the 'Model Training' page first.")
        model_status = st.empty()
    else:
        st.success(f"✅ Using trained **{st.session_state.model_name}** model.")

    # --- Prediction Button ---
    if st.button("🚀 Predict"):
        st.info("Making prediction...")

        # Preprocess
        preprocessed_input = preprocess_text(user_input)

        # Check if model exists
        if st.session_state.trained_model is not None:
            model = st.session_state.trained_model
            vectorizer = st.session_state.vectorizer
            label_columns = st.session_state.label_columns

            input_tfidf = vectorizer.transform([preprocessed_input])
        else:
            
            model_status.info("Training a default SVM model...")

            label_columns = [
                'fuel_negative', 'fuel_neutral', 'fuel_positive',
                'machine_negative', 'machine_neutral', 'machine_positive',
                'others_negative', 'others_neutral', 'others_positive',
                'part_negative', 'part_neutral', 'part_positive',
                'price_negative', 'price_neutral', 'price_positive',
                'service_negative', 'service_neutral', 'service_positive'
            ]

            y_multilabel = create_multilabel_target(df)
            X_train, _, y_train, _ = train_test_split(
                df['translated'], y_multilabel, test_size=0.2, random_state=42
            )

            vectorizer = create_vectorizer(max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            input_tfidf = vectorizer.transform([preprocessed_input])

            model = get_multilabel_classifier(
                "SVM", C=10, kernel='linear', gamma='scale'
            )
            model.fit(X_train_tfidf, y_train)
            

            st.session_state.trained_model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.model_name = "SVM (default)"
            st.session_state.label_columns = label_columns

        # --- Make Prediction ---
        prediction = model.predict(input_tfidf)

        st.success("🎉 Prediction complete!")

        # --- Display Input & Preprocessed ---
        st.markdown("## 📝 Review Details")
        st.write("**Original Input:**")
        st.info(user_input)
        st.write("**After Preprocessing:**")
        st.code(preprocessed_input)

        # --- Display Predictions ---
        st.markdown("## 🏷️ Prediction Results")

        results = [label_columns[i] for i in range(len(label_columns)) if prediction.toarray()[0, i] == 1]

        if results:
            # Group results by category
            categories = {
                'Fuel Sentiment': ('fuel_', 'red'),
                'Machine Sentiment': ('machine_', 'green'),
                'Part Sentiment': ('part_', 'blue'),
                'Other Sentiment': ('others_', 'orange'),
                'Price Sentiment': ('price_', 'purple'),
                'Service Sentiment': ('service_', 'brown')
            }

            cols = st.columns(6)
            for idx, (category_name, (prefix, color)) in enumerate(categories.items()):
                with cols[idx]:
                    st.markdown(f"**{category_name}**")
                    matched_labels = [label.replace(prefix, '') for label in results if label.startswith(prefix)]

                    if matched_labels:
                        for lbl in matched_labels:
                            st.markdown(f"<span style='color:{color};'>- {lbl}</span>", unsafe_allow_html=True)
                    else:
                        st.write("No prediction.")
        else:
            st.info("No labels predicted for this input.")
