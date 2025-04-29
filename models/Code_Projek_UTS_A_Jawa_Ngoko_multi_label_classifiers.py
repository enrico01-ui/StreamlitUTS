from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
import pandas as pd


def create_multilabel_target(df):
    """
    Create a multilabel target DataFrame from the original dataset
    """
    y_multilabel = pd.DataFrame()
    for sentiment in ['fuel', 'machine','others', 'part' , 'price', 'service']:
        for label in ['negative', 'neutral', 'positive']:
            col_name = f"{sentiment}_{label}"
            y_multilabel[col_name] = (df[sentiment] == label).astype(int)
    return y_multilabel


def get_multilabel_classifier(model_name, **params):
    """
    Return a multi-label classifier with the specified base model
    """
    if model_name == "KKN":
        base_classifier = KNeighborsClassifier(
            n_neighbors=params.get('classifier__n_neighbors', 5),
            p=params.get('classifier__p', 2),
            weights=params.get('classifier__weights', 'uniform')
        )
    elif model_name == "SVM":
        base_classifier = SVC(
            C=params.get('C', 1.0),
            probability=True,
            random_state=42
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return BinaryRelevance(classifier=base_classifier)


def create_vectorizer(max_features=5000):
    """
    Create a TF-IDF vectorizer
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),      # unigram + bigram
        min_df=3,                # buang kata yg jarang banget
        max_df=0.9,              # buang kata yg terlalu umum
        sublinear_tf=True,       # skala TF biar smooth
        strip_accents='unicode', # bantu normalisasi huruf
        norm='l2',               # normalisasi fitur
        lowercase=True           # tetap lowercase (walau udah kita bersihin juga)
    )


def evaluate_multilabel_model(model, X_test, y_test, label_columns):
    """
    Evaluate a multi-label model and return performance metrics
    """
    from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, f1_score, hamming_loss

# Prediksi
    y_pred = model.predict(X_test)

    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    f1_micro = f1_score(y_test, y_pred, average='micro')  # Tambahan
    hamming = hamming_loss(y_test, y_pred)                # Tambahan
    mcm = multilabel_confusion_matrix(y_test, y_pred.toarray())

    # Buat DataFrame perbandingan
    comparison_df = pd.DataFrame()
    for i, label_col in enumerate(label_columns):
        comparison_df[f'{label_col}_actual'] = y_test[label_col].reset_index(drop=True)
        comparison_df[f'{label_col}_predicted'] = y_pred.toarray()[:, i]
        comparison_df[f'{label_col}_match'] = comparison_df[f'{label_col}_actual'] == comparison_df[f'{label_col}_predicted']

    # Tampilkan hasil
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"F1-Micro: {f1_micro:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Return kalau dipakai dalam fungsi
    return accuracy, f1_micro, hamming, mcm, comparison_df, y_pred
