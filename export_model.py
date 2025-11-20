import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# 1. Load dataset
# pastikan ada kolom 'cleaned_content' dan 'label'
df = pd.read_csv("5000_balanced_edit.csv")

# 2. Pisahkan X dan y
X = df["cleaned_content"].astype(str)
y = df["label"]

# 3. Encode label
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# 4. TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# 6. Train SVM dengan class_weight balanced (biar tidak bias ke label mayoritas)
clf = SVC(kernel="linear", class_weight="balanced", probability=True)
clf.fit(X_train, y_train)

# 7. Evaluasi
y_pred = clf.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8. Simpan artifact (model + vectorizer + encoder)
artifact = {
    "model": clf,
    "vectorizer": vectorizer,
    "label_encoder": label_encoder
}

joblib.dump(artifact, "svm_edit.pkl")
print("✅ Model berhasil disimpan ke svm_final.pkl")
