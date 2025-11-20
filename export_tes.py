import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib

# 1. Load dataset
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

count = df['label'].value_counts()
print("Jumlah data per label:")
for label, count in count.items():
    print(f"{label}: {count}")

# 5. Inisialisasi Stratified K-Fold (agar distribusi label tetap seimbang)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
fold = 1

# 6. Loop setiap fold
for train_index, test_index in kf.split(X_vec, y_enc):
    print(f"\n🟩 Fold {fold} -----------------------------")

    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y_enc[train_index], y_enc[test_index]

    # 7. Train model
    clf = SVC(kernel="linear", class_weight="balanced", probability=True)
    clf.fit(X_train, y_train)

    # 8. Evaluasi
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Akurasi Fold {fold}: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    fold += 1

# 9. Rata-rata hasil
print("\n📊 Rata-rata Akurasi dari 5-Fold:")
print(f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

# 10. Simpan model terakhir (opsional)
# artifact = {
#     "model": clf,
#     "vectorizer": vectorizer,
#     "label_encoder": label_encoder
# }
# joblib.dump(artifact, "svm_kfold_final.pkl")
# print("✅ Model terakhir berhasil disimpan ke svm_kfold_final.pkl")
