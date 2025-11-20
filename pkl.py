import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# 1. Load Data
df = pd.read_csv("5000_balanced_final_custom.csv")

# 2. Encode label (multi-class)
le = LabelEncoder()
y = le.fit_transform(df['label'])
X_text = df['cleaned_content']

# 3. TF-IDF vectorization (max_features=5000)
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X_text)

count = df['label'].value_counts()
print("Jumlah data per label:")
for label, count in count.items():
    print(f"{label}: {count}")

# 4. K-Fold + SVM Linear
kf = KFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []
accuracies = []

# Tambahkan list untuk menyimpan semua y_test dan y_pred dari semua fold
all_y_true = []
all_y_pred = []

for i, (train_idx, test_idx) in enumerate(kf.split(X_tfidf, y), 1):
    X_train, X_test = X_tfidf[train_idx], X_tfidf[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train SVM with linear kernel
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n=== Fold {i} ===")
    print("Accuracy:", round(acc, 4))
    print("F1-score (macro):", round(f1, 4))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    accuracies.append(acc)
    f1_scores.append(f1)

    # ⬇ Simpan hasil prediksi dari fold ini
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# 5. Confusion Matrix GLOBAL (dari semua fold)
cm_global = confusion_matrix(all_y_true, all_y_pred)

# Plot Confusion Matrix Global
plt.figure(figsize=(8, 6))
sns.heatmap(cm_global, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# 6. Summary of all folds
print("\n=== RATA-RATA FOLD ===")
for i in range(len(accuracies)):
    print(f"Fold {i+1} - Accuracy: {round(accuracies[i], 4)}, F1-score: {round(f1_scores[i], 4)}")
print("Average Accuracy:", round(np.mean(accuracies), 4))
print("Average F1-score (macro):", round(np.mean(f1_scores), 4))

# 7. Train final model dengan semua data dan simpan sebagai .pkl
final_model = SVC(kernel='linear', random_state=42)
final_model.fit(X_tfidf, y)

with open("svm_final.pkl", "wb") as f:
    pickle.dump({
        "model": final_model,
        "vectorizer": vectorizer,
        "label_encoder": le
    }, f)

print("✅ Model final berhasil disimpan ke svm_final.pkl")
