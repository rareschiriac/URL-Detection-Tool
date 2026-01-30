import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLS = [
    "url_length", "valid_url", "at_symbol", "sensitive_words_count", "path_length",
    "isHttps", "nb_dots", "nb_hyphens", "nb_and", "nb_or", "nb_www", "nb_com", "nb_underscore"
]

df = pd.read_csv("data/feature_dataset.csv")


X = df[FEATURE_COLS].copy()
y = df["target"].astype(int)

print("Label distribution:\n", y.value_counts())
print("Label distribution (%):\n", y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced_subsample"
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))
print("\nReport:\n", classification_report(y_test, preds, digits=4))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/url_model.joblib")
print("\nSaved models/url_model.joblib")
