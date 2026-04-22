import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import extract_features, FEATURE_COLS

print("Loading dataset...")
df_raw = pd.read_csv("data/phishing_site_urls.csv")

print(f"Dataset loaded: {len(df_raw)} rows")
print(f"Columns: {list(df_raw.columns)}")
print(f"\nLabel distribution:")
print(df_raw.iloc[:, -1].value_counts())

# Get URL and label columns
url_col = df_raw.columns[0]
label_col = df_raw.columns[-1]

print(f"\nURL column: {url_col}")
print(f"Label column: {label_col}")

# Convert labels to 0 and 1
df_raw[label_col] = df_raw[label_col].str.strip().str.lower()
df_raw["target"] = df_raw[label_col].map({"good": 0, "legitimate": 0, "benign": 0, "bad": 1, "phishing": 1, "malicious": 1})
df_raw = df_raw.dropna(subset=["target"])

print(f"\nAfter label mapping:")
print(df_raw["target"].value_counts())

print("\nExtracting features from URLs...")
records = []
for i, row in df_raw.iterrows():
    try:
        feats = extract_features(str(row[url_col]))
        feats["target"] = int(row["target"])
        records.append(feats)
    except Exception:
        pass
    if (len(records)) % 10000 == 0:
        print(f"  Processed {len(records)}/{len(df_raw)} URLs...")
df = pd.DataFrame(records)
df = df.dropna()

print(f"\nFinal dataset shape: {df.shape}")
print(f"\nLabel distribution:")
print(df["target"].value_counts())
print(df["target"].value_counts(normalize=True).round(3))

X = df[FEATURE_COLS]
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print(f"\nClassification Report:")
print(classification_report(y_test, preds, digits=4))

print("\nTop 10 most important features:")
importances = sorted(
    zip(FEATURE_COLS, model.feature_importances_),
    key=lambda x: x[1], reverse=True
)
for feat, score in importances[:10]:
    print(f"  {feat}: {score:.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/url_model.joblib")
joblib.dump(FEATURE_COLS, "models/feature_cols.joblib")

print("\nModel saved to models/url_model.joblib")
print("Feature columns saved to models/feature_cols.joblib")
