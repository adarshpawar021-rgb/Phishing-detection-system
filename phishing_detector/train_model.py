"""
train_model.py — Hybrid Ensemble Training Pipeline

Trains a soft-voting ensemble of three complementary models:
  1. Random Forest          — strong on structural features, fast
  2. Gradient Boosting      — iterative boosting, catches subtle patterns
  3. Logistic Regression    — calibrated probability, linear boundary

The ensemble uses soft voting (average probabilities), giving well-calibrated
confidence scores. The trained VotingClassifier is saved as phishing_model.pkl.
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(BASE_DIR, "data", "PhiUSIIL_Phishing_URL_Dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")


# ============================================================
# 30 FEATURES (MUST MATCH feature_extractor.MODEL_FEATURES)
# ============================================================

FEATURES = [
    # Original 21
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoofobfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOFQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "ISHTTPS",
    "Bank",
    "Pay",
    "Crypto",
    # New 9
    "URLEntropy",
    "DomainEntropy",
    "IsPunycode",
    "IsSuspiciousTLD",
    "HasRedirectInURL",
    "URLDepth",
    "SubdomainLength",
    "HasBrandMismatch",
    "DomainHyphenCount",
]


# ============================================================
# COLUMN NAME NORMALIZATION
# ============================================================

def normalize_column_name(name):
    """
    Converts column names into a standard comparison format.

    Example:
        NoofobfuscatedChar -> noofobfuscatedchar
        NoOfObfuscatedChar -> noofobfuscatedchar
        NOOFQMarkInURL     -> noofqmarkinurl
        NoOFQMarkInURL     -> noofqmarkinurl
        ISHTTPS            -> ishttps
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


# ============================================================
# COMPUTE DERIVED FEATURES FROM DATASET URL COLUMN
# ============================================================

def _compute_derived_features(df):
    """
    The PhiUSIIL dataset contains many URL features natively, but the 9 new
    features must be computed from the URL column.  This function adds them.
    """
    try:
        from phishing_detector.feature_extractor import extract_url_features
    except ImportError:
        from feature_extractor import extract_url_features

    NEW_FEATURES = [
        "URLEntropy", "DomainEntropy", "IsPunycode", "IsSuspiciousTLD",
        "HasRedirectInURL", "URLDepth", "SubdomainLength",
        "HasBrandMismatch", "DomainHyphenCount",
    ]

    # Check if the URL column exists
    url_col = None
    for col in df.columns:
        if col.strip().lower() == "url":
            url_col = col
            break

    if url_col is None:
        print("  Warning: No 'URL' column found — new features will be set to 0.")
        for feat in NEW_FEATURES:
            df[feat] = 0
        return df

    print(f"  Computing {len(NEW_FEATURES)} new features from URL column ...")
    print("  This may take 1-2 minutes for large datasets ...")

    extracted = [extract_url_features(u) for u in df[url_col]]
    feat_df = pd.DataFrame(extracted)

    for feat in NEW_FEATURES:
        if feat in feat_df.columns:
            df[feat] = feat_df[feat].values
        else:
            df[feat] = 0

    return df


# ============================================================
# TRAINING PIPELINE
# ============================================================

def train_model():
    print("\n" + "=" * 60)
    print("       HYBRID PHISHING ENSEMBLE — TRAINING")
    print("=" * 60)

    # ---- Load dataset ----
    print("\n[1/6] Loading dataset ...")
    if not os.path.exists(DATASET_PATH):
        print(f"\nERROR: Dataset not found at {DATASET_PATH}")
        raise SystemExit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ---- Normalize column names ----
    print("\n[2/6] Normalizing column names ...")
    normalized_columns = {}
    for column in df.columns:
        normalized_name = normalize_column_name(column)
        if normalized_name not in normalized_columns:
            normalized_columns[normalized_name] = column

    # Map original 21 features that already exist in the dataset
    column_mapping = {}
    for feature in FEATURES:
        normalized_feature = normalize_column_name(feature)
        if normalized_feature in normalized_columns:
            actual_column = normalized_columns[normalized_feature]
            column_mapping[actual_column] = feature

    # Map label column
    normalized_label = normalize_column_name("label")
    label_column = normalized_columns.get(normalized_label)

    df = df.rename(columns=column_mapping)
    if label_column is not None and label_column in df.columns:
        df = df.rename(columns={label_column: "label"})

    print(f"  Mapped {len(column_mapping)} feature columns")

    # ---- Compute new derived features ----
    print("\n[3/6] Computing new derived features ...")
    df = _compute_derived_features(df)

    # ---- Validate required columns ----
    required_columns = FEATURES + ["label"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"\nERROR: Missing required columns after mapping: {missing_columns}")
        raise SystemExit(1)

    # ---- Prepare X, y ----
    print("\n[4/6] Preparing feature matrix ...")
    X = df[FEATURES].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df["label"], errors="coerce")
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    print(f"  Usable samples: {len(X):,}")
    print(f"  Class distribution:\n    {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ---- Build ensemble ----
    print("\n[5/6] Building and training the soft-voting ensemble ...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.10,
        max_depth=5,
        subsample=0.80,
        random_state=42,
    )

    # Logistic Regression needs feature scaling
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("lr", lr_pipeline),
        ],
        voting="soft",
        weights=[3, 3, 1],   # RF and GB carry more weight than LR
        n_jobs=1,             # VotingClassifier handles parallelism internally
    )

    ensemble.fit(X_train, y_train)
    print("  Ensemble training completed!")

    # ---- Evaluate ----
    print("\n[6/6] Evaluating on held-out test set ...")
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  Accuracy  : {accuracy * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ---- Save ----
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ensemble, MODEL_PATH)
    joblib.dump(FEATURES, FEATURES_PATH)

    print(f"\n  Model saved  : {MODEL_PATH}")
    print(f"  Features saved: {FEATURES_PATH}")
    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_model()