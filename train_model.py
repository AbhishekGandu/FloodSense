"""
Rising Waters: Flood Prediction System
========================================
This script loads historical rainfall/weather data, performs exploratory
data analysis, preprocesses the dataset, trains multiple classification
models, compares their performance, and saves the best model along with
the fitted StandardScaler and feature column names.
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Try to import XGBoost; fall back to GradientBoostingClassifier if unavailable
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using GradientBoostingClassifier as fallback.")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path configuration (works regardless of where the script is run from)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "flood dataset.xlsx")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_dataset(path):
    """Load the Excel dataset and return a pandas DataFrame."""
    print("=" * 60)
    print("STEP 1: Loading Dataset")
    print("=" * 60)
    df = pd.read_excel(path)
    return df


def explore_dataset(df):
    """Print basic dataset information and summaries."""
    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Dataset Info ---")
    df.info()

    print("\n--- Statistical Summary ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    print(missing_df[missing_df["Missing Count"] > 0] if missing.sum() > 0 else "No missing values found.")
    if missing.sum() == 0:
        print(missing_df)


def univariate_analysis(df, output_dir):
    """
    Create histograms for numerical columns and box plots for outlier detection.
    Plots are saved to the analysis folder.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Univariate Analysis")
    print("=" * 60)

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Histograms / distribution plots
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    hist_path = os.path.join(output_dir, "histograms.png")
    plt.savefig(hist_path, dpi=120)
    plt.close()
    print(f"Histograms saved to: {hist_path}")

    # Box plots for outlier detection
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        sns.boxplot(y=df[col], ax=axes[i], color="lightblue")
        axes[i].set_title(f"Box Plot: {col}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    box_path = os.path.join(output_dir, "boxplots.png")
    plt.savefig(box_path, dpi=120)
    plt.close()
    print(f"Box plots saved to: {box_path}")


def multivariate_analysis(df, output_dir):
    """Create a correlation heatmap for numerical features."""
    print("\n" + "=" * 60)
    print("STEP 3: Multivariate Analysis")
    print("=" * 60)

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
    )
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=120)
    plt.close()
    print(f"Correlation heatmap saved to: {heatmap_path}")


def handle_missing_values(df):
    """Fill numerical missing values with median; categorical with mode."""
    print("\n" + "=" * 60)
    print("STEP 4: Handling Missing Values")
    print("=" * 60)

    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"  Filled '{col}' missing values with median: {median_val}")
            else:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"  Filled '{col}' missing values with mode: {mode_val}")

    print("Missing value handling complete.")
    return df_clean


def cap_outliers_iqr(df, exclude_cols=None):
    """
    Cap outliers using the IQR method for numerical columns.
    Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are capped.
    """
    print("\n" + "=" * 60)
    print("STEP 5: Outlier Handling (IQR Capping)")
    print("=" * 60)

    if exclude_cols is None:
        exclude_cols = []

    df_capped = df.copy()
    numerical_cols = df_capped.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [c for c in numerical_cols if c not in exclude_cols]

    for col in numerical_cols:
        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_before = ((df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)).sum()
        df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"  '{col}': capped {outliers_before} outlier(s) to [{lower_bound:.2f}, {upper_bound:.2f}]")

    return df_capped


def encode_categorical(df, target_col="flood"):
    """Convert categorical columns to numeric using LabelEncoder."""
    print("\n" + "=" * 60)
    print("STEP 6: Encoding Categorical Columns")
    print("=" * 60)

    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=["object", "category"]).columns.tolist()

    if not categorical_cols:
        print("No categorical columns found. Skipping encoding.")
        return df_encoded

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded '{col}' with LabelEncoder")

    return df_encoded


def xgboost_model():
    """
    Return an XGBClassifier if available, otherwise GradientBoostingClassifier.
    Function name is kept as xgboost_model per project requirements.
    """
    if XGBOOST_AVAILABLE:
        return XGBClassifier(random_state=10, use_label_encoder=False, eval_metric="logloss")
    return GradientBoostingClassifier(random_state=10)


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Train a model and print accuracy, confusion matrix, and classification report."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {model_name} ---")
    print(f"Accuracy Score: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, acc


def main():
    """Main training pipeline."""
    # 1. Load dataset
    df = load_dataset(DATASET_PATH)

    # 2. Exploratory data analysis
    explore_dataset(df)
    univariate_analysis(df, ANALYSIS_DIR)
    multivariate_analysis(df, ANALYSIS_DIR)

    # 3. Preprocessing
    df = handle_missing_values(df)
    df = cap_outliers_iqr(df, exclude_cols=["flood"])
    df = encode_categorical(df)

    # 4. Split features (X) and target (y)
    print("\n" + "=" * 60)
    print("STEP 7: Feature-Target Split")
    print("=" * 60)

    target_col = "flood"
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols]
    y = df[target_col]

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Target distribution:\n{y.value_counts()}")

    # 5. Train-test split
    print("\n" + "=" * 60)
    print("STEP 8: Train-Test Split")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=10
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 6. Feature scaling (fit on train only, transform test)
    print("\n" + "=" * 60)
    print("STEP 9: Feature Scaling (StandardScaler)")
    print("=" * 60)

    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)  # Important: only transform, never fit on test data

    # 7. Train multiple models
    print("\n" + "=" * 60)
    print("STEP 10: Model Training and Evaluation")
    print("=" * 60)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=10),
        "Random Forest": RandomForestClassifier(random_state=10),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgboost_model(),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        trained_model, acc = train_and_evaluate(
            model, name, X_train_scaled, X_test_scaled, y_train, y_test
        )
        results[name] = acc
        trained_models[name] = trained_model

    # 8. Compare model accuracies
    print("\n" + "=" * 60)
    print("STEP 11: Model Comparison")
    print("=" * 60)

    comparison_df = pd.DataFrame(
        {"Model": list(results.keys()), "Accuracy": list(results.values())}
    ).sort_values("Accuracy", ascending=False)
    comparison_df.reset_index(drop=True, inplace=True)
    print("\nModel Accuracy Comparison Table:")
    print(comparison_df.to_string(index=False))

    # 9. Select and save the best model
    best_model_name = comparison_df.iloc[0]["Model"]
    best_accuracy = comparison_df.iloc[0]["Accuracy"]
    best_model = trained_models[best_model_name]

    print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

    model_path = os.path.join(MODELS_DIR, "floods.save")
    scaler_path = os.path.join(MODELS_DIR, "transform.save")
    features_path = os.path.join(MODELS_DIR, "features.save")

    joblib.dump(best_model, model_path)
    joblib.dump(sc, scaler_path)
    joblib.dump(feature_cols, features_path)

    print("\n" + "=" * 60)
    print("STEP 12: Saving Artifacts")
    print("=" * 60)
    print(f"Best model saved to:       {model_path}")
    print(f"StandardScaler saved to:   {scaler_path}")
    print(f"Feature names saved to:    {features_path}")
    print("\nTraining complete! Model, scaler, and feature names are saved successfully.")


if __name__ == "__main__":
    main()
