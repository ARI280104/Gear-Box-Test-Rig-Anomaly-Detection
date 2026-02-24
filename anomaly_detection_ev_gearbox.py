"""
=============================================================
  ANOMALY DETECTION — EV MOTOR GEARBOX TEST BENCH DATA
  Trained on: Pocket 1–5, CODE RESULT section
  Rule: NOK if Actual VALUE > HIGH  OR  Actual VALUE < LOW
  Algorithm: Random Forest Classifier (supervised ML)
=============================================================

FOLDER STRUCTURE:
  your_project/
  ├── data/
  │   ├── 4311SS00255_7.xlsx
  │   ├── 4311SS00255_8.xlsx
  │   └── ... more files ...
  └── anomaly_detection_ev_gearbox.py   ← this script

HOW TO RUN:
  pip install pandas openpyxl scikit-learn matplotlib seaborn joblib
  python anomaly_detection_ev_gearbox.py

OUTPUTS:
  anomaly_results.xlsx   — full results with OK/NOK predictions
  trained_model.pkl      — saved model (reuse on new files)
  anomaly_report.png     — visual report
=============================================================
"""

# ─────────────────────────────────────────────
# STEP 0: IMPORTS
# ─────────────────────────────────────────────
import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# STEP 1: CONFIGURATION  ← Edit these
# ─────────────────────────────────────────────

EXCEL_FOLDER  = "GEAR TEST PROJECT"              # Folder with your .xlsx files
SHEET_NAME    = "01-08-2025"        # Sheet tab name in Excel
OUTPUT_FILE   = ".xlsx"
MODEL_FILE    = "trained_model.pkl"
REPORT_IMAGE  = "anomaly_report.png"

# Row indices (0-based) in the sheet — adjust if your layout differs
ROW_SERIAL_NUM    = 1   # Row that has SERIAL NUM / file info
ROW_HEADER_RPM    = 4   # Row with MODE, NG/OK, TIME, RPM headers
ROW_DATA_START    = 5   # First data row (POCKET_1 RPM row)
ROW_DATA_END      = 10  # Last RPM data row (inclusive)

ROW_HEADER_CODE   = 12  # Row with MODE, CH, SHAFT CODE, PARAMETER headers
ROW_CODE_START    = 13  # First CODE RESULT data row
# CODE RESULT ends at last row of file

# Column names after parsing CODE RESULT section
COL_MODE     = "MODE"
COL_CH       = "CH"
COL_SHAFT    = "SHAFT CODE"
COL_PARAM    = "PARAMETER"
COL_ORDER    = "Order Number"
COL_LOW      = "LOW"
COL_ACTUAL   = "Actual VALUE"
COL_HIGH     = "HIGH"
COL_UNIT     = "UNIT"
COL_OKNOK    = "OK/NOK"       # Ground truth label in training files


# ─────────────────────────────────────────────
# STEP 2: LOAD AND PARSE ALL EXCEL FILES
# ─────────────────────────────────────────────

def parse_excel_file(filepath: str) -> pd.DataFrame:
    """
    Reads one .xlsx file and extracts the CODE RESULT section
    (rows 13 onward, columns A–J) into a clean DataFrame.
    Also extracts SERIAL NUM as a metadata column.
    """
    raw = pd.read_excel(filepath, sheet_name=SHEET_NAME, header=None)

    # ── Extract Serial Number from row 1 ──
    serial_num = ""
    for col_idx in range(raw.shape[1]):
        cell = raw.iloc[ROW_SERIAL_NUM, col_idx]
        if isinstance(cell, str) and "4311" in cell:
            serial_num = cell.strip()
            break
    if not serial_num:
        serial_num = os.path.splitext(os.path.basename(filepath))[0]

    # ── Extract CODE RESULT header row ──
    header_row = raw.iloc[ROW_HEADER_CODE].tolist()
    # Find the column indices for our target columns
    col_map = {}
    wanted = [COL_MODE, COL_CH, COL_SHAFT, COL_PARAM,
              COL_ORDER, COL_LOW, COL_ACTUAL, COL_HIGH,
              COL_UNIT, COL_OKNOK]
    for idx, cell in enumerate(header_row):
        if isinstance(cell, str):
            for w in wanted:
                if w.lower() in cell.lower():
                    col_map[w] = idx
                    break

    # ── Extract data rows from CODE RESULT section ──
    data_rows = []
    for row_idx in range(ROW_CODE_START, raw.shape[0]):
        row = raw.iloc[row_idx]
        # Stop if row is completely empty
        if row.isna().all():
            continue

        record = {"source_file": os.path.basename(filepath),
                  "serial_num": serial_num}
        for col_name, col_idx in col_map.items():
            record[col_name] = row.iloc[col_idx] if col_idx < len(row) else np.nan
        data_rows.append(record)

    df = pd.DataFrame(data_rows)
    return df


def load_all_files(folder: str) -> pd.DataFrame:
    pattern = os.path.join(folder, "*.xlsx")
    files   = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No .xlsx files found in '{folder}'.\n"
            "Please check the EXCEL_FOLDER path."
        )

    print(f"\n[STEP 2] Found {len(files)} Excel file(s):")
    all_dfs = []
    for fp in sorted(files):
        try:
            df = parse_excel_file(fp)
            all_dfs.append(df)
            print(f"   ✔ {os.path.basename(fp)}  →  {len(df)} code-result rows")
        except Exception as e:
            print(f"   ✘ Skipped {os.path.basename(fp)}: {e}")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n   Total rows combined: {len(combined)}")
    return combined


# ─────────────────────────────────────────────
# STEP 3: APPLY RULE-BASED LABELLING
#   NOK if Actual VALUE > HIGH  OR  Actual VALUE < LOW
#   OK  otherwise
# ─────────────────────────────────────────────

def apply_rule_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a definitive 'true_label' column using your business rule:
      • Actual VALUE > HIGH  → NOK
      • Actual VALUE < LOW   → NOK
      • Otherwise            → OK
    This is used as ground truth for ML training.
    """
    print(f"\n[STEP 3] Applying rule-based labelling (LOW ≤ Actual ≤ HIGH = OK)...")

    # Convert to numeric, coerce errors to NaN
    df[COL_ACTUAL] = pd.to_numeric(df[COL_ACTUAL], errors="coerce")
    df[COL_LOW]    = pd.to_numeric(df[COL_LOW],    errors="coerce")
    df[COL_HIGH]   = pd.to_numeric(df[COL_HIGH],   errors="coerce")

    def label_row(row):
        actual = row[COL_ACTUAL]
        low    = row[COL_LOW]
        high   = row[COL_HIGH]

        if pd.isna(actual):
            return "SKIP"           # No measurement → skip

        # If HIGH exists, check upper bound
        if pd.notna(high) and actual > high:
            return "NOK"

        # If LOW exists, check lower bound
        if pd.notna(low) and actual < low:
            return "NOK"

        return "OK"

    df["true_label"] = df.apply(label_row, axis=1)

    # Drop rows with no measurable value
    df = df[df["true_label"] != "SKIP"].copy()

    ok_count  = (df["true_label"] == "OK").sum()
    nok_count = (df["true_label"] == "NOK").sum()
    print(f"   OK  rows: {ok_count}")
    print(f"   NOK rows: {nok_count}")

    # Cross-check against Excel's own OK/NOK column (if present)
    if COL_OKNOK in df.columns:
        df[COL_OKNOK] = df[COL_OKNOK].astype(str).str.strip().str.upper()
        match = (df[COL_OKNOK].str.contains("OK") == (df["true_label"] == "OK")).mean()
        print(f"   Agreement with Excel's OK/NOK column: {match*100:.1f}%")

    return df


# ─────────────────────────────────────────────
# STEP 4: FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates ML-ready features from the raw columns:
      - Numeric: Actual VALUE, LOW, HIGH, Order Number
      - Derived:  actual_vs_high  (headroom above HIGH)
                  actual_vs_low   (margin below LOW)
                  range_span      (HIGH - LOW)
                  actual_pct      (where Actual sits in LOW–HIGH range)
      - Encoded:  MODE, CH, PARAMETER (converted to integers)
    """
    print(f"\n[STEP 4] Engineering features...")

    df = df.copy()

    # Derived numeric features
    df["actual_vs_high"] = df[COL_HIGH]   - df[COL_ACTUAL]   # negative = over limit
    df["actual_vs_low"]  = df[COL_ACTUAL] - df[COL_LOW]       # negative = under limit
    df["range_span"]     = df[COL_HIGH]   - df[COL_LOW]
    df["actual_pct"]     = np.where(
        df["range_span"] > 0,
        (df[COL_ACTUAL] - df[COL_LOW]) / df["range_span"],
        0.5
    )
    df[COL_ORDER] = pd.to_numeric(df[COL_ORDER], errors="coerce").fillna(-1)

    # Label-encode categorical columns
    le = LabelEncoder()
    for cat_col in [COL_MODE, COL_CH, COL_PARAM]:
        if cat_col in df.columns:
            df[cat_col + "_enc"] = le.fit_transform(
                df[cat_col].astype(str).str.strip()
            )

    feature_cols = [
        COL_ACTUAL,
        COL_LOW,
        COL_HIGH,
        COL_ORDER,
        "actual_vs_high",
        "actual_vs_low",
        "range_span",
        "actual_pct",
        COL_MODE  + "_enc",
        COL_CH    + "_enc",
        COL_PARAM + "_enc",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Fill any remaining NaN
    df[feature_cols] = df[feature_cols].fillna(0)

    print(f"   Feature columns ({len(feature_cols)}): {feature_cols}")
    return df, feature_cols


# ─────────────────────────────────────────────
# STEP 5: TRAIN RANDOM FOREST CLASSIFIER
# ─────────────────────────────────────────────

def train_model(df: pd.DataFrame, feature_cols: list):
    """
    Trains a Random Forest on all labelled rows.
    Saves the model to disk.
    Returns: trained model, X_test, y_test (for evaluation).
    """
    print(f"\n[STEP 5] Training Random Forest Classifier...")

    X = df[feature_cols].values
    y = df["true_label"].values          # "OK" or "NOK"

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    print(f"   Class distribution: {dict(zip(unique, counts))}")

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",    # handles imbalanced OK/NOK
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(f"   Cross-validation F1 (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Save model
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_FILE)
    print(f"   ✔ Model saved → {MODEL_FILE}")

    return model, X_test, y_test


# ─────────────────────────────────────────────
# STEP 6: EVALUATE THE MODEL
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Prints accuracy, precision, recall, F1 and saves confusion matrix.
    """
    print(f"\n[STEP 6] Evaluating model on test set ({len(y_test)} rows)...")

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n   Accuracy : {acc*100:.2f}%")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["NOK", "OK"]))

    return y_pred


# ─────────────────────────────────────────────
# STEP 7: PREDICT ON FULL DATASET
# ─────────────────────────────────────────────

def predict_full_dataset(model, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Runs the trained model on every row and adds:
      predicted_label  — what the ML model says (OK / NOK)
      prediction_match — does it agree with rule-based label?
      confidence_%     — model's confidence (probability)
    """
    print(f"\n[STEP 7] Predicting on all {len(df)} rows...")

    X = df[feature_cols].values
    df["predicted_label"] = model.predict(X)
    proba = model.predict_proba(X)
    class_labels = model.classes_

    # Confidence = probability of predicted class
    pred_class_indices = [list(class_labels).index(p) for p in df["predicted_label"]]
    df["confidence_%"] = [round(proba[i, ci] * 100, 1)
                          for i, ci in enumerate(pred_class_indices)]

    df["prediction_match"] = (df["true_label"] == df["predicted_label"])

    nok_count = (df["predicted_label"] == "NOK").sum()
    ok_count  = (df["predicted_label"] == "OK").sum()
    print(f"   ✔ OK predicted  : {ok_count}")
    print(f"   🔴 NOK predicted : {nok_count}")

    return df


# ─────────────────────────────────────────────
# STEP 8: SAVE RESULTS TO EXCEL
# ─────────────────────────────────────────────

def save_results(df: pd.DataFrame, output_path: str):
    """
    Writes 4 sheets:
      1. All Results       — every row with labels + predictions
      2. NOK Rows Only     — just the flagged anomalies
      3. Pocket Summary    — NOK count per pocket per file
      4. Parameter Summary — which parameters fail most
    """
    print(f"\n[STEP 8] Saving results to '{output_path}'...")

    # Friendly display columns
    display_cols = [
        "source_file", "serial_num",
        COL_MODE, COL_CH, COL_SHAFT, COL_PARAM,
        COL_LOW, COL_ACTUAL, COL_HIGH, COL_UNIT,
        "true_label", "predicted_label", "confidence_%", "prediction_match"
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    out_df = df[display_cols].copy()

    # NOK only
    nok_df = out_df[out_df["predicted_label"] == "NOK"].copy()

    # Pocket summary
    pocket_summary = df.groupby([COL_MODE, "source_file"]).apply(
        lambda g: pd.Series({
            "Total Rows"    : len(g),
            "NOK Count"     : (g["predicted_label"] == "NOK").sum(),
            "NOK %"         : round((g["predicted_label"] == "NOK").mean() * 100, 1),
        })
    ).reset_index()

    # Parameter summary
    param_summary = df.groupby(COL_PARAM).apply(
        lambda g: pd.Series({
            "Total Rows" : len(g),
            "NOK Count"  : (g["predicted_label"] == "NOK").sum(),
            "NOK %"      : round((g["predicted_label"] == "NOK").mean() * 100, 1),
        })
    ).reset_index().sort_values("NOK Count", ascending=False)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        out_df.to_excel(writer, sheet_name="All Results",       index=False)
        nok_df.to_excel(writer, sheet_name="NOK Rows Only",     index=False)
        pocket_summary.to_excel(writer, sheet_name="Pocket Summary",    index=False)
        param_summary.to_excel(writer, sheet_name="Parameter Summary",  index=False)

    print(f"   ✔ Saved → {output_path}")
    print(f"      Sheet 1: All Results       ({len(out_df)} rows)")
    print(f"      Sheet 2: NOK Rows Only     ({len(nok_df)} rows)")
    print(f"      Sheet 3: Pocket Summary")
    print(f"      Sheet 4: Parameter Summary")


# ─────────────────────────────────────────────
# STEP 9: VISUALISE RESULTS
# ─────────────────────────────────────────────

def create_report(df: pd.DataFrame, model, feature_cols: list, X_test, y_test, image_path: str):
    """
    Creates a 2×3 dashboard:
      [0,0] NOK count per pocket per file (grouped bar)
      [0,1] Actual VALUE distribution OK vs NOK
      [1,0] Confusion matrix
      [1,1] Feature importance
      [2,0] NOK % per parameter
      [2,1] Actual vs HIGH scatter (coloured by label)
    """
    print(f"\n[STEP 9] Creating visual report...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle("EV Motor Gearbox — Anomaly Detection Report",
                 fontsize=16, fontweight="bold", y=0.98)

    RED   = "#e74c3c"
    GREEN = "#2ecc71"
    BLUE  = "#3498db"

    # ── Chart 1: NOK per Pocket per File ──
    ax = axes[0, 0]
    pivot = df.groupby([COL_MODE, "source_file"])["predicted_label"].apply(
        lambda s: (s == "NOK").sum()
    ).unstack(fill_value=0)
    pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", width=0.7)
    ax.set_title("NOK Count per Pocket per File", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pocket"); ax.set_ylabel("NOK Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="File", fontsize=7)
    for bar in ax.patches:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    int(bar.get_height()), ha="center", fontsize=7)

    # ── Chart 2: Actual VALUE distribution OK vs NOK ──
    ax = axes[0, 1]
    ok_vals  = df[df["true_label"] == "OK"][COL_ACTUAL].dropna()
    nok_vals = df[df["true_label"] == "NOK"][COL_ACTUAL].dropna()
    ax.hist(ok_vals,  bins=30, alpha=0.6, color=GREEN, label="OK",  edgecolor="white")
    ax.hist(nok_vals, bins=30, alpha=0.8, color=RED,   label="NOK", edgecolor="white")
    ax.set_title("Actual VALUE Distribution: OK vs NOK", fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual VALUE (m/s²)"); ax.set_ylabel("Frequency")
    ax.legend()

    # ── Chart 3: Confusion Matrix ──
    ax = axes[1, 0]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=["NOK", "OK"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NOK", "OK"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")

    # ── Chart 4: Feature Importance ──
    ax = axes[1, 1]
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    colors = [RED if "vs_high" in f or "vs_low" in f or "actual" in f.lower()
              else BLUE for f in feat_series.index]
    feat_series.plot(kind="barh", ax=ax, color=colors, edgecolor="black")
    ax.set_title("Feature Importance (Random Forest)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    red_p  = mpatches.Patch(color=RED,  label="Derived features")
    blue_p = mpatches.Patch(color=BLUE, label="Raw features")
    ax.legend(handles=[red_p, blue_p], fontsize=8)

    # ── Chart 5: NOK % per Parameter ──
    ax = axes[2, 0]
    param_nok = df.groupby(COL_PARAM)["predicted_label"].apply(
        lambda s: (s == "NOK").mean() * 100
    ).sort_values(ascending=False)
    bar_colors = [RED if v > 20 else "#e67e22" if v > 0 else GREEN
                  for v in param_nok]
    param_nok.plot(kind="bar", ax=ax, color=bar_colors, edgecolor="black")
    ax.set_title("NOK % per Parameter Type", fontsize=12, fontweight="bold")
    ax.set_xlabel("Parameter"); ax.set_ylabel("NOK %")
    ax.tick_params(axis="x", rotation=30)

    # ── Chart 6: Actual vs HIGH scatter ──
    ax = axes[2, 1]
    colors_scatter = [RED if l == "NOK" else GREEN for l in df["true_label"]]
    ax.scatter(df[COL_HIGH], df[COL_ACTUAL], c=colors_scatter, alpha=0.5, s=15)
    # Draw y=x reference line
    lim_max = max(df[COL_HIGH].max(), df[COL_ACTUAL].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, label="Actual = HIGH limit")
    ax.set_title("Actual VALUE vs HIGH Limit (Red = NOK)", fontsize=12, fontweight="bold")
    ax.set_xlabel("HIGH Limit (m/s²)"); ax.set_ylabel("Actual VALUE (m/s²)")
    ok_p  = mpatches.Patch(color=GREEN, label="OK")
    nok_p = mpatches.Patch(color=RED,   label="NOK")
    ax.legend(handles=[ok_p, nok_p])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✔ Chart saved → {image_path}")


# ─────────────────────────────────────────────
# STEP 10: PREDICT ON NEW FILES (optional)
# ─────────────────────────────────────────────

def predict_new_files(new_folder: str, output_file: str = "new_file_predictions.xlsx"):
    """
    Load a previously trained model and run it on brand-new Excel files.
    No retraining required.

    Usage:
        predict_new_files("new_data/")
    """
    if not os.path.exists(MODEL_FILE):
        print(f"Model file '{MODEL_FILE}' not found. Run main pipeline first.")
        return

    saved    = joblib.load(MODEL_FILE)
    model    = saved["model"]
    features = saved["feature_cols"]
    print(f"✔ Loaded model from '{MODEL_FILE}'")

    raw       = load_all_files(new_folder)
    labelled  = apply_rule_label(raw)
    feat_df, _ = build_features(labelled)

    # Make sure all expected feature columns exist
    for fc in features:
        if fc not in feat_df.columns:
            feat_df[fc] = 0

    X = feat_df[features].fillna(0).values
    feat_df["predicted_label"] = model.predict(X)
    proba = model.predict_proba(X)
    class_labels = list(model.classes_)
    feat_df["confidence_%"] = [
        round(proba[i, class_labels.index(p)] * 100, 1)
        for i, p in enumerate(feat_df["predicted_label"])
    ]

    save_results(feat_df, output_file)
    print(f"Done! Results saved to '{output_file}'")


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  EV GEARBOX ANOMALY DETECTION — FULL ML PIPELINE")
    print("=" * 65)

    # STEP 2: Load all Excel files
    raw_df = load_all_files(EXCEL_FOLDER)

    # STEP 3: Apply rule-based labelling (NOK if out of LOW–HIGH range)
    labelled_df = apply_rule_label(raw_df)

    # STEP 4: Build ML features
    feat_df, feature_cols = build_features(labelled_df)

    # STEP 5: Train Random Forest
    model, X_test, y_test = train_model(feat_df, feature_cols)

    # STEP 6: Evaluate
    evaluate_model(model, X_test, y_test)

    # STEP 7: Predict on full dataset
    result_df = predict_full_dataset(model, feat_df, feature_cols)

    # STEP 8: Save Excel output
    save_results(result_df, OUTPUT_FILE)

    # STEP 9: Save visual report
    create_report(result_df, model, feature_cols, X_test, y_test, REPORT_IMAGE)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Results  : {OUTPUT_FILE}")
    print(f"  Model    : {MODEL_FILE}")
    print(f"  Chart    : {REPORT_IMAGE}")
    print("=" * 65)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────
# QUICK REFERENCE
# ─────────────────────────────────────────────
"""
LABELLING RULE (matches your requirement exactly):
  Actual VALUE > HIGH  → NOK
  Actual VALUE < LOW   → NOK
  LOW ≤ Actual ≤ HIGH  → OK

ML ALGORITHM: Random Forest Classifier
  • Supervised learning — learns from your OK/NOK labels
  • 300 decision trees vote on each row
  • class_weight="balanced" handles cases where NOK rows are rare
  • Outputs probability (confidence) for each prediction

FEATURES USED:
  1. Actual VALUE         — the measurement itself
  2. LOW                  — lower limit from Excel
  3. HIGH                 — upper limit from Excel
  4. actual_vs_high       — headroom to HIGH limit (negative = exceeded)
  5. actual_vs_low        — margin above LOW (negative = below limit)
  6. range_span           — HIGH − LOW (how tight the spec is)
  7. actual_pct           — where measurement sits in LOW–HIGH range
  8. Order Number         — encoded numeric
  9. MODE encoded         — POCKET_1 … POCKET_5
  10. CH encoded          — CHANNEL 1 / CHANNEL 2
  11. PARAMETER encoded   — MOTOR VIBRATION / TRANSMISSION VIBRATION

OUTPUTS:
  anomaly_results.xlsx → Sheet 1: All rows with OK/NOK flag
                       → Sheet 2: Only flagged NOK rows
                       → Sheet 3: NOK count per pocket per file
                       → Sheet 4: Which parameters fail most
  trained_model.pkl    → Reuse on new files without retraining
  anomaly_report.png   → 6-panel visual dashboard
"""
