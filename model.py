"""
Sink Hole Node Detection in AODV MANETs
Hybrid Cascade: XGBoost + Autoencoder
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# CONFIG
# =========================================================
FILE_PATH = "engineered_dataset_v11.csv"   # <-- update to your dataset path

BASE_COLS = ["RREQ_Sent", "RREQ_Recv", "RREP_Sent", "RREP_Recv",
             "Pkts_Forwarded", "Pkts_Dropped"]

TRAIN_SCENARIOS = ['0_nodes', '3_nodes', '7_nodes']
TEST_SCENARIOS  = ['5_nodes', '10_nodes']

# =========================================================
# 1. LOAD & PREPROCESS
# =========================================================
def load_and_preprocess(filepath: str) -> pd.DataFrame:
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    for col in BASE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Time-window aggregation (5-second windows)
    df['Time_Window'] = df['Time'] // 5
    agg_dict = {col: 'sum' for col in BASE_COLS}
    agg_dict['Label'] = 'max'
    df_w = (df.groupby(['Scenario', 'Run', 'NodeID', 'Time_Window'])
              .agg(agg_dict).reset_index())

    # Remove silent nodes
    df_w = df_w[df_w[BASE_COLS].sum(axis=1) > 0].copy()
    df_w['Scenario'] = df_w['Scenario'].astype(str)

    # Engineered ratios
    df_w['Drop_Ratio'] = df_w['Pkts_Dropped'] / (
        df_w['Pkts_Forwarded'] + df_w['Pkts_Dropped'] + 1e-5)
    df_w['RREP_to_RREQ_Ratio'] = df_w['RREP_Sent'] / (
        df_w['RREQ_Recv'] + 1e-5)
    return df_w


# =========================================================
# 2. TRAIN / TEST SPLIT
# =========================================================
def split_scenarios(df: pd.DataFrame):
    df_train = df[df['Scenario'].isin(TRAIN_SCENARIOS)]
    df_test  = df[df['Scenario'].isin(TEST_SCENARIOS)]
    return df_train, df_test


# =========================================================
# 3. STAGE 1 – XGBOOST
# =========================================================
def train_xgboost(X_train, y_train):
    print("\n--- Training Stage 1: XGBoost ---")
    try:
        smote = SMOTE(sampling_strategy=0.25, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
    except ValueError:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_res, y_res)
    return model


# =========================================================
# 4. STAGE 2 – AUTOENCODER
# =========================================================
def train_autoencoder(X_normal_raw):
    print("--- Training Stage 2: Autoencoder ---")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_normal_raw)

    dim = X_scaled.shape[1]
    inp = Input(shape=(dim,))
    enc = Dense(8,   activation='relu')(inp)
    enc = Dense(4,   activation='relu')(enc)
    dec = Dense(8,   activation='relu')(enc)
    out = Dense(dim, activation='linear')(dec)

    ae = Model(inputs=inp, outputs=out)
    ae.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ae.fit(X_scaled, X_scaled, epochs=40, batch_size=128,
           validation_split=0.1, callbacks=[es], verbose=0)

    # Critical threshold = 2× max reconstruction error on clean training data
    train_preds = ae.predict(X_scaled, verbose=0)
    train_mse   = np.mean(np.power(X_scaled - train_preds, 2), axis=1)
    threshold   = np.max(train_mse) * 2.0
    print(f"Critical Zero-Day Threshold: {threshold:.6f}")
    return ae, scaler, threshold


# =========================================================
# 5. EVALUATE
# =========================================================
def evaluate(df_test, xgb_model, ae_model, scaler, threshold, activity_cols):
    X_test = df_test[activity_cols]
    X_test_scaled = scaler.transform(X_test)

    preds_xgb = xgb_model.predict(X_test)

    test_preds = ae_model.predict(X_test_scaled, verbose=0)
    test_mse   = np.mean(np.power(X_test_scaled - test_preds, 2), axis=1)
    preds_ae   = (test_mse > threshold).astype(int)

    df_eval = df_test.copy()
    df_eval['Pred_XGB']      = preds_xgb
    df_eval['Pred_AE']       = preds_ae
    df_eval['Pred_Cascade']  = df_eval[['Pred_XGB', 'Pred_AE']].max(axis=1)

    # Node-level aggregation
    node_eval = (df_eval.groupby(['Scenario', 'NodeID'])
                 .agg({'Label': 'max', 'Pred_XGB': 'max', 'Pred_Cascade': 'max'})
                 .reset_index())

    print("\n" + "#"*60)
    print("FINAL RESULTS: NODE-LEVEL COMPARISON")
    print("#"*60)

    for scenario in TEST_SCENARIOS:
        subset = node_eval[node_eval['Scenario'] == scenario]
        print(f"\n{'='*56}")
        print(f"SCENARIO: {scenario}")
        print(f"{'='*56}")

        print("\n[BASELINE] PURE XGBOOST:")
        print(classification_report(subset['Label'], subset['Pred_XGB'], zero_division=0))
        print(f"Confusion Matrix:\n{confusion_matrix(subset['Label'], subset['Pred_XGB'])}\n")

        print("\n[ARCHITECTURE 2] SEQUENTIAL CASCADE (XGBoost + Critical AE):")
        print(classification_report(subset['Label'], subset['Pred_Cascade'], zero_division=0))
        print(f"Confusion Matrix:\n{confusion_matrix(subset['Label'], subset['Pred_Cascade'])}\n")

    return node_eval


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    df_w = load_and_preprocess(FILE_PATH)
    df_train, df_test = split_scenarios(df_w)

    ACTIVITY_COLS = BASE_COLS + ['Drop_Ratio', 'RREP_to_RREQ_Ratio']

    # Clean outliers from training set
    df_train_clean = df_train[df_train['RREQ_Sent'] < 500]

    X_train = df_train_clean[ACTIVITY_COLS]
    y_train = df_train_clean['Label']

    xgb_model = train_xgboost(X_train, y_train)

    X_normal = df_train_clean[df_train_clean['Label'] == 0][ACTIVITY_COLS]
    ae_model, scaler, threshold = train_autoencoder(X_normal)

    evaluate(df_test, xgb_model, ae_model, scaler, threshold, ACTIVITY_COLS)
