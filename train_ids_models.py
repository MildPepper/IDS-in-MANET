import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


META_COLS = {"Time", "NodeID", "Scenario", "Run"}
LABEL_COL = "Label"


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and c != LABEL_COL]


def group_train_test_split(df: pd.DataFrame, test_size=0.2, random_state=42):
    """
    Leakage-safe split: keep whole (Scenario, Run) groups together.
    """
    groups = df["Scenario"].astype(str) + "::" + df["Run"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df[LABEL_COL], groups=groups))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def stage1_xgb_window_score(train_df: pd.DataFrame, test_df: pd.DataFrame):
    feats = _feature_cols(train_df)
    X_train = train_df[feats].to_numpy(np.float32)
    y_train = train_df[LABEL_COL].to_numpy(int)
    X_test = test_df[feats].to_numpy(np.float32)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = n_neg / max(n_pos, 1)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    # Window-level suspicion score in [0,1]
    train_prob = clf.predict_proba(X_train_s)[:, 1]
    test_prob = clf.predict_proba(X_test_s)[:, 1]

    # Set an event threshold using only normal training windows (controls false alarms)
    normal_train_prob = train_prob[y_train == 0]
    win_thr = float(np.quantile(normal_train_prob, 0.995))

    return test_prob, win_thr


def stage1_ae_window_score(train_df: pd.DataFrame, test_df: pd.DataFrame, encoding_dim=8):
    """
    Semi-supervised anomaly scoring:
    - Fit AE only on normal windows (Label=0)
    - Score each window by reconstruction MSE (higher => more suspicious)
    - Convert MSE to [0,1] using train-normal distribution percentiles
    """
    feats = _feature_cols(train_df)
    X_train = train_df[feats].to_numpy(np.float32)
    y_train = train_df[LABEL_COL].to_numpy(int)
    X_test = test_df[feats].to_numpy(np.float32)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_normal = X_train_s[y_train == 0]

    ae = MLPRegressor(
        hidden_layer_sizes=(encoding_dim,),
        activation="relu",
        solver="adam",
        max_iter=60,
        batch_size=1024,
        learning_rate_init=1e-3,
        random_state=42,
        verbose=False,
    )
    ae.fit(X_train_normal, X_train_normal)

    def recon_mse(Xs):
        X_hat = ae.predict(Xs)
        return np.mean((Xs - X_hat) ** 2, axis=1)

    train_mse = recon_mse(X_train_normal)
    test_mse = recon_mse(X_test_s)

    # Event threshold on reconstruction error (train-normal quantile)
    win_thr = float(np.quantile(train_mse, 0.995))
    return test_mse, win_thr


def stage2_accumulate_and_predict_nodes(
    df_test: pd.DataFrame,
    win_score: np.ndarray,
    win_threshold: float,
    alpha=0.90,
    threshold=3.0,
):
    """
    Accumulate suspicion per (Scenario,Run,NodeID) over Time.

    suspicion_t = alpha*suspicion_{t-1} + win_score_t

    Node is predicted malicious if suspicion exceeds threshold for >=k consecutive windows.
    """
    work = df_test[["Scenario", "Run", "NodeID", "Time", LABEL_COL]].copy()
    work["win_score"] = win_score.astype(np.float32)
    # Convert raw score into sparse "evidence" events.
    # This prevents idle periods from slowly accumulating suspicion.
    work["event"] = (work["win_score"] >= float(win_threshold)).astype(np.float32)

    # Sort in temporal order per node
    work.sort_values(["Scenario", "Run", "NodeID", "Time"], inplace=True)

    preds = []
    truths = []

    for (scenario, run, node_id), g in work.groupby(["Scenario", "Run", "NodeID"], sort=False):
        s = 0.0
        consec = 0
        alerted = False

        for ev in g["event"].to_numpy():
            s = alpha * s + float(ev)
            if s >= threshold:
                alerted = True
                break

        y_true = int(g[LABEL_COL].max())  # node label is constant in your data
        y_pred = 1 if alerted else 0

        truths.append(y_true)
        preds.append(y_pred)

    return np.array(truths, dtype=int), np.array(preds, dtype=int)


def run_pipeline(df: pd.DataFrame):
    if not META_COLS.issubset(df.columns):
        missing = sorted(META_COLS - set(df.columns))
        raise ValueError(
            f"Dataset is missing required IDS columns: {missing}. "
            "Use engineered_dataset_v3.csv generated from merge.py + feature_v2.py."
        )

    train_df, test_df = group_train_test_split(df)
    print("Train rows:", len(train_df), "Test rows:", len(test_df))
    print("Test groups (Scenario,Run):", test_df.groupby(["Scenario", "Run"]).ngroups)

    print("\n=== Stage 1A: XGBoost window suspicion ===")
    xgb_win, xgb_thr = stage1_xgb_window_score(train_df, test_df)
    print(f"Window event threshold (train-normal 99.5% quantile): {xgb_thr:.4f}")
    y_true_nodes, y_pred_nodes = stage2_accumulate_and_predict_nodes(
        test_df, xgb_win, xgb_thr, alpha=0.90, threshold=3.0
    )
    cm = confusion_matrix(y_true_nodes, y_pred_nodes)
    print("Node-level confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm)
    print("\nNode-level classification report:")
    print(classification_report(y_true_nodes, y_pred_nodes, digits=4))

    print("\n=== Stage 1B: Autoencoder window suspicion ===")
    ae_win, ae_thr = stage1_ae_window_score(train_df, test_df, encoding_dim=8)
    print(f"Window event threshold (train-normal 99.5% quantile): {ae_thr:.6f}")
    y_true_nodes2, y_pred_nodes2 = stage2_accumulate_and_predict_nodes(
        test_df, ae_win, ae_thr, alpha=0.90, threshold=3.0
    )
    cm2 = confusion_matrix(y_true_nodes2, y_pred_nodes2)
    print("Node-level confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm2)
    print("\nNode-level classification report:")
    print(classification_report(y_true_nodes2, y_pred_nodes2, digits=4))


def main():
    df = pd.read_csv("engineered_dataset_v3.csv")
    print("Loaded:", df.shape, "columns:", len(df.columns))
    run_pipeline(df)


if __name__ == "__main__":
    main()

