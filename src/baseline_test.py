"""
Baseline Model Test Suite (v2)

Determines whether the current feature set contains ANY predictive signal
for the directional classification task by training simple non-sequential
models that mirror the pipeline's exact data pipeline.

Fixes over v1:
  1. LOOKBACK=24 / HORIZON=8 aligned with pipeline.py (was 72/12)
  2. Bearish regime oversampling (2x) applied before training
  3. Close price excluded from X features (via DataPreprocessor logic)
  4. Per-group evaluation matching the pipeline's group structure
  5. Real feature names from FeatureEngineer.get_feature_columns()
  6. Statistical significance testing (binomial + bootstrap CI)
  7. Label-skew-aware interpretation (naive-majority baseline calibrated)
  8. Dynamic min_test_rows matching pipeline logic
"""

import numpy as np
import json
import sys
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

LOOKBACK_WINDOW = 24
FORECAST_HORIZON = 8
BEARISH_OVERSAMPLE_MULTIPLIER = 2
DATA_PATH = "data/raw"


def _load_pipeline_groups():
    """Load groups from the master manifest or tournament CSV if available."""
    manifest_path = Path("models/master_manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest:
            return [r.get("stocks", []) for r in manifest]

    csv_path = Path("results/group_tournament.csv")
    if csv_path.exists():
        import csv as _csv
        groups = []
        with open(csv_path, newline="") as f:
            for row in _csv.DictReader(f):
                stocks_str = row.get("stocks", "")
                if stocks_str and stocks_str != "FAILED":
                    groups.append([s.strip() for s in stocks_str.split(",") if s.strip()])
        if groups:
            return groups

    return None


def _load_all_tickers():
    """Fallback: load all selected tickers if no group structure available."""
    from lstm_model import get_high_beta_stocks
    pool = get_high_beta_stocks(DATA_PATH, pool_size=200, num_select=15, min_volume=300000)
    return pool


def load_preprocessed_data():
    """
    Load data per-group through the exact same pipeline as train_group():
      - FeatureEngineer.compute_indicators()
      - _make_preprocessor(LOOKBACK, HORIZON, ...)
      - make_direction_onehot_from_raw() with triple-barrier labels
      - Close excluded from X (inherent in DataPreprocessor.create_sequences)
      - Bearish regime oversampling (2x) on training set
      - Per-group min_test_rows gating
      - Label drift check (>0.07 skips ticker)

    Returns dict of {group_key: {X_train_2d, y_train, X_val_2d, y_val,
                                 X_test_2d, y_test, feature_names, tickers}}
    """
    from data_loader import DataLoader
    from data_preprocessor import DataPreprocessor
    from feature_engineer import FeatureEngineer
    from lstm_model import make_direction_onehot_from_raw
    from ensemble_model import _make_preprocessor

    pipeline_groups = _load_pipeline_groups()
    if pipeline_groups is None:
        all_tickers = _load_all_tickers()
        group_size = 5
        pipeline_groups = [all_tickers[i:i + group_size]
                           for i in range(0, len(all_tickers), group_size)]
        print(f"No pipeline groups found. Using {len(pipeline_groups)} "
              f"ad-hoc groups of ~{group_size} tickers each.")

    group_datasets = {}
    global_fe_columns = None

    for gid, stock_list in enumerate(pipeline_groups):
        print(f"\n--- Group {gid}: {stock_list} ---")
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        all_X_test, all_y_test = [], []
        valid_tickers = []

        for ticker in stock_list:
            print(f"  Loading {ticker}...", end=" ")
            try:
                loader = DataLoader(DATA_PATH, ticker)
                raw_df = loader.load_data()
                loader.validate_data()

                fe = FeatureEngineer()
                enriched_df = fe.compute_indicators(raw_df)

                pre, splits, enriched_trunc = _make_preprocessor(
                    LOOKBACK_WINDOW, FORECAST_HORIZON, enriched_df, None,
                )

                min_test_rows = 7_300 if pre.use_full_split else 470
                if len(splits["X_test"]) < min_test_rows:
                    print(f"SKIP (test={len(splits['X_test'])} < {min_test_rows})")
                    continue

                direction_labels, direction_masks, _ = make_direction_onehot_from_raw(
                    enriched_df=enriched_trunc,
                    split_indices={
                        "idx_train": splits["idx_train"],
                        "idx_val": splits["idx_val"],
                        "idx_test": splits["idx_test"],
                    },
                    lookback_window=LOOKBACK_WINDOW,
                    forecast_horizon=FORECAST_HORIZON,
                )

                train_up_pct = direction_labels["train"][:, 1].mean()
                val_up_pct = direction_labels["val"][:, 1].mean()
                drift = val_up_pct - train_up_pct
                if abs(drift) > 0.07:
                    print(f"SKIP (label drift {drift:+.3f})")
                    continue

                train_mask = direction_masks["train"]
                val_mask = direction_masks["val"]
                test_mask = direction_masks["test"]

                # Extract LAST timestep (same as EnsembleModel._extract_2d).
                # Close is ALREADY excluded from X by DataPreprocessor.create_sequences.
                X_train_last = splits["X_train"][train_mask, -1, :]
                X_val_last = splits["X_val"][val_mask, -1, :]
                X_test_last = splits["X_test"][test_mask, -1, :]

                # Apply the same mask to the labels so X and y stay aligned.
                # Without this, tickers whose lookback/horizon warmup zeros out
                # leading rows of *_mask leave y_* longer than X_*, and any
                # downstream metric call raises "inconsistent numbers of samples".
                y_train_int = np.argmax(direction_labels["train"], axis=1)[train_mask]
                y_val_int   = np.argmax(direction_labels["val"],   axis=1)[val_mask]
                y_test_int  = np.argmax(direction_labels["test"],  axis=1)[test_mask]

                all_X_train.append(X_train_last)
                all_y_train.append(y_train_int)
                all_X_val.append(X_val_last)
                all_y_val.append(y_val_int)
                all_X_test.append(X_test_last)
                all_y_test.append(y_test_int)
                valid_tickers.append(ticker)

                if global_fe_columns is None:
                    fe_cols = fe.get_feature_columns()
                    close_idx = fe.get_close_index()
                    global_fe_columns = [
                        c for i, c in enumerate(fe_cols) if i != close_idx
                    ]

                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

        if not valid_tickers:
            print(f"  Group {gid}: no valid tickers, skipping.")
            continue

        X_train = np.concatenate(all_X_train)
        y_train = np.concatenate(all_y_train)
        X_val = np.concatenate(all_X_val)
        y_val = np.concatenate(all_y_val)
        X_test = np.concatenate(all_X_test)
        y_test = np.concatenate(all_y_test)

        # --- Bearish regime oversampling (matches train_group logic) ---
        # sharpe_col_idx = -1 is the last feature in the 2D slice
        # (after Close removal, sharpe_ratio_20 is still the final column)
        # In 3D the pipeline does X[:,:,−1].mean(axis=1), but our X_train
        # is already 2D (samples × features), so a single column is 1-D.
        sharpe_col_idx = -1
        mean_sharpe = X_train[:, sharpe_col_idx]
        bearish_mask = mean_sharpe < 0
        n_bearish = int(bearish_mask.sum())

        if n_bearish > 0:
            X_bear = X_train[bearish_mask]
            y_bear = y_train[bearish_mask]
            X_train = np.concatenate(
                [X_train] + [X_bear] * (BEARISH_OVERSAMPLE_MULTIPLIER - 1),
                axis=0,
            )
            y_train = np.concatenate(
                [y_train] + [y_bear] * (BEARISH_OVERSAMPLE_MULTIPLIER - 1),
                axis=0,
            )
            rng = np.random.default_rng(seed=42)
            perm = rng.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]
            print(f"  Bearish oversampling: {n_bearish} sequences "
                  f"duplicated {BEARISH_OVERSAMPLE_MULTIPLIER - 1}x "
                  f"-> train={len(X_train)}")
        else:
            print("  Bearish oversampling: no bearish sequences found.")

        group_key = f"group_{gid}"
        group_datasets[group_key] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "tickers": valid_tickers,
        }

    if not group_datasets:
        raise ValueError("No data loaded successfully for any group.")

    return group_datasets, global_fe_columns


def print_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"            DOWN    UP")
    print(f"Actual DOWN {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"Actual UP   {cm[1,0]:5d}  {cm[1,1]:5d}")
    print(f"\nClass Distribution in Predictions:")
    print(f"  Predicted DOWN: {np.sum(y_pred == 0)} ({np.mean(y_pred == 0)*100:.1f}%)")
    print(f"  Predicted UP:   {np.sum(y_pred == 1)} ({np.mean(y_pred == 1)*100:.1f}%)")

    return acc


def binomial_test(accuracy, n_samples, null_acc=None):
    """
    One-sided binomial test: H0 = model accuracy equals null_acc.
    If null_acc is None, uses 0.5 (random guessing).
    Returns (p_value, z_stat).
    """
    if null_acc is None:
        null_acc = 0.5
    p = null_acc
    n_success = int(round(accuracy * n_samples))
    z = (accuracy - p) / np.sqrt(p * (1 - p) / n_samples) if n_samples > 0 else 0.0
    p_value = 1.0 - stats.norm.cdf(z)
    return p_value, z


def bootstrap_ci(y_true, y_pred, n_bootstrap=5000, ci=0.95):
    """
    Bootstrap confidence interval for accuracy.
    Returns (lower, upper) bounds.
    """
    rng = np.random.default_rng(seed=42)
    n = len(y_true)
    boot_accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_accs[i] = accuracy_score(y_true[idx], y_pred[idx])
    alpha = (1.0 - ci) / 2.0
    return np.percentile(boot_accs, 100 * alpha), np.percentile(boot_accs, 100 * (1 - alpha))


def evaluate_group(group_key, group_data, feature_names):
    """Run all baseline tests for a single group and return results dict."""
    X_train = group_data["X_train"]
    y_train = group_data["y_train"]
    X_val = group_data["X_val"]
    y_val = group_data["y_val"]
    X_test = group_data["X_test"]
    y_test = group_data["y_test"]

    print(f"\n{'#'*60}")
    print(f"# GROUP: {group_key} — tickers: {group_data['tickers']}")
    print(f"{'#'*60}")
    print(f"  Train: {X_train.shape[0]:,} samples x {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]:,} samples")
    print(f"  Test:  {X_test.shape[0]:,} samples")

    train_up = np.mean(y_train == 1) * 100
    val_up = np.mean(y_val == 1) * 100
    test_up = np.mean(y_test == 1) * 100
    print(f"  Label balance — Train UP: {train_up:.1f}%  Val UP: {val_up:.1f}%  Test UP: {test_up:.1f}%")

    # --- Naive majority baseline ---
    majority_class = 1 if np.mean(y_train) > 0.5 else 0
    majority_name = "UP" if majority_class == 1 else "DOWN"
    naive_pred_val = np.full(len(y_val), majority_class)
    naive_pred_test = np.full(len(y_test), majority_class)
    naive_val_acc = accuracy_score(y_val, naive_pred_val)
    naive_test_acc = accuracy_score(y_test, naive_pred_test)

    print(f"\n  NAIVE (always predict {majority_name}):")
    print(f"    Val accuracy:  {naive_val_acc:.4f}")
    print(f"    Test accuracy: {naive_test_acc:.4f}")

    # --- Logistic Regression ---
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)

    val_pred_lr = lr.predict(X_val)
    test_pred_lr = lr.predict(X_test)
    val_acc_lr = print_metrics(y_val, val_pred_lr, f"LOGISTIC REGRESSION — {group_key} VAL")
    test_acc_lr = print_metrics(y_test, test_pred_lr, f"LOGISTIC REGRESSION — {group_key} TEST")

    if feature_names and len(feature_names) == X_train.shape[1]:
        coef = lr.coef_[0]
        importance = np.abs(coef)
        top_idx = np.argsort(importance)[-10:][::-1]
        print(f"\n  Top 10 LR features (by |coefficient|):")
        for idx in top_idx:
            print(f"    {feature_names[idx]:25s}: {coef[idx]:+.4f} (|{importance[idx]:.4f}|)")
    else:
        print(f"\n  (Feature names unavailable; skipping LR importance)")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rf.fit(X_train, y_train)

    val_pred_rf = rf.predict(X_val)
    test_pred_rf = rf.predict(X_test)
    val_acc_rf = print_metrics(y_val, val_pred_rf, f"RANDOM FOREST — {group_key} VAL")
    test_acc_rf = print_metrics(y_test, test_pred_rf, f"RANDOM FOREST — {group_key} TEST")

    if feature_names and len(feature_names) == X_train.shape[1]:
        importance_rf = rf.feature_importances_
        top_idx_rf = np.argsort(importance_rf)[-10:][::-1]
        print(f"\n  Top 10 RF features (by Gini importance):")
        for idx in top_idx_rf:
            print(f"    {feature_names[idx]:25s}: {importance_rf[idx]:.4f}")
    else:
        print(f"\n  (Feature names unavailable; skipping RF importance)")

    # --- Statistical significance tests ---
    max_test_acc = max(test_acc_lr, test_acc_rf)
    n_test = len(y_test)
    best_name = "LR" if test_acc_lr >= test_acc_rf else "RF"
    best_pred = test_pred_lr if test_acc_lr >= test_acc_rf else test_pred_rf

    p_vs_random, z_vs_random = binomial_test(max_test_acc, n_test, null_acc=0.5)
    p_vs_naive, z_vs_naive = binomial_test(max_test_acc, n_test, null_acc=naive_test_acc)
    ci_lo, ci_hi = bootstrap_ci(y_test, best_pred)

    print(f"\n  STATISTICAL SIGNIFICANCE ({best_name} vs baselines):")
    print(f"    vs Random (50%):    z={z_vs_random:.3f}, p={p_vs_random:.4f} {'***' if p_vs_random < 0.001 else '**' if p_vs_random < 0.01 else '*' if p_vs_random < 0.05 else '(n.s.)'}")
    print(f"    vs Naive ({naive_test_acc:.1f}%):  z={z_vs_naive:.3f}, p={p_vs_naive:.4f} {'***' if p_vs_naive < 0.001 else '**' if p_vs_naive < 0.01 else '*' if p_vs_naive < 0.05 else '(n.s.)'}")
    print(f"    Bootstrap 95% CI:   [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "group_key": group_key,
        "tickers": group_data["tickers"],
        "naive_val_acc": naive_val_acc,
        "naive_test_acc": naive_test_acc,
        "lr_val_acc": val_acc_lr,
        "lr_test_acc": test_acc_lr,
        "rf_val_acc": val_acc_rf,
        "rf_test_acc": test_acc_rf,
        "max_test_acc": max_test_acc,
        "best_model": best_name,
        "test_up_pct": test_up,
        "p_vs_random": p_vs_random,
        "p_vs_naive": p_vs_naive,
        "z_vs_random": z_vs_random,
        "z_vs_naive": z_vs_naive,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_test": n_test,
    }


def print_summary(all_results):
    print("\n" + "=" * 80)
    print("CROSS-GROUP SUMMARY")
    print("=" * 80)

    print(f"\n{'Group':<12} {'Tickers':<20} {'Naive':>7} {'LR':>7} {'RF':>7} {'Best':>7} "
          f"{'p(rand)':>9} {'p(naive)':>9} {'95% CI':>16}")
    print("-" * 104)

    for r in all_results:
        ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        tickers_str = ",".join(r["tickers"][:3]) + ("..." if len(r["tickers"]) > 3 else "")
        sig_rand = "***" if r["p_vs_random"] < 0.001 else "**" if r["p_vs_random"] < 0.01 else "*" if r["p_vs_random"] < 0.05 else ""
        sig_naive = "***" if r["p_vs_naive"] < 0.001 else "**" if r["p_vs_naive"] < 0.01 else "*" if r["p_vs_naive"] < 0.05 else ""
        print(f"{r['group_key']:<12} {tickers_str:<20} "
              f"{r['naive_test_acc']:>7.4f} {r['lr_test_acc']:>7.4f} {r['rf_test_acc']:>7.4f} "
              f"{r['max_test_acc']:>7.4f} {r['p_vs_random']:>7.4f}{sig_rand:<2} "
              f"{r['p_vs_naive']:>7.4f}{sig_naive:<2} {ci_str:>16}")

    # Pooled averages
    avg_naive = np.mean([r["naive_test_acc"] for r in all_results])
    avg_lr = np.mean([r["lr_test_acc"] for r in all_results])
    avg_rf = np.mean([r["rf_test_acc"] for r in all_results])
    avg_best = np.mean([r["max_test_acc"] for r in all_results])
    avg_test_up = np.mean([r["test_up_pct"] for r in all_results])

    n_sig_random = sum(1 for r in all_results if r["p_vs_random"] < 0.05)
    n_sig_naive = sum(1 for r in all_results if r["p_vs_naive"] < 0.05)

    print("-" * 104)
    print(f"{'AVERAGE':<12} {'':<20} {avg_naive:>7.4f} {avg_lr:>7.4f} {avg_rf:>7.4f} {avg_best:>7.4f}")
    print()
    print(f"  Groups with significant signal vs random (p<0.05):  {n_sig_random}/{len(all_results)}")
    print(f"  Groups with significant signal vs naive (p<0.05):  {n_sig_naive}/{len(all_results)}")
    print(f"  Average test-set UP%:                               {avg_test_up:.1f}%")

    # --- Diagnostic interpretation ---
    print("\n" + "=" * 80)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 80)

    # Use the naive-adjusted signal: how much better than naive is the best model?
    naive_adj_edge = avg_best - avg_naive

    print(f"\n  Average naive baseline:     {avg_naive:.4f} ({avg_naive*100:.1f}%)")
    print(f"  Average best model:         {avg_best:.4f} ({avg_best*100:.1f}%)")
    print(f"  Naive-adjusted edge:        {naive_adj_edge:+.4f} ({naive_adj_edge*100:+.1f}pp)")

    # Interpretation calibrated to label skew:
    # - "No signal" = can't beat naive majority baseline
    # - "Weak signal" = beats naive but by < 2pp
    # - "Usable signal" = beats naive by 2-5pp
    # - "Strong signal" = beats naive by > 5pp (LSTM should definitely use this)
    if n_sig_naive == 0 and naive_adj_edge <= 0:
        print("\n  NO PREDICTIVE SIGNAL")
        print("  Baselines CANNOT beat the naive majority predictor.")
        print("  The features do not contain directional information.")
        print("\n  RECOMMENDED ACTIONS:")
        print("    1. Lower triple-barrier thresholds (try 0.3-0.5% instead of current)")
        print("    2. Add volume-weighted features and order flow proxies")
        print("    3. Add multi-timeframe features (daily indicators on hourly data)")
        print("    4. Consider adding sentiment/alternative data sources")
        print("    5. Try predicting next-bar direction instead of longer horizons")
        print("\n  DO NOT iterate on LSTM architecture — the problem is upstream.")

    elif n_sig_naive <= len(all_results) // 2 and naive_adj_edge < 0.02:
        print("\n  WEAK BUT DETECTABLE SIGNAL")
        print("  Some groups beat naive, but the edge is thin (< 2pp).")
        print("  Not enough for reliable trading on its own.")
        print("\n  RECOMMENDED ACTIONS:")
        print("    1. Enhance current features (derivatives, interactions)")
        print("    2. Try the 'No Signal' actions above as well")
        print("    3. LSTM may capture temporal patterns trees miss, but gains will be marginal")
        print("    4. Focus on groups where p_vs_naive < 0.05 — those have the most signal")

    elif naive_adj_edge < 0.05:
        print("\n  USABLE SIGNAL — LSTM MAY UNDERPERFORM")
        print("  Baselines reliably beat naive by 2-5pp. Features contain directional info.")
        print("  If the LSTM can't beat these baselines, it has an architectural issue.")
        print("\n  RECOMMENDED ACTIONS:")
        print("    1. Reduce LSTM lookback if trees already capture the signal at last-bar")
        print("    2. Ensure LSTM is using sequential info (check if full-sequence > last-bar)")
        print("    3. Add bidirectional LSTM or CNN preprocessing layer")
        print("    4. Increase model capacity or adjust regularization")
        print("    5. Compare per-ticker models vs pooled training")

    else:
        print("\n  STRONG SIGNAL — LSTM IS UNDERPERFORMING")
        print("  Baselines beat naive by > 5pp. The features clearly predict direction.")
        print("  If the LSTM can't capitalise, it has a serious architectural problem.")
        print("\n  RECOMMENDED ACTIONS:")
        print("    1. Debug LSTM training (check loss curves, gradient flow)")
        print("    2. Reduce sequence length (trees use only last-bar and already win)")
        print("    3. Try simpler LSTM: fewer units, no excess regularization")
        print("    4. Verify data pipeline: are train/val/test sequences correct?")
        print("    5. Consider replacing LSTM with gradient-boosted trees entirely")


def main():
    print("\n" + "=" * 60)
    print("BASELINE MODEL TEST SUITE v2")
    print("=" * 60)
    print(f"Lookback:  {LOOKBACK_WINDOW}h  |  Horizon: {FORECAST_HORIZON}h  |  Oversampling: {BEARISH_OVERSAMPLE_MULTIPLIER}x bearish")
    print(f"Close excluded from X  |  Per-group evaluation  |  Statistical significance tests")
    print("\nObjective: Determine if features contain predictive signal")
    print("Method: Train simple non-sequential models on last-bar features")
    print("  matching the exact pipeline data processing and labels")

    group_datasets, feature_names = load_preprocessed_data()

    print(f"\nLoaded {len(group_datasets)} groups with {len(feature_names or [])} features each")
    if feature_names:
        print(f"Feature names: {feature_names}")

    all_results = []
    for group_key, group_data in group_datasets.items():
        result = evaluate_group(group_key, group_data, feature_names)
        all_results.append(result)

    if all_results:
        print_summary(all_results)

    print(f"\n{'='*60}")
    print("Baseline test complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during baseline test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
