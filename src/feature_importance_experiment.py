# src/feature_importance_experiment.py
"""
Feature Importance: LASSO + RF ranking on FULL stock_scan_cache.json.
Uses pipeline-matching config. Run: python src/feature_importance_experiment.py
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from lstm_model import make_direction_onehot_from_raw

LOOKBACK = 24; HORIZON = 8; DATA_PATH = "data/raw"
SCAN_CACHE = Path("models/stock_scan_cache.json")

def run():
    with open(SCAN_CACHE) as f:
        stocks = json.load(f)
    print(f"Stocks: {len(stocks)}")

    all_X, all_y = [], []
    fe = FeatureEngineer()
    fnames = None; skipped = 0; total = 0

    for ticker in stocks:
        try:
            loader = DataLoader(DATA_PATH, ticker)
            raw_df = loader.load_data()
            if len(raw_df) < 500: skipped += 1; continue
            enriched = fe.compute_indicators(raw_df)
            if len(enriched) < 500: skipped += 1; continue
            if fnames is None:
                all_feats = enriched.columns.tolist()
                ci = all_feats.index('Close') if 'Close' in all_feats else -1
                fnames = [f for i, f in enumerate(all_feats) if i != ci]

            pre = DataPreprocessor(lookback_window=LOOKBACK, forecast_horizon=HORIZON, use_walk_forward=True)
            splits = pre.preprocess(enriched)
            labels, masks, _ = make_direction_onehot_from_raw(enriched, splits, LOOKBACK, HORIZON)

            for sk, mk in [("train", "train"), ("val", "val")]:
                X3 = splits[f"X_{sk}"]; yc = labels[sk]
                m = masks.get(mk)
                if m is not None: X3 = X3[m]; yc = yc[m]
                if len(X3) == 0: continue
                all_X.append(X3[:, -1, :])
                all_y.append(np.argmax(yc, axis=1))
                total += len(X3)
        except Exception: skipped += 1; continue

    if not all_X: print("ERROR"); return
    X = np.concatenate(all_X); y = np.concatenate(all_y)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    up_pct = np.sum(y == 1) / len(y) * 100
    scaler = StandardScaler(); Xs = scaler.fit_transform(X)

    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, max_iter=1000, random_state=42)
    lasso.fit(Xs, y); coefs = lasso.coef_[0]
    dl = pd.DataFrame({'Feature': fnames, 'Abs': np.abs(coefs)}).sort_values('Abs', ascending=False)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y); imps = rf.feature_importances_
    dr = pd.DataFrame({'Feature': fnames, 'Imp': imps}).sort_values('Imp', ascending=False)

    dl['LRk'] = range(1, len(dl)+1); dr['RRk'] = range(1, len(dr)+1)
    mg = pd.merge(dl[['Feature','LRk','Abs']], dr[['Feature','RRk','Imp']], on='Feature')
    mg['Avg'] = (mg['LRk']+mg['RRk'])/2; mg = mg.sort_values('Avg')

    print(f"\n{len(fnames)} features, {total} samples, UP={up_pct:.1f}%")
    print(f"{'Feature':<25} {'LRk':>4} {'LASSO':>8} {'RRk':>4} {'RF':>6} {'Avg':>5}")
    for _, r in mg.iterrows():
        print(f"{r['Feature']:<25} {int(r['LRk']):>4} {r['Abs']:>8.4f} {int(r['RRk']):>4} {r['Imp']:>6.4f} {r['Avg']:>5.1f}")

    nelim = int(np.sum(np.abs(coefs) < 1e-6))
    lr_cv = cross_val_score(LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42), Xs, y, cv=5, scoring='accuracy', n_jobs=-1)
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    le = (lr_cv.mean() - up_pct/100)*100; re = (rf_cv.mean() - up_pct/100)*100

    print(f"\nLASSO elim: {nelim}/{len(fnames)}")
    print(f"LR CV: {lr_cv.mean()*100:.2f}% +/- {lr_cv.std()*100:.2f}%  edge: {le:+.2f}pp")
    print(f"RF CV: {rf_cv.mean()*100:.2f}% +/- {rf_cv.std()*100:.2f}%  edge: {re:+.2f}pp")
    print(f"METRIC lr_edge_pp={le}")
    print(f"METRIC rf_edge_pp={re}")
    print(f"METRIC lasso_eliminated={nelim}")
    print(f"METRIC total_samples={total}")

if __name__ == "__main__":
    run()
