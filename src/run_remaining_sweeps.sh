#!/bin/bash
set -e
cd "$(dirname "$0")/.."

SWEEP_DIR="results/sweep"
mkdir -p "$SWEEP_DIR"

# Backup the FIXED pipeline.py (from commit 04b2ca0)
cp src/pipeline.py       src/pipeline.py.sweep_bak
cp src/ensemble_model.py src/ensemble_model.py.sweep_bak
cp src/lstm_model.py     src/lstm_model.py.sweep_bak
cp src/PredictionEngine.py src/PredictionEngine.py.sweep_bak

restore_from_backup() {
    cp src/pipeline.py.sweep_bak       src/pipeline.py
    cp src/ensemble_model.py.sweep_bak src/ensemble_model.py
    cp src/lstm_model.py.sweep_bak     src/lstm_model.py
    cp src/PredictionEngine.py.sweep_bak src/PredictionEngine.py
}

run_config() {
    local NAME="$1"; shift
    echo ""
    echo "################################################################"
    echo "  SWEEP: $NAME"
    echo "################################################################"
    "$@"
    PYTHONIOENCODING=utf-8 python src/pipeline.py --threshold 0.5 --n_stocks 50 2>&1 | tee "$SWEEP_DIR/${NAME}.log"
    restore_from_backup
    echo "  -> $NAME done"
}

patch_high_leverage() {
    python3 -c "
with open('src/pipeline.py','r',encoding='utf-8') as f: s=f.read()
s=s.replace('\"base_trade_size_pct\":   0.20,\n        \"leverage_min\":          2.0,\n        \"leverage_max\":          5.0,\n        \"short_size_multiplier\": 0.50,\n    },\n    \"CRISIS\"',
            '\"base_trade_size_pct\":   0.30,\n        \"leverage_min\":          3.0,\n        \"leverage_max\":          7.0,\n        \"short_size_multiplier\": 0.50,\n    },\n    \"CRISIS\"')
s=s.replace('\"base_trade_size_pct\":   0.20,\n        \"leverage_min\":          2.0,\n        \"leverage_max\":          5.0,\n        \"short_size_multiplier\": 0.50,\n    },\n    \"BEAR\"',
            '\"base_trade_size_pct\":   0.25,\n        \"leverage_min\":          2.5,\n        \"leverage_max\":          6.0,\n        \"short_size_multiplier\": 0.50,\n    },\n    \"BEAR\"')
s=s.replace('\"base_trade_size_pct\":   0.12,\n        \"leverage_min\":          1.5,\n        \"leverage_max\":          3.0,',
            '\"base_trade_size_pct\":   0.20,\n        \"leverage_min\":          2.0,\n        \"leverage_max\":          5.0,')
with open('src/pipeline.py','w',encoding='utf-8') as f: f.write(s)
print('high_leverage patched')
"
}

patch_tight_stops() {
    python3 -c "
with open('src/pipeline.py','r',encoding='utf-8') as f: s=f.read()
s=s.replace('\"long_safety_sl\":        0.05,\n        \"short_safety_sl\":       0.05,\n        \"base_trade_size_pct\":   0.20',
            '\"long_safety_sl\":        0.025,\n        \"short_safety_sl\":       0.03,\n        \"base_trade_size_pct\":   0.20')
s=s.replace('\"long_safety_sl\":        0.03,\n        \"short_safety_sl\":       0.05,',
            '\"long_safety_sl\":        0.02,\n        \"short_safety_sl\":       0.03,')
with open('src/pipeline.py','w',encoding='utf-8') as f: f.write(s)
with open('src/PredictionEngine.py','r',encoding='utf-8') as f: s=f.read()
s=s.replace('profit_pct >= 0.020','profit_pct >= 0.015')
s=s.replace('profit_pct >= 0.015','profit_pct >= 0.010')
with open('src/PredictionEngine.py','w',encoding='utf-8') as f: f.write(s)
print('tight_stops patched')
"
}

patch_no_vol_filter() {
    python3 -c "
with open('src/pipeline.py','r',encoding='utf-8') as f: s=f.read()
s=s.replace('VOL_REGIME_PERCENTILE = 95','VOL_REGIME_PERCENTILE = 100')
with open('src/pipeline.py','w',encoding='utf-8') as f: f.write(s)
with open('src/ensemble_model.py','r',encoding='utf-8') as f: s=f.read()
s=s.replace('MIN_VOL_RATIO         = 0.004','MIN_VOL_RATIO         = 0.0')
s=s.replace('MIN_VOL_RATIO   = 0.004','MIN_VOL_RATIO   = 0.0')
s=s.replace('VOL_REGIME_PERCENTILE = 95','VOL_REGIME_PERCENTILE = 100')
with open('src/ensemble_model.py','w',encoding='utf-8') as f: f.write(s)
print('no_vol_filter patched')
"
}

patch_bear_shorts() {
    python3 -c "
with open('src/pipeline.py','r',encoding='utf-8') as f: s=f.read()
s=s.replace('\"short_size_multiplier\": 0.50,\n    },\n    \"CRISIS\"',
            '\"short_size_multiplier\": 0.80,\n    },\n    \"CRISIS\"')
s=s.replace('\"short_size_multiplier\": 0.50,\n    },\n    \"BEAR\"',
            '\"short_size_multiplier\": 1.00,\n    },\n    \"BEAR\"')
s=s.replace('\"short_size_multiplier\": 0.80,\n    },\n}',
            '\"short_size_multiplier\": 1.00,\n    },\n}')
with open('src/pipeline.py','w',encoding='utf-8') as f: f.write(s)
with open('src/ensemble_model.py','r',encoding='utf-8') as f: s=f.read()
s=s.replace('SHORT_THRESHOLD       = 0.45','SHORT_THRESHOLD       = 0.47')
s=s.replace('SHORT_THRESHOLD = 0.45','SHORT_THRESHOLD = 0.47')
with open('src/ensemble_model.py','w',encoding='utf-8') as f: f.write(s)
print('bear_shorts patched')
"
}

# ── Configs 3-7: reuse trained tournament models (fast ~2-3 min each) ──
run_config "high_leverage"  patch_high_leverage
run_config "tight_stops"    patch_tight_stops
run_config "no_vol_filter"  patch_no_vol_filter
run_config "bear_shorts"    patch_bear_shorts

# ── Config 8: wide_barriers — must retrain (~15 min) ────────────────────
echo ""
echo "################################################################"
echo "  SWEEP: wide_barriers (RETRAINING FROM SCRATCH)"
echo "################################################################"
# Wipe tournament so pipeline retrains
rm -f results/group_tournament_lstm.csv models/master_manifest.json
python3 -c "
with open('src/lstm_model.py','r',encoding='utf-8') as f: s=f.read()
s=s.replace('SL_ATR_RATIO = 0.450','SL_ATR_RATIO = 0.300')
s=s.replace('PT_MIN_PCT = 0.0045','PT_MIN_PCT = 0.003')
s=s.replace('PT_MAX_PCT = 0.060','PT_MAX_PCT = 0.040')
s=s.replace('SL_MIN_PCT = 0.003','SL_MIN_PCT = 0.002')
s=s.replace('SL_MAX_PCT = 0.040','SL_MAX_PCT = 0.025')
with open('src/lstm_model.py','w',encoding='utf-8') as f: f.write(s)
print('wide_barriers patched (tighter ATR barriers)')
"
PYTHONIOENCODING=utf-8 python src/pipeline.py --threshold 0.5 --n_stocks 50 2>&1 | tee "$SWEEP_DIR/wide_barriers.log"
restore_from_backup
echo "  -> wide_barriers done"

echo ""
echo "################################################################"
echo "  ALL REMAINING SWEEPS COMPLETE"
echo "################################################################"
