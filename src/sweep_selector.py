"""
sweep_selector.py - Auto-run sweep configs and select best performer.

Usage:
    python src/sweep_selector.py [--bear] [--live] [--threshold 0.5]

Runs all 20 sweep configs, captures Sharpe ratios, saves best to models/optimized_config.json
"""

import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
import sys

# 20 sweep configs (reduced from 31)
SWEEP_CONFIGS = [
    "high_leverage",
    "tight_stops",
    "no_vol_filter",
    "aggressive",
    "conservative",
    "conservative_leverage",
    "aggressive_novol",
    "highleverage_novol",
    "trade_size_half",
    "trade_size_double",
    "sl_tight_1pct",
    "sl_wide_10pct",
    "bear_short_sl_tight",
    "bear_short_sl_wide",
    "bear_short_threshold_lower",
    "bear_short_threshold_higher",
    "bear_short_larger_trades",
    "bear_short_higher_leverage",
    "bear_no_longs",
    "bear_short_combined",
    "exit_breakeven_trail",
	"SPECIALIST_BLEND_50_50",
	"SPECIALIST_BLEND_60_40",
	"SPECIALIST_BLEND_70_30",
]


def run_sweep_config(config_name, extra_args=None, idx=0, total=20):
    """Run a single sweep config and extract Sharpe ratio from output."""
    extra_args = extra_args or []
    cmd = ["python", "src/sweep_stage3.py", config_name] + extra_args

    print(f"\n{'='*60}")
    print(f"[{idx+1}/{total}] Running: {config_name}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Starting at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  (this may take 5-10 minutes per config...)")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per config
        )

        # Extract Sharpe from output
        sharpe_match = re.search(r"Sharpe\s*:\s*([+-]?\d+\.?\d*)", result.stdout)
        return_match = re.search(r"Return\s*:\s*([+-]?\d+\.?\d*)%", result.stdout)
        trades_match = re.search(r"Trades\s*:\s*(\d+)", result.stdout)

        if sharpe_match:
            sharpe = float(sharpe_match.group(1))
            ret = float(return_match.group(1)) if return_match else 0.0
            trades = int(trades_match.group(1)) if trades_match else 0
            print(f"  -> COMPLETED at {datetime.now().strftime('%H:%M:%S')}")
            print(f"  -> Sharpe: {sharpe:.3f} | Return: {ret:+.2f}% | Trades: {trades}")
            return {
                "config": config_name,
                "sharpe": sharpe,
                "return_pct": ret,
                "trades": trades,
                "stdout": result.stdout[-500:],  # Last 500 chars for debugging
            }
        else:
            print(f"  -> FAILED: Could not extract Sharpe from output")
            print(f"  -> stderr: {result.stderr[-200:]}")
            return {
                "config": config_name,
                "sharpe": -999.0,
                "return_pct": 0.0,
                "trades": 0,
                "stdout": result.stdout[-500:],
                "error": "Sharpe extraction failed",
            }

    except subprocess.TimeoutExpired:
        print(f"  -> TIMEOUT after 10 minutes")
        return {"config": config_name, "sharpe": -999.0, "error": "timeout"}
    except Exception as e:
        print(f"  -> ERROR: {e}")
        return {"config": config_name, "sharpe": -999.0, "error": str(e)}


def detect_regime_from_output(outputs):
    """Detect dominant regime from outputs (simplified)."""
    # Look for regime distribution in any output
    for output in outputs:
        stdout = output.get("stdout", "")
        if "BULL:" in stdout and "BEAR:" in stdout:
            # Extract regime distribution
            bull_match = re.search(r"BULL:\s*(\d+)\s+timesteps", stdout)
            bear_match = re.search(r"BEAR:\s*(\d+)\s+timesteps", stdout)
            crisis_match = re.search(r"CRISIS:\s*(\d+)\s+timesteps", stdout)

            counts = {}
            if bull_match:
                counts["BULL"] = int(bull_match.group(1))
            if bear_match:
                counts["BEAR"] = int(bear_match.group(1))
            if crisis_match:
                counts["CRISIS"] = int(crisis_match.group(1))

            if counts:
                return max(counts, key=counts.get)

    return "UNKNOWN"


def main():
    # Parse args
    extra_args = []
    if "--bear" in sys.argv:
        extra_args.append("--bear")
    if "--live" in sys.argv:
        extra_args.append("--live")
    if "--threshold" in sys.argv:
        idx = sys.argv.index("--threshold")
        if idx + 1 < len(sys.argv):
            extra_args.extend(["--threshold", sys.argv[idx + 1]])
    if "--oos_holdout_pct" in sys.argv:
        idx = sys.argv.index("--oos_holdout_pct")
        if idx + 1 < len(sys.argv):
            extra_args.extend(["--oos_holdout_pct", sys.argv[idx + 1]])
    if "--sentiment_alpha" in sys.argv:
        idx = sys.argv.index("--sentiment_alpha")
        if idx + 1 < len(sys.argv):
            extra_args.extend(["--sentiment_alpha", sys.argv[idx + 1]])
    if "--oos_holdout_pct" in sys.argv:
        idx = sys.argv.index("--oos_holdout_pct")
        if idx + 1 < len(sys.argv):
            extra_args.extend(["--oos_holdout_pct", sys.argv[idx + 1]])
    if "--sentiment_alpha" in sys.argv:
        idx = sys.argv.index("--sentiment_alpha")
        if idx + 1 < len(sys.argv):
            extra_args.extend(["--sentiment_alpha", sys.argv[idx + 1]])

    print(f"\n{'='*60}")
    print("SWEEP SELECTOR: Running 20 configs to find best performer")
    print(f"{'='*60}")
    print(f"Configs to test: {len(SWEEP_CONFIGS)}")
    print(f"Extra args: {extra_args}\n")

    # Run all configs
    results = []
    for i, config in enumerate(SWEEP_CONFIGS):
        result = run_sweep_config(config, extra_args, idx=i, total=len(SWEEP_CONFIGS))
        results.append(result)

    # Find best by Sharpe
    valid_results = [r for r in results if r["sharpe"] > -900]
    if not valid_results:
        print("\n" + "="*60)
        print("ERROR: No valid results from any config")
        print("="*60)
        sys.exit(1)

    best = max(valid_results, key=lambda x: x["sharpe"])

    # Detect regime
    regime = detect_regime_from_output(results)

    print(f"\n{'='*60}")
    print("SWEEP RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\nRank | Config{' '*30} | Sharpe | Return | Trades")
    print("-" * 70)

    sorted_results = sorted(valid_results, key=lambda x: x["sharpe"], reverse=True)
    for i, r in enumerate(sorted_results[:10], 1):  # Top 10
        config_name = r["config"][:35].ljust(35)
        print(f"{i:2d}   | {config_name} | {r['sharpe']:6.3f} | {r['return_pct']:+6.2f}% | {r['trades']:6d}")

    print(f"\n{'='*60}")
    print(f"WINNER: {best['config']}")
    print(f"  Sharpe: {best['sharpe']:.3f}")
    print(f"  Return: {best['return_pct']:+.2f}%")
    print(f"  Trades: {best['trades']}")
    print(f"  Detected Regime: {regime}")
    print(f"{'='*60}")

    # Save to optimized_config.json
    output = {
        "config": best["config"],
        "sharpe": best["sharpe"],
        "return_pct": best["return_pct"],
        "trades": best["trades"],
        "regime": regime,
        "timestamp": datetime.now().isoformat(),
        "all_results": [
            {"config": r["config"], "sharpe": r["sharpe"], "return_pct": r["return_pct"], "trades": r["trades"]}
            for r in sorted_results
        ],
    }

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    config_path = models_dir / "optimized_config.json"

    with open(config_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved optimized config to: {config_path}")
    print(f"\nTo use this config in demo mode:")
    print(f"  python src/demo_runner.py --use-optimized")


if __name__ == "__main__":
    main()
