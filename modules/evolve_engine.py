"""EVOLVE engine — autoresearch loop for APEX config optimization.

Reads REFLECT metrics, picks ONE param to tune (based on current market
state), generates 3 variants (current, +step, -step), backtests each
against historical trades, promotes the winner if it beats baseline.

One param. One change. Verified before applied.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("evolve_engine")

# Param definitions: (step_size, min, max)
# Nano bounds — conservative range safe for $50 accounts
PARAM_SPACE = {
    "radar_score_threshold":      (10,  160,  220),   # don't go below 160 (too noisy) or above 220 (no trades)
    "pulse_confidence_threshold": (5.0, 60.0, 90.0),  # don't go below 60 (too noisy)
    "daily_loss_limit":           (1.0,  5.0, 15.0),  # $5–$15 range for $50 account
    "max_same_direction":         (1,    1,    1),     # always 1 with nano — no stacking
}


@dataclass
class EvolveResult:
    param: str
    old_value: float
    new_value: float
    old_net_pnl: float
    new_net_pnl: float
    direction: str   # "up" | "down" | "no_change"
    reason: str
    applied: bool


def run(data_dir: str = "/data") -> Optional[EvolveResult]:
    """Run one EVOLVE cycle. Returns result or None if skipped."""
    data_path = Path(data_dir)
    trades_path = data_path / "cli" / "trades.jsonl"
    evolve_dir = data_path / "evolve"
    evolve_dir.mkdir(parents=True, exist_ok=True)
    config_override_path = evolve_dir / "apex_config.json"

    # --- Step 1: Load REFLECT metrics ---
    metrics = _load_latest_reflect_metrics(data_path)
    if metrics is None:
        log.info("EVOLVE: no REFLECT metrics found, skipping")
        return None

    total_trades = metrics.get("total_round_trips", 0)
    if total_trades < 5:
        log.info("EVOLVE: only %d round trips — need 5+ to evolve", total_trades)
        return None

    if not trades_path.exists():
        log.info("EVOLVE: no trades file at %s, skipping", trades_path)
        return None

    # --- Step 2: Load current config ---
    from modules.apex_config import ApexConfig
    current_config = _load_config(config_override_path)

    # --- Step 3: Pick ONE param to tune ---
    param, reason = _pick_param(metrics, current_config)
    if param is None:
        log.info("EVOLVE: no param to tune right now (%s)", reason)
        return None

    step, lo, hi = PARAM_SPACE[param]
    current_val = getattr(current_config, param)

    # --- Step 4: Generate 3 variants: current, +step, -step ---
    candidates = {
        "baseline": current_val,
        "up":       min(hi, current_val + step),
        "down":     max(lo, current_val - step),
    }
    # Deduplicate
    candidates = {k: v for k, v in candidates.items() if v != current_val or k == "baseline"}

    log.info("EVOLVE: tuning %s (current=%s) — testing %s", param, current_val, candidates)

    # --- Step 5: Backtest each variant ---
    results = {}
    for label, val in candidates.items():
        pnl = _backtest_variant(current_config, param, val, trades_path)
        if pnl is not None:
            results[label] = (val, pnl)
            log.info("  %s: %s=%s → net_pnl=%.2f", label, param, val, pnl)

    if not results:
        log.warning("EVOLVE: all backtests failed/rejected")
        return None

    # --- Step 6: Pick winner ---
    best_label = max(results, key=lambda k: results[k][1])
    best_val, best_pnl = results[best_label]
    baseline_pnl = results.get("baseline", (current_val, None))[1]

    if baseline_pnl is None:
        baseline_pnl = 0.0

    # Only apply if strictly better than baseline
    if best_label == "baseline" or best_pnl <= baseline_pnl:
        log.info("EVOLVE: baseline wins (%.2f) — no change to %s", baseline_pnl, param)
        direction = "no_change"
        applied = False
        new_val = current_val
    else:
        direction = best_label  # "up" or "down"
        new_val = best_val
        applied = True

        # Write config override
        config_dict = _config_to_dict(current_config)
        config_dict[param] = new_val
        config_override_path.write_text(json.dumps(config_dict, indent=2))
        log.info(
            "EVOLVE: promoted %s %s -> %s (net_pnl: %.2f -> %.2f)",
            param, current_val, new_val, baseline_pnl, best_pnl,
        )

    # --- Step 7: Write evolve log ---
    result = EvolveResult(
        param=param,
        old_value=current_val,
        new_value=new_val,
        old_net_pnl=baseline_pnl,
        new_net_pnl=best_pnl,
        direction=direction,
        reason=reason,
        applied=applied,
    )
    _append_evolve_log(evolve_dir, result, metrics)
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _pick_param(metrics: dict, config) -> tuple[Optional[str], str]:
    """Pick the single most impactful param to tune given current metrics.

    Priority:
    1. Emergency (fees > gross PnL) → radar_score_threshold
    2. Critical FDR (>30%) → radar_score_threshold
    3. Low win rate (<40%) → pulse_confidence_threshold
    4. Loss streak (>=5) → daily_loss_limit
    5. Direction imbalance → max_same_direction
    6. Healthy → try lowering radar_score_threshold to get more trades
    """
    total = metrics.get("total_round_trips", 0)
    fdr = metrics.get("fdr", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    net_pnl = metrics.get("net_pnl", 0.0)
    gross_pnl = metrics.get("gross_pnl", 0.0)
    total_fees = metrics.get("total_fees", 0.0)
    consec_losses = metrics.get("max_consecutive_losses", 0)
    long_pnl = metrics.get("long_pnl", 0.0)
    short_pnl = metrics.get("short_pnl", 0.0)

    # Emergency
    if total >= 3 and total_fees > abs(gross_pnl):
        return "radar_score_threshold", "EMERGENCY: fees exceed gross PnL"

    # Critical FDR
    if fdr > 30:
        return "radar_score_threshold", f"FDR critical ({fdr:.1f}%): filter low-quality entries"

    # Low win rate
    if win_rate < 40 and total >= 5:
        return "pulse_confidence_threshold", f"Win rate low ({win_rate:.1f}%): require higher conviction"

    # Loss streak
    if consec_losses >= 5:
        return "daily_loss_limit", f"Loss streak ({consec_losses}): reduce daily limit"

    # Direction imbalance
    if long_pnl < 0 and short_pnl > 0 and metrics.get("long_count", 0) >= 3:
        return "max_same_direction", "Long bias losing: limit same-direction slots"
    if short_pnl < 0 and long_pnl > 0 and metrics.get("short_count", 0) >= 3:
        return "max_same_direction", "Short bias losing: limit same-direction slots"

    # Healthy — try to open up more trades
    if win_rate >= 50 and net_pnl > 0 and fdr < 15 and total >= 5:
        cur_threshold = getattr(config, "radar_score_threshold", 170)
        if cur_threshold >= 140:
            return "radar_score_threshold", f"Healthy strategy: try lowering radar threshold to capture more trades"

    return None, "no clear direction from metrics"


def _backtest_variant(config, param: str, value, trades_path: Path) -> Optional[float]:
    """Run backtest_apex.py with a single param overridden. Returns net_pnl or None."""
    config_dict = _config_to_dict(config)
    config_dict[param] = value

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_dict, f)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, "scripts/backtest_apex.py",
             "--config", tmp_path,
             "--trades", str(trades_path)],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip()
        if output.startswith("REJECT"):
            return None
        # Parse "net_pnl: 12.34" format
        for line in output.splitlines():
            if line.startswith("net_pnl:"):
                return float(line.split(":")[1].strip())
    except Exception as e:
        log.warning("EVOLVE: backtest error for %s=%s: %s", param, value, e)
    finally:
        os.unlink(tmp_path)

    return None


def _load_latest_reflect_metrics(data_path: Path) -> Optional[dict]:
    """Load the most recent REFLECT metrics from JSON summaries or report."""
    # Try reflect metrics JSON first
    reflect_dir = data_path / "reflect"
    if not reflect_dir.exists():
        return None

    # Look for metrics JSON files (newest first)
    metric_files = sorted(reflect_dir.glob("*.json"), reverse=True)
    for f in metric_files:
        try:
            return json.loads(f.read_text())
        except Exception:
            continue

    # Fall back: parse latest .md report for key metrics
    md_files = sorted(reflect_dir.glob("*.md"), reverse=True)
    for f in md_files:
        metrics = _parse_reflect_md(f.read_text())
        if metrics:
            return metrics

    return None


def _parse_reflect_md(text: str) -> Optional[dict]:
    """Extract key metrics from a REFLECT markdown report."""
    import re
    metrics = {}
    patterns = {
        "win_rate": r"Win Rate[^\d]+([\d.]+)%",
        "net_pnl": r"Net PnL[^\d$+-]*([\+\-]?[\d.]+)",
        "fdr": r"FDR[^\d]+([\d.]+)%",
        "total_round_trips": r"Round Trips[^\d]+(\d+)",
        "profit_factor": r"Profit Factor[^\d]+([\d.]+)",
        "max_consecutive_losses": r"Max Consec.*?Losses[^\d]+(\d+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                metrics[key] = float(m.group(1))
            except ValueError:
                pass
    return metrics if len(metrics) >= 3 else None


def _load_config(config_path: Path):
    """Load ApexConfig from JSON override or return defaults."""
    from modules.apex_config import ApexConfig
    if config_path.exists():
        try:
            return ApexConfig.from_json(str(config_path))
        except Exception as e:
            log.warning("EVOLVE: failed to load config override: %s", e)
    return ApexConfig()


def _config_to_dict(config) -> dict:
    """Serialize ApexConfig to a plain dict."""
    try:
        return asdict(config)
    except Exception:
        # Fallback for non-dataclass
        return {k: v for k, v in vars(config).items()
                if not k.startswith("_")}


def _append_evolve_log(evolve_dir: Path, result: EvolveResult, metrics: dict) -> None:
    """Append EVOLVE cycle result to the evolve log."""
    log_path = evolve_dir / "evolve_log.jsonl"
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "param": result.param,
        "old_value": result.old_value,
        "new_value": result.new_value,
        "old_net_pnl": result.old_net_pnl,
        "new_net_pnl": result.new_net_pnl,
        "direction": result.direction,
        "reason": result.reason,
        "applied": result.applied,
        "metrics_snapshot": {
            k: metrics.get(k)
            for k in ["win_rate", "net_pnl", "fdr", "total_round_trips", "profit_factor"]
        },
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
