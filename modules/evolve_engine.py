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

# Param rotation — one param per week, in order.
# If the week's net_pnl doesn't improve, rotate to next param.
PARAM_ROTATION = [
    "radar_score_threshold",
    "pulse_confidence_threshold",
    "daily_loss_limit",
]

# (step_size, min, max) per param
PARAM_SPACE = {
    "radar_score_threshold":      (10,  160, 220),
    "pulse_confidence_threshold": (5.0, 60.0, 90.0),
    "daily_loss_limit":           (2.0, 10.0, 25.0),
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
    # Write config override where the runner reads it
    apex_dir = data_path / "apex"
    apex_dir.mkdir(parents=True, exist_ok=True)
    config_override_path = apex_dir / "config-override.json"

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
    param, reason = _pick_param(metrics, current_config, evolve_dir)
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

def _pick_param(metrics: dict, config, evolve_dir: Path) -> tuple[Optional[str], str]:
    """Pick param based on weekly rotation state.

    Tunes the current week's param every day.
    At the end of each week (7 daily cycles), checks if net_pnl improved.
    If yes → stay on same param next week.
    If no → rotate to next param in PARAM_ROTATION.
    """
    total = metrics.get("total_round_trips", 0)
    if total < 5:
        return None, "need 5+ round trips before tuning"

    state = _load_weekly_state(evolve_dir)
    current_param = state.get("current_param", PARAM_ROTATION[0])
    week_start_pnl = state.get("week_start_pnl", metrics.get("net_pnl", 0.0))
    days_on_param = state.get("days_on_param", 0)

    # End of week — evaluate and maybe rotate
    if days_on_param >= 7:
        current_pnl = metrics.get("net_pnl", 0.0)
        improved = current_pnl > week_start_pnl

        if improved:
            # Keep same param, reset week
            reason = f"week improved (${current_pnl:.2f} vs ${week_start_pnl:.2f}) — continuing {current_param}"
            _save_weekly_state(evolve_dir, current_param, current_pnl, 0)
        else:
            # Rotate to next param
            idx = PARAM_ROTATION.index(current_param) if current_param in PARAM_ROTATION else 0
            next_param = PARAM_ROTATION[(idx + 1) % len(PARAM_ROTATION)]
            reason = f"week flat/negative (${current_pnl:.2f} vs ${week_start_pnl:.2f}) — rotating to {next_param}"
            current_param = next_param
            _save_weekly_state(evolve_dir, current_param, current_pnl, 0)
    else:
        # Mid-week — increment day count, keep tuning
        _save_weekly_state(evolve_dir, current_param, week_start_pnl, days_on_param + 1)
        reason = f"day {days_on_param + 1}/7 on {current_param}"

    return current_param, reason


def _load_weekly_state(evolve_dir: Path) -> dict:
    state_path = evolve_dir / "weekly_state.json"
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            pass
    return {}


def _save_weekly_state(evolve_dir: Path, param: str, week_start_pnl: float, days: int) -> None:
    state_path = evolve_dir / "weekly_state.json"
    state_path.write_text(json.dumps({
        "current_param": param,
        "week_start_pnl": week_start_pnl,
        "days_on_param": days,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


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
