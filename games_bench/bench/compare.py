from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPORT_VERSION = "compare-v1"
CompareMetricHook = Callable[[dict[str, Any]], dict[str, float]]


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_dir: Path
    run_config: dict[str, Any]
    summary: dict[str, Any]

    @property
    def game(self) -> str:
        return (
            _nonempty_str(self.run_config.get("game"))
            or _nonempty_str(self.summary.get("game"))
            or ""
        )

    @property
    def spec(self) -> str:
        explicit_spec = _nonempty_str(self.run_config.get("spec")) or _nonempty_str(
            self.summary.get("spec")
        )
        if explicit_spec is not None:
            return explicit_spec
        spec_base = _nonempty_str(self.run_config.get("spec_base")) or _nonempty_str(
            self.summary.get("spec_base")
        )
        if spec_base is not None and self.interaction_mode:
            return f"{spec_base}-{self.interaction_mode}"
        return spec_base or ""

    @property
    def interaction_mode(self) -> str:
        explicit_mode = _nonempty_str(
            self.run_config.get("interaction_mode")
        ) or _nonempty_str(self.summary.get("interaction_mode"))
        if explicit_mode is not None:
            return explicit_mode
        stateless_raw = self.run_config.get("stateless", self.summary.get("stateless"))
        if isinstance(stateless_raw, bool):
            return "stateless" if stateless_raw else "stateful"
        return ""

    @property
    def provider(self) -> str:
        return (
            _nonempty_str(self.run_config.get("provider"))
            or _nonempty_str(self.summary.get("provider"))
            or ""
        )

    @property
    def model(self) -> str:
        return (
            _nonempty_str(self.run_config.get("model"))
            or _nonempty_str(self.summary.get("model"))
            or ""
        )

    @property
    def match_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.game,
            self.spec,
            self.interaction_mode,
            self.provider,
            self.model,
        )


def _nonempty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid JSON object in file: {path}")
    return data


def _load_run_record(run_dir: Path) -> RunRecord:
    run_config = _read_json(run_dir / "run_config.json")
    summary = _read_json(run_dir / "summary.json")
    return RunRecord(run_dir=run_dir, run_config=run_config, summary=summary)


def _discover_run_records(path: Path) -> list[RunRecord]:
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"Path does not exist: {path}")

    direct_config = path / "run_config.json"
    direct_summary = path / "summary.json"
    if direct_config.exists() and direct_summary.exists():
        return [_load_run_record(path)]

    run_dirs: set[Path] = set()
    for config_path in path.rglob("run_config.json"):
        run_dir = config_path.parent
        if (run_dir / "summary.json").exists():
            run_dirs.add(run_dir)

    records = [_load_run_record(run_dir) for run_dir in sorted(run_dirs)]
    if not records:
        raise SystemExit(
            f"No scored run directories found under {path}. "
            "Expected run_config.json + summary.json."
        )
    return records


def _load_thresholds(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = _read_json(path)
    metrics = data.get("metrics", {})
    gating = data.get("gating", {})
    if metrics is not None and not isinstance(metrics, dict):
        raise SystemExit("thresholds.metrics must be an object")
    if gating is not None and not isinstance(gating, dict):
        raise SystemExit("thresholds.gating must be an object")
    return data


def _require_bool(value: Any, *, field: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise SystemExit(f"thresholds.gating['{field}'] must be a boolean")


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL file: {path}:{lineno}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"Invalid JSONL row at {path}:{lineno}: expected object")
        rows.append(payload)
    return rows


def _extract_binary_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    numeric = _to_float(value)
    if numeric is None:
        return None
    return 1.0 if numeric != 0.0 else 0.0


def _extract_numeric_metric(row: dict[str, Any], key: str) -> float | None:
    return _to_float(row.get(key))


_BOOTSTRAP_EXTRACTORS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "solve_rate": lambda row: _extract_binary_metric(row, "solved"),
    "deadlock_rate": lambda row: _extract_binary_metric(row, "deadlocked"),
    "avg_illegal_moves": lambda row: _extract_numeric_metric(row, "illegal_moves"),
    "avg_tool_calls": lambda row: _extract_numeric_metric(row, "tool_calls"),
    "avg_moves": lambda row: _extract_numeric_metric(row, "move_count"),
    "avg_pushes": lambda row: _extract_numeric_metric(row, "push_count"),
    "avg_boxes_on_goals_ratio": lambda row: _extract_numeric_metric(
        row, "boxes_on_goals_ratio"
    ),
    "avg_move_ratio": lambda row: _extract_numeric_metric(row, "move_ratio"),
    "avg_push_ratio": lambda row: _extract_numeric_metric(row, "push_ratio"),
}


def _extract_bootstrap_series(
    episodes: list[dict[str, Any]],
    *,
    metric_name: str,
) -> list[float]:
    extractor = _BOOTSTRAP_EXTRACTORS.get(metric_name)
    if extractor is None:
        return []
    values: list[float] = []
    for row in episodes:
        value = extractor(row)
        if value is None:
            continue
        values.append(float(value))
    return values


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    q = max(0.0, min(1.0, q))
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_difference(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    samples: int,
    confidence: float,
    rng: random.Random,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if not baseline_values or not candidate_values:
        raise ValueError("baseline/candidate values must be non-empty")
    n_baseline = len(baseline_values)
    n_candidate = len(candidate_values)
    deltas: list[float] = []
    for _ in range(samples):
        baseline_mean = (
            sum(baseline_values[rng.randrange(n_baseline)] for _ in range(n_baseline))
            / n_baseline
        )
        candidate_mean = (
            sum(
                candidate_values[rng.randrange(n_candidate)] for _ in range(n_candidate)
            )
            / n_candidate
        )
        deltas.append(candidate_mean - baseline_mean)

    deltas.sort()
    alpha = 1.0 - confidence
    ci_low = _quantile(deltas, alpha / 2.0)
    ci_high = _quantile(deltas, 1.0 - (alpha / 2.0))
    frac_le_zero = sum(1 for delta in deltas if delta <= 0.0) / samples
    frac_ge_zero = sum(1 for delta in deltas if delta >= 0.0) / samples
    p_two_sided = min(1.0, 2.0 * min(frac_le_zero, frac_ge_zero))

    observed_delta = (sum(candidate_values) / n_candidate) - (
        sum(baseline_values) / n_baseline
    )
    return {
        "status": "ok",
        "baseline_n": n_baseline,
        "candidate_n": n_candidate,
        "observed_delta": observed_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence_level": confidence,
        "p_value_two_sided": p_two_sided,
        "significant": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def _bootstrap_metrics_for_pair(
    baseline: RunRecord,
    candidate: RunRecord,
    *,
    metric_names: list[str],
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    baseline_rows = _read_jsonl(baseline.run_dir / "episodes.jsonl")
    candidate_rows = _read_jsonl(candidate.run_dir / "episodes.jsonl")
    if not baseline_rows or not candidate_rows:
        return {
            metric_name: {
                "status": "missing",
                "reason": "episodes.jsonl missing or empty for baseline/candidate run",
            }
            for metric_name in metric_names
        }

    rng = random.Random(seed)
    results: dict[str, Any] = {}
    for metric_name in metric_names:
        baseline_values = _extract_bootstrap_series(
            baseline_rows,
            metric_name=metric_name,
        )
        candidate_values = _extract_bootstrap_series(
            candidate_rows,
            metric_name=metric_name,
        )
        if not baseline_values or not candidate_values:
            results[metric_name] = {
                "status": "missing",
                "reason": "metric unsupported for bootstrap or values missing in episodes",
            }
            continue
        results[metric_name] = _bootstrap_difference(
            baseline_values,
            candidate_values,
            samples=samples,
            confidence=confidence,
            rng=rng,
        )
    return results


def _compare_metric(
    *,
    metric_name: str,
    baseline_value: Any,
    candidate_value: Any,
    rule: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline = _to_float(baseline_value)
    candidate = _to_float(candidate_value)
    result: dict[str, Any] = {
        "metric": metric_name,
        "baseline": baseline,
        "candidate": candidate,
        "delta": None,
        "direction": None,
        "threshold": None,
        "status": "ok",
        "regression": False,
        "reason": None,
    }

    if baseline is None or candidate is None:
        result["status"] = "missing"
        result["reason"] = "baseline or candidate value is non-numeric"
        return result

    delta = candidate - baseline
    result["delta"] = delta

    if rule is None:
        return result

    direction = str(rule.get("direction", "")).strip().lower()
    result["direction"] = direction

    if direction == "higher_better":
        max_drop_raw = rule.get("max_drop", 0.0)
        max_drop = _to_float(max_drop_raw)
        if max_drop is None or max_drop < 0:
            raise SystemExit(
                f"Invalid max_drop for metric '{metric_name}': {max_drop_raw!r}"
            )
        result["threshold"] = {"max_drop": max_drop}
        if delta < -max_drop:
            result["status"] = "regression"
            result["regression"] = True
            result["reason"] = (
                f"candidate dropped by {abs(delta):.6f} > allowed {max_drop:.6f}"
            )
        return result

    if direction == "lower_better":
        max_increase_raw = rule.get("max_increase", 0.0)
        max_increase = _to_float(max_increase_raw)
        if max_increase is None or max_increase < 0:
            raise SystemExit(
                f"Invalid max_increase for metric '{metric_name}': {max_increase_raw!r}"
            )
        result["threshold"] = {"max_increase": max_increase}
        if delta > max_increase:
            result["status"] = "regression"
            result["regression"] = True
            result["reason"] = (
                f"candidate increased by {delta:.6f} > allowed {max_increase:.6f}"
            )
        return result

    if direction == "equal":
        max_abs_diff_raw = rule.get("max_abs_diff", 0.0)
        max_abs_diff = _to_float(max_abs_diff_raw)
        if max_abs_diff is None or max_abs_diff < 0:
            raise SystemExit(
                f"Invalid max_abs_diff for metric '{metric_name}': {max_abs_diff_raw!r}"
            )
        result["threshold"] = {"max_abs_diff": max_abs_diff}
        if abs(delta) > max_abs_diff:
            result["status"] = "regression"
            result["regression"] = True
            result["reason"] = (
                f"absolute diff {abs(delta):.6f} > allowed {max_abs_diff:.6f}"
            )
        return result

    raise SystemExit(
        f"Invalid direction for metric '{metric_name}': {direction!r}. "
        "Expected one of: higher_better, lower_better, equal."
    )


def _gating_violations(
    baseline: RunRecord,
    candidate: RunRecord,
    gating: dict[str, Any],
) -> list[str]:
    require_same_spec = _require_bool(
        gating.get("require_same_spec"),
        field="require_same_spec",
        default=True,
    )
    require_same_mode = _require_bool(
        gating.get("require_same_interaction_mode"),
        field="require_same_interaction_mode",
        default=True,
    )
    require_same_game = _require_bool(
        gating.get("require_same_game"),
        field="require_same_game",
        default=True,
    )
    require_same_provider = _require_bool(
        gating.get("require_same_provider"),
        field="require_same_provider",
        default=True,
    )
    require_same_model = _require_bool(
        gating.get("require_same_model"),
        field="require_same_model",
        default=True,
    )

    violations: list[str] = []
    if require_same_game and baseline.game != candidate.game:
        violations.append(
            f"game mismatch: baseline={baseline.game!r}, candidate={candidate.game!r}"
        )
    if require_same_spec and baseline.spec != candidate.spec:
        violations.append(
            f"spec mismatch: baseline={baseline.spec!r}, candidate={candidate.spec!r}"
        )
    if require_same_mode and baseline.interaction_mode != candidate.interaction_mode:
        violations.append(
            "interaction_mode mismatch: "
            f"baseline={baseline.interaction_mode!r}, candidate={candidate.interaction_mode!r}"
        )
    if require_same_provider and baseline.provider != candidate.provider:
        violations.append(
            f"provider mismatch: baseline={baseline.provider!r}, candidate={candidate.provider!r}"
        )
    if require_same_model and baseline.model != candidate.model:
        violations.append(
            f"model mismatch: baseline={baseline.model!r}, candidate={candidate.model!r}"
        )
    return violations


def _match_records(
    baseline_records: list[RunRecord],
    candidate_records: list[RunRecord],
) -> tuple[list[tuple[RunRecord, RunRecord]], list[str], list[str]]:
    if len(baseline_records) == 1 and len(candidate_records) == 1:
        return (
            [(baseline_records[0], candidate_records[0])],
            [],
            [],
        )

    baseline_map = _keyed_records_map(baseline_records, label="baseline")
    candidate_map = _keyed_records_map(candidate_records, label="candidate")

    matched_pairs: list[tuple[RunRecord, RunRecord]] = []
    for key in sorted(set(baseline_map) & set(candidate_map)):
        matched_pairs.append((baseline_map[key], candidate_map[key]))

    unmatched_baseline = [
        str(baseline_map[key].run_dir)
        for key in sorted(set(baseline_map) - set(candidate_map))
    ]
    unmatched_candidate = [
        str(candidate_map[key].run_dir)
        for key in sorted(set(candidate_map) - set(baseline_map))
    ]

    return matched_pairs, unmatched_baseline, unmatched_candidate


def _keyed_records_map(
    records: list[RunRecord],
    *,
    label: str,
) -> dict[tuple[str, str, str, str, str], RunRecord]:
    result: dict[tuple[str, str, str, str, str], RunRecord] = {}
    for record in records:
        key = record.match_key
        if key in result:
            raise SystemExit(
                f"Duplicate {label} run key detected for {key}. "
                f"Conflicting runs: {result[key].run_dir} and {record.run_dir}"
            )
        result[key] = record
    return result


def _resolve_compare_metric_hooks(
    records: list[RunRecord],
) -> dict[str, CompareMetricHook | None]:
    game_names = sorted({record.game for record in records if record.game})
    if not game_names:
        return {}

    from games_bench.bench.registry import get_benchmark, load_builtin_benchmarks

    load_builtin_benchmarks()
    hooks: dict[str, CompareMetricHook | None] = {}
    for game_name in game_names:
        try:
            benchmark = get_benchmark(game_name)
        except KeyError:
            hooks[game_name] = None
            continue
        hooks[game_name] = benchmark.compare_metrics
    return hooks


def _extract_metrics_from_summary(
    record: RunRecord,
    *,
    hook: CompareMetricHook | None,
    side: str,
) -> dict[str, Any]:
    if hook is None:
        payload = record.summary.get("overall", {})
        if not isinstance(payload, dict):
            raise SystemExit(
                "Invalid summary format: expected object at summary.overall "
                f"for {side} run {record.run_dir}"
            )
        return dict(payload)

    metrics = hook(record.summary)
    if not isinstance(metrics, dict):
        raise SystemExit(
            "Invalid compare_metrics hook result for game "
            f"{record.game!r}: expected object, got {type(metrics).__name__}"
        )
    return {str(name): value for name, value in metrics.items()}


def compare_runs(
    *,
    baseline_path: Path,
    candidate_path: Path,
    thresholds_path: Path | None,
    bootstrap_samples: int = 0,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
    bootstrap_metrics: list[str] | None = None,
) -> dict[str, Any]:
    if bootstrap_samples < 0:
        raise SystemExit("bootstrap_samples must be >= 0")
    if not (0.0 < bootstrap_confidence < 1.0):
        raise SystemExit("bootstrap_confidence must be within (0, 1).")

    baseline_records = _discover_run_records(baseline_path)
    candidate_records = _discover_run_records(candidate_path)
    compare_metric_hooks = _resolve_compare_metric_hooks(
        baseline_records + candidate_records
    )

    thresholds = _load_thresholds(thresholds_path)
    metric_rules = thresholds.get("metrics") or {}
    gating = thresholds.get("gating") or {}
    for metric_name, rule in metric_rules.items():
        if not isinstance(rule, dict):
            raise SystemExit(f"thresholds.metrics['{metric_name}'] must be an object")

    matched_pairs, unmatched_baseline, unmatched_candidate = _match_records(
        baseline_records,
        candidate_records,
    )

    if not matched_pairs:
        raise SystemExit(
            "No comparable run pairs found. "
            "For directory mode, pairs are matched on "
            "(game, spec, interaction_mode, provider, model)."
        )

    comparisons: list[dict[str, Any]] = []
    total_regressions = 0
    total_metric_checks = 0
    total_missing_metric_values = 0
    total_gating_violations = 0

    for baseline, candidate in matched_pairs:
        baseline_overall = _extract_metrics_from_summary(
            baseline,
            hook=compare_metric_hooks.get(baseline.game),
            side="baseline",
        )
        candidate_overall = _extract_metrics_from_summary(
            candidate,
            hook=compare_metric_hooks.get(candidate.game),
            side="candidate",
        )

        if metric_rules:
            metric_names = sorted(metric_rules.keys())
        else:
            metric_names = []
            for name in sorted(
                set(baseline_overall.keys()) | set(candidate_overall.keys())
            ):
                if (
                    _to_float(baseline_overall.get(name)) is None
                    and _to_float(candidate_overall.get(name)) is None
                ):
                    continue
                metric_names.append(name)

        metric_results: dict[str, Any] = {}
        regression_count = 0
        for metric_name in metric_names:
            rule = metric_rules.get(metric_name)
            metric_result = _compare_metric(
                metric_name=metric_name,
                baseline_value=baseline_overall.get(metric_name),
                candidate_value=candidate_overall.get(metric_name),
                rule=rule if isinstance(rule, dict) else None,
            )
            metric_results[metric_name] = metric_result
            if metric_result.get("regression"):
                regression_count += 1
            if metric_result.get("status") == "missing":
                total_missing_metric_values += 1
            else:
                total_metric_checks += 1

        bootstrap_payload: dict[str, Any] = {}
        if bootstrap_samples > 0:
            target_metrics = (
                sorted(set(bootstrap_metrics)) if bootstrap_metrics else metric_names
            )
            if target_metrics:
                seed_material = (
                    f"{baseline.run_dir.resolve()}::{candidate.run_dir.resolve()}"
                ).encode("utf-8")
                seed_hash = int(hashlib.sha256(seed_material).hexdigest()[:12], 16)
                pair_seed = int(seed_hash + int(bootstrap_seed))
                bootstrap_payload = _bootstrap_metrics_for_pair(
                    baseline,
                    candidate,
                    metric_names=target_metrics,
                    samples=int(bootstrap_samples),
                    confidence=float(bootstrap_confidence),
                    seed=pair_seed,
                )

        violations = _gating_violations(baseline, candidate, gating)
        total_gating_violations += len(violations)
        regression_count += len(violations)

        comparisons.append(
            {
                "match_key": {
                    "game": baseline.game,
                    "spec": baseline.spec,
                    "interaction_mode": baseline.interaction_mode,
                    "provider": baseline.provider,
                    "model": baseline.model,
                },
                "baseline_run_dir": str(baseline.run_dir),
                "candidate_run_dir": str(candidate.run_dir),
                "metrics": metric_results,
                "bootstrap": bootstrap_payload,
                "gating_violations": violations,
                "regression_count": regression_count,
            }
        )
        total_regressions += regression_count

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "report_version": REPORT_VERSION,
        "generated_at": now,
        "baseline_path": str(baseline_path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
        "thresholds_path": str(thresholds_path.resolve()) if thresholds_path else None,
        "thresholds": thresholds,
        "bootstrap": {
            "enabled": bool(bootstrap_samples > 0),
            "samples": int(bootstrap_samples),
            "confidence": float(bootstrap_confidence),
            "seed": int(bootstrap_seed),
            "metrics": sorted(set(bootstrap_metrics)) if bootstrap_metrics else None,
        },
        "summary": {
            "pairs": len(comparisons),
            "metric_checks": total_metric_checks,
            "missing_metric_values": total_missing_metric_values,
            "regressions": total_regressions,
            "gating_violations": total_gating_violations,
            "unmatched_baseline": unmatched_baseline,
            "unmatched_candidate": unmatched_candidate,
        },
        "comparisons": comparisons,
    }


def _format_metric_line(metric_name: str, payload: dict[str, Any]) -> str:
    baseline = payload.get("baseline")
    candidate = payload.get("candidate")
    delta = payload.get("delta")
    status = payload.get("status", "ok")
    reason = payload.get("reason")

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    parts = [f"{metric_name}: baseline={_fmt(baseline)} candidate={_fmt(candidate)}"]
    if delta is not None:
        parts.append(f"delta={_fmt(delta)}")
    parts.append(f"status={status}")
    if reason:
        parts.append(f"reason={reason}")
    return " | ".join(parts)


def _format_bootstrap_line(metric_name: str, payload: dict[str, Any]) -> str:
    status = str(payload.get("status", "missing"))
    if status != "ok":
        reason = payload.get("reason")
        suffix = f" reason={reason}" if reason else ""
        return f"{metric_name}: status={status}{suffix}"
    ci_low = payload.get("ci_low")
    ci_high = payload.get("ci_high")
    p_value = payload.get("p_value_two_sided")
    observed_delta = payload.get("observed_delta")
    return (
        f"{metric_name}: delta={observed_delta:.6f} "
        f"ci=[{ci_low:.6f}, {ci_high:.6f}] "
        f"p={p_value:.6f} significant={bool(payload.get('significant'))}"
    )


def _print_human_report(report: dict[str, Any]) -> None:
    summary = report.get("summary", {})
    print(
        "Compare summary: "
        f"pairs={summary.get('pairs', 0)} "
        f"regressions={summary.get('regressions', 0)} "
        f"gating_violations={summary.get('gating_violations', 0)}"
    )
    missing_metric_values = int(summary.get("missing_metric_values", 0))
    if missing_metric_values > 0:
        print(
            "Warnings: "
            f"{missing_metric_values} metric value(s) unavailable/non-numeric "
            "and skipped from regression checks."
        )

    unmatched_baseline = summary.get("unmatched_baseline") or []
    unmatched_candidate = summary.get("unmatched_candidate") or []
    if unmatched_baseline:
        print("Unmatched baseline runs:")
        for item in unmatched_baseline:
            print(f"  - {item}")
    if unmatched_candidate:
        print("Unmatched candidate runs:")
        for item in unmatched_candidate:
            print(f"  - {item}")

    for comp in report.get("comparisons", []):
        key = comp.get("match_key", {})
        key_text = (
            f"game={key.get('game')} spec={key.get('spec')} "
            f"mode={key.get('interaction_mode')} "
            f"provider={key.get('provider')} model={key.get('model')}"
        )
        print(f"\nPair: {key_text}")

        violations = comp.get("gating_violations") or []
        if violations:
            print("  Gating violations:")
            for issue in violations:
                print(f"    - {issue}")

        metrics = comp.get("metrics", {})
        for metric_name in sorted(metrics.keys()):
            print("  " + _format_metric_line(metric_name, metrics[metric_name]))
        bootstrap = comp.get("bootstrap") or {}
        if bootstrap:
            print("  Bootstrap:")
            for metric_name in sorted(bootstrap.keys()):
                print(
                    "    " + _format_bootstrap_line(metric_name, bootstrap[metric_name])
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare benchmark runs and apply regression thresholds."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline run directory or parent directory of run directories.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate run directory or parent directory of run directories.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Path to thresholds JSON file.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return exit code 1 if any regression is detected.",
    )
    parser.add_argument(
        "--report-file",
        default="compare_report.json",
        help="Path to write machine-readable compare report JSON.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help=(
            "Number of bootstrap resamples for uncertainty reporting. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals (default: 0.95).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Base RNG seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--bootstrap-metric",
        action="append",
        dest="bootstrap_metrics",
        default=None,
        help=(
            "Metric name to bootstrap (repeatable). Defaults to compared metrics "
            "when omitted."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    report = compare_runs(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        thresholds_path=(Path(args.thresholds) if args.thresholds else None),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_confidence=float(args.bootstrap_confidence),
        bootstrap_seed=int(args.bootstrap_seed),
        bootstrap_metrics=list(args.bootstrap_metrics or []),
    )

    report_file = Path(args.report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2))

    _print_human_report(report)
    print(f"\nMachine report: {report_file.resolve()}")

    regressions = int(report.get("summary", {}).get("regressions", 0))
    if args.fail_on_regression and regressions > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
