from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

from compare_decomposition_runs import _quality_entry, model_order_checks, resolve_result_files

from icaim_py.common import json_ready, load_results_file, save_json


def _finite_or_inf(value: float | None) -> float:
    if value is None:
        return float("inf")
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return value_float if math.isfinite(value_float) else float("inf")


def _bool_penalty(value: Any) -> int:
    return 1 if value is True else 0


def _fit_rank(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _finite_or_inf(entry.get("reduced_chi2_ICA")),
        -float(entry.get("variance_explained_ICA", float("-inf"))),
        -float(entry.get("chi2_gain_ICA_vs_PCA_pct", float("-inf"))),
        _bool_penalty(entry.get("ARD_too_many_components")),
        _finite_or_inf(entry.get("ARD_ratio")),
        entry["path"],
    )


def _family_label(entry: dict[str, Any]) -> str:
    return " | ".join(
        [
            f"centering={entry.get('centering_type')}/{entry.get('centering_function')}",
            f"pca={entry.get('PCA_fit_method')}",
            f"init={entry.get('ICA_net_init')}",
        ]
    )


def _best_by_n_components(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(int(entry["n_components"]), []).append(entry)
    winners = [min(group, key=_fit_rank) for _, group in sorted(grouped.items())]
    return winners


def _checks_lookup(checks: list[dict[str, Any]]) -> dict[tuple[str | None, int, int], dict[str, Any]]:
    lookup: dict[tuple[str | None, int, int], dict[str, Any]] = {}
    for check in checks:
        key = (
            check.get("config_signature"),
            int(check["from_n_components"]),
            int(check["to_n_components"]),
        )
        lookup[key] = check
    return lookup


def _family_recommendations(entries: list[dict[str, Any]], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str | None, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(entry.get("config_signature"), []).append(entry)

    check_lookup = _checks_lookup(checks)
    recommendations: list[dict[str, Any]] = []
    for signature, family_entries in grouped.items():
        ordered = sorted(family_entries, key=lambda item: (int(item["n_components"]), item["path"]))
        preferred = ordered[0]
        decisions: list[dict[str, Any]] = []
        stopped = False
        for current, nxt in zip(ordered, ordered[1:]):
            key = (signature, int(current["n_components"]), int(nxt["n_components"]))
            check = check_lookup.get(key)
            significant = False
            reason_parts: list[str] = []
            if check is None:
                reason_parts.append("model-order check missing")
            else:
                f_comp = _finite_or_inf(check.get("F_comp_ICA"))
                f_crit = _finite_or_inf(check.get("F_crit_ICA_95"))
                significant = math.isfinite(f_comp) and math.isfinite(f_crit) and f_comp > f_crit
                reason_parts.append(
                    f"F_ICA={check.get('F_comp_ICA'):.4g} vs Fcrit={check.get('F_crit_ICA_95'):.4g}"
                    if math.isfinite(f_comp) and math.isfinite(f_crit)
                    else "F-test not available"
                )
            ard_stop = bool(nxt.get("ARD_too_many_components"))
            if significant:
                reason_parts.append("F-test significant")
            else:
                reason_parts.append("F-test not significant")
            if ard_stop:
                reason_parts.append("ARD suggests too many components")

            accepted = significant and not ard_stop
            if accepted:
                preferred = nxt
            else:
                stopped = True
            decisions.append(
                {
                    "from_n_components": int(current["n_components"]),
                    "to_n_components": int(nxt["n_components"]),
                    "accepted": accepted,
                    "reason": "; ".join(reason_parts),
                    "from_path": current["path"],
                    "to_path": nxt["path"],
                }
            )
            if stopped:
                break

        recommendations.append(
            {
                "family_label": _family_label(ordered[0]),
                "config_signature": signature,
                "available_n_components": [int(item["n_components"]) for item in ordered],
                "preferred_n_components": int(preferred["n_components"]),
                "preferred_path": preferred["path"],
                "preferred_label": preferred["label"],
                "decisions": decisions,
            }
        )
    recommendations.sort(key=lambda item: (item["preferred_n_components"], item["family_label"]))
    return recommendations


def _recommended_order(
    best_by_n: list[dict[str, Any]],
    families: list[dict[str, Any]],
) -> tuple[int, dict[str, Any], dict[int, int]]:
    vote_counter = Counter(int(item["preferred_n_components"]) for item in families)
    best_lookup = {int(entry["n_components"]): entry for entry in best_by_n}
    if vote_counter:
        top_votes = max(vote_counter.values())
        candidates = [n_components for n_components, votes in vote_counter.items() if votes == top_votes]
        candidates.sort(key=lambda n_components: (_fit_rank(best_lookup[n_components]), n_components))
        recommended_n = candidates[0]
    else:
        best_entry = min(best_by_n, key=_fit_rank)
        recommended_n = int(best_entry["n_components"])
    return recommended_n, best_lookup[recommended_n], dict(sorted(vote_counter.items()))


def _render_markdown(
    best_by_n: list[dict[str, Any]],
    families: list[dict[str, Any]],
    recommended_n: int,
    recommended_run: dict[str, Any],
    votes: dict[int, int],
) -> str:
    lines = [
        "# ICAIM Batch Selection Report",
        "",
        "## Recommendation",
        "",
        f"- Recommended `n_components`: **{recommended_n}**",
        f"- Recommended run: `{recommended_run['label']}`",
        f"- Path: `{recommended_run['path']}`",
        f"- `reduced_chi2_ICA={recommended_run['reduced_chi2_ICA']:.6g}`, `variance_explained_ICA={recommended_run['variance_explained_ICA']:.6g}`, `ARD_ratio={recommended_run['ARD_ratio']:.6g}`",
    ]
    reproduce_script = Path(recommended_run["path"]).resolve().parent / "reproduce_run.sh"
    if reproduce_script.exists():
        lines.append(f"- Reproduce script: `{reproduce_script}`")
    if votes:
        vote_summary = ", ".join(f"{n}:{count}" for n, count in votes.items())
        lines.append(f"- Family votes: `{vote_summary}`")

    lines.extend(
        [
            "",
            "## Best Run For Each Component Count",
            "",
        ]
    )
    for entry in best_by_n:
        lines.append(
            f"- `n={entry['n_components']}`: `{entry['label']}` | `reduced_chi2_ICA={entry['reduced_chi2_ICA']:.6g}` | `variance_explained_ICA={entry['variance_explained_ICA']:.6g}` | `ARD_ratio={entry['ARD_ratio']:.6g}` | `{entry['path']}`"
        )

    lines.extend(
        [
            "",
            "## Family Recommendations",
            "",
        ]
    )
    for family in families:
        lines.append(
            f"- `{family['family_label']}` -> preferred `n={family['preferred_n_components']}` | `{family['preferred_path']}`"
        )
        for decision in family["decisions"]:
            lines.append(
                f"  transition `{decision['from_n_components']}->{decision['to_n_components']}`: {'accepted' if decision['accepted'] else 'stopped'} ({decision['reason']})"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best ICAIM decomposition runs from one or more result files/directories using fit metrics, F-tests, and ARD diagnostics."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Result files or directories. Directories are scanned recursively for all_python.npz.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        type=Path,
        help="Optional JSON report file.",
    )
    parser.add_argument(
        "--output-markdown",
        default=None,
        type=Path,
        help="Optional Markdown report file.",
    )
    args = parser.parse_args()

    files = resolve_result_files(args.inputs)
    if not files:
        raise ValueError("No result files found.")

    loaded = [(path, load_results_file(path)) for path in files]
    entries = [_quality_entry(path, results) for path, results in loaded]
    entries.sort(key=lambda item: (int(item["n_components"]), item["path"]))
    checks = model_order_checks(entries)
    best_by_n = _best_by_n_components(entries)
    families = _family_recommendations(entries, checks)
    recommended_n, recommended_run, votes = _recommended_order(best_by_n, families)

    print("Best run per n_components")
    for entry in best_by_n:
        print(
            " | ".join(
                [
                    f"n={entry['n_components']}",
                    f"label={entry['label']}",
                    f"redchi2_ICA={entry['reduced_chi2_ICA']:.6g}",
                    f"varExp_ICA={entry['variance_explained_ICA']:.6g}",
                    f"ARD_ratio={entry['ARD_ratio']:.6g}",
                    f"path={entry['path']}",
                ]
            )
        )

    print()
    print("Family recommendations")
    for family in families:
        print(
            " | ".join(
                [
                    family["family_label"],
                    f"preferred_n={family['preferred_n_components']}",
                    f"path={family['preferred_path']}",
                ]
            )
        )
        for decision in family["decisions"]:
            print(
                f"  {decision['from_n_components']}->{decision['to_n_components']}: "
                f"{'accepted' if decision['accepted'] else 'stopped'} ({decision['reason']})"
            )

    print()
    print("Recommended")
    print(
        " | ".join(
            [
                f"n={recommended_n}",
                f"label={recommended_run['label']}",
                f"redchi2_ICA={recommended_run['reduced_chi2_ICA']:.6g}",
                f"varExp_ICA={recommended_run['variance_explained_ICA']:.6g}",
                f"ARD_ratio={recommended_run['ARD_ratio']:.6g}",
                f"path={recommended_run['path']}",
            ]
        )
    )

    report = {
        "files": [str(path) for path in files],
        "best_by_n_components": best_by_n,
        "family_recommendations": families,
        "recommended_n_components": recommended_n,
        "recommended_run": recommended_run,
        "family_votes": votes,
        "model_order_checks": checks,
    }

    if args.output_json is not None:
        save_json(args.output_json, report)
    if args.output_markdown is not None:
        args.output_markdown.write_text(
            _render_markdown(best_by_n, families, recommended_n, recommended_run, votes)
        )


if __name__ == "__main__":
    main()
