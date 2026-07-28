#!/usr/bin/env python
"""Aggregate two or more blinded manuscript-review sheets."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

SCORE_FIELDS = (
    "artifact_grounded_factuality_0_2",
    "task_fulfillment_0_2",
    "uncertainty_handling_0_2",
    "unsupported_claims_0_2",
)


def _parse_score(value: Any, *, field: str, path: Path, row_number: int) -> int:
    try:
        score = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row_number}: {field} must be an integer from 0 to 2") from exc
    if score not in {0, 1, 2}:
        raise ValueError(f"{path}:{row_number}: {field} must be an integer from 0 to 2")
    return score


def load_reviews(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """Load completed sheets and validate IDs, coverage fields, and ordinal scores."""
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                review_id = str(row.get("review_id") or "").strip()
                reviewer_id = str(row.get("reviewer_id") or "").strip()
                if not review_id or not reviewer_id:
                    raise ValueError(f"{path}:{row_number}: review_id and reviewer_id are required")
                identity = (review_id, reviewer_id)
                if identity in seen:
                    raise ValueError(
                        f"{path}:{row_number}: duplicate review {review_id!r} "
                        f"from reviewer {reviewer_id!r}"
                    )
                seen.add(identity)
                rows.append(
                    {
                        "review_id": review_id,
                        "reviewer_id": reviewer_id,
                        **{
                            field: _parse_score(
                                row.get(field),
                                field=field,
                                path=path,
                                row_number=row_number,
                            )
                            for field in SCORE_FIELDS
                        },
                    }
                )
    return rows


def weighted_cohen_kappa(pairs: Iterable[Tuple[int, int]]) -> float | None:
    """Quadratic-weighted Cohen's kappa for the ordinal scale 0, 1, 2."""
    pairs_list = list(pairs)
    if not pairs_list:
        return None
    observed = sum(((left - right) / 2) ** 2 for left, right in pairs_list) / len(pairs_list)
    left_counts = [sum(left == score for left, _ in pairs_list) for score in range(3)]
    right_counts = [sum(right == score for _, right in pairs_list) for score in range(3)]
    expected = (
        sum(
            left_counts[left] * right_counts[right] * ((left - right) / 2) ** 2
            for left in range(3)
            for right in range(3)
        )
        / len(pairs_list) ** 2
    )
    return 1.0 if expected == 0 and observed == 0 else 1 - observed / expected


def aggregate_reviews(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize scores, coverage, exact agreement, and pairwise weighted kappa."""
    by_review: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_review[str(row["review_id"])].append(row)
    if not by_review:
        raise ValueError("No completed human reviews were provided")

    reviewer_ids = sorted({str(row["reviewer_id"]) for row in rows})
    under_reviewed = sorted(
        review_id for review_id, ratings in by_review.items() if len(ratings) < 2
    )
    dimensions: Dict[str, Dict[str, Any]] = {}
    for field in SCORE_FIELDS:
        scores = [int(row[field]) for row in rows]
        exactly_agreed = sum(
            len({int(row[field]) for row in ratings}) == 1
            for ratings in by_review.values()
            if len(ratings) >= 2
        )
        reviewed_by_multiple = sum(len(ratings) >= 2 for ratings in by_review.values())

        kappas = []
        for left_reviewer, right_reviewer in itertools.combinations(reviewer_ids, 2):
            pairs = []
            for ratings in by_review.values():
                indexed = {str(row["reviewer_id"]): int(row[field]) for row in ratings}
                if left_reviewer in indexed and right_reviewer in indexed:
                    pairs.append((indexed[left_reviewer], indexed[right_reviewer]))
            kappa = weighted_cohen_kappa(pairs)
            if kappa is not None:
                kappas.append(kappa)

        dimensions[field] = {
            "ratings": len(scores),
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "exact_agreement": (
                exactly_agreed / reviewed_by_multiple if reviewed_by_multiple else None
            ),
            "mean_pairwise_quadratic_weighted_kappa": (statistics.mean(kappas) if kappas else None),
        }

    return {
        "responses": len(by_review),
        "reviewers": reviewer_ids,
        "ratings": len(rows),
        "responses_with_at_least_two_reviews": sum(
            len(ratings) >= 2 for ratings in by_review.values()
        ),
        "under_reviewed_response_ids": under_reviewed,
        "dimensions": dimensions,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Blinded Human Review Summary",
        "",
        f"- Responses: {summary['responses']}",
        f"- Reviewers: {len(summary['reviewers'])}",
        (
            "- Responses with at least two reviews: "
            f"{summary['responses_with_at_least_two_reviews']}"
        ),
        "",
        "| Dimension | Mean | Median | Exact agreement | Weighted kappa |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, values in summary["dimensions"].items():
        agreement = values["exact_agreement"]
        kappa = values["mean_pairwise_quadratic_weighted_kappa"]
        agreement_text = f"{agreement:.1%}" if agreement is not None else "n/a"
        kappa_text = f"{kappa:.3f}" if kappa is not None else "n/a"
        lines.append(
            f"| {field} | {values['mean']:.3f} | {values['median']:.3f} | "
            f"{agreement_text} | {kappa_text} |"
        )
    if summary["under_reviewed_response_ids"]:
        lines.extend(
            [
                "",
                "Warning: some responses have fewer than two completed reviews; "
                "see the JSON output for their anonymized IDs.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheets", nargs="+", type=Path, help="Completed reviewer CSV files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = aggregate_reviews(load_reviews(args.sheets))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "human_review_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    (args.output_dir / "human_review_summary.md").write_text(_markdown(summary))


if __name__ == "__main__":
    main()
