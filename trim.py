"""
trim_concerns.py
=================
Fixes Pydantic validation errors from `contextmatch build-calibration`:
    "List should have at most 5 items after validation"

The CalibrationReview model caps corrected_assessment.concerns at 5 items
(see models.py: concerns: list[str] = Field(..., max_length=5)). This
script truncates any over-long concerns lists in calibration/reviews.json
down to 5 items, keeping the first 5 (presumed most salient / most
specific) and dropping the rest.

Usage:
    python trim_concerns.py
    python trim_concerns.py --input calibration/reviews.json --max-items 5
"""

import argparse
import json
from pathlib import Path


def trim_concerns(reviews: list[dict], max_items: int = 5) -> tuple[list[dict], int]:
    trimmed_count = 0
    for review in reviews:
        ca = review.get("corrected_assessment")
        if not ca:
            continue
        concerns = ca.get("concerns")
        if isinstance(concerns, list) and len(concerns) > max_items:
            ca["concerns"] = concerns[:max_items]
            trimmed_count += 1
    return reviews, trimmed_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="calibration/reviews.json")
    ap.add_argument("--max-items", type=int, default=5)
    ap.add_argument("--in-place", action="store_true", default=True)
    args = ap.parse_args()

    path = Path(args.input)
    reviews = json.loads(path.read_text(encoding="utf-8"))

    reviews, trimmed_count = trim_concerns(reviews, max_items=args.max_items)

    path.write_text(json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Trimmed {trimmed_count} review(s) with >{args.max_items} concerns "
          f"down to {args.max_items} items.")
    print(f"Wrote updated file: {path}")


if __name__ == "__main__":
    main()
