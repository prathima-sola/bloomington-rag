#!/usr/bin/env python3
"""Fail CI when a candidate retrieval result regresses from a stored baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--metric", default="hit_at_3")
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    expected = float(baseline["summary"]["overall"][args.metric])
    actual = float(candidate["summary"]["overall"][args.metric])
    minimum = expected - args.tolerance
    print(f"{args.metric}: candidate={actual:.4f} baseline={expected:.4f} minimum={minimum:.4f}")
    if actual < minimum:
        print("Retrieval regression detected")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
