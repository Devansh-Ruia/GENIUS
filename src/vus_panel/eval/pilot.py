"""Prepare and execute the frozen Findlay/Evo 2 timing pilot.

Usage: python -m vus_panel.eval.pilot prepare --download
       python -m vus_panel.eval.pilot run --selection data/splits/pilot20.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vus_panel import config
from vus_panel.eval.pilot_data import prepare


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible Findlay/Evo 2 calibration pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare", help="validate pinned data and create or verify the immutable selection")
    prep.add_argument("--download", action="store_true", help="download missing publisher/reference files")
    run = subparsers.add_parser("run", help="score each frozen ID once, retaining every outcome")
    run.add_argument("--selection", type=Path, default=config.splits_dir() / "pilot20.json")
    run.add_argument("--backend", help="defaults to EVO2_BACKEND_URL")
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--billing-rate-usd-per-hour", type=float, default=None,
                     help="actual instance billing rate, if known; not an assumed rate")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            selection, variants = prepare(args.download)
            print(f"Validated {len(variants)} variants; frozen {selection['n_selected']} pilot IDs at {config.splits_dir() / 'pilot20.json'}")
            return 0
        from vus_panel.eval.pilot_run import run_pilot
        directory, complete = run_pilot(args.selection, backend_url=args.backend, timeout=args.timeout,
                                       billing_rate=args.billing_rate_usd_per_hour)
        print(f"Pilot artifacts: {directory}")
        return 0 if complete else 1
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
