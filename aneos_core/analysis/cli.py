"""Existing analysis launcher backend: one actual pipeline, JSON or Markdown."""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from .pipeline import create_analysis_pipeline
from .reporting import build_screening_report, render_screening_markdown


def main(argv=None, pipeline=None):
    parser = argparse.ArgumentParser(description='Screen a NEO using actual source evidence; no calibrated origin model is installed.')
    parser.add_argument('designation')
    parser.add_argument('--json', action='store_true', help='Write the versioned machine-readable report')
    parser.add_argument('--force-refresh', action='store_true', help='Bypass source cache')
    parser.add_argument('--output', type=Path, help='Also create a report file; existing files are never overwritten')
    args = parser.parse_args(argv)
    if not args.designation.strip():
        parser.error('designation must not be blank')
    try:
        if args.output and args.output.exists():
            raise FileExistsError(f'Report path already exists: {args.output}')
        result = asyncio.run((pipeline or create_analysis_pipeline()).analyze_neo(
            args.designation, force_refresh=args.force_refresh))
        report = build_screening_report(result)
        text = json.dumps(report, indent=2, allow_nan=False)+'\n' if args.json else render_screening_markdown(report)
        if args.output:
            with args.output.open('x', encoding='utf-8') as target:
                target.write(text)
        print(text, end='')
        return 1 if report['errors'] else 0
    except Exception as exc:
        print(f'Analysis/report unavailable: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
