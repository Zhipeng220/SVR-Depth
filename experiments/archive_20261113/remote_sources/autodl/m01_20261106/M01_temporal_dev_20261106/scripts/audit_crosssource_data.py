#!/usr/bin/env python3
"""Validate canonical manifests for the proposed five-source P_multi training pool."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svr_depth.datasets.crosssource import audit_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Cross-source JSON config")
    parser.add_argument("--metadata-only", action="store_true", help="Validate paths/metadata without decoding images")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    result = audit_config(args.config, decode_samples=not args.metadata_only)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
