#!/usr/bin/env python3
"""Exercise the installed SynPlanner with pinned public assets and no LLM calls."""

import argparse
import hashlib
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from cs_copilot.storage import S3, is_s3_enabled
from cs_copilot.tools.chemistry.synplanner_assets import (
    SYNPLANNER_REVISION,
    resolve_synplanner_assets,
)
from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerToolkit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/synplanner-smoke.json"))
    args = parser.parse_args()
    if is_s3_enabled():
        raise RuntimeError("Run this smoke with USE_S3=false")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    paths = resolve_synplanner_assets()
    S3.set_session_prefix("sessions/synplanner-integration-smoke")
    toolkit = SynPlannerToolkit(max_time=90, max_iterations=100, enable_retry_profiles=False)
    started = time.monotonic()
    result = toolkit.plan_synthesis("CC(=O)Oc1ccccc1C(=O)O")
    result = result["synthesis_report_data"]
    assert result["routes"], result.get("attempts")
    assert result["routes"][0]["num_steps"] > 0
    assert result["routes"][0]["score"] is not None
    assert result["visualizations"], "The native route depiction must work"
    for route in result["visualizations"]:
        for key in ("svg_path", "png_path"):
            assert route[key] and Path(route[key]).is_file(), (key, route)
    receipt = {
        "kind": "software_integration_smoke_not_manuscript_benchmark",
        "synplanner": version("SynPlanner"),
        "asset_revision": SYNPLANNER_REVISION,
        "asset_sha256": {
            key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()
        },
        "elapsed_seconds": time.monotonic() - started,
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(f"SynPlanner {receipt['synplanner']}: {len(result['routes'])} route(s), SVG/PNG verified")


if __name__ == "__main__":
    main()
