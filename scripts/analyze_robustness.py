#!/usr/bin/env python
# coding: utf-8
"""
CLI script for analyzing robustness test results using the evaluator agent.

Usage:
    python scripts/analyze_robustness.py --test chembl_download --timestamp 20250122_120000
    python scripts/analyze_robustness.py --test chembl_interactivity --latest
    python scripts/analyze_robustness.py --list chembl_download
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from cs_copilot.agents import get_robustness_evaluator_agent
from cs_copilot.model_config import load_model_from_config
from cs_copilot.tools.analysis import RobustnessAnalysisToolkit
from cs_copilot.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze robustness test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze specific test run
  python scripts/analyze_robustness.py --test chembl_download --timestamp 20250122_120000

  # Analyze latest test run
  python scripts/analyze_robustness.py --test chembl_interactivity --latest

  # List available test runs
  python scripts/analyze_robustness.py --list chembl_download

  # Compare multiple test runs
  python scripts/analyze_robustness.py --test chembl_download --compare 20250122_100000 20250122_120000

  # Save report to file
  python scripts/analyze_robustness.py --test chembl_download --latest --output report.md
        """,
    )

    parser.add_argument(
        "--test",
        type=str,
        help="Test name (e.g., chembl_download, chembl_interactivity, gtm_optimization)",
    )

    parser.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp of the test run (format: YYYYMMDD_HHMMSS)",
    )

    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest test run",
    )

    parser.add_argument(
        "--list",
        type=str,
        metavar="TEST_NAME",
        help="List all available test runs for the given test",
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="TIMESTAMP",
        help="Compare multiple test runs (provide multiple timestamps)",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for the report (default: print to console)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json", "csv"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model to use (overrides .modelconf model_id)",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["deepseek", "ollama"],
        help="LLM provider (overrides .modelconf provider)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    # Handle --list command
    if args.list:
        logger.info(f"Listing available test runs for: {args.list}")
        toolkit = RobustnessAnalysisToolkit()
        runs = toolkit.list_available_test_runs(args.list)

        if not runs:
            logger.info(f"No test runs found for '{args.list}'")
            return 0

        logger.info(f"\nFound {len(runs)} test runs:")
        for i, ts in enumerate(runs, 1):
            logger.info(f"  {i}. {ts}")

        return 0

    # Validate required arguments for analysis
    if not args.test:
        parser.error("--test is required (or use --list to list available tests)")

    if not args.timestamp and not args.latest and not args.compare:
        parser.error("Either --timestamp, --latest, or --compare must be specified")

    # Create agent – CLI flags override .modelconf / env vars for this run
    import os

    if args.provider:
        os.environ["MODEL_PROVIDER"] = args.provider
    if args.model:
        os.environ["MODEL_ID"] = args.model
    model = load_model_from_config()
    logger.info(f"Creating robustness evaluator agent with model: {model.id}")
    agent = get_robustness_evaluator_agent(
        model=model,
        debug_mode=args.debug,
    )

    # Build prompt
    if args.compare:
        # Compare multiple runs
        timestamps_str = ", ".join(args.compare)
        prompt = f"Compare the {args.test} test runs from timestamps: {timestamps_str}"
    elif args.latest:
        # Analyze latest run
        prompt = f"Analyze the latest {args.test} test run"
    else:
        # Analyze specific run
        prompt = f"Analyze the {args.test} test from {args.timestamp}"

    # Add format specification
    if args.output:
        prompt += f". Export the report as {args.format} to '{args.output}'"

    # Run agent
    logger.info(f"Running analysis: {prompt}")
    logger.info("-" * 80)

    try:
        response = agent.run(prompt)
        content = response.content

        # Print or save output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(content)
            logger.info(f"\n✅ Report saved to: {output_path}")
        else:
            print("\n" + "=" * 80)
            print("ANALYSIS REPORT")
            print("=" * 80)
            print(content)
            print("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
