#!/usr/bin/env python
# coding: utf-8
"""
Script to manually download GTM model from HuggingFace.
"""

import os
from pathlib import Path

# Add the src directory to the path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cs_copilot.tools.constants import HUGGINGFACE_GTM_REPO, DEFAULT_GTM_MODEL_PATH


def download_gtm_model(target_dir: str = None):
    """Download the default GTM model from HuggingFace.

    Args:
        target_dir: Directory to save the model (default: DEFAULT_GTM_MODEL_PATH)
    """
    print("=" * 70)
    print("GTM Model Download from HuggingFace")
    print("=" * 70)
    print()
    print(f"Repository: {HUGGINGFACE_GTM_REPO}")
    print()

    # Check if huggingface_hub is available
    print("Checking if huggingface_hub is installed...")
    try:
        from huggingface_hub import snapshot_download, get_token

        print("✓ huggingface_hub is installed")
    except ImportError:
        print("✗ huggingface_hub not found")
        print("Please install it with: pip install huggingface_hub")
        return False

    print()

    # Set up target directory
    if target_dir is None:
        target_dir = Path(DEFAULT_GTM_MODEL_PATH).expanduser()
    else:
        target_dir = Path(target_dir).expanduser()

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory: {target_dir}")
    print()

    # Get HuggingFace token (optional)
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        try:
            hf_token = get_token()
        except Exception:
            hf_token = None

    if hf_token:
        print("✓ Using HuggingFace token for authentication")
    else:
        print("No HuggingFace token found (public repos don't need one)")
    print()

    # Download the model
    print("Starting download (this may take a while)...")
    try:
        sd_kwargs = dict(
            repo_id=HUGGINGFACE_GTM_REPO,
            revision="main",
            local_dir=str(target_dir),
            allow_patterns=["*.pkl", "*.pkl.gz", "*.dill", "*.pt"],
        )
        if hf_token:
            sd_kwargs["token"] = hf_token

        # Disable progress bars for cleaner output
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        snapshot_download(**sd_kwargs)

        print()
        print("✓ Download completed successfully!")
        print(f"  Files saved to: {target_dir}")
        print()

        # List downloaded files
        print("Downloaded files:")
        for file in sorted(target_dir.glob("*")):
            if file.is_file():
                size = file.stat().st_size / (1024 * 1024)  # Size in MB
                print(f"  - {file.name} ({size:.2f} MB)")

        return True

    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download GTM model from HuggingFace")
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help=f"Directory to save the model (default: {DEFAULT_GTM_MODEL_PATH})",
    )

    args = parser.parse_args()

    success = download_gtm_model(args.target_dir)
    sys.exit(0 if success else 1)
