#!/usr/bin/env python3
"""
Test script to verify S3 integration is working correctly
"""

import sys

import pandas as pd
from dotenv import load_dotenv

from cs_copilot.storage import S3


def test_s3_operations():
    """Test basic S3 operations"""
    load_dotenv()
    print("Testing S3 integration...")

    try:
        # Test 1: Write a simple CSV file
        print("1. Testing CSV write operation...")
        test_df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["test1", "test2", "test3"], "value": [10.5, 20.3, 30.1]}
        )

        with S3.open("test_data.csv", "w") as f:
            test_df.to_csv(f, index=False)
        print("   ✅ CSV write successful")

        # Test 2: Read the CSV file back
        print("2. Testing CSV read operation...")
        with S3.open("test_data.csv", "r") as f:
            loaded_df = pd.read_csv(f)

        print(f"   ✅ CSV read successful. Shape: {loaded_df.shape}")
        print(f"   Data preview:\n{loaded_df.head()}")

        # Test 3: Test binary file operations (for GTM models)
        print("3. Testing binary file operations...")
        import pickle

        test_data = {"model": "test", "accuracy": 0.95}

        with S3.open("test_model.pkl", "wb") as f:
            pickle.dump(test_data, f)
        print("   ✅ Binary write successful")

        with S3.open("test_model.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        print(f"   ✅ Binary read successful. Data: {loaded_data}")

        # Test 4: Test gzipped file operations
        print("4. Testing gzipped file operations...")
        import gzip

        import dill

        test_gtm_data = {"gtm_model": "test_gtm", "nodes": 100}

        with S3.open("test_gtm.pkl.gz", "wb") as f:
            with gzip.open(f, "wb") as gz:
                dill.dump(test_gtm_data, gz)
        print("   ✅ Gzipped write successful")

        with S3.open("test_gtm.pkl.gz", "rb") as f:
            with gzip.open(f, "rb") as gz:
                loaded_gtm = dill.load(gz)
        print(f"   ✅ Gzipped read successful. Data: {loaded_gtm}")

        print("\n🎉 All S3 operations completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ S3 operations failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_s3_operations()
    sys.exit(0 if success else 1)
