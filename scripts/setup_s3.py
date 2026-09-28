#!/usr/bin/env python
# coding: utf-8
"""
Setup script for S3/MinIO integration
Description: Interactive setup script to configure S3 integration
"""

from pathlib import Path


def get_user_input(prompt: str, default: str = "") -> str:
    """Get user input with a default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def create_env_file():
    """Create a .env file with S3 configuration."""
    print("S3/MinIO Integration Setup")
    print("=" * 50)
    print()

    # Get configuration from user
    print("Please provide your S3/MinIO configuration:")
    print()

    endpoint_url = get_user_input("S3/MinIO endpoint URL", "http://localhost:9000")

    access_key = get_user_input("Access key", "cs_copilot")

    secret_key = get_user_input("Secret key", "chempwd123")

    bucket_name = get_user_input("Bucket name", "chatbot-assets")

    session_id = get_user_input("Session ID (leave empty for auto-generated)", "")

    use_s3 = get_user_input("Use S3 (true/false)", "true")

    # Create .env file content
    env_content = f"""# S3/MinIO Configuration
MINIO_ENDPOINT={endpoint_url}
MINIO_ACCESS_KEY={access_key}
MINIO_SECRET_KEY={secret_key}
ASSETS_BUCKET={bucket_name}
SESSION_ID={session_id if session_id else ""}
USE_S3={use_s3}

# Optional: Logging
LOG_LEVEL=INFO
"""

    # Write to .env file
    env_file = Path(".env")
    with open(env_file, "w") as f:
        f.write(env_content)

    print(f"\nConfiguration saved to {env_file}")
    return env_file


def test_connection():
    """Test the S3 connection."""
    print("\nTesting S3 connection...")

    try:
        import os

        from cs_copilot.storage import S3

        # Check if S3 is enabled
        use_s3 = os.getenv("USE_S3", "true").lower() == "true"
        if not use_s3:
            print("⚠️  S3 is disabled. Check your configuration.")
            return False

        # Test basic operations
        print("✅ S3 client initialized successfully")

        # Test bucket access by trying to write a test file
        try:
            import pandas as pd

            test_df = pd.DataFrame({"test": [1, 2, 3]})

            with S3.open("test_connection.csv", "w") as f:
                test_df.to_csv(f, index=False)
            print("✅ Bucket write access successful")

            # Clean up test file
            try:
                import fsspec

                from cs_copilot.storage import get_s3_config

                config = get_s3_config()
                fs = fsspec.filesystem("s3", **config.to_storage_options())
                fs.rm(f"s3://{config.bucket_name}/{S3.prefix}/test_connection.csv")
            except Exception:
                pass  # Ignore cleanup errors

        except Exception as e:
            print(f"⚠️  Bucket access failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def create_example_data():
    """Create example data to test the integration."""
    print("\nCreating example data...")

    try:
        import numpy as np
        import pandas as pd

        from cs_copilot.storage import S3

        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "id": range(1, 11),
                "name": [f"Compound_{i}" for i in range(1, 11)],
                "smiles": ["CCO", "CCCO", "CCCC", "CCCCC", "CCCCCC"] * 2,
                "activity": np.random.randn(10),
            }
        )

        # Save to S3
        with S3.open("data/example_compounds.csv", "w") as f:
            df.to_csv(f, index=False)
        print(f"✅ Example data saved to: {S3.path('data/example_compounds.csv')}")

        # Test reading back
        with S3.open("data/example_compounds.csv", "r") as f:
            loaded_df = pd.read_csv(f)
        print(f"✅ Example data loaded successfully. Shape: {loaded_df.shape}")

        return True

    except Exception as e:
        print(f"❌ Failed to create example data: {e}")
        return False


def main():
    """Main setup function."""
    print("Cs_copilot S3 Integration Setup")
    print("=" * 50)
    print()

    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        overwrite = get_user_input(".env file already exists. Overwrite? (y/n)", "n")
        if overwrite.lower() != "y":
            print("Setup cancelled.")
            return

    # Create configuration
    env_file = create_env_file()

    # Load environment variables
    from dotenv import load_dotenv

    load_dotenv(env_file)

    # Test connection
    if test_connection():
        print("\n✅ S3 integration setup successful!")

        # Create example data
        if create_example_data():
            print("\n✅ Example data created successfully!")

        print("\nNext steps:")
        print("1. Start your MinIO server (if using MinIO)")
        print("   docker build -f Dockerfile.minio --target server -t cs_copilot-minio:local .")
        print("   docker run -d --name minio \\")
        print("     -p 9000:9000 -p 9001:9001 \\")
        print("     -v /mnt/data:/data \\")
        print("     -e MINIO_ROOT_USER=minioadmin \\")
        print("     -e MINIO_ROOT_PASSWORD=minioadmin \\")
        print('     cs_copilot-minio:local server /data --console-address ":9001"')
        print("2. Access MinIO console at: http://localhost:9001")
        print("3. Run: python test_s3_integration.py")
        print("4. Check the documentation: docs/S3_INTEGRATION.md")

    else:
        print("\n❌ S3 integration setup failed.")
        print("Please check your configuration and try again.")
        print("You can edit the .env file manually and run this script again.")


if __name__ == "__main__":
    main()
