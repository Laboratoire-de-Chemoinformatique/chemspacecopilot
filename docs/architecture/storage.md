# Storage System

Provides a **unified S3/local filesystem abstraction** with session-scoped storage.

**Location**: `src/cs_copilot/storage/`

## Usage

```python
from cs_copilot.storage import S3

# Relative paths are session-scoped: sessions/{SESSION_ID}/results.csv
S3.open("results.csv", "w")

# Get full S3 URL for a relative path
S3.path("results.csv")  # → s3://bucket/sessions/{SESSION_ID}/results.csv

# Absolute S3 URLs work directly
S3.open("s3://bucket/data.csv", "r")

# Local absolute paths work too
S3.open("/tmp/data.csv", "r")
```

## Key Features

- **Session ID**: Auto-generated (timestamp + UUID) or env-configured
- **Backend Toggle**: Local filesystem by default; S3/MinIO only when `USE_S3=true`
- **Configuration Fallbacks**: Supports multiple env var names (MINIO_ENDPOINT, S3_ENDPOINT_URL, etc.)

## Integration Pattern

All file I/O should use the storage abstraction:

```python
from cs_copilot.storage import S3

# ✅ Good — session-scoped, works with S3 or local
with S3.open("output.csv", "w") as f:
    df.to_csv(f)

# ❌ Bad — hardcoded local path
with open("/tmp/output.csv", "w") as f:
    df.to_csv(f)
```

## Incomplete multipart upload lifecycle

Invocation cleanup removes staged objects after ordinary failures, but a
process or host can terminate while an S3 multipart upload is still in
progress. In that case, uploaded parts continue to consume storage until the
upload is completed or explicitly aborted. Object-expiration rules do not
replace multipart-upload cleanup.

Configure every production S3 or MinIO bucket used by ChemSpace Copilot with an
`AbortIncompleteMultipartUpload` lifecycle action. One day after initiation is
the recommended default:

```json
{
  "Rules": [
    {
      "ID": "abort-incomplete-multipart-uploads",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "sessions/"
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 1
      }
    }
  ]
}
```

Adjust the prefix if the deployment uses a different storage layout. Keep the
rule even when application-level cleanup is enabled: it is the recovery path
for termination that application code cannot observe. See the
[AWS S3 lifecycle documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpu-abort-incomplete-mpu-lifecycle-config.html)
for the action semantics and provider-specific deployment examples.
