"""Offline provenance and crash-aware execution ledger for manuscript studies."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

TERMINAL = {"completed", "error", "timeout", "interrupted"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Required local file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(256)
        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Git LFS pointer is not a scientific asset: {path}")
        if not head:
            raise ValueError(f"Required local file is empty: {path}")
        digest.update(head)
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def software_identity() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)
    versions, distributions = {}, {}
    packages = (
        "rdkit",
        "numpy",
        "torch",
        "deepchemography",
        "SynPlanner",
        "CGRtools",
        "cgrtools-stable",
        "chython",
        "chython-synplan",
        "pytorch-lightning",
        "torch-geometric",
        "ray",
        "mini-racer",
    )
    for package in packages:
        try:
            distribution = importlib.metadata.distribution(package)
            versions[package] = distribution.version
            identity = {
                "version": distribution.version,
                "location": str(distribution.locate_file("")),
            }
            direct = distribution.read_text("direct_url.json")
            if direct:
                source = json.loads(direct)
                parsed = urlsplit(source.get("url", ""))
                # Keep provenance, never URL credentials, access tokens, or query strings.
                host = parsed.hostname or ""
                if parsed.port:
                    host += f":{parsed.port}"
                identity["source_url"] = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
                if source.get("vcs_info"):
                    identity["vcs"] = source["vcs_info"].get("vcs")
                    identity["commit_id"] = source["vcs_info"].get("commit_id")
            distributions[package] = identity
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    module_origins = {}
    for module in ("synplan", "CGRtools", "chython", "pytorch_lightning", "torch_geometric", "ray"):
        active = sys.modules.get(module)
        module_spec = getattr(active, "__spec__", None) or importlib.util.find_spec(module)
        module_origins[module] = getattr(active, "__file__", None) or getattr(
            module_spec, "origin", None
        )
    unmet = []
    requirements = importlib.metadata.requires("SynPlanner") if versions["SynPlanner"] else []
    from packaging.requirements import Requirement

    for text in requirements or []:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None or not requirement.specifier.contains(installed, prereleases=True):
            unmet.append({"requirement": text, "installed_distribution_version": installed})
    return {
        "git_commit": commit,
        "working_tree_dirty": bool(dirty),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "packages": versions,
        "distributions": distributions,
        "imported_module_origins": module_origins,
        "synplanner_declared_requirements": requirements,
        "synplanner_unmet_named_requirements": unmet,
    }


def apply_seed(seed: int) -> dict[str, Any]:
    """Apply seeds immediately before the stochastic operation, not just record them."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "python_random": True,
        "numpy_random": True,
        "torch": True,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cuda_seeded": torch.cuda.is_available(),
        "limitations": "Determinism is scoped to these RNGs and the recorded software/hardware; external backend RNGs may differ.",
    }


def _alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def recover_interrupted(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("hostname") != platform.node():
        raise ValueError("Cannot verify a running job on another host; inspect it before resuming")
    if any(_alive(record.get(key)) for key in ("controller_pid", "worker_pid")):
        raise ValueError(
            "A recorded study controller or worker is still alive; refusing concurrent resume"
        )
    return {
        **record,
        "status": "interrupted",
        "finished_at": now(),
        "error": "Previous process exited without a terminal record; retained without retry",
    }


def _stop_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


@contextmanager
def _controller_lock(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".controller.lock"
    try:
        handle = lock_path.open("x")
    except FileExistsError:
        owner = json.loads(lock_path.read_text())
        if owner.get("hostname") != platform.node() or _alive(owner.get("pid")):
            raise ValueError(
                "Study controller lock is owned by a live or unverifiable process"
            ) from None
        lock_path.unlink()
        handle = lock_path.open("x")
    with handle:
        json.dump({"hostname": platform.node(), "pid": os.getpid()}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _verify_artifacts(record: dict[str, Any]) -> None:
    for asset in record.get("result", {}).get("artifacts", {}).values():
        if file_identity(Path(asset["path"])) != asset:
            raise ValueError("A completed study artifact changed; refusing silent reuse")


def run_jobs(
    specification: dict[str, Any],
    output_dir: Path,
    *,
    worker_script: Path,
    resume: bool = False,
    prepare_only: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    with _controller_lock(output_dir):
        return _run_jobs(
            specification,
            output_dir,
            worker_script=worker_script,
            resume=resume,
            prepare_only=prepare_only,
        )


def _run_jobs(
    specification: dict[str, Any],
    output_dir: Path,
    *,
    worker_script: Path,
    resume: bool = False,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """Resume only never-started jobs; a failed attempt is never silently rerun."""
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Study already exists; use --resume to continue never-started cases")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("specification") != specification:
            raise ValueError(
                "Resume configuration/provenance differs from the frozen study manifest"
            )
    else:
        if resume:
            raise ValueError("Cannot resume a study without its manifest")
        if output_dir.exists() and any(p.name != ".controller.lock" for p in output_dir.iterdir()):
            raise ValueError("Use a new empty output directory for a study")
        manifest = {
            "schema": "chemspacecopilot.scientific_study.v1",
            "created_at": now(),
            "specification": specification,
        }
        write_json(manifest_path, manifest)
    jobs = specification["jobs"]
    for job in jobs:
        case_dir = output_dir / job["id"]
        record_path = case_dir / "execution.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("status") == "running":
                record = recover_interrupted(record)
                write_json(record_path, record)
            if record.get("status") == "completed":
                _verify_artifacts(record)
            if record.get("status") not in TERMINAL:
                raise ValueError(f"Unrecognized execution status: {record_path}")
            continue
        if prepare_only:
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        worker_spec = {"specification": specification, "job": job, "output_dir": str(case_dir)}
        write_json(case_dir / "worker_spec.json", worker_spec)
        record = {
            "id": job["id"],
            "seed": job["seed"],
            "status": "running",
            "started_at": now(),
            "hostname": platform.node(),
            "controller_pid": os.getpid(),
            "worker_pid": None,
        }
        write_json(record_path, record)
        start = time.monotonic()
        env = dict(os.environ)
        env.update(
            {
                "USE_S3": "false",
                "SESSION_ID": "study_" + job["id"],
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONHASHSEED": str(job["seed"]),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        # Keep the current interpreter's explicit dependency/PYTHONPATH setup, but
        # make its entries absolute before changing each worker's working directory.
        env["PYTHONPATH"] = os.pathsep.join(
            str(Path(p or ".").resolve()) for p in env.get("PYTHONPATH", "").split(os.pathsep)
        )
        process = None
        try:
            with (
                (case_dir / "stdout.log").open("w") as stdout,
                (case_dir / "stderr.log").open("w") as stderr,
            ):
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(worker_script.resolve()),
                        "--worker-spec",
                        str(case_dir / "worker_spec.json"),
                    ],
                    cwd=case_dir,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=os.name == "posix",
                )
                record["worker_pid"] = process.pid
                write_json(record_path, record)
                try:
                    returncode = process.wait(timeout=specification["case_timeout_seconds"])
                    result_path = case_dir / "result.json"
                    result = json.loads(result_path.read_text()) if result_path.exists() else None
                    if isinstance(result, dict):
                        record["result"] = result
                    if returncode != 0 or not isinstance(result, dict):
                        raise RuntimeError(
                            f"Worker exited {returncode}; inspect stderr.log/result.json"
                        )
                    record.update({"status": "completed", "result": result})
                    _verify_artifacts(record)
                except subprocess.TimeoutExpired:
                    _stop_worker(process)
                    record.update(
                        {"status": "timeout", "error": "Per-case wall-clock limit exceeded"}
                    )
        except KeyboardInterrupt:
            if process is not None:
                _stop_worker(process)
            record.update({"status": "interrupted", "error": "Controller interrupted"})
            raise
        except Exception as exc:
            if process is not None:
                _stop_worker(process)
            record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            record.update({"finished_at": now(), "wall_seconds": time.monotonic() - start})
            write_json(record_path, record)
    records = []
    for job in jobs:
        path = output_dir / job["id"] / "execution.json"
        records.append(
            json.loads(path.read_text())
            if path.exists()
            else {"id": job["id"], "status": "not_started"}
        )
    summary = {
        "planned_case_count": len(jobs),
        "attempted_case_count": sum(r["status"] != "not_started" for r in records),
        "status_counts": {
            status: sum(r["status"] == status for r in records)
            for status in ["not_started", *sorted(TERMINAL)]
        },
        "cases": records,
        "interpretation": "Every predeclared case remains in this denominator. Completed means the worker reported an outcome, not scientific success. Failed/interrupted cases are never retried on resume.",
    }
    write_json(output_dir / "study_summary.json", summary)
    return summary
