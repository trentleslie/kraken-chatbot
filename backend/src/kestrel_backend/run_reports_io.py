"""Durable, fail-safe artifact I/O for the per-node performance report.

Surface-agnostic: this module owns disk emission only (atomic write, owner-only
permissions, retention pruning, git SHA). It raises normally — the fail-safe
wrapper that guarantees a reporting failure can never break a user-facing run
lives in the reporting node (plan Unit 5). Retention pruning, however, is
swallowed inside :func:`write_report` so a prune failure never blocks a write.

Security (plan Unit 3): artifacts contain the raw user query and cost data.
The directory is created mode 700 without a world-readable window, and each file
is created mode 600 *at creation* (``O_EXCL``), never world-readable even
transiently. Filenames use an opaque run-id, never a query slug.
"""

import functools
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RETENTION = 200
ENV_RETENTION = "REPORT_RETENTION_MAX"
# A live atomic write holds its .tmp for milliseconds; any .tmp older than this was
# orphaned by a hard abort (SIGKILL/OOM) mid-write and is safe to sweep.
_TMP_STALE_SECONDS = 300

# backend/run_reports/. This file is backend/src/kestrel_backend/run_reports_io.py,
# so parents[2] is the backend/ root.
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = _BACKEND_ROOT / "run_reports"


def new_run_id() -> str:
    """Opaque, non-reversible run identifier (used in filenames and the report)."""
    return uuid.uuid4().hex


def utc_timestamp() -> str:
    """UTC timestamp for filenames, matching the brown harness convention."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@functools.lru_cache(maxsize=1)
def get_git_sha() -> str:
    """Deployed commit SHA, computed once and memoized (not per-request).

    Catches only the expected failures and returns ``"unknown"`` without logging
    the repo path (avoids leaking an absolute filesystem path into logs).
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(_BACKEND_ROOT),
            timeout=5,  # never hang the event loop on a stalled/locked git (esp. GDrive-synced repo)
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("git_sha lookup failed; using 'unknown'")
        return "unknown"


def _ensure_dir(d: Path) -> None:
    """Create ``d`` mode 700, owner-only at every moment.

    No ``os.umask`` (a process-global side effect that races under concurrency):
    ``mkdir(mode=0o700)`` has no group/other bits for umask to *add*, so the dir is
    never world-readable even at creation, and the explicit ``chmod`` enforces 700
    on a dir that pre-existed with looser perms.
    """
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(d, 0o700)


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically, mode 600 at creation.

    Uses ``O_EXCL`` so the temp file is created mode 600 from the first byte
    (never world-readable pre-chmod), then ``os.replace`` for an atomic rename —
    a crash before the rename leaves no partial final file.
    """
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(str(tmp), str(path))
    except BaseException:
        # Clean up the temp on any failure (write or rename) so no partial file lingers.
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _resolve_retention(max_pairs: int | None) -> int:
    if max_pairs is not None:
        return max_pairs
    raw = os.environ.get(ENV_RETENTION)
    if raw:
        try:
            return int(raw)
        except ValueError:
            logger.warning(
                "invalid %s=%r; using default %d", ENV_RETENTION, raw, DEFAULT_RETENTION
            )
    return DEFAULT_RETENTION


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except FileNotFoundError:
        return 0.0  # concurrently deleted — sort it to the end


def _sweep_stale_temps(d: Path) -> None:
    """Delete orphaned ``*.tmp`` older than the stale threshold.

    A concurrent writer's temp is milliseconds old, so it is never swept; only temps
    left behind by a hard abort (SIGKILL/OOM between os.open and rename) are removed.
    """
    now = time.time()
    for tmp in d.glob("*.tmp"):
        try:
            if now - tmp.stat().st_mtime > _TMP_STALE_SECONDS:
                tmp.unlink()
        except FileNotFoundError:
            pass  # concurrently removed


def prune(out_dir: Path, *, max_pairs: int | None = None) -> int:
    """Keep the newest ``max_pairs`` report pairs by mtime; delete older ones.

    Best-effort under concurrent runs (``SDK_SEMAPHORE`` allows parallel writers):
    ignores in-flight ``*.tmp`` files and swallows ``FileNotFoundError`` from a
    concurrent pruner. Returns the number of pairs targeted for removal.
    """
    d = Path(out_dir)
    if not d.is_dir():
        return 0
    # Sweep orphaned *.tmp left by a hard abort mid-write (O_EXCL can't self-heal, and
    # the pair-cap loop below ignores *.tmp). Done regardless of the retention cap.
    _sweep_stale_temps(d)
    cap = _resolve_retention(max_pairs)
    if cap <= 0:
        return 0
    jsons = [p for p in d.glob("*.json") if not p.name.endswith(".tmp")]
    jsons.sort(key=_mtime, reverse=True)  # newest first
    stale_pairs = jsons[cap:]
    for stale in stale_pairs:
        for path in (stale, stale.with_suffix(".md")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass  # concurrent pruner already removed it
    return len(stale_pairs)


def write_report(
    report_json: dict,
    markdown: str,
    *,
    run_id: str | None = None,
    out_dir: Path | str | None = None,
    timestamp: str | None = None,
    retention: int | None = None,
) -> dict:
    """Write the JSON + Markdown report pair to disk and prune old reports.

    Returns ``{"json": Path, "md": Path, "run_id": str}``. May raise on a real
    I/O failure (disk full, permissions) — the caller (reporting node) wraps this
    so the run is never broken. Pruning is internally swallowed.
    """
    directory = Path(out_dir) if out_dir is not None else DEFAULT_DIR
    rid = run_id or new_run_id()
    ts = timestamp or utc_timestamp()
    _ensure_dir(directory)
    base = f"{ts}_{rid}"
    json_path = directory / f"{base}.json"
    md_path = directory / f"{base}.md"
    _atomic_write(json_path, json.dumps(report_json, indent=2, default=str))
    try:
        _atomic_write(md_path, markdown)
    except BaseException:
        # Keep the pair atomic: don't leave an orphaned .json (which holds the raw
        # query) on disk if the markdown half fails.
        try:
            os.unlink(json_path)
        except FileNotFoundError:
            pass
        raise
    try:
        prune(directory, max_pairs=retention)
    except Exception:  # noqa: BLE001 - prune must never block a successful write
        logger.warning("run_reports prune failed", exc_info=True)
    return {"json": json_path, "md": md_path, "run_id": rid}
