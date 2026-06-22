"""Unit 3: artifact I/O — atomic write, perms, retention, git_sha."""

import os
import stat
import subprocess

import pytest

from kestrel_backend import run_reports_io as io


def _mode(p) -> int:
    return stat.S_IMODE(os.stat(p).st_mode)


def test_write_report_produces_pair_with_owner_only_perms(tmp_path):
    result = io.write_report({"report_version": 1}, "# md", out_dir=tmp_path / "rr")
    assert result["json"].exists() and result["md"].exists()
    assert result["json"].suffix == ".json" and result["md"].suffix == ".md"
    assert _mode(result["json"]) == 0o600
    assert _mode(result["md"]) == 0o600
    assert _mode(tmp_path / "rr") == 0o700


def test_filename_uses_run_id_not_query(tmp_path):
    result = io.write_report(
        {"query": "secret patient query"}, "# md", out_dir=tmp_path, run_id="abc123"
    )
    assert "abc123" in result["json"].name
    assert "secret" not in result["json"].name


def test_temp_file_is_600_at_creation(tmp_path, monkeypatch):
    # Capture the temp file's mode at the moment of rename — it must be 600 the
    # whole time, never world-readable pre-chmod.
    captured = {}
    real_replace = os.replace

    def spy_replace(src, dst):
        captured["mode"] = stat.S_IMODE(os.stat(src).st_mode)
        return real_replace(src, dst)

    monkeypatch.setattr(io.os, "replace", spy_replace)
    io._atomic_write(tmp_path / "x.json", "data")
    assert captured["mode"] == 0o600


def test_atomic_write_leaves_no_tmp_on_success(tmp_path):
    io.write_report({"a": 1}, "md", out_dir=tmp_path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_failure_leaves_no_partial_final(tmp_path, monkeypatch):
    # If the write fails before rename, the final path must not exist.
    target = tmp_path / "fail.json"

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(io.os, "replace", boom)
    with pytest.raises(OSError):
        io._atomic_write(target, "data")
    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []  # temp cleaned up


def test_prune_keeps_newest_cap_pairs(tmp_path):
    # Create 205 pairs with strictly increasing mtimes; cap 200 keeps the newest 200.
    for i in range(205):
        base = tmp_path / f"r{i:03d}"
        j = base.with_suffix(".json")
        m = base.with_suffix(".md")
        j.write_text("{}")
        m.write_text("md")
        os.utime(j, (i, i))  # older index = older mtime
        os.utime(m, (i, i))
    removed = io.prune(tmp_path, max_pairs=200)
    assert removed == 5
    remaining = sorted(p.stem for p in tmp_path.glob("*.json"))
    assert len(remaining) == 200
    assert "r000" not in remaining  # 5 oldest gone
    assert "r004" not in remaining
    assert "r005" in remaining


def test_prune_ignores_fresh_tmp_files(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "inflight.json.tmp").write_text("{}")  # fresh => an in-flight write
    io.prune(tmp_path, max_pairs=200)
    assert (tmp_path / "inflight.json.tmp").exists()  # in-flight write untouched


def test_prune_sweeps_stale_tmp_files(tmp_path):
    # An orphaned .tmp (old mtime) left by a crashed write is swept; a fresh one is kept.
    stale = tmp_path / "orphan.json.tmp"
    stale.write_text("{}")
    os.utime(stale, (1, 1))  # ancient mtime
    fresh = tmp_path / "inflight.json.tmp"
    fresh.write_text("{}")
    io.prune(tmp_path, max_pairs=200)
    assert not stale.exists()  # orphan swept
    assert fresh.exists()  # live write preserved


def test_prune_sweeps_stale_tmp_even_when_retention_disabled(tmp_path):
    stale = tmp_path / "orphan.json.tmp"
    stale.write_text("{}")
    os.utime(stale, (1, 1))
    io.prune(tmp_path, max_pairs=0)  # cap disabled, but temps still swept
    assert not stale.exists()


def test_prune_tolerates_concurrent_deletion(tmp_path):
    # Two prune passes target the same stale files; the second swallows FileNotFoundError.
    for i in range(3):
        base = tmp_path / f"r{i}"
        base.with_suffix(".json").write_text("{}")
        base.with_suffix(".md").write_text("md")
        os.utime(base.with_suffix(".json"), (i, i))
        os.utime(base.with_suffix(".md"), (i, i))
    assert io.prune(tmp_path, max_pairs=1) == 2  # removes r0, r1
    # Second pass with same cap: nothing left to remove, no error.
    assert io.prune(tmp_path, max_pairs=1) == 0
    assert (tmp_path / "r2.json").exists()  # newest survives


def test_prune_respects_env_default(tmp_path, monkeypatch):
    monkeypatch.setenv(io.ENV_RETENTION, "1")
    for i in range(3):
        base = tmp_path / f"r{i}"
        base.with_suffix(".json").write_text("{}")
        os.utime(base.with_suffix(".json"), (i, i))
    assert io.prune(tmp_path) == 2  # env cap = 1


def test_prune_invalid_env_falls_back_to_default(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setenv(io.ENV_RETENTION, "not_a_number")
    with caplog.at_level(logging.WARNING):
        assert io._resolve_retention(None) == io.DEFAULT_RETENTION
    assert any("invalid" in rec.message.lower() for rec in caplog.records)


def test_write_report_unlinks_json_if_md_fails(tmp_path, monkeypatch):
    # The pair must be atomic: a failed .md write must not leave an orphaned .json
    # (which holds the raw query) behind.
    calls = {"n": 0}
    real_atomic = io._atomic_write

    def flaky(path, content):
        calls["n"] += 1
        if calls["n"] == 2:  # second call is the .md write
            raise OSError("md write failed")
        return real_atomic(path, content)

    monkeypatch.setattr(io, "_atomic_write", flaky)
    with pytest.raises(OSError):
        io.write_report({"query": "secret"}, "md", out_dir=tmp_path, run_id="r1")
    assert list(tmp_path.glob("*.json")) == []  # json half cleaned up
    assert list(tmp_path.glob("*.md")) == []


def test_write_report_prune_failure_does_not_block_write(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "prune", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    result = io.write_report({"a": 1}, "md", out_dir=tmp_path)
    assert result["json"].exists()  # report still persisted despite prune failure


def test_get_git_sha_returns_hex_in_repo():
    io.get_git_sha.cache_clear()
    sha = io.get_git_sha()
    # In this repo it's a 40-char hex; tolerate "unknown" if git is somehow absent.
    assert sha == "unknown" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


def test_get_git_sha_unknown_when_git_missing(tmp_path, monkeypatch, caplog):
    import logging

    io.get_git_sha.cache_clear()

    def no_git(*_a, **_k):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(io.subprocess, "run", no_git)
    with caplog.at_level(logging.WARNING):
        assert io.get_git_sha() == "unknown"
    io.get_git_sha.cache_clear()  # don't poison other tests
    # No absolute path leaked into the log message.
    assert all("/home/" not in rec.message for rec in caplog.records)
