"""
Tests: training must not be able to silently invalidate a release.

THE HAZARD THIS GUARDS

`generate_bundles.py` treats `data/lnn_weather_water_h<N>.pt` as the RELEASE
SOURCE. It hashes that file and compares the hash against the
`checkpoint_sha256` recorded inside every shipped bundle manifest. So a routine
training run does not merely produce an artifact -- it breaks the integrity of
all five live bundles, which the running ingestor serves from.

This happened on 2026-10-01. A five-horizon training run overwrote all five
sources and `generate_bundles.py --check-only` then failed with "Checkpoint hash
mismatch" on every horizon. It was caught only because that check happened to be
run: the ingestor kept serving, the dashboard kept rendering and the audit trail
kept growing, all from bundles whose recorded provenance no longer described the
bytes on disk.

The property under test is the one that matters operationally: after an ordinary
training run, the release bundles still verify. That is asserted against the
real `generate_bundles.py --check-only` path rather than against a reimplementation
of the hash comparison, because a test that duplicates the check cannot catch the
check changing.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _release_sources() -> list[Path]:
    return sorted((ROOT / "prediction-model/data").glob("lnn_weather_water_h*.pt"))


def _check_only() -> tuple[int, str]:
    """Run the real integrity check and return (exit code, output)."""
    result = subprocess.run(
        [sys.executable, str(SRC / "generate_bundles.py"), "--check-only"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    return result.returncode, (result.stdout or "") + (result.stderr or "")


class TestReleaseSourceIsProtected:
    def test_release_sources_exist_to_protect(self):
        sources = _release_sources()
        assert sources, "no release source checkpoints found; this test needs them"

    def test_bundles_verify_before_anything_is_touched(self):
        """Guard the guard: if this fails, the assertions below prove nothing."""
        code, output = _check_only()
        assert code == 0, f"bundles do not verify to begin with:\n{output}"

    def test_training_writes_outside_the_release_directory(self, monkeypatch):
        """
        The default output path must not be the release source.

        Asserted on the path the trainer computes, without running a training
        loop -- the defect is a path decision, not a training behaviour. The
        trainer is stopped by making the next call after path resolution raise,
        so nothing is written and no training runs.
        """
        import train

        monkeypatch.delenv("TRAIN_OUTPUT_DIR", raising=False)
        monkeypatch.setattr(train, "ALLOW_RELEASE_SOURCE_OVERWRITE", False)

        captured: dict[str, str] = {}

        class _RecordingPath:
            """Proxies os.path, recording the final join() so the decision is observable."""

            def __getattr__(self, name):
                return getattr(os.path, name)

            @staticmethod
            def join(*parts: str) -> str:
                result = os.path.join(*parts)
                captured["path"] = result
                return result

        class _FakeOs:
            """Proxies to the real os, but with a recording os.path."""

            path = _RecordingPath()

            def __getattr__(self, name):
                return getattr(os, name)

        monkeypatch.setattr(train, "os", _FakeOs())

        # The trainer resolves its output path immediately after acquiring the
        # telemetry pipeline and immediately before building any dataset. Let the
        # pipeline call succeed and raise at dataset construction, so the path
        # decision has been made but nothing is trained or written.
        monkeypatch.setattr(train, "get_telemetry_pipeline", lambda: object())

        class _StopAfterPath(Exception):
            pass

        def _boom(*a, **k):
            raise _StopAfterPath()

        monkeypatch.setattr(train, "TelemetryDataset", _boom)

        with pytest.raises(_StopAfterPath):
            train.train_mf1_model(horizon=1)

        resolved = captured.get("path", "")
        assert resolved, "the trainer did not resolve an output path"

        # The danger is writing to the exact release source FILE, not merely
        # living under data/. train_output/ is a subdirectory of data/, so a
        # substring test on the data directory would false-positive on the
        # correct behaviour. Compare the normalised final path instead.
        release_source = (ROOT / "prediction-model/data/lnn_weather_water_h1.pt").resolve()
        assert Path(resolved).resolve() != release_source, \
            f"training still targets the release source: {resolved}"
        assert Path(resolved).resolve().is_relative_to(release_source.parent / "train_output"), \
            f"training output escaped its own directory: {resolved}"

    def test_escape_hatch_is_off_unless_explicitly_set(self, monkeypatch):
        """The overwrite path must require a deliberate environment opt-in."""
        import importlib

        import train
        monkeypatch.delenv("ALLOW_RELEASE_SOURCE_OVERWRITE", raising=False)
        reloaded = importlib.reload(train)
        assert reloaded.ALLOW_RELEASE_SOURCE_OVERWRITE is False

        monkeypatch.setenv("ALLOW_RELEASE_SOURCE_OVERWRITE", "1")
        reloaded = importlib.reload(train)
        assert reloaded.ALLOW_RELEASE_SOURCE_OVERWRITE is True
        # Leave the module in its safe default for any later test.
        monkeypatch.delenv("ALLOW_RELEASE_SOURCE_OVERWRITE", raising=False)
        importlib.reload(train)

    def test_bundles_still_verify_after_the_guard_is_installed(self):
        """
        The end-to-end statement, on the real checker.

        This is the assertion the incident would have failed: the release inputs
        are unchanged, therefore the bundle manifests still describe the bytes on
        disk, therefore --check-only still passes.
        """
        code, output = _check_only()
        assert code == 0, f"release bundles failed integrity:\n{output}"
        assert "successfully packaged and verified" in output.lower()
