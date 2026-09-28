"""Offline checks for source bootstrap and checksum protection."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from src.config import load_params


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_source.py"


class FetchSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.fixture = self.root / "upstream.csv"
        self.content = b"Title,Overview\nExample,An example movie\n"
        self.fixture.write_bytes(self.content)
        self.target = self.root / "sources" / "9000plus.csv"
        self.params_path = self.root / "params.yaml"

    def write_params(self, sha256: str) -> None:
        params = load_params(str(ROOT / "params.yaml"))
        version = params["collect"]["version"]
        params["collect"]["sources"][version] = ["sources/9000plus.csv"]
        params["collect"]["source"].update(
            file="9000plus.csv", url=self.fixture.as_uri(), sha256=sha256
        )
        self.params_path.write_text(
            yaml.safe_dump(params, allow_unicode=True), encoding="utf-8"
        )

    def run_fetch(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--params", str(self.params_path)],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_download_and_repeat_without_network(self) -> None:
        self.write_params(hashlib.sha256(self.content).hexdigest())
        first = self.run_fetch()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(self.target.read_bytes(), self.content)

        self.fixture.unlink()
        repeated = self.run_fetch()
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(self.target.read_bytes(), self.content)

    def test_checksum_mismatch_preserves_existing_file(self) -> None:
        self.write_params("0" * 64)
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b"keep me")

        result = self.run_fetch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256 mismatch", result.stderr)
        self.assertNotIn(self.fixture.as_uri(), result.stderr)
        self.assertEqual(self.target.read_bytes(), b"keep me")
        self.assertEqual(list(self.target.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
