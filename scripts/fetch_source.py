#!/usr/bin/env python3
"""Download the pinned public source snapshot and verify it before use."""

import argparse
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_params, source_files  # noqa: E402

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_source(params_path: Path) -> Path:
    """Fetch the selected source only when its pinned checksum is absent."""
    params_path = params_path.resolve()
    params = load_params(str(params_path))
    files = source_files(params, require_exists=False)
    if len(files) != 1:
        raise SystemExit(
            "collect.sources for the selected version must contain one file"
        )

    source = params["collect"]["source"]
    expected = source["sha256"].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise SystemExit("collect.source.sha256 must be a 64-digit hexadecimal SHA256")
    target = params_path.parent / files[0]
    if target.name != source["file"]:
        raise SystemExit("collect.source.file does not match the selected source path")
    if target.is_file() and sha256_file(target) == expected:
        print(f"Source already verified: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as output:
            try:
                with urlopen(source["url"], timeout=60) as response:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        digest.update(chunk)
            except HTTPError as exc:
                raise SystemExit(f"Source download failed: HTTP {exc.code}") from None
            except URLError as exc:
                raise SystemExit(
                    f"Source download failed: {type(exc).__name__}"
                ) from None
            except Exception as exc:
                raise SystemExit(
                    f"Source download failed: {type(exc).__name__}"
                ) from None

        actual = digest.hexdigest()
        if actual != expected:
            raise SystemExit(
                f"Source SHA256 mismatch: expected {expected}, got {actual}"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Source downloaded and verified: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params", type=Path, default=Path("params.yaml"), help="params.yaml path"
    )
    args = parser.parse_args()
    fetch_source(args.params)


if __name__ == "__main__":
    main()
