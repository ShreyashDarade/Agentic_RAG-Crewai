#!/usr/bin/env python3
"""Download a BEIR dataset (default: scifact) into .dev/beir/<name>. The data is licensed for non-commercial use: it is never committed."""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def main(name: str = "scifact") -> int:
    target = ROOT / ".dev" / "beir"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / f"{name}.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL.format(name=name), archive)  # noqa: S310 - fixed https URL
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    print(target / name)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
