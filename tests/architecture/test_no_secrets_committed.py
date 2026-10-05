"""Security: no credential-shaped string is committed (a repo-level check, not a substitute for rotating a leaked key)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PATTERNS = {
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{32,}"),
    "AWS access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "bearer JWT": re.compile(r"\bBearer\s+eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\."),
    "assigned long secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"'][A-Za-z0-9/+_-]{32,}[\"']"
    ),
}


def _tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return [ROOT / name for name in out.stdout.decode().split("\0") if name]


def test_no_tracked_file_contains_a_credential_shaped_string() -> None:
    hits: list[str] = []
    for path in _tracked_files():
        if not path.is_file() or path.suffix in {".lock", ".png", ".jpg", ".db", ".whl", ".zip"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                hits.append(f"{path.relative_to(ROOT)}: {label}")
    assert not hits, "credential-shaped strings are committed:\n" + "\n".join(hits)


def test_the_environment_file_is_not_tracked() -> None:
    names = {p.relative_to(ROOT).as_posix() for p in _tracked_files()}
    assert "config/.env" not in names and ".env" not in names
    assert ".env.example" in names


def test_the_scanner_actually_matches_what_it_claims_to() -> None:
    assert PATTERNS["OpenAI-style key"].search("key = sk-" + "a" * 40)
    assert PATTERNS["AWS access key id"].search("AKIA" + "A" * 16)
    assert PATTERNS["assigned long secret"].search('api_key = "' + "x" * 40 + '"')
    assert not PATTERNS["OpenAI-style key"].search("sk-test-0123456789")
