"""Guard against committing secrets (hermetic; runs in CI on every PR).
Run: python tests/test_no_secrets.py
Scans every git-TRACKED file for key-shaped strings and checks that the ignore rules cover secret files.
GitHub secret scanning + push protection is the second layer; this one fails the build before a PR can merge.
"""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SHAPES = {
    "Groq key": r"gsk_[A-Za-z0-9]{20,}",
    "Langfuse public key": r"pk-lf-[0-9a-f-]{20,}",
    "Langfuse secret key": r"sk-lf-[0-9a-f-]{20,}",
    "database URL with a real password": r"postgres(?:ql)?://[^\s:@/]+:(?!<|\[|\$|\{)[^\s@/]{4,}@",
    "JWT": r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.",
    "AWS access key": r"AKIA[0-9A-Z]{16}",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "personal gmail address": r"[A-Za-z0-9._%+-]+@gmail\.com",
}
FORBIDDEN_NAMES = re.compile(r"(^|/)(\.env(\.(?!example$)[^/]+)?|secrets\.toml|id_rsa|[^/]+\.(pem|key|p12))$")
MUST_BE_IGNORED = [".env", ".env.local", ".env.production", ".streamlit/secrets.toml", "docs/", "reports/", "server.pem", "private.key"]


def _tracked() -> list[str]:
    return [f for f in subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n") if f]


def test_no_secret_shaped_strings_in_tracked_files():
    problems = []
    for f in _tracked():
        path = ROOT / f
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary (e.g. the favicon)
        for label, rx in SHAPES.items():
            if re.search(rx, text):
                problems.append(f"{f}: {label}")  # names the file and kind, never the value
    assert not problems, "secret-shaped strings in tracked files: " + "; ".join(problems)


def test_no_secret_files_are_tracked():
    bad = [f for f in _tracked() if FORBIDDEN_NAMES.search(f)]
    assert not bad, f"secret files tracked: {bad}"


def test_ignore_rules_cover_secret_files():
    missing = [p for p in MUST_BE_IGNORED if subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", p]).returncode != 0]
    assert not missing, f"not ignored: {missing}"
    # the placeholder templates must stay committable
    for p in (".env.example", ".streamlit/secrets.toml.example"):
        assert subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", p]).returncode != 0, f"{p} must not be ignored"


def test_scanner_catches_a_planted_key():
    fake = "gsk_" + "A1b2C3d4E5f6G7h8I9j0K1l2"  # assembled at runtime so this file itself stays clean
    assert re.search(SHAPES["Groq key"], f"KEY={fake}")
    fake_url = "postgresql://u:" + "hunter2pass" + "@host:5432/db"  # assembled at runtime: a literal here would flag this very file
    assert re.search(SHAPES["database URL with a real password"], fake_url)
    assert not re.search(SHAPES["database URL with a real password"], "postgresql://postgres:[password]@[host]:5432/postgres")
    assert not re.search(SHAPES["database URL with a real password"], 'postgresql://postgres.<project-ref>:<password>@aws-0.pooler.supabase.com:5432/postgres')


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
