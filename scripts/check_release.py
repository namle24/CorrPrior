"""Bounded source, link and artifact audit for the standalone repository."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {"data", "checkpoints", "runs", "private_runs", "manuscript", "paper", ".aws", ".ssh"}
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".state", ".safetensors", ".onnx", ".zip", ".tar", ".gz", ".tex", ".docx", ".pdf", ".pem", ".key"}
SECRET_PATTERNS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{60,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def source_files():
    if (ROOT / ".git").exists():
        result = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        return [ROOT / name for name in sorted(set(result.decode().split("\0"))) if name]
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache"}
    return [p for p in ROOT.rglob("*") if p.is_file()
            and p.relative_to(ROOT).parts[0] not in FORBIDDEN_PARTS
            and not (set(p.relative_to(ROOT).parts) & excluded)]


def audit():
    errors = []
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            errors.append(f"Missing retained source: {record['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["exported_sha256"]:
            errors.append(f"Changed retained source: {record['path']} (review and update provenance)")
    files = source_files()
    for path in files:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"Symlink requires review: {relative}")
            continue
        if (relative.parts[0] in FORBIDDEN_PARTS
                or set(relative.parts) & {".aws", ".ssh"}
                or path.suffix.lower() in FORBIDDEN_SUFFIXES):
            errors.append(f"Forbidden release artifact: {relative}")
        if path.stat().st_size > 2 * 1024 * 1024:
            errors.append(f"Unexpected file larger than 2 MiB: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"Unexpected binary file: {relative}")
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"Possible credential in {relative}; value not printed")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(relative))
            except SyntaxError as error:
                errors.append(f"Python syntax error in {relative}: line {error.lineno}")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                target = target.split("#", 1)[0]
                if not target or re.match(r"[a-z]+:", target):
                    continue
                if not (path.parent / target).exists():
                    errors.append(f"Broken local link in {relative}: {target}")
    return errors, len(files), len(manifest["files"])


if __name__ == "__main__":
    failures, count, retained = audit()
    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: {count} source files audited; {retained} retained-source hashes match.")
    print("PASS: Python syntax and local Markdown links; no flagged release artifacts or credential patterns.")
    print("This bounded check does not certify complete reproducibility or an exhaustive security review.")
