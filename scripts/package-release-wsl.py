#!/usr/bin/env python3
"""Package the Windows release for WSL2 from a container export of its Linux system.

    docker export CONTAINER | python3 scripts/package-release-wsl.py

scripts/qwen38/build-release.sh runs this in the release container after scripts/package-release.sh.
The export on stdin is the release's WSL distribution: Ubuntu 24.04 with curl, the Linux release
unpacked at /opt/ninfer-5060-qwen and scripts/qwen38/windows/package/wsl.conf as /etc/wsl.conf. It
becomes dist/ninfer-5060-qwen-<tag>-windows-wsl-x64.zip with its .sha256: Install.cmd,
Start-Qwen38.cmd and Uninstall.cmd with their scripts, the README (docs/release-archive-wsl.md), the
licences and the distribution as ninfer-5060-qwen-<tag>-wsl.tar.gz for `wsl --import`.
"""

import hashlib
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Docker manages these per container; WSL writes its own when the distribution starts.
DROPPED = {".dockerenv", "etc/hostname", "etc/hosts", "etc/resolv.conf"}
# Files whose absence means the export is not the release's distribution.
REQUIRED = {
    "etc/wsl.conf",
    "opt/ninfer-5060-qwen/bin/ninfer-serve",
    "opt/ninfer-5060-qwen/serve.sh",
    "opt/ninfer-5060-qwen/download-model.sh",
    "opt/ninfer-5060-qwen/windows/Start-Qwen38.ps1",
    "usr/bin/curl",
}
WINDOWS = REPO / "scripts/qwen38/windows/package"


def fail(message: str) -> None:
    sys.exit(f"package-release-wsl.py: {message}")


def write_distribution(source, target: Path) -> None:
    seen = set()
    with tarfile.open(fileobj=source, mode="r|") as export, \
            tarfile.open(target, mode="w:gz", compresslevel=6, format=tarfile.PAX_FORMAT) as rootfs:
        for member in export:
            name = member.name.removeprefix("./")
            if name in DROPPED:
                continue
            seen.add(name)
            rootfs.addfile(member, export.extractfile(member) if member.isfile() else None)
    missing = sorted(REQUIRED - seen)
    if missing:
        fail(f"the export lacks {', '.join(missing)}")


def crlf(source: Path, target: Path) -> None:
    target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    tag = (REPO / "VERSION").read_text().strip()
    notes = f"RELEASE_NOTES_{tag}.md"
    dist = REPO / "dist"
    linux = dist / f"ninfer-5060-qwen-{tag}-linux-x64"
    if not (linux / "licenses").is_dir():
        fail(f"missing {linux}: run scripts/package-release.sh first")
    name = f"ninfer-5060-qwen-{tag}-windows-wsl-x64"
    product, archive = dist / name, dist / f"{name}.zip"
    checksum = dist / f"{name}.zip.sha256"
    # The checksum goes first and is written last, so a failed run leaves no stale pair to publish.
    for stale in (checksum, archive):
        stale.unlink(missing_ok=True)
    shutil.rmtree(product, ignore_errors=True)
    (product / "scripts").mkdir(parents=True)

    write_distribution(sys.stdin.buffer, product / f"ninfer-5060-qwen-{tag}-wsl.tar.gz")
    for script in ("Install.cmd", "Start-Qwen38.cmd", "Uninstall.cmd"):
        crlf(WINDOWS / script, product / script)
    for script in ("Install.ps1", "Uninstall.ps1"):
        crlf(WINDOWS / script, product / "scripts" / script)
    shutil.copyfile(REPO / "docs/release-archive-wsl.md", product / "README.md")
    for file in ("VERSION", "LICENSE", notes):
        shutil.copyfile(REPO / file, product / file)
    shutil.copytree(linux / "licenses", product / "licenses")

    files = sorted(path for path in product.rglob("*") if path.is_file())
    (product / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(product).as_posix()}\n" for path in files))
    files.append(product / "SHA256SUMS.txt")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(files):
            # The distribution is gzip already; deflating it again only costs time.
            stored = path.suffix == ".gz"
            bundle.write(path, f"{name}/{path.relative_to(product).as_posix()}",
                         compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n")
    print(f"{archive.stat().st_size} bytes  {archive}")
    print(checksum.read_text(), end="")


if __name__ == "__main__":
    main()
