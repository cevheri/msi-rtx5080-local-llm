"""The Hugging Face cache, shared by every runtime: hub/models--<owner>--<name>/{blobs,snapshots/<revision>}/.

download_file writes the same layout huggingface_hub does, from a pinned revision, and checks every file
against the SHA-256 that Hugging Face publishes for it before linking it into the snapshot.
The token is only ever sent in a request header; it is never printed.
"""

import hashlib
import http.client
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from launcher.errors import DownloadError, UsageError

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1 << 20


@dataclass(frozen=True)
class HfAuth:
    token: str | None
    env_file: Path | None


def hf_home(env) -> Path:
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]).expanduser()
    return Path(env.get("HOME", "~")).expanduser() / ".cache" / "huggingface"


def repo_dirname(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def repo_dir(home: Path, repo: str) -> Path:
    return home / "hub" / repo_dirname(repo)


def snapshot_dir(home: Path, repo: str, revision: str) -> Path:
    return repo_dir(home, repo) / "snapshots" / revision


def read_auth(env, env_file: Path | None) -> HfAuth:
    """The token from --hf-env-file (HF_TOKEN=... lines, as docker --env-file reads them) or from HF_TOKEN."""
    if env_file is None:
        return HfAuth(env.get("HF_TOKEN") or None, None)
    if not env_file.is_file():
        raise UsageError(f"--hf-env-file {env_file} does not exist")
    for line in env_file.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip()
            if token:
                return HfAuth(token, env_file)
    raise UsageError(f"no HF_TOKEN= line in {env_file}")


def _gib(size: int) -> str:
    return f"{size / 2**30:.2f} GiB"


def download_file(host, home: Path, repo: str, revision: str, filename: str, token: str | None,
                  out: TextIO) -> Path:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    headers = {"User-Agent": "llmctl"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        head = host.http("HEAD", url, headers=headers, follow_redirects=False, timeout=30)
    except (OSError, http.client.HTTPException) as exc:
        raise DownloadError(f"{filename}: could not reach Hugging Face ({exc}); check the network and pull again") from None
    if head.status in (401, 403):
        raise DownloadError(f"{repo} refused the download (HTTP {head.status}); a gated model needs a Hugging Face "
                            "token: export HF_TOKEN or pass --hf-env-file")
    if head.status == 404:
        raise DownloadError(f"{filename} does not exist in {repo} at revision {revision}")
    if head.status not in (200, 301, 302, 303, 307, 308):
        raise DownloadError(f"Hugging Face answered HTTP {head.status} for {url}")
    sha256 = head.headers.get("x-linked-etag", "").strip('"')
    if not _SHA256.fullmatch(sha256):
        raise DownloadError(f"Hugging Face gave no SHA-256 for {filename}; it is not stored with Git LFS, "
                            "so llmctl cannot verify it")
    size = int(head.headers.get("x-linked-size") or 0)

    blobs = repo_dir(home, repo) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    blob = blobs / sha256
    if not (blob.is_file() and blob.stat().st_size == size):
        partial = blobs / f"{sha256}.incomplete"
        hasher = hashlib.sha256()
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            partial.unlink()
            offset = 0
        if offset:
            with open(partial, "rb") as handle:
                while chunk := handle.read(_CHUNK):
                    hasher.update(chunk)
            out.write(f"{filename}: resuming at {_gib(offset)} of {_gib(size)}\n")
        else:
            out.write(f"{filename}: downloading {_gib(size)}\n")
        get_headers = dict(headers)
        if offset:
            get_headers["Range"] = f"bytes={offset}-"
        try:
            host.download(url, get_headers, partial, offset, hasher.update)
        except (OSError, http.client.HTTPException) as exc:
            raise DownloadError(f"{filename}: the download stopped ({exc}); the partial file is kept, so running "
                                "./llmctl pull again resumes it") from None
        if partial.stat().st_size != size or hasher.hexdigest() != sha256:
            partial.unlink()
            raise DownloadError(f"{filename}: the checksum or size does not match what Hugging Face publishes; "
                                "the partial file was deleted, pull again")
        partial.rename(blob)
    link = snapshot_dir(home, repo, revision) / filename
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(os.path.relpath(blob, link.parent), link)
    out.write(f"{filename}: ok ({sha256[:12]})\n")
    return link
