"""The Hugging Face cache, shared by every runtime: hub/models--<owner>--<name>/{blobs,snapshots/<revision>}/.

download_file writes the same layout huggingface_hub does, from a pinned revision, and checks every file
against the SHA-256 that Hugging Face publishes for it before linking it into the snapshot.
snapshot_problems and cli_download serve the runtimes that download a whole snapshot with an image's hf CLI.
build_overlay builds, aside and then renamed into place, a folder of links into a snapshot plus a few real files
that a runtime mounts in the snapshot's place.
The token goes only into a request header (download_file) or into docker run's environment or --env-file
(cli_download); it never appears in an argv and is never printed.
"""

import errno
import hashlib
import http.client
import os
import re
import shutil
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


@dataclass(frozen=True)
class HfRepoModel:
    repo: str
    revision: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    size_bytes: int
    served_name: str


def snapshot_problems(model: HfRepoModel, home: Path) -> list[str]:
    """Complete means every link resolves and the files add up to the recipe's size; hf links small files first."""
    snapshot = snapshot_dir(home, model.repo, model.revision)
    if not snapshot.is_dir():
        return [f"missing snapshot {model.revision} of {model.repo}"]
    entries = [path for path in snapshot.rglob("*") if path.is_symlink() or path.is_file()]
    dangling = sorted(path.name for path in entries if not path.exists())
    if dangling:
        return [f"the download is incomplete: {', '.join(dangling[:3])} missing in snapshot {model.revision}"]
    if not (snapshot / "config.json").is_file():
        return [f"missing config.json in snapshot {model.revision}"]
    total = sum(path.stat().st_size for path in entries)
    if total != model.size_bytes:
        return [f"the snapshot holds {total} bytes, the recipe expects {model.size_bytes}; the download is incomplete"]
    return []


def cli_download(host, recipe_id: str, model: HfRepoModel, image: str, entrypoint: str, command: tuple[str, ...],
                 home: Path, auth: HfAuth, out) -> None:
    """Download the repository at its pinned revision with the hf CLI from a pinned runtime image."""
    uid, gid = host.uid_gid()
    argv = ["docker", "run", "--rm", "--user", f"{uid}:{gid}", "-e", "HOME=/tmp", "-e", "HF_HOME=/hf"]
    env_extra = {}
    if auth.env_file is not None:
        argv += ["--env-file", str(auth.env_file)]
    elif auth.token:
        argv += ["-e", "HF_TOKEN"]
        env_extra["HF_TOKEN"] = auth.token
    argv += ["-v", f"{home}:/hf", "--memory", "8g", "--memory-swap", "8g",
             "--entrypoint", entrypoint, image, *command, "download", model.repo, "--revision", model.revision]
    for pattern in model.include:
        argv += ["--include", pattern]
    for pattern in model.exclude:
        argv += ["--exclude", pattern]
    out.write(f"{recipe_id}: downloading {model.repo} at {model.revision[:12]} with the hf CLI\n")
    result = host.run(argv, env_extra=env_extra or None)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-5:])
        raise DownloadError(f"{recipe_id}: hf download exited {result.returncode}:\n{tail}")
    problems = snapshot_problems(model, home)
    if problems:
        raise DownloadError(f"{recipe_id}: " + "; ".join(problems))


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.unlink(path)


def _raise(error: OSError) -> None:
    raise error


def build_overlay(folder: Path, snapshot: Path, link_root: str, files: dict[str, bytes]) -> int:
    """Rebuild folder so that each key of files is a real top-level file holding its bytes, in place of any
    snapshot entry of that name, and every other file of snapshot is a link to f"{link_root}/<path>"; return the
    number of links. The live folder is never edited in place, and every failure is the OSError that caused it."""
    if not snapshot.is_dir():
        raise FileNotFoundError(errno.ENOENT, "the snapshot folder does not exist", str(snapshot))
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = folder.parent / f".{folder.name}.llmctl-tmp"
    old = folder.parent / f".{folder.name}.llmctl-old"
    _remove(tmp)
    _remove(old)
    tmp.mkdir()
    links = 0
    for dirpath, dirnames, filenames in os.walk(snapshot, onerror=_raise):
        dirnames.sort()
        filenames.sort()
        relative = Path(dirpath).relative_to(snapshot)
        for name in dirnames:
            (tmp / relative / name).mkdir()
        for name in filenames:
            if relative == Path(".") and name in files:
                continue
            path = relative / name
            os.symlink(f"{link_root}/{path.as_posix()}", tmp / path)
            links += 1
    for name in sorted(files):
        path = tmp / name
        with open(path, "wb") as handle:
            handle.write(files[name])
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o644)
    if os.path.lexists(folder):
        os.rename(folder, old)
    os.rename(tmp, folder)
    _remove(old)
    return links


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
