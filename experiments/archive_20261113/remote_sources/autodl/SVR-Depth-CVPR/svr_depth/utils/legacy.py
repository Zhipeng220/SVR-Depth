"""Materialize the byte-preserved historical runtime into an output directory.

This compatibility layer keeps old source-hash-locked experiments separate
from the public API while their replacements are validated.
"""
from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

COMPAT = Path(__file__).resolve().parents[1] / '_compat'
ARCHIVE = COMPAT / 'historical_m04_m05_snapshot.zip'

def _matches_after_short_wait(path: Path, data: bytes) -> bool:
    expected = hashlib.sha256(data).digest()
    # A concurrent launcher may have created the file but not finished writing.
    for _ in range(40):
        if path.is_file() and hashlib.sha256(path.read_bytes()).digest() == expected:
            return True
        time.sleep(0.05)
    return False

def materialize() -> Path:
    configured = os.environ.get('SVR_DEPTH_RUNTIME_DIR')
    target = Path(configured).expanduser().resolve() if configured else (Path.cwd() / 'outputs' / 'legacy_runtime').resolve()
    prefix = 'SVR-Depth-M04-M05'
    with ZipFile(ARCHIVE) as archive:
        entries = []
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.is_dir():
                continue
            if not parts or parts[0] != prefix or any(p in ('..', '.') for p in parts):
                raise ValueError(f'Invalid compatibility archive member: {info.filename}')
            dst = target.joinpath(*parts[1:])
            if not dst.resolve().is_relative_to(target.resolve()):
                raise ValueError('Archive member escapes compatibility directory')
            data = archive.read(info)
            if dst.exists() and not _matches_after_short_wait(dst, data):
                raise RuntimeError(f'Compatibility source changed; refusing overwrite: {dst}')
            entries.append((dst, data))
        for dst, data in entries:
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with dst.open('xb') as stream:
                        stream.write(data)
                except FileExistsError:
                    if not _matches_after_short_wait(dst, data):
                        raise RuntimeError(f'Compatibility source changed: {dst}')
    return target
