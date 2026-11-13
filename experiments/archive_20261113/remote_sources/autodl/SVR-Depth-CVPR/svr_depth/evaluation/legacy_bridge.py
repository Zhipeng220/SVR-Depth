"""Stage the three exact manifest files required by historical experiments."""
from pathlib import Path
from svr_depth.utils.provenance import sha256_file

PAIR_HASHES = {
    'crowd2': 'f7bf9226edcedf6c05d63938e29f92bf9364878c9291dbd79172c973c296f350',
    'crowd3': '9f4da8360b8f6b3039330dc7fe9dba6dd7219d9427e774b89006b09269fa17ad',
    'balloon2': 'afbd46ebed192aaf69f580822993fb7c756c5656de0595f68143993859918b06',
}

def stage_locked_pairs(runtime: Path, source: Path) -> None:
    source = source.expanduser().resolve()
    destination = runtime / 'outputs/gt4a2_exp1'
    files = []
    for sequence, expected in PAIR_HASHES.items():
        name = f'gt4a2_exp1_{sequence}_pairs_used.csv'
        src, dst = source / name, destination / name
        if not src.is_file() or sha256_file(src) != expected:
            raise ValueError(f'Missing or changed historically locked manifest: {src}')
        if dst.exists() and (not dst.is_file() or sha256_file(dst) != expected):
            raise RuntimeError(f'Refusing to replace compatibility manifest: {dst}')
        files.append((src, dst))
    destination.mkdir(parents=True, exist_ok=True)
    for src, dst in files:
        if not dst.exists():
            with dst.open('xb') as f:
                f.write(src.read_bytes())
