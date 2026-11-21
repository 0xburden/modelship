#!/usr/bin/env python3
"""Package the exact GPU-tested binaries; never build or download a substitute."""
import argparse
import gzip
import hashlib
import stat
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'generation': (
        ('modelship-sd-worker', 115618872, '5e1e684468576b74040a7c201eff289452511136f34477c9f9fbdcfafdbba1e1'),
        ('modelship-sd-probe', 72735736, '92ff72e75d53e502c0e5dffd1cf618201c28eb34a229b44775134d87b385c134'),
    ),
    'edit': (
        ('modelship-sd-edit-worker', 115614808, '519b0f987681f72da95d75df2d40e801cb23490b5a840d417510399fedabdaa2'),
        ('modelship-sd-probe', 72731640, 'e5f7357f4e920d336d04215c71cd605e37459de62f23ae5735a5724701352c3e'),
    ),
}


def package(profile, source, output):
    name = f'sd-532d2c1-u2204-cuda12.8-{profile}-r1.tar.gz'
    artifact = output / name
    binaries = PROFILES[profile]
    for filename, size, digest in binaries:
        path = source / filename
        info = path.lstat()
        with path.open('rb') as stream:
            if not stat.S_ISREG(info.st_mode) or info.st_size != size or hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                raise SystemExit(f'{filename} does not match the GPU-tested binary')
    notice = ROOT / 'releases/diffusion/THIRD_PARTY_NOTICES.txt'
    with artifact.open('xb') as archive:
        try:
            with gzip.GzipFile(fileobj=archive, filename='', mode='wb', mtime=0, compresslevel=6) as zipped:
                with tarfile.open(fileobj=zipped, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
                    for path, mode in [(notice, 0o400)] + [(source / p[0], 0o500) for p in binaries]:
                        entry = tarfile.TarInfo(path.name)
                        entry.size, entry.mode = path.stat().st_size, mode
                        with path.open('rb') as stream:
                            tar.addfile(entry, stream)
        except BaseException:
            artifact.unlink()
            raise
    # Check the archived bytes too, not just the input paths checked before packaging.
    with tarfile.open(artifact, mode='r:gz') as tar:
        assert [m.name for m in tar] == [notice.name] + [p[0] for p in binaries]
        for filename, size, digest in binaries:
            member = tar.getmember(filename)
            with tar.extractfile(member) as stream:
                if not member.isfile() or member.size != size or hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    artifact.unlink()
                    raise SystemExit(f'archived runtime bytes do not match: {filename}')
    with artifact.open('rb') as f:
        print(hashlib.file_digest(f, 'sha256').hexdigest(), artifact.stat().st_size, artifact.name)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('profile', choices=PROFILES)
    cli.add_argument('source', type=Path, help='directory containing the exact reviewed executable pair')
    cli.add_argument('output', type=Path, help='existing output directory')
    args = cli.parse_args()
    package(args.profile, args.source, args.output)
