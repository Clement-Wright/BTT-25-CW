"""Verify a staged T15 release against its expanded-file receipts, without labels."""
import argparse
import json
from pathlib import Path
import time

from btt25.assets import sha256
from btt25.data import write_json


def verify(root, manifest_path):
    root = Path(root).resolve()
    manifest_path = Path(manifest_path)
    expected = {}
    for row in json.loads(manifest_path.read_text(encoding='utf-8'))['files']:
        key = (row['session_id'], row['file'])
        if key in expected or not key[0].startswith('t15.') or Path(key[0]).name != key[0] or key[1] not in (
                'data_train.hdf5', 'data_val.hdf5', 'data_test.hdf5'):
            raise ValueError(f'Invalid or duplicate release receipt entry: {key}')
        expected[key] = row
    present = {(p.parent.name, p.name) for p in root.glob('t15.*/data_*.hdf5')}
    if present != expected.keys():
        raise ValueError(f'Release file coverage mismatch: {len(present)} present, {len(expected)} expected')
    start = time.monotonic()
    total = 0
    for key, row in sorted(expected.items()):
        path = root.joinpath(*key).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError(f'Release checksum or size mismatch: {key}')
        total += row['bytes']
    return {'files':len(expected), 'bytes':total, 'original_receipt_sha256':sha256(manifest_path),
            'verification_seconds':time.monotonic()-start, 'references_loaded':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--hashes', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    receipt = verify(args.data_root, args.hashes)
    write_json(args.output, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
