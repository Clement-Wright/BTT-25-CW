"""Extract and verify exactly the released HDF5 files, without reading labels."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import zipfile


def check_existing(path, row):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return path.stat().st_size == row['bytes'] and digest.hexdigest() == row['sha256']


def expand(archive, root, hashes):
    root = Path(root)
    start = time.monotonic()
    expected = {(row['session_id'], row['file']):row for row in json.loads(Path(hashes).read_text())['files']}
    with zipfile.ZipFile(archive) as release:
        inputs = {}
        for info in release.infolist():
            parts = PurePosixPath(info.filename).parts
            if len(parts)<3 or parts[-3]!='hdf5_data_final' or parts[-2:] not in expected:
                continue
            key = parts[-2:]
            if key in inputs or info.file_size != expected[key]['bytes']:
                raise ValueError(f'Unexpected ZIP entry: {info.filename}')
            inputs[key] = info
        if inputs.keys()!=expected.keys():
            raise ValueError(f'Archive contains {len(inputs)} verified file names of {len(expected)} required')
        for key, row in expected.items():
            session, filename = key
            if not session.startswith('t15.') or PurePosixPath(session).name!=session or filename not in ('data_train.hdf5','data_val.hdf5','data_test.hdf5'):
                raise ValueError(f'Unsafe file receipt entry: {key}')
            destination = root/session/filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                if not check_existing(destination,row):
                    raise ValueError(f'Staged HDF5 file differs: {key}')
                continue
            partial = destination.with_suffix('.hdf5.partial')
            digest = hashlib.sha256()
            with release.open(inputs[key]) as src, partial.open('wb') as output:
                for block in iter(lambda: src.read(1024*1024), b''):
                    digest.update(block)
                    output.write(block)
            if partial.stat().st_size!=row['bytes'] or digest.hexdigest()!=row['sha256']:
                raise ValueError(f'Expanded HDF5 bytes differ: {key}')
            partial.replace(destination)
    return {'files':len(expected), 'bytes':sum(row['bytes'] for row in expected.values()),
            'elapsed_seconds':time.monotonic()-start, 'references_loaded':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--hashes', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args=parser.parse_args()
    result=expand(args.archive,args.data_root,args.hashes)
    temporary=args.receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    temporary.replace(args.receipt)
    print(json.dumps(result))


if __name__=='__main__':
    main()
