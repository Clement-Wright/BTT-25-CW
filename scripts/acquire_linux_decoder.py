"""Acquire the tested Linux x86-64 CPython 3.12 Flashlight wheel by hash."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from acquire_extras import fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    digest = '6570039ce70429cecb84ea3c6bae212607c5f0b97c0cd769aed7276ad4a70361'
    with urlopen('https://pypi.org/pypi/flashlight-text/0.0.7/json', timeout=30) as response:
        entries = json.load(response)['urls']
    wheel, = [row for row in entries if row['digests']['sha256'] == digest]
    if wheel['size'] != 1_304_673 or not wheel['filename'].startswith('flashlight_text-0.0.7-cp312-cp312-manylinux'):
        raise ValueError('The pinned Linux wheel metadata changed')
    print(json.dumps(fetch({'url': wheel['url'], 'sha256': digest, 'bytes': wheel['size']}, args.output)))


if __name__ == '__main__':
    main()
