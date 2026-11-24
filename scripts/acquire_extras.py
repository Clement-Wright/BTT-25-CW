"""Fetch pinned dictionary, metadata and optional scorer; reuse verified files."""
import argparse
import json
from pathlib import Path
import shutil
import time
import urllib.request

from btt25.assets import sha256
from btt25.data import write_json


def fetch(row, output):
    output=Path(output)
    if output.exists():
        if output.stat().st_size!=row['bytes'] or sha256(output)!=row['sha256']:
            raise ValueError(f'Existing asset differs from pinned source: {output}')
        return {'file':output.name,'reused':True,'seconds':0,'bytes_downloaded':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.download')
    started=time.monotonic()
    request=urllib.request.Request(row.get('url',row.get('source_url')),headers={'User-Agent':'BTT-25-CW/0.1'})
    with urllib.request.urlopen(request,timeout=120) as response,temporary.open('wb') as stream:
        shutil.copyfileobj(response,stream,length=1024*1024)
    if temporary.stat().st_size!=row['bytes'] or sha256(temporary)!=row['sha256']:
        raise ValueError(f'Downloaded asset checksum differs: {output.name}')
    temporary.replace(output)
    return {'file':output.name,'reused':False,'seconds':time.monotonic()-started,'bytes_downloaded':row['bytes']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir',default='assets/local')
    parser.add_argument('--with-scorer',action='store_true')
    args=parser.parse_args()
    catalog=json.loads(Path('assets/external_assets.json').read_text(encoding='utf-8'))
    output=Path(args.asset_dir)
    events=[]
    for name,row in catalog['cmudict']['files'].items():
        events.append(fetch(row,output/'cmudict'/name))
    write_json(output/'cmudict/manifest.json',catalog['cmudict'])
    events.append(fetch(catalog['corpus_metadata'],output/catalog['corpus_metadata']['filename']))
    if args.with_scorer:
        for row in catalog['distilgpt2']['files']:
            events.append(fetch(row,output/'distilgpt2'/row['file']))
        write_json(output/'distilgpt2/manifest.json',catalog['distilgpt2'])
    report={'scope':'First-time setup; excluded from bounded scientific job timing','files':events,
            'download_seconds':sum(row['seconds'] for row in events),
            'download_bytes':sum(row['bytes_downloaded'] for row in events)}
    write_json(output/'extras_acquisition.json',report)
    print(json.dumps(report))


if __name__=='__main__': main()
