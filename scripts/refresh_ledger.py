"""Refresh the local evidence ledger without opening dataset labels."""
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def main():
    root=Path('runs')
    rows=[]
    mirror=root/'gilbreth/workspace/runs'
    directories=[*sorted(root.iterdir()),*sorted(mirror.iterdir())] if mirror.exists() else sorted(root.iterdir())
    entries=[]
    for directory in directories:
        entries.append((directory.name,directory))
        for condition in sorted((directory/'A').glob('A*')):
            entries.append((f'{directory.name}/dev_report/{condition.name}',condition))
        if (directory/'submission.csv').is_file():
            entries.append((f'{directory.name}/test_decode',directory/'decode'))
        for trial in sorted(directory.glob('weight-*')):
            for stage in ('decode','dev_tune'):
                entries.append((f'{directory.name}/{trial.name}/{stage}',trial/stage))
        for split in ('dev_report','official_val'):
            for condition in sorted((directory/split).glob('*')):
                entries.append((f'{directory.name}/{split}/{condition.name}',condition))
    for run_id,directory in entries:
        if not directory.is_dir(): continue
        state=read(directory/'run_manifest.json')
        metric=read(directory/'metrics.json')
        resources=read(directory/'resources.json')
        pipeline=read(directory/'pipeline_manifest.json')
        if pipeline:
            state=pipeline
            metric=read(directory/'evaluate/metrics.json')
            resources=read(directory/'supervisor.json')
        if not state and not metric and not resources: continue
        predictions=directory/('decode/predictions.jsonl' if pipeline else 'predictions.jsonl')
        prediction_count=state.get('saved_trials',metric.get('evaluated_count',0))
        if predictions.exists():
            with predictions.open(encoding='utf-8') as stream:
                prediction_count=sum(1 for line in stream if line.strip())
        notes=state.get('error',state.get('reason',metric.get('scope','')))
        rows.append({'run_id':run_id,'stage':state.get('stage',state.get('mode',state.get('decoder_mode',metric.get('score_split','evaluation')))),
                     'status':state.get('state',resources.get('outcome','scored')),
                     'artifact':directory.as_posix(),
                     'elapsed_seconds':resources.get('elapsed_seconds',''),
                     'predictions':prediction_count,
                     'per':metric.get('per',''),'wer':metric.get('wer',''),'notes':notes})
    with (root/'ledger.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['run_id','stage','status','artifact','elapsed_seconds','predictions','per','wer','notes'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} evidence rows to runs/ledger.csv')


if __name__=='__main__': main()
